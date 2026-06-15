"""Shared ridge-inversion core for the hidden-state and attention-score
attacks.

Both attacks are the same math at different observables: fit a ridge map
from the exposed operand ``X`` to the token embedding, pick the alpha by
validation top-1, then cosine-match the predicted embedding against a
candidate pool to read off the recovered token id (TTRSR).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..ridge import evaluate_inversion, fit_ridge, predict_ridge
from ..splits import train_val_test_split, vocab_disjoint_train_val_test_split


def ridge_inversion(
    X: np.ndarray,
    y: np.ndarray,
    embed_table: torch.Tensor,
    *,
    n_train: int = 1024,
    n_val: int = 128,
    n_test: int = 128,
    topk: int = 10,
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0),
    candidate_pool_size: int = 2048,
    seed: int = 20260615,
    split_mode: str = "vocab",
) -> dict[str, Any] | None:
    """Return a metrics dict (or ``None`` if there aren't enough rows to
    form a train/test split). The dict carries TTRSR top-1/top-10, the
    selected alpha + its validation scan, and the embedding cosine.
    """
    if X.shape[0] == 0:
        return None

    splitter = (
        vocab_disjoint_train_val_test_split
        if split_mode == "vocab"
        else train_val_test_split
    )
    Xtr, ytr, Xva, yva, Xte, yte = splitter(
        X, y, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed
    )
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        return None

    Xtr_t = torch.from_numpy(Xtr).to(torch.float32)
    Xva_t = torch.from_numpy(Xva).to(torch.float32)
    Xte_t = torch.from_numpy(Xte).to(torch.float32)
    ytr_emb = embed_table[torch.from_numpy(ytr)].to(torch.float32)
    yva_ids = torch.from_numpy(yva)
    yte_ids = torch.from_numpy(yte)

    rng = np.random.default_rng(seed + 1)
    vocab = embed_table.shape[0]
    pool = torch.from_numpy(
        rng.choice(vocab, size=min(candidate_pool_size, vocab), replace=False)
    ).to(torch.long)
    pool = torch.unique(torch.cat([pool, yva_ids, yte_ids]))

    best_alpha, best_val, best_model = None, -1.0, None
    alpha_scan: list[dict[str, float]] = []
    for alpha in ridge_alphas:
        model = fit_ridge(Xtr_t, ytr_emb, ridge_alpha=float(alpha))
        val_pred = predict_ridge(model, Xva_t)
        vm = evaluate_inversion(
            predicted_embeddings=val_pred,
            true_ids=yva_ids,
            candidate_ids=pool,
            embed_table=embed_table,
            topk=topk,
        )
        v = float(vm["token_top1_recovery_rate"])
        alpha_scan.append({"ridge_alpha": float(alpha), "val_top1": v})
        if v > best_val:
            best_val, best_alpha, best_model = v, float(alpha), model

    pred = predict_ridge(best_model, Xte_t)
    m = evaluate_inversion(
        predicted_embeddings=pred,
        true_ids=yte_ids,
        candidate_ids=pool,
        embed_table=embed_table,
        topk=topk,
    )
    return {
        "ttrsr_top1": float(m["token_top1_recovery_rate"]),
        "ttrsr_top10": float(m["token_top10_recovery_rate"]),
        "embedding_cosine_similarity": float(m["embedding_cosine_similarity"]),
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "best_ridge_alpha": best_alpha,
        "ridge_alpha_val_scan": alpha_scan,
        "candidate_pool_size": int(pool.shape[0]),
        "split_mode": split_mode,
    }
