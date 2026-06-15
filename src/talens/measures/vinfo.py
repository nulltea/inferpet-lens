"""V-usable information and pointwise V-information (PVI).

Ethayarajh, Choi & Swayamdipta (ICML 2022); Xu et al. (ICLR 2020).
For representation ``X`` (the exposed operand) and secret ``Y`` (the
token id), with predictive family ``V`` = the softmax probe:

    PVI(x → y) = log₂ q[X](y | x) − log₂ q[∅](y)
    I_V(X → Y) = mean PVI = H_V(Y | ∅) − H_V(Y | X)   (bits)

``q[X]`` is the probe trained on the representation; ``q[∅]`` is the
class prior (the null model). Higher I_V means more token-identity
information a *bounded* adversary can use — the quantity hypothesised to
predict inversion-attack recovery.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._probe import (
    class_log_prior,
    probe_log_softmax,
    row_split,
    to_class_indices,
    train_softmax_probe,
)

_LN2 = np.log(2.0)


def v_information(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_frac: float = 0.7,
    max_classes: int = 4096,
    l2: float = 1e-4,
    max_iter: int = 100,
    seed: int = 20260615,
    return_pvi: bool = False,
) -> dict[str, Any]:
    """Estimate I_V(X→Y) in bits with a softmax-probe family. If the
    corpus has more than ``max_classes`` distinct ids, restricts to the
    most frequent ``max_classes`` (rows of rarer ids are dropped).
    """
    if X.shape[0] < 4:
        return {"v_information_bits": None, "note": "too few rows"}

    y_idx_all, classes = to_class_indices(y)
    if classes.size > max_classes:
        counts = np.bincount(y_idx_all, minlength=classes.size)
        keep = np.argsort(counts)[::-1][:max_classes]
        keep_mask = np.isin(y_idx_all, keep)
        X, y = X[keep_mask], y[keep_mask]
        y_idx_all, classes = to_class_indices(y)
    num_classes = int(classes.size)

    tr, te = row_split(X.shape[0], train_frac, seed)
    if tr.size == 0 or te.size == 0:
        return {"v_information_bits": None, "note": "empty split"}

    probe = train_softmax_probe(
        X[tr], y_idx_all[tr], num_classes, l2=l2, max_iter=max_iter, seed=seed
    )
    logq = probe_log_softmax(probe, X[te])          # (n_te, C) natural log
    log_prior = class_log_prior(y_idx_all[tr], num_classes)  # (C,)

    yte = y_idx_all[te]
    cond_nats = logq[np.arange(yte.size), yte]      # log q[X](y|x)
    prior_nats = log_prior[yte]                      # log q[∅](y)
    pvi_bits = (cond_nats - prior_nats) / _LN2

    out: dict[str, Any] = {
        "v_information_bits": float(pvi_bits.mean()),
        "h_y_given_x_bits": float(-cond_nats.mean() / _LN2),
        "h_y_prior_bits": float(-prior_nats.mean() / _LN2),
        "num_classes": num_classes,
        "n_train": int(tr.size),
        "n_test": int(te.size),
    }
    if return_pvi:
        out["pvi_bits"] = pvi_bits
    return out


def v_information_retrieval(
    X: np.ndarray,
    y: np.ndarray,
    embed_table: "torch.Tensor",
    *,
    n_train: int = 1024,
    n_val: int = 128,
    n_test: int = 128,
    ridge_alpha: float = 1e-2,
    candidate_pool_size: int = 2048,
    seed: int = 20260615,
    split_mode: str = "vocab",
    return_pvi: bool = False,
) -> dict[str, Any]:
    """V-information under the **retrieval family** (resolution B).

    The predictive family is the inversion attack itself: a ridge
    ``X→embedding`` map with a softmax over cosine similarity to a
    candidate pool. Generalises to unseen ids, so it runs **vocab-disjoint**
    (default) — the same regime as the honest attack. PVI here *is* the
    attack's pointwise usable information. See ``docs/research/attacks_setting.md``.
    """
    import torch

    from ..splits import train_val_test_split, vocab_disjoint_train_val_test_split
    from ._retrieval import (
        build_candidate_pool,
        fit_inverter,
        log_prob_true,
        pick_temperature,
        predict_embeddings,
    )

    if X.shape[0] < 6:
        return {"v_information_bits": None, "note": "too few rows"}

    splitter = (
        vocab_disjoint_train_val_test_split if split_mode == "vocab" else train_val_test_split
    )
    Xtr, ytr, Xva, yva, Xte, yte = splitter(
        X, y, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed
    )
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0 or Xva.shape[0] == 0:
        return {"v_information_bits": None, "note": "empty split"}

    model = fit_inverter(Xtr, ytr, embed_table, ridge_alpha=ridge_alpha)
    pool = build_candidate_pool(
        yva, yte, vocab_size=embed_table.shape[0], pool_size=candidate_pool_size, seed=seed
    )
    yva_t, yte_t = torch.from_numpy(yva), torch.from_numpy(yte)

    tau = pick_temperature(predict_embeddings(model, Xva), yva_t, pool, embed_table)

    pred_te = predict_embeddings(model, Xte)
    logq_x = log_prob_true(pred_te, yte_t, pool, embed_table, tau)        # log q[X](y|x)
    # Null: best input-free predictor = mean train target embedding.
    ebar = embed_table[torch.from_numpy(ytr)].to(torch.float32).mean(dim=0, keepdim=True)
    pred_null = ebar.expand(yte_t.shape[0], -1)
    logq_0 = log_prob_true(pred_null, yte_t, pool, embed_table, tau)      # log q[∅](y)

    pvi_bits = (logq_x - logq_0).numpy() / _LN2
    out: dict[str, Any] = {
        "v_information_bits": float(pvi_bits.mean()),
        "family": "retrieval",
        "temperature": float(tau),
        "candidate_pool_size": int(pool.shape[0]),
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "split_mode": split_mode,
    }
    if return_pvi:
        out["pvi_bits"] = pvi_bits
    return out
