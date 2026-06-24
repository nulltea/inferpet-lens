"""Shared inversion cores for the hidden-state / attention-score attacks.

Three inverters at one observable, sharing the same split + candidate-pool +
TTRSR evaluator so they are directly comparable on the same ``(X, y)`` rows:

* ``ridge_inversion`` — linear standardised ridge map ``X → token embedding``
  (the aloepri IMA/ISA attack; the default inverter).
* ``nn_inversion`` — cosine nearest-neighbour over *train activations*: predict
  the embedding of the nearest train row's token (the aloepri ``nn`` baseline).
  Under a vocab-disjoint split this reads the **generalisation floor** (the
  nearest train token is never a test token), so it bounds how much of any
  inverter's recovery is mere memorisation.
* ``learned_inversion`` — a learned 2-layer MLP head ``X → token embedding``
  (the aloepri ``ima_paper_like`` learned inverter; the per-position analog of
  a 2-layer transformer, since each row is a single-token sequence). Tests
  whether a non-linear inverter extracts more than linear ridge at depth.

All three pick on validation top-1, then cosine-match the predicted embedding
against a candidate pool to read off the recovered token id (TTRSR).
``control="shuffle"`` permutes the labels before the split (breaking X↔Y); the
inverter then collapses to the frequency floor (M1) — ``selectivity =
real − shuffled``. See ``docs/dev/control-tasks.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..ridge import evaluate_inversion, fit_ridge, predict_ridge
from ..splits import train_val_test_split, vocab_disjoint_train_val_test_split


def _split_pool(
    X: np.ndarray,
    y: np.ndarray,
    embed_table: torch.Tensor,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    split_mode: str,
    control: str,
    control_seed: int,
    candidate_pool_size: int,
    device: str | None,
):
    """Shared prep: optional label-shuffle control, train/val/test split, the
    candidate pool (random ids ∪ val ∪ test ids), and the on-device tensors.

    Returns ``None`` when there aren't enough rows; otherwise a dict with the
    device tensors (``Xtr_t``/``Xva_t``/``Xte_t``), the embedding targets
    (``ytr_emb``), the id arrays (``ytr``/``yva_ids``/``yte_ids``), the
    ``pool`` and the resolved ``dev``."""
    if X.shape[0] == 0:
        return None
    if control == "shuffle":
        y = y[np.random.default_rng(control_seed).permutation(y.shape[0])]

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

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtr_t = torch.from_numpy(Xtr).to(torch.float32).to(dev)
    Xva_t = torch.from_numpy(Xva).to(torch.float32).to(dev)
    Xte_t = torch.from_numpy(Xte).to(torch.float32).to(dev)
    ytr_emb = embed_table[torch.from_numpy(ytr)].to(torch.float32).to(dev)
    yva_ids = torch.from_numpy(yva)
    yte_ids = torch.from_numpy(yte)

    rng = np.random.default_rng(seed + 1)
    vocab = embed_table.shape[0]
    pool = torch.from_numpy(
        rng.choice(vocab, size=min(candidate_pool_size, vocab), replace=False)
    ).to(torch.long)
    pool = torch.unique(torch.cat([pool, yva_ids, yte_ids]))

    return {
        "Xtr_t": Xtr_t, "Xva_t": Xva_t, "Xte_t": Xte_t,
        "ytr": torch.from_numpy(ytr), "ytr_emb": ytr_emb,
        "yva_ids": yva_ids, "yte_ids": yte_ids,
        "pool": pool, "dev": dev,
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
    }


def _result(m_test: dict, *, n_train: int, n_test: int, split_mode: str,
            control: str, pool_size: int, extra: dict | None = None) -> dict:
    out = {
        "ttrsr_top1": float(m_test["token_top1_recovery_rate"]),
        "ttrsr_top10": float(m_test["token_top10_recovery_rate"]),
        "embedding_cosine_similarity": float(m_test["embedding_cosine_similarity"]),
        "n_train": n_train,
        "n_test": n_test,
        "candidate_pool_size": int(pool_size),
        "split_mode": split_mode,
        "control": control,
        "top1_hits": m_test.get("top1_hits"),
    }
    if extra:
        out.update(extra)
    return out


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
    control: str = "none",
    control_seed: int = 20260616,
    device: str | None = None,
) -> dict[str, Any] | None:
    """Linear ridge inverter. Returns a metrics dict (or ``None`` if there
    aren't enough rows) carrying TTRSR top-1/top-10, the selected alpha + its
    validation scan, and the embedding cosine."""
    sp = _split_pool(
        X, y, embed_table, n_train=n_train, n_val=n_val, n_test=n_test,
        seed=seed, split_mode=split_mode, control=control,
        control_seed=control_seed, candidate_pool_size=candidate_pool_size,
        device=device,
    )
    if sp is None:
        return None

    best_alpha, best_val, best_model = None, -1.0, None
    alpha_scan: list[dict[str, float]] = []
    for alpha in ridge_alphas:
        model = fit_ridge(sp["Xtr_t"], sp["ytr_emb"], ridge_alpha=float(alpha))
        val_pred = predict_ridge(model, sp["Xva_t"])
        vm = evaluate_inversion(
            predicted_embeddings=val_pred, true_ids=sp["yva_ids"],
            candidate_ids=sp["pool"], embed_table=embed_table, topk=topk,
        )
        v = float(vm["token_top1_recovery_rate"])
        alpha_scan.append({"ridge_alpha": float(alpha), "val_top1": v})
        if v > best_val:
            best_val, best_alpha, best_model = v, float(alpha), model

    pred = predict_ridge(best_model, sp["Xte_t"])
    m = evaluate_inversion(
        predicted_embeddings=pred, true_ids=sp["yte_ids"],
        candidate_ids=sp["pool"], embed_table=embed_table, topk=topk,
    )
    return _result(
        m, n_train=sp["n_train"], n_test=sp["n_test"], split_mode=split_mode,
        control=control, pool_size=sp["pool"].shape[0],
        extra={"best_ridge_alpha": best_alpha, "ridge_alpha_val_scan": alpha_scan},
    )


def nn_inversion(
    X: np.ndarray,
    y: np.ndarray,
    embed_table: torch.Tensor,
    *,
    n_train: int = 1024,
    n_val: int = 128,
    n_test: int = 128,
    topk: int = 10,
    candidate_pool_size: int = 2048,
    seed: int = 20260615,
    split_mode: str = "vocab",
    control: str = "none",
    control_seed: int = 20260616,
    device: str | None = None,
    **_ignored: Any,
) -> dict[str, Any] | None:
    """Cosine nearest-neighbour inverter (aloepri ``nn`` baseline). For each
    test row, take the cosine-nearest *train* activations and predict their
    token ids **directly** (top-1 = nearest neighbour's id; top-10 = true id
    among the 10 nearest neighbours' ids). This is the memorisation baseline:
    under a vocab-disjoint split the nearest train token is never a test token,
    so recovery is ~0 by construction — the floor ridge/learned must beat to
    claim *generalising* inversion. (Direct-id scoring, not the embedding-pool
    TTRSR, so embedding-space clustering can't lend nn spurious credit.)"""
    sp = _split_pool(
        X, y, embed_table, n_train=n_train, n_val=n_val, n_test=n_test,
        seed=seed, split_mode=split_mode, control=control,
        control_seed=control_seed, candidate_pool_size=candidate_pool_size,
        device=device,
    )
    if sp is None:
        return None
    Xtr_n = sp["Xtr_t"] / sp["Xtr_t"].norm(dim=1, keepdim=True).clamp_min(1e-8)
    Xte_n = sp["Xte_t"] / sp["Xte_t"].norm(dim=1, keepdim=True).clamp_min(1e-8)
    ytr_ids = sp["ytr"].to(sp["dev"])
    yte_ids = sp["yte_ids"].to(sp["dev"])
    sims = Xte_n @ Xtr_n.T  # [n_test, n_train] cosine
    k = min(topk, sims.shape[1])
    nbr_idx = torch.topk(sims, k=k, dim=1).indices  # [n_test, k]
    nbr_ids = ytr_ids[nbr_idx]  # predicted ids of the k nearest train neighbours
    true = yte_ids.unsqueeze(1)
    top1_hits = nbr_ids[:, 0].eq(yte_ids)
    top10_hits = nbr_ids.eq(true).any(dim=1)
    m = {
        "token_top1_recovery_rate": float(top1_hits.to(torch.float32).mean().item()),
        "token_top10_recovery_rate": float(top10_hits.to(torch.float32).mean().item()),
        "embedding_cosine_similarity": float("nan"),  # nn predicts ids, not embeddings
        "top1_hits": top1_hits.to(torch.uint8).cpu().numpy(),
    }
    return _result(
        m, n_train=sp["n_train"], n_test=sp["n_test"], split_mode=split_mode,
        control=control, pool_size=sp["pool"].shape[0],
    )


class _MLP2(torch.nn.Module):
    """Two-hidden-layer MLP head ``d_in → h → h → d_emb`` (GELU). The
    per-position analog of the aloepri ``ima_paper_like`` 2-layer transformer:
    each TTRSR row is a single-token sequence, so self-attention is vacuous and
    the learned inverter reduces to this stacked feed-forward head."""

    def __init__(self, d_in: int, d_out: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def learned_inversion(
    X: np.ndarray,
    y: np.ndarray,
    embed_table: torch.Tensor,
    *,
    n_train: int = 1024,
    n_val: int = 128,
    n_test: int = 128,
    topk: int = 10,
    candidate_pool_size: int = 2048,
    hidden: int = 1024,
    epochs: int = 150,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    patience: int = 20,
    seed: int = 20260615,
    split_mode: str = "vocab",
    control: str = "none",
    control_seed: int = 20260616,
    device: str | None = None,
    **_ignored: Any,
) -> dict[str, Any] | None:
    """Learned 2-layer MLP inverter (aloepri ``ima_paper_like``). Trains
    ``X → standardised token embedding`` with Adam, early-stops on validation
    TTRSR top-1, then evaluates against the candidate pool. Standardises the
    embedding target the same way ridge does so the two are comparable."""
    sp = _split_pool(
        X, y, embed_table, n_train=n_train, n_val=n_val, n_test=n_test,
        seed=seed, split_mode=split_mode, control=control,
        control_seed=control_seed, candidate_pool_size=candidate_pool_size,
        device=device,
    )
    if sp is None:
        return None
    dev = sp["dev"]
    torch.manual_seed(seed)
    Xtr, Xva, Xte = sp["Xtr_t"], sp["Xva_t"], sp["Xte_t"]
    ytr_emb = sp["ytr_emb"]

    # Per-feature standardisation (input + target), mirroring fit_ridge.
    x_mean = Xtr.mean(0, keepdim=True)
    x_std = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    y_mean = ytr_emb.mean(0, keepdim=True)
    y_std = ytr_emb.std(0, keepdim=True).clamp_min(1e-6)
    Xtr_z = (Xtr - x_mean) / x_std
    Xva_z = (Xva - x_mean) / x_std
    Xte_z = (Xte - x_mean) / x_std
    ytr_z = (ytr_emb - y_mean) / y_std

    model = _MLP2(Xtr.shape[1], ytr_emb.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    n = Xtr_z.shape[0]

    def _val_top1() -> float:
        model.eval()
        with torch.no_grad():
            pred = model(Xva_z) * y_std + y_mean
        vm = evaluate_inversion(
            predicted_embeddings=pred, true_ids=sp["yva_ids"],
            candidate_ids=sp["pool"], embed_table=embed_table, topk=topk,
        )
        return float(vm["token_top1_recovery_rate"])

    best_val, best_state, since = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_z[idx]), ytr_z[idx])
            loss.backward()
            opt.step()
        v = _val_top1()
        if v > best_val:
            best_val, since = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(Xte_z) * y_std + y_mean
    m = evaluate_inversion(
        predicted_embeddings=pred, true_ids=sp["yte_ids"],
        candidate_ids=sp["pool"], embed_table=embed_table, topk=topk,
    )
    return _result(
        m, n_train=sp["n_train"], n_test=sp["n_test"], split_mode=split_mode,
        control=control, pool_size=sp["pool"].shape[0],
        extra={"val_top1": best_val, "hidden": hidden, "epochs_max": epochs},
    )


INVERTERS = {
    "ridge": ridge_inversion,
    "nn": nn_inversion,
    "mlp2": learned_inversion,
}
