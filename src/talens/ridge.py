"""Ridge inverter primitives — the shared attack math.

Faithful reimplementation of the aloepri / AloePri ridge inverter
(``vendor/aloepri-py/src/security_qwen/ima.py`` ``_fit_ridge_regressor``
/ ``_predict_ridge`` / ``_evaluate_inversion_predictions``), as clean
standalone torch functions with no vendored-package dependency.

The inverter fits a standardised ridge map from an observation ``X``
(a representation row) to a token *embedding* ``Y``, then recovers the
token id by cosine-matching the predicted embedding against a candidate
pool of real token embeddings. Top-1 match rate is the TTRSR.
"""

from __future__ import annotations

from typing import Any

import torch


def fit_ridge(
    x_train: torch.Tensor, y_train: torch.Tensor, *, ridge_alpha: float
) -> dict[str, torch.Tensor]:
    """Closed-form standardised ridge with an un-penalised bias column.

    Standardises ``X`` and ``Y`` per-feature, augments ``X`` with a ones
    column, and solves ``(XᵀX + αI)·W = XᵀY`` with the bias row excluded
    from the penalty. Returns the weight plus the standardisation stats
    needed by :func:`predict_ridge`.
    """
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    x_norm = (x_train - x_mean) / x_std
    y_norm = (y_train - y_mean) / y_std
    ones = torch.ones((x_norm.shape[0], 1), dtype=x_norm.dtype, device=x_norm.device)
    x_aug = torch.cat([x_norm, ones], dim=1)
    dim = x_aug.shape[1]
    identity = torch.eye(dim, dtype=x_norm.dtype, device=x_norm.device)
    identity[-1, -1] = 0.0  # don't penalise the bias
    lhs = x_aug.T @ x_aug + ridge_alpha * identity
    rhs = x_aug.T @ y_norm
    # Solve on-device via Cholesky (rocSOLVER potrf). torch.linalg.solve's LU
    # path goes through getrs/trsm (``hipblasStrsm``), whose workspace
    # ALLOC_FAILs on gfx1151 for wide systems (d≳2560 — kqv_out d=4096 dies);
    # potrf does not. ``lhs = XᵀX + αI`` is SPD in the over-determined regime
    # (n_train ≥ d, the full-corpus case); a tiny jitter on the unpenalised
    # bias diagonal guarantees PD with no effect on the fit (Δ < 1e-6 vs the
    # CPU LU solve, verified on gfx1151). Fall back to the robust CPU LU solve
    # for the rare under-determined + tiny-α system that is too ill-conditioned
    # for float32 Cholesky. Weight stays on the inputs' device.
    try:
        lhs_pd = lhs.clone()
        lhs_pd[-1, -1] += 1e-3
        weight = torch.cholesky_solve(rhs, torch.linalg.cholesky(lhs_pd))
    except (torch.linalg.LinAlgError, RuntimeError):
        weight = torch.linalg.solve(lhs.cpu(), rhs.cpu()).to(x_aug.device)
    return {
        "weight": weight,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def predict_ridge(model: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Apply a fitted ridge map to new observations, undoing the
    output standardisation."""
    x_norm = (x - model["x_mean"]) / model["x_std"]
    ones = torch.ones((x_norm.shape[0], 1), dtype=x_norm.dtype, device=x_norm.device)
    x_aug = torch.cat([x_norm, ones], dim=1)
    y_norm = x_aug @ model["weight"]
    return y_norm * model["y_std"] + model["y_mean"]


def _row_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    yn = y / y.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (xn * yn).sum(dim=1)


def evaluate_inversion(
    *,
    predicted_embeddings: torch.Tensor,
    true_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    embed_table: torch.Tensor,
    topk: int = 10,
) -> dict[str, Any]:
    """Cosine-match predicted embeddings to a candidate token-embedding
    pool; report top-1 / top-k recovery and the mean predicted/true
    embedding cosine. Mirrors the aloepri evaluator exactly.
    """
    # The big embed_table stays on CPU; only the small candidate/true
    # slices move to the prediction's device so the cosine matmul runs
    # wherever the inverter ran (GPU when available).
    dev = predicted_embeddings.device
    cand = embed_table[candidate_ids].to(dev)
    cand_ids = candidate_ids.to(dev)
    true_dev = true_ids.to(dev)
    pred_n = predicted_embeddings / predicted_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cand_n = cand / cand.norm(dim=1, keepdim=True).clamp_min(1e-8)
    scores = pred_n @ cand_n.T
    k = min(topk, scores.shape[1])
    topk_idx = torch.topk(scores, k=k, dim=1).indices
    pred_ids = cand_ids[topk_idx]
    hits = pred_ids.eq(true_dev.unsqueeze(1))
    cos = _row_cosine(predicted_embeddings, embed_table[true_ids].to(dev))
    return {
        "token_top1_recovery_rate": float(hits[:, 0].to(torch.float32).mean().item()),
        "token_top10_recovery_rate": float(
            hits[:, : min(10, hits.shape[1])].any(dim=1).to(torch.float32).mean().item()
        ),
        "embedding_cosine_similarity": float(cos.mean().item()),
        "predicted_ids_top1": pred_ids[:, 0].cpu(),
    }
