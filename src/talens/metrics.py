"""Attack-result schema and the shared recovery metrics.

Dependency-light (numpy only) so the attacks, measures, and tests run
without the model/capture stack. The ``ttrsr`` metric and the
``AttackResult`` dataclass are carried over from the aloepri harness so
result tables stay directly comparable; ``risk_level`` thresholds match
its ``metrics.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def ttrsr(predicted_ids: np.ndarray, ground_truth_ids: np.ndarray) -> float:
    """Token-level Recovery Success Rate — fraction of positions where
    the top-1 predicted id matches the ground-truth id. This is the
    uniform regression target the IT measures are calibrated against.
    """
    if predicted_ids.shape != ground_truth_ids.shape:
        raise ValueError(
            f"ttrsr: shape mismatch — predicted {predicted_ids.shape} "
            f"vs ground truth {ground_truth_ids.shape}"
        )
    if predicted_ids.size == 0:
        return 0.0
    return float((predicted_ids == ground_truth_ids).mean())


def topk_recovery(
    predicted_ids_topk: np.ndarray, ground_truth_ids: np.ndarray
) -> float:
    """Fraction of positions where the ground-truth id is in the
    predicted top-k set. ``predicted_ids_topk`` is ``(n, k)``,
    ``ground_truth_ids`` is ``(n,)``.
    """
    if predicted_ids_topk.shape[0] != ground_truth_ids.shape[0]:
        raise ValueError("topk_recovery: row mismatch")
    if predicted_ids_topk.size == 0:
        return 0.0
    hits = (predicted_ids_topk == ground_truth_ids[:, None]).any(axis=1)
    return float(hits.mean())


def classify_risk_level(primary_metric_value: float | None) -> str:
    """≥0.30 → "high", ≥0.10 → "medium", <0.10 → "low", None → "unknown"."""
    if primary_metric_value is None:
        return "unknown"
    if primary_metric_value >= 0.30:
        return "high"
    if primary_metric_value >= 0.10:
        return "medium"
    return "low"


@dataclass
class AttackResult:
    """One row of attack output, paired downstream with an IT-measure
    row over the same (representation kind, layer, transform).
    """

    attack: str
    transform: str          # name of the applied Transform (Identity by default)
    model_id: str
    kind: str               # representation kind, e.g. "resid_post", "attn_score"
    layer: int
    n_prompts: int
    n_train: int
    n_test: int
    ttrsr_top1: float | None
    ttrsr_top10: float | None
    risk_level: str
    # The single comparable "recovery" scalar the calibration regresses
    # the IT measures against. Token attacks set it to ttrsr_top1; the
    # cover-break attack sets it to its p95 cosine recovery. Always read
    # ``primary_metric_name`` to know which it is.
    primary_metric_value: float | None = None
    primary_metric_name: str = "token_top1_recovery_rate"
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}
