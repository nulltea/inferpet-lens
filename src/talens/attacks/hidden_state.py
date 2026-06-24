"""Hidden-state inversion attack (IMA / ISA).

Fits the ridge inverter from a layer's residual-stream operand to the
token embedding and reads back the token id. This is the aloepri
IMA (shallow layer) and ISA (deep layer) attack unified — they are the
same math at different ``layer`` values, so the per-layer sweep this
repo cares about is just calling :func:`run` across all layers.
"""

from __future__ import annotations

import numpy as np
import torch

from ..capture.types import CaptureSet
from ..metrics import AttackResult, classify_risk_level
from ..transforms import Identity, Transform
from ._inversion import INVERTERS, ridge_inversion


def run(
    capture: CaptureSet,
    embed_table: torch.Tensor,
    *,
    layer: int,
    kind: str = "resid_post",
    transform: Transform | None = None,
    n_train: int | None = None,
    n_val: int = 128,
    n_test: int = 128,
    topk: int = 10,
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0),
    candidate_pool_size: int = 2048,
    seed: int = 20260615,
    split_mode: str = "vocab",
    control: str = "none",
    control_seed: int = 20260616,
    attack_name: str = "hidden_state_inversion",
    inverter: str = "ridge",
    inverter_kwargs: dict | None = None,
    xy: tuple[np.ndarray, np.ndarray] | None = None,
) -> AttackResult:
    transform = transform or Identity()
    if inverter not in INVERTERS:
        raise ValueError(f"unknown inverter {inverter!r}; choose from {sorted(INVERTERS)}")
    # n_train=None → full row-split: a huge request makes the splitter
    # auto-scale to 70/15/15 of all available rows (the aloepri row-split).
    # The old fixed 1024 under-trained the inverter on ~9k-row corpora and
    # understated recovery (esp. for the wide kqv_out surface).
    if n_train is None:
        n_train = 10**9
    # ``xy`` lets the caller pass an already-stacked (X, y) so the operand
    # flattening isn't repeated (the orchestrator stacks once per block).
    X, y = xy if xy is not None else capture.stack(kind, layer, transform=transform)[:2]
    metrics = INVERTERS[inverter](
        X,
        y,
        embed_table,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        topk=topk,
        ridge_alphas=ridge_alphas,
        candidate_pool_size=candidate_pool_size,
        seed=seed,
        split_mode=split_mode,
        control=control,
        control_seed=control_seed,
        **(inverter_kwargs or {}),
    )
    if metrics is None:
        return AttackResult(
            attack=attack_name,
            transform=transform.name,
            model_id=capture.model_id,
            kind=kind,
            layer=layer,
            n_prompts=capture.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            primary_metric_value=None,
            extra={"note": "not enough rows for a train/test split"},
        )
    top1 = metrics["ttrsr_top1"]
    return AttackResult(
        attack=attack_name,
        transform=transform.name,
        model_id=capture.model_id,
        kind=kind,
        layer=layer,
        n_prompts=capture.n_prompts(),
        n_train=metrics["n_train"],
        n_test=metrics["n_test"],
        ttrsr_top1=top1,
        ttrsr_top10=metrics["ttrsr_top10"],
        risk_level=classify_risk_level(top1),
        primary_metric_value=top1,
        primary_metric_name="token_top1_recovery_rate",
        extra={k: v for k, v in metrics.items() if k not in {"ttrsr_top1", "ttrsr_top10", "n_train", "n_test"}},
    )
