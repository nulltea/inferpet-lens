"""Model-free test of the thread-parallel orchestration (calibrate_capture).

Builds a synthetic CaptureSet across several layers and confirms the
pass-1 driver runs every block in parallel and emits a well-formed report
(one record per block, the three calibration entries) — without the
nnsight/model stack.
"""

from __future__ import annotations

import numpy as np
import torch

from talens.capture.types import CaptureSet
from talens.cli import calibrate_capture

VOCAB, DIM, N_PROMPTS, SEQ, N_LAYERS = 60, 24, 40, 12, 6


def _embed():
    g = torch.Generator().manual_seed(0)
    return torch.randn(VOCAB, DIM, generator=g)


def _capture(embed):
    rng = np.random.default_rng(1)
    token_ids, ops = [], {}
    for p in range(N_PROMPTS):
        ids = rng.integers(0, VOCAB, size=SEQ)
        token_ids.append(ids.tolist())
        for L in range(N_LAYERS):
            op = embed[torch.from_numpy(ids)] + 0.05 * torch.randn(SEQ, DIM)
            ops.setdefault(("resid_post", L), []).append(op)
    return CaptureSet(model_id="synthetic", prompt_token_ids=token_ids, operands=ops)


def test_calibrate_capture_parallel_report():
    emb = _embed()
    cap = _capture(emb)
    report = calibrate_capture(cap, emb, attack_split_mode="row")

    # one record per (kind, layer) block
    assert len(report["records"]) == N_LAYERS
    assert {r["layer"] for r in report["records"]} == set(range(N_LAYERS))
    for r in report["records"]:
        assert r["primary_metric_value"] is not None
        assert r["v_information_bits"] is not None
        assert r["club_mi_bits"] is not None

    # the three calibrations are present and computed over the blocks
    cal = report["calibration"]
    assert set(cal) == {"v_information_bits", "mdl_surplus_bits", "club_mi_bits"}
    assert cal["v_information_bits"]["n"] == N_LAYERS
