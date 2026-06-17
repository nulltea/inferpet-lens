"""Control-task tests — the memorisation floors (docs/dev/control-tasks.md).

The locked control set is: a Hewitt–Liang label-shuffle floor on every
estimator (``selectivity = real − shuffled``) and the attack's
vocab-disjoint memorisation gap. These tests assert the floors behave as
designed on the synthetic informative capture: the shuffle floor sits well
below the real signal, and the pipeline emits the selectivity / mem-gap
columns.
"""

from __future__ import annotations

import numpy as np
import torch

from talens.attacks import hidden_state
from talens.capture.types import CaptureSet
from talens.cli import _parse_controls, _process_block, _selectivity
from talens.measures import club_mi_upper_bound, v_information
from talens.transforms import Identity

VOCAB, DIM, N_PROMPTS, SEQ = 60, 24, 50, 10


def _embed_table(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, DIM, generator=g)


def _make_capture(embed: torch.Tensor, *, seed: int = 1) -> CaptureSet:
    rng = np.random.default_rng(seed)
    token_ids: list[list[int]] = []
    resid_ops: list[torch.Tensor] = []
    for _ in range(N_PROMPTS):
        ids = rng.integers(0, VOCAB, size=SEQ)
        token_ids.append(ids.tolist())
        resid_ops.append(embed[torch.from_numpy(ids)] + 0.05 * torch.randn(SEQ, DIM))
    return CaptureSet(
        model_id="synthetic",
        prompt_token_ids=token_ids,
        operands={("resid_post", 0): resid_ops},
    )


# --- helpers --------------------------------------------------------------

def test_parse_controls():
    assert _parse_controls("none") == frozenset()
    assert _parse_controls("shuffle") == frozenset({"shuffle"})
    assert _parse_controls("all") == frozenset({"shuffle", "vocab"})


def test_selectivity_handles_missing():
    assert _selectivity(0.9, 0.1) == 0.8
    assert _selectivity(None, 0.1) is None
    assert _selectivity(0.9, None) is None


# --- shuffle floor sits below the real signal -----------------------------

def test_pvi_shuffle_floor_below_real():
    emb = _embed_table()
    X, y, _ = _make_capture(emb).stack("resid_post", 0)
    real = v_information(X, y)["v_information_bits"]
    floor = v_information(X, y, control="shuffle")["v_information_bits"]
    # Breaking X↔Y must collapse PVI toward the prior (≈0); real keeps signal.
    assert real > 0.5
    assert floor < 0.3
    assert real - floor > 0.3


def test_club_shuffle_floor_below_real():
    emb = _embed_table()
    X, y, _ = _make_capture(emb).stack("resid_post", 0)
    Y = emb[torch.from_numpy(y)].numpy()
    real = club_mi_upper_bound(X, Y, steps=200)["club_mi_bits"]
    floor = club_mi_upper_bound(X, Y, steps=200, control="shuffle")["club_mi_bits"]
    assert real > floor


def test_attack_shuffle_floor_below_real():
    emb = _embed_table()
    cap = _make_capture(emb)
    real = hidden_state.run(cap, emb, layer=0, split_mode="row").ttrsr_top1
    floor = hidden_state.run(
        cap, emb, layer=0, split_mode="row", control="shuffle"
    ).ttrsr_top1
    assert real > floor


def test_shuffle_is_deterministic():
    emb = _embed_table()
    X, y, _ = _make_capture(emb).stack("resid_post", 0)
    a = v_information(X, y, control="shuffle", control_seed=123)["v_information_bits"]
    b = v_information(X, y, control="shuffle", control_seed=123)["v_information_bits"]
    assert a == b


# --- pipeline emits the control columns -----------------------------------

def test_process_block_emits_control_fields():
    emb = _embed_table()
    cap = _make_capture(emb)
    rec = _process_block(
        cap, emb, "resid_post", 0,
        transform=Identity(), attack_split_mode="row",
        controls=frozenset({"shuffle", "vocab"}),
    )
    # shuffle floor + selectivity for every estimator
    for base in ("v_information_bits", "mdl_surplus_bits", "club_mi_bits"):
        assert f"{base}_shuffle" in rec
    for sel in ("v_information_selectivity", "mdl_selectivity",
                "club_mi_selectivity", "ttrsr_selectivity"):
        assert sel in rec
        assert rec[sel] is None or np.isfinite(rec[sel])
    # vocab-disjoint memorisation gap (attack)
    assert "ttrsr_top1_row" in rec and "ttrsr_top1_vocab" in rec
    assert "ttrsr_mem_gap" in rec
    # informative synthetic data: PVI selectivity should be clearly positive
    assert rec["v_information_selectivity"] > 0.3


def test_process_block_no_controls_is_unchanged():
    emb = _embed_table()
    cap = _make_capture(emb)
    rec = _process_block(
        cap, emb, "resid_post", 0,
        transform=Identity(), attack_split_mode="row",
    )
    assert "v_information_selectivity" not in rec
    assert "ttrsr_mem_gap" not in rec
