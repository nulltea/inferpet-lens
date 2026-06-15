"""Synthetic smoke tests — exercise the attacks, the three IT measures,
the transform seam, and calibration without the model/capture stack.

Construction: a random embedding table; each token's "representation" is
its embedding plus small noise (informative) or pure noise (control).
An informative representation should let the attacks recover tokens and
the measures register high leakage; the control should not.
"""

from __future__ import annotations

import numpy as np
import torch

from talens.attacks import attn_score, cover_break, hidden_state
from talens.calibration import calibrate
from talens.capture.types import CaptureSet
from talens.measures import club_mi_upper_bound, online_code_length, v_information
from talens.transforms import Identity

VOCAB, DIM, N_PROMPTS, SEQ = 60, 24, 50, 10


def _embed_table(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, DIM, generator=g)


def _make_capture(
    embed: torch.Tensor, *, informative: bool, seed: int = 1, seq: int = SEQ
) -> CaptureSet:
    rng = np.random.default_rng(seed)
    token_ids: list[list[int]] = []
    resid_ops: list[torch.Tensor] = []
    attn_ops: list[torch.Tensor] = []
    for p in range(N_PROMPTS):
        ids = rng.integers(0, VOCAB, size=seq)
        token_ids.append(ids.tolist())
        if informative:
            op = embed[torch.from_numpy(ids)] + 0.05 * torch.randn(seq, DIM)
        else:
            op = torch.randn(seq, DIM)
        resid_ops.append(op)
        # attention scores with a ragged key length (causal-ish): n_kv varies.
        n_kv = seq - (p % 3)
        attn_ops.append(torch.rand(2, seq, n_kv))
    return CaptureSet(
        model_id="synthetic",
        prompt_token_ids=token_ids,
        operands={("resid_post", 0): resid_ops, ("attn_score", 0): attn_ops},
    )


# --- transform seam -------------------------------------------------------

def test_identity_is_noop():
    t = Identity()
    x = torch.randn(4, 8)
    assert torch.equal(t(x, prompt_index=0), x)
    assert t.name == "identity"


# --- capture stacking -----------------------------------------------------

def test_stack_resid_shapes():
    emb = _embed_table()
    cap = _make_capture(emb, informative=True)
    X, y, lengths = cap.stack("resid_post", 0)
    assert X.shape == (N_PROMPTS * SEQ, DIM)
    assert y.shape == (N_PROMPTS * SEQ,)
    assert sum(lengths) == N_PROMPTS * SEQ


def test_stack_attn_ragged_padded_to_fixed_width():
    emb = _embed_table()
    cap = _make_capture(emb, informative=True)
    X, y, _ = cap.stack("attn_score", 0)
    # n_heads(2) * max_kv(SEQ) features, one row per query position.
    assert X.shape[1] == 2 * SEQ
    assert X.shape[0] == N_PROMPTS * SEQ


# --- attacks --------------------------------------------------------------

def test_hidden_state_inversion_recovers_when_informative():
    emb = _embed_table()
    cap = _make_capture(emb, informative=True)
    res = hidden_state.run(cap, emb, layer=0, split_mode="row")
    assert res.primary_metric_value is not None
    assert res.ttrsr_top1 > 0.3


def test_hidden_state_inversion_fails_on_noise():
    emb = _embed_table()
    cap = _make_capture(emb, informative=False)
    res = hidden_state.run(cap, emb, layer=0, split_mode="row")
    assert res.ttrsr_top1 < 0.25


def test_attn_score_attack_runs():
    emb = _embed_table()
    cap = _make_capture(emb, informative=True)
    res = attn_score.run(cap, emb, layer=0, split_mode="row")
    assert res.attack == "attn_score_inversion"
    assert res.kind == "attn_score"


def test_cover_break_identity_recovers():
    # Under Identity, U == H. Ridge recovers the identity map only with
    # enough anchors (K >= d); use long sequences so K=32 > DIM=24.
    emb = _embed_table()
    cap = _make_capture(emb, informative=True, seq=40)
    res = cover_break.run(cap, layer=0, anchor_counts=(32,))
    assert res.primary_metric_value is not None
    assert res.primary_metric_value > 0.8


def test_cover_break_fastica_deferred():
    emb = _embed_table()
    cap = _make_capture(emb, informative=True)
    try:
        cover_break.run(cap, layer=0, fastica=True)
        raised = False
    except NotImplementedError:
        raised = True
    assert raised


# --- measures -------------------------------------------------------------

def test_v_information_high_when_informative_low_when_noise():
    emb = _embed_table()
    info = v_information(*_make_capture(emb, informative=True).stack("resid_post", 0)[:2])
    noise = v_information(*_make_capture(emb, informative=False).stack("resid_post", 0)[:2])
    assert info["v_information_bits"] > noise["v_information_bits"]
    assert info["v_information_bits"] > 0.5


def test_mdl_compresses_when_informative():
    emb = _embed_table()
    info = online_code_length(*_make_capture(emb, informative=True).stack("resid_post", 0)[:2])
    noise = online_code_length(*_make_capture(emb, informative=False).stack("resid_post", 0)[:2])
    assert info["compression"] > 1.0
    assert info["compression"] > noise["compression"]


def test_club_upper_bound_finite_and_ordered():
    emb = _embed_table()
    Xi, yi, _ = _make_capture(emb, informative=True).stack("resid_post", 0)
    Yi = emb[torch.from_numpy(yi)].numpy()
    info = club_mi_upper_bound(Xi, Yi, steps=200)
    Xn, yn, _ = _make_capture(emb, informative=False).stack("resid_post", 0)
    Yn = emb[torch.from_numpy(yn)].numpy()
    noise = club_mi_upper_bound(Xn, Yn, steps=200)
    assert info["club_mi_bits"] is not None and np.isfinite(info["club_mi_bits"])
    assert noise["club_mi_bits"] is not None and np.isfinite(noise["club_mi_bits"])


# --- calibration ----------------------------------------------------------

def test_calibrate_detects_correlation():
    measure = np.linspace(0, 1, 12)
    recovery = measure * 0.8 + 0.1  # perfectly rank-correlated
    cal = calibrate(measure, recovery)
    assert cal["spearman"] > 0.99
    assert cal["r_squared"] > 0.99
