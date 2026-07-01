"""Synthetic tests for the Hidden No More vocab-matching attack
(:mod:`talens.attacks.vocab_match`) — model-free, CPU.

The attack runs a ``forward_fn`` over candidate sequences and matches the
resulting hidden state (L1) to the observed target. We stand in a
*synthetic* forward map: a token's hidden state is a deterministic,
mildly-contextual function of the sequence (last token's embedding plus a
small contribution from the previous token). The capture's targets are
built from the **same** map, so greedy search must recover ~everything;
a forward map from a **different** embedding table must collapse to the
chance floor. This is exactly the cover-agnostic argument — recovery
works iff the attacker's forward map shares the targets' space.
"""

from __future__ import annotations

import numpy as np
import torch

from talens.attacks import vocab_match
from talens.capture.types import CaptureSet

VOCAB, DIM, N_PROMPTS, SEQ = 40, 16, 12, 8


def _embed(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((VOCAB, DIM)).astype(np.float32)


def _make_forward_fn(embed: np.ndarray):
    """hidden(seq) = embed[last] + 0.1 * embed[prev] (prev only if len>=2)."""

    def forward_fn(seqs: np.ndarray) -> np.ndarray:
        seqs = np.asarray(seqs, dtype=np.int64)
        out = embed[seqs[:, -1]].copy()
        if seqs.shape[1] >= 2:
            out += 0.1 * embed[seqs[:, -2]]
        return out

    return forward_fn


def _make_capture(embed: np.ndarray, seed: int = 1) -> CaptureSet:
    """Targets at each position = forward_fn over the *true* prefix, so the
    observed hidden state at position p is the deterministic image of the
    true tokens up to p."""
    fwd = _make_forward_fn(embed)
    rng = np.random.default_rng(seed)
    token_ids: list[list[int]] = []
    ops: list[torch.Tensor] = []
    for _ in range(N_PROMPTS):
        ids = rng.integers(0, VOCAB, size=SEQ).astype(np.int64)
        token_ids.append(ids.tolist())
        rows = np.stack([fwd(ids[: p + 1][None, :])[0] for p in range(SEQ)])
        ops.append(torch.from_numpy(rows.astype(np.float32)))
    return CaptureSet(
        model_id="synthetic", prompt_token_ids=token_ids,
        operands={("resid_post", 0): ops},
    )


def test_recovers_with_matching_forward_map():
    embed = _embed(0)
    cap = _make_capture(embed)
    res = vocab_match.run(
        cap, _make_forward_fn(embed), layer=0, kind="resid_post",
        candidate_ids=np.arange(VOCAB, dtype=np.int64),
    )
    assert res.attack == "vocab_match_inversion"
    assert res.n_train == 0
    # Deterministic, noise-free, shared space -> near-perfect recovery.
    assert res.ttrsr_top1 > 0.98
    assert res.ttrsr_top10 == 1.0
    assert res.risk_level == "high"


def test_wrong_forward_map_collapses_to_floor():
    cap = _make_capture(_embed(0))
    # Attacker's forward map uses a *different* embedding table (wrong
    # space / wrong weights) -> recovery near the 1/VOCAB chance floor.
    res = vocab_match.run(
        cap, _make_forward_fn(_embed(99)), layer=0, kind="resid_post",
        candidate_ids=np.arange(VOCAB, dtype=np.int64),
    )
    assert res.ttrsr_top1 < 0.2


def test_max_positions_caps_recovered_length():
    embed = _embed(0)
    cap = _make_capture(embed)
    res = vocab_match.run(
        cap, _make_forward_fn(embed), layer=0, kind="resid_post",
        candidate_ids=np.arange(VOCAB, dtype=np.int64), max_positions=3,
    )
    assert res.n_test == N_PROMPTS * 3


def test_default_candidate_pool_used_when_unspecified():
    embed = _embed(0)
    cap = _make_capture(embed)
    res = vocab_match.run(cap, _make_forward_fn(embed), layer=0, kind="resid_post")
    # Pool defaults to the corpus's observed ids (<= VOCAB distinct).
    assert 0 < res.extra["candidate_pool_size"] <= VOCAB
    assert res.ttrsr_top1 > 0.98
