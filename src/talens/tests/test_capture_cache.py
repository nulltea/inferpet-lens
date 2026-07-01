"""Model-free tests for the capture/embedding disk cache.

Exercises the round-trip persistence, the layer-subset view, and the
reuse-decision truth table — all without nnsight or a model, on a
synthetic CaptureSet.
"""

from __future__ import annotations

import numpy as np
import torch

from talens.capture.cache import (
    can_reuse,
    capture_cache_path,
    embed_cache_path,
    load_capture,
    load_embed,
    present_layers,
    save_capture,
    save_embed,
    subset_capture,
)
from talens.capture.types import CaptureSet

KINDS = ("resid_post", "attn_score")


def _synthetic(layers=(0, 1, 2, 3), n_prompts=3, d=8, seq=5) -> CaptureSet:
    rng = np.random.default_rng(0)
    token_ids, operands = [], {}
    for p in range(n_prompts):
        token_ids.append(rng.integers(0, 50, size=seq).tolist())
        for L in layers:
            operands.setdefault(("resid_post", L), []).append(torch.randn(seq, d))
    return CaptureSet(model_id="synthetic", prompt_token_ids=token_ids, operands=operands)


def test_capture_round_trip(tmp_path):
    cap = _synthetic()
    path = capture_cache_path(tmp_path, "synthetic", ["a", "b"], KINDS)
    save_capture(cap, path, capture_layers=[0, 1, 2, 3])
    loaded, spec = load_capture(path)

    assert spec == [0, 1, 2, 3]
    assert loaded.model_id == cap.model_id
    assert loaded.prompt_token_ids == cap.prompt_token_ids
    assert present_layers(loaded) == {0, 1, 2, 3}
    X0, y0, _ = cap.stack("resid_post", 2)
    X1, y1, _ = loaded.stack("resid_post", 2)
    assert np.allclose(X0, X1) and np.array_equal(y0, y1)


def test_embed_round_trip(tmp_path):
    emb = torch.randn(20, 8)
    path = embed_cache_path(tmp_path, "Qwen/Qwen3-4B")
    save_embed(emb, path)
    assert torch.allclose(load_embed(path), emb)


def test_cache_key_depends_on_corpus_and_kinds(tmp_path):
    a = capture_cache_path(tmp_path, "m", ["p1", "p2"], KINDS)
    b = capture_cache_path(tmp_path, "m", ["p1", "p3"], KINDS)
    c = capture_cache_path(tmp_path, "m", ["p1", "p2"], ("resid_post",))
    assert a != b and a != c


def test_subset_capture_restricts_layers():
    cap = _synthetic(layers=(0, 1, 2, 3))
    sub = subset_capture(cap, [1, 3])
    assert present_layers(sub) == {1, 3}
    # unchanged when None
    assert present_layers(subset_capture(cap, None)) == {0, 1, 2, 3}


def test_can_reuse_truth_table():
    present = {0, 1, 2, 3}
    # cache made for ALL layers (spec=None) covers any request
    assert can_reuse(present, None, None) is True
    assert can_reuse(present, None, [2, 3]) is True
    # cache made for a specific subset covers a request iff layers present
    assert can_reuse(present, [0, 1, 2, 3], [1, 2]) is True
    assert can_reuse(present, [0, 1], [1, 5]) is False
    # an all-layers request is NOT served by a subset cache
    assert can_reuse(present, [0, 1, 2, 3], None) is False
