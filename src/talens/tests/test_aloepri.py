"""Tests for the AloePri defense (scripts/defenses/aloepri.py) — model-free, CPU.

Covers Algorithm 1's invertibility invariant (``P̂ Q̂ = I``), the faithful
obfuscated-table generator feeding the VMA τ-recovery family (recovery high
undefended, decaying with the ``α_e`` noise knob), and the activation covers.
"""

from __future__ import annotations

import numpy as np
import torch

from defenses.aloepri import (
    AloePriKeyMatCover,
    AloePriPermCover,
    keymat_gen,
    obfuscate_embedding_table,
)
from talens.weights import vma


def test_keymat_is_a_left_inverse():
    """Algorithm 1's core invariant: P̂ Q̂ = I_d despite P̂ ≠ Q̂⁻¹ obviously."""
    for d, h in [(16, 8), (32, 16), (24, 12)]:
        P, Q = keymat_gen(d, h, lam=0.1, seed=d)
        assert P.shape == (d, d + 2 * h)
        assert Q.shape == (d + 2 * h, d)
        I = P.astype(np.float64) @ Q.astype(np.float64)
        assert np.allclose(I, np.eye(d), atol=1e-3), np.abs(I - np.eye(d)).max()


def test_keymat_rejects_odd_h():
    try:
        keymat_gen(16, 7)
    except ValueError:
        return
    raise AssertionError("expected ValueError for odd h")


def _plaintext(n=96, d=48, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, d)).astype(np.float32)
    spike = rng.random((n, d)) < (4.0 / d)
    W += (spike * rng.normal(0.0, 8.0, (n, d))).astype(np.float32)
    return W


def test_keymat_defeats_sorted_quantile_vma():
    """The dense Algorithm-1 key matrix is NOT a column permutation, so it breaks
    the RowSort/sorted-quantile VMA → recovery falls to chance even undefended.
    The full keymat IS the defense against this attack (a finding for B2/B4)."""
    W = _plaintext()
    pair = obfuscate_embedding_table(W, alpha_e=0.0, keymat=True, seed=3)
    assert pair.obf.shape[1] > pair.plain.shape[1]        # d̃ = d + 2h wider
    res = vma.run(pair, bins=32, match="hungarian")
    assert res.ttrsr_top1 < 0.15                          # keymat defeats sorted VMA
    assert res.primary_metric_name == "permutation_recovery_rate"


def test_permcore_recoverable_undefended_and_decays():
    """The permutation-core regime (perm + col-perm + noise) is the VMA-vulnerable
    one: high recovery undefended, monotone decay with the α_e noise knob."""
    W = _plaintext()
    recs = [
        vma.run(obfuscate_embedding_table(W, alpha_e=a, keymat=False, seed=3),
                bins=32, match="hungarian").ttrsr_top1
        for a in (0.0, 0.5, 1.0, 2.0)
    ]
    assert recs[0] >= recs[1] >= recs[2] >= recs[3]       # monotone defense curve
    assert recs[0] > 0.95                                 # undefended perm-core leaks Π
    assert recs[-1] < 0.30                                # heavy noise → defended


def test_permutation_core_keeps_width():
    W = _plaintext()
    und = obfuscate_embedding_table(W, alpha_e=0.0, keymat=False, seed=3)
    assert und.obf.shape == und.plain.shape               # perm-core keeps width


def test_perm_cover_is_norm_preserving_bijection():
    cover = AloePriPermCover(seed=1)
    x = torch.randn(10, 32)
    u = cover(x, prompt_index=0)
    assert u.shape == x.shape
    assert torch.allclose(u.norm(dim=-1), x.norm(dim=-1), atol=1e-5)   # channel perm
    # same prompt → same permutation (pure)
    assert torch.equal(cover(x, prompt_index=5), u)


def test_keymat_cover_widens_and_is_linearly_invertible():
    cover = AloePriKeyMatCover(h=16, lam=0.1, seed=2)
    x = torch.randn(10, 32)
    u = cover(x, prompt_index=0)
    assert u.shape[-1] == 32 + 2 * 16                     # widened to d+2h
    # P̂ has a left inverse Q̂, so x is linearly recoverable from u
    P, Q = keymat_gen(32, 16, lam=0.1, seed=2)
    recon = u.numpy() @ Q
    assert np.allclose(recon, x.numpy(), atol=1e-2)
