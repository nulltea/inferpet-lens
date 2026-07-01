"""Unit tests for the geometry-only BNN/MAP error bounds (B1 sanity).

Model-free, synthetic, CPU — runs on the host ``.venv``. Validates the
five-theorem proof package (``refine-logs/PROOF_PACKAGE.md``):
T-a σ→0 limits, T-b orthonormal closed-form, T-c exact-Q ≤ Bhattacharyya,
T-d Ĥ_M unbiased vs brute-force H(V|Y), T-e independence-by-construction
(no observation argument), T-f σ-monotonicity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from talens.probes.channel_error_bounds import (
    fano_equivocation,
    union_bhattacharyya,
)


def _orthonormal_pool(K: int, d: int) -> np.ndarray:
    """K orthonormal rows in R^d (K<=d): identity-block. ‖Δ‖=√2 for all pairs."""
    E = np.zeros((K, d), dtype=np.float32)
    for i in range(K):
        E[i, i] = 1.0
    return E


def _brute_force_h_cond(E: np.ndarray, sigma: float, n_mc: int, seed: int) -> float:
    """Reference H(V|Y) in bits via plain (non-stratified) MC: draw v~Unif,
    ε~N(0,σ²I), accumulate −log2 p(v|Y). Independent code path from the
    stratified estimator under test."""
    rng = np.random.default_rng(seed)
    K, d = E.shape
    sq = (E * E).sum(1)
    acc = 0.0
    for _ in range(n_mc):
        v = rng.integers(K)
        y = E[v] + rng.standard_normal(d).astype(np.float32) * sigma
        d2 = sq - 2.0 * (y @ E.T) + (y @ y)
        logits = -d2 / (2.0 * sigma ** 2)
        m = logits.max()
        lse = m + math.log(np.exp(logits - m).sum())
        acc += (lse - logits[v]) / math.log(2.0)
    return acc / n_mc


# ── T-a: σ→0 limits ──────────────────────────────────────────────────────────
def test_sigma_zero_limits():
    E = _orthonormal_pool(8, 16)
    ub = union_bhattacharyya(E, 0.0)
    lb = fano_equivocation(E, 0.0, M=16, seed=0)
    assert ub["p_e_ub"] == 0.0 and ub["p_e_ub_bhat"] == 0.0
    assert lb["h_cond_bits"] == 0.0 and lb["p_e_lb"] == 0.0


# ── T-b: orthonormal closed form ──────────────────────────────────────────────
def test_orthonormal_upper_bound_closed_form():
    # ‖Δ‖=√2 for every pair → exact-pairwise union = (K-1)·Q(√2/(2σ)).
    K, d, sigma = 6, 32, 0.5
    E = _orthonormal_pool(K, d)
    from scipy.special import erfc

    q = 0.5 * erfc((math.sqrt(2) / (2 * sigma)) / math.sqrt(2))
    expected = (K - 1) * q
    got = union_bhattacharyya(E, sigma)["p_e_ub_raw"]
    assert got == pytest.approx(expected, rel=1e-6)


# ── T-c: exact-Q ≤ Bhattacharyya (T1 inequality) ──────────────────────────────
def test_exact_q_le_bhattacharyya():
    rng = np.random.default_rng(1)
    E = rng.standard_normal((20, 24)).astype(np.float32)
    for sigma in (0.2, 0.5, 1.0, 2.0):
        r = union_bhattacharyya(E, sigma)
        assert r["p_e_ub_raw"] <= r["p_e_ub_bhat_raw"] + 1e-9


# ── T-d: Ĥ_M unbiased vs brute-force H(V|Y) ───────────────────────────────────
def test_fano_equivocation_unbiased():
    K, d, sigma = 10, 12, 0.7
    rng = np.random.default_rng(2)
    E = (rng.standard_normal((K, d)).astype(np.float32))
    h_stratified = fano_equivocation(E, sigma, M=4000, seed=3)["h_cond_bits"]
    h_brute = _brute_force_h_cond(E, sigma, n_mc=40000, seed=4)
    # both are MC estimates of the same H(V|Y); agree within sampling noise
    assert h_stratified == pytest.approx(h_brute, abs=0.05)


# ── T-e: independence by construction (no observation argument) ───────────────
def test_no_observation_argument():
    import inspect

    for fn in (union_bhattacharyya, fano_equivocation):
        params = set(inspect.signature(fn).parameters)
        # the probe must NOT accept the attack's observations / labels
        assert not (params & {"Y", "X", "obs", "y", "labels", "Y_obs"}), (
            f"{fn.__name__} exposes an observation argument — breaks T3 independence"
        )


def test_independence_of_synthetic_seed_is_only_randomness():
    # Same codebook+σ, different seed → only MC noise differs; the upper
    # bound (deterministic) is identical, equivocation agrees within SE.
    rng = np.random.default_rng(5)
    E = rng.standard_normal((30, 20)).astype(np.float32)
    a = union_bhattacharyya(E, 0.6)
    b = union_bhattacharyya(E, 0.6)
    assert a["p_e_ub_raw"] == b["p_e_ub_raw"]  # deterministic, no data
    f1 = fano_equivocation(E, 0.6, M=500, seed=11)
    f2 = fano_equivocation(E, 0.6, M=500, seed=22)
    assert abs(f1["h_cond_bits"] - f2["h_cond_bits"]) < 6 * (f1["se"] + f2["se"]) + 1e-6


# ── T-f: σ-monotonicity (T5) ──────────────────────────────────────────────────
def test_monotonicity_in_sigma():
    rng = np.random.default_rng(6)
    E = rng.standard_normal((40, 24)).astype(np.float32)
    sigmas = [0.1, 0.3, 0.6, 1.0, 2.0]
    ubs = [union_bhattacharyya(E, s)["p_e_ub_raw"] for s in sigmas]
    hs = [fano_equivocation(E, s, M=400, seed=7)["h_cond_bits"] for s in sigmas]
    # upper bound strictly non-decreasing in σ
    assert all(ubs[i] <= ubs[i + 1] + 1e-9 for i in range(len(ubs) - 1))
    # equivocation non-decreasing in σ (allow small MC slack)
    assert all(hs[i] <= hs[i + 1] + 0.05 for i in range(len(hs) - 1))


# ── bonus: bracketing holds on a synthetic codebook (T1+T2 end-to-end) ────────
def test_bracketing_synthetic():
    """measured uniform-prior NN error ∈ [p_e_lb, p_e_ub] across σ."""
    rng = np.random.default_rng(8)
    K, d = 50, 32
    E = rng.standard_normal((K, d)).astype(np.float32)
    sq = (E * E).sum(1)
    for sigma in (0.3, 0.6, 1.0):
        ub = union_bhattacharyya(E, sigma)["p_e_ub"]
        lb = fano_equivocation(E, sigma, M=64, seed=9)["p_e_lb"]
        # measure NN (uniform-prior MAP) error by MC
        n = 4000
        v = rng.integers(K, size=n)
        Y = E[v] + rng.standard_normal((n, d)).astype(np.float32) * sigma
        d2 = (Y * Y).sum(1)[:, None] - 2.0 * (Y @ E.T) + sq[None, :]
        pred = d2.argmin(1)
        err = float((pred != v).mean())
        assert lb - 0.02 <= err <= ub + 0.02, (
            f"σ={sigma}: err={err:.3f} not in [{lb:.3f}, {ub:.3f}]"
        )


# ── σ=0 collision honesty (round-2/3 review regression) ───────────────────────
def test_sigma0_distinct_codewords_full_leakage():
    """σ=0, distinct codewords: deterministic injective channel ⇒ i_channel=log₂K,
    H(V|Y)=0, and the union/Bhattacharyya error upper bound is exactly 0."""
    E = np.eye(8, dtype=np.float32)
    fa = fano_equivocation(E, sigma=0.0)
    assert abs(fa["i_channel_bits"] - 3.0) < 1e-9  # log2(8)
    assert abs(fa["h_cond_bits"] - 0.0) < 1e-9
    assert union_bhattacharyya(E, sigma=0.0)["p_e_ub"] == 0.0


def test_sigma0_collision_does_not_overstate_leakage_or_understate_error():
    """σ=0 with collisions (4 distinct codewords each duplicated → K=8, G=4): the
    deterministic channel is NON-injective, so i_channel must drop to log₂G=2 (not
    overstate as log₂K=3) AND the MAP error upper bound must rise to (K−G)/K=0.5
    (not the false 0 the distinctness assumption gives)."""
    E = np.repeat(np.eye(4, dtype=np.float32), 2, axis=0)  # K=8, 4 groups of 2
    fa = fano_equivocation(E, sigma=0.0)
    assert abs(fa["i_channel_bits"] - 2.0) < 1e-9  # log2(4), NOT log2(8)
    assert abs(fa["h_cond_bits"] - 1.0) < 1e-9     # 1 bit ambiguity within each pair
    ub = union_bhattacharyya(E, sigma=0.0)
    assert ub["min_dist"] == 0.0
    assert abs(ub["p_e_ub"] - 0.5) < 1e-9          # (8-4)/8 — recovery floor = 0.5, not 1
