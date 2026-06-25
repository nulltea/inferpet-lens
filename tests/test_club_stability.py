"""CLUB numerical-stability regression (model-free, CPU).

Heavy-tailed / outlier inputs (e.g. Laplace-noised activations in the B4
cross-scheme sweep) blew up ``p_mu`` → non-finite loss → nan estimate (seed-1,
20/72 cells). The fix (grad clipping + non-finite skip + a None guard) must make
the estimate either finite or an explicit ``None`` — never nan.
"""

from __future__ import annotations

import numpy as np

from talens.probes.club import club_mi_upper_bound


def _heavy_tailed(seed: int, n=400, d=64):
    """Correlated X,Y with fat Laplace tails + a few extreme outlier rows —
    the regime that diverged the unclipped trainer."""
    rng = np.random.default_rng(seed)
    z = rng.laplace(0.0, 1.0, (n, d))
    X = z + rng.laplace(0.0, 0.5, (n, d))
    Y = z + rng.laplace(0.0, 0.5, (n, d))
    out = rng.random(n) < 0.03                       # ~3% extreme outliers
    X[out] *= 50.0
    return X.astype(np.float32), Y.astype(np.float32)


def test_no_nan_on_heavy_tailed_inputs():
    for s in range(6):
        X, Y = _heavy_tailed(s)
        r = club_mi_upper_bound(X, Y, steps=200, hidden_size=64, seed=s)
        bits = r["club_mi_bits"]
        # never nan: either a finite estimate or an explicit None (filtered downstream)
        assert bits is None or np.isfinite(bits), f"seed {s}: {bits}"


def test_finite_estimate_is_sane_on_clean_correlated_input():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((400, 32))
    X = (z + 0.1 * rng.standard_normal((400, 32))).astype(np.float32)
    Y = (z + 0.1 * rng.standard_normal((400, 32))).astype(np.float32)
    r = club_mi_upper_bound(X, Y, steps=300, hidden_size=64, seed=0)
    assert r["club_mi_bits"] is not None and np.isfinite(r["club_mi_bits"])
    assert r["club_mi_bits"] > 0.0                   # correlated → positive MI bound
    assert r.get("diverged") is False
