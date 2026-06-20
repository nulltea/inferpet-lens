"""M0 sanity for the capacity-matched V-information families.

A healthy estimator on a *separable* synthetic task must (1) report positive
PVI (signal present), (2) have a shuffle-control floor ≈ 0 (NOT the −48 b of the
overfit free softmax), and (3) fall monotonically as the signal is buried in
noise. These run model-free on the host CPU venv.
"""

from __future__ import annotations

import numpy as np
import pytest

from talens.measures.vinfo_capacity import v_information_capacity

FAMILIES = ["pca_softmax", "randproj_softmax", "gauss", "knn"]


def _separable(n_per_class: int, d: int, num_classes: int, noise: float, seed: int = 0):
    """High-d (d > n_val regime), linearly separable class means + Gaussian
    noise. At large ``noise`` the per-dim SNR collapses and PCA picks noise
    directions, so a healthy measure must decay."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((num_classes, d)) * 3.0
    X, y = [], []
    for c in range(num_classes):
        X.append(centers[c] + noise * rng.standard_normal((n_per_class, d)))
        y.append(np.full(n_per_class, c))
    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    p = rng.permutation(y.size)
    return X[p], y[p]


@pytest.mark.parametrize("family", FAMILIES)
def test_signal_positive_and_shuffle_floor_near_zero(family: str):
    # d=512 >> rows/class: the regime where the free softmax overfits to −48 b.
    X, y = _separable(n_per_class=40, d=512, num_classes=8, noise=1.0, seed=1)
    kw = dict(family=family, dim=32, n_neighbors=11, max_classes=8, seed=7)

    real = v_information_capacity(X, y, **kw)["v_information_bits"]
    shuf = v_information_capacity(X, y, control="shuffle", **kw)["v_information_bits"]

    assert real is not None and shuf is not None
    # signal present on separable data
    assert real > 0.5, f"{family}: expected positive PVI, got {real:.3f}"
    # healthy shuffle floor ≈ 0 — the whole point vs class-PVI's −48 b
    assert -2.0 < shuf < 1.0, f"{family}: shuffle floor {shuf:.3f} not ≈ 0"
    # selectivity is clearly positive
    assert real - shuf > 0.5


@pytest.mark.parametrize("family", FAMILIES)
def test_monotone_under_noise(family: str):
    kw = dict(family=family, dim=32, n_neighbors=11, max_classes=8, seed=3)
    vals = []
    for noise in [0.5, 6.0, 30.0]:
        X, y = _separable(n_per_class=60, d=512, num_classes=8, noise=noise, seed=2)
        vals.append(v_information_capacity(X, y, **kw)["v_information_bits"])
    # non-increasing (allow a small tolerance for estimator jitter)
    assert vals[0] + 0.3 >= vals[1] >= vals[2] - 0.3, f"{family}: non-monotone {vals}"
    assert vals[0] > vals[2] + 0.2, f"{family}: no decay under noise {vals}"


def test_eff_dim_reported_below_full():
    X, y = _separable(n_per_class=40, d=512, num_classes=8, noise=1.0, seed=1)
    out = v_information_capacity(X, y, family="pca_softmax", dim=32, max_classes=8)
    assert out["eff_dim"] <= 32  # capacity bounded well below d=512
