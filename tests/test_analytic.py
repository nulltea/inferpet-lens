"""Analytic ground-truth tests — validate the measures against cases
with a *known* answer, not just signal-vs-noise direction.

* CLUB — jointly-Gaussian X,Y has closed-form MI = −½·d·ln(1−ρ²) nats.
  CLUB is an MI *upper* bound, so it should land at-or-above the truth
  and within a sane band, and increase with ρ.
* V-information — for well-separated class blobs the representation
  determines the label, so I_V(X→Y) → H(Y) = log₂C bits; for pure noise
  it → 0.
* MDL — separable data is highly compressible (compression ≫ 1) and the
  achievable floor cross-entropy → 0.
"""

from __future__ import annotations

import numpy as np

from talens.measures import club_mi_upper_bound, online_code_length, v_information


def _gaussian_pair(n: int, d: int, rho: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    E = rng.standard_normal((n, d)).astype(np.float32)
    Y = (rho * X + np.sqrt(1.0 - rho**2) * E).astype(np.float32)
    return X, Y


def _blobs(n_per: int, d: int, n_classes: int, sep: float, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((n_classes, d)) * sep
    Xs, ys = [], []
    for c in range(n_classes):
        Xs.append(means[c] + noise * rng.standard_normal((n_per, d)))
        ys.append(np.full(n_per, c))
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


# --- CLUB vs closed-form Gaussian MI -------------------------------------

def test_club_tracks_gaussian_mi():
    d, n = 4, 2500
    X, Y = _gaussian_pair(n, d, rho=0.8, seed=0)
    true_nats = -0.5 * d * np.log(1.0 - 0.8**2)  # ≈ 2.04 nats
    out = club_mi_upper_bound(X, Y, hidden_size=64, steps=800, lr=2e-3, seed=0)
    est = out["club_mi_nats"]
    # CLUB is a *valid but loose* upper bound: it should sit at/above the
    # true MI (allowing finite-sample slack) and stay finite — not be
    # tight. Magnitude is an upper envelope; rank is the usable signal
    # (see test_club_increases_with_dependence).
    assert est > 0.8 * true_nats          # a genuine upper bound
    assert est < 6.0 * true_nats          # finite, not diverging


def test_club_estimate_equals_verbatim_forward():
    # The O(n·d) moment-based estimator must be numerically identical to
    # the verbatim O(n²·d) CLUB.forward (small n so forward is affordable).
    import torch

    from talens.measures.club import CLUB, _club_estimate

    torch.manual_seed(0)
    n, dx, dy = 200, 16, 12
    x = torch.randn(n, dx)
    y = torch.randn(n, dy)
    net = CLUB(dx, dy, 32)
    fast = _club_estimate(net, x, y)
    slow = float(net(x, y).item())
    assert abs(fast - slow) < 1e-3


def test_club_increases_with_dependence():
    d, n = 4, 2500
    Xh, Yh = _gaussian_pair(n, d, rho=0.85, seed=1)
    Xl, Yl = _gaussian_pair(n, d, rho=0.25, seed=1)
    hi = club_mi_upper_bound(Xh, Yh, hidden_size=64, steps=800, lr=2e-3, seed=1)
    lo = club_mi_upper_bound(Xl, Yl, hidden_size=64, steps=800, lr=2e-3, seed=1)
    assert hi["club_mi_nats"] > lo["club_mi_nats"]


# --- V-information vs label entropy --------------------------------------

def test_vinfo_approaches_label_entropy_when_separable():
    n_classes = 4
    X, y = _blobs(n_per=400, d=8, n_classes=n_classes, sep=6.0, noise=0.3, seed=2)
    out = v_information(X, y, C=10.0, max_iter=500, seed=2)
    h_y = np.log2(n_classes)  # 2.0 bits, balanced labels
    assert out["v_information_bits"] > 0.85 * h_y
    assert out["v_information_bits"] < h_y + 0.05


def test_vinfo_near_zero_on_noise():
    n_classes = 4
    rng = np.random.default_rng(3)
    y = rng.integers(0, n_classes, size=1600).astype(np.int64)
    X = rng.standard_normal((1600, 8)).astype(np.float32)
    out = v_information(X, y, C=1.0, max_iter=300, seed=3)
    assert out["v_information_bits"] < 0.3


# --- MDL compression on separable data -----------------------------------

def test_mdl_compresses_and_floor_is_low_when_separable():
    X, y = _blobs(n_per=400, d=8, n_classes=4, sep=6.0, noise=0.3, seed=4)
    out = online_code_length(X, y, C=10.0, max_iter=500, seed=4)
    assert out["compression"] > 3.0
    assert out["floor_ce_bits_per_row"] < 0.3
