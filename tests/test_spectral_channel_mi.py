"""Model-free unit tests for the geometry-only spectral channel-MI probe.

Runs on the host CPU `.venv` (no model, no GPU). Locks the verified identities of
`claim:spectral-channel-mi-embedding-inversion`.
"""
import math

import numpy as np
import pytest

from talens.measures.spectral_channel_mi import spectral_channel_mi, _invert_gamma


def _random_cov(d, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    return scale * (A @ A.T) / d  # SPD


def test_gaussian_exact_logdet():
    # T2 with equality for the Gaussian case: I_G == ½ log2 det(I + Σ/σ²).
    d, sigma = 16, 0.7
    cov = _random_cov(d, seed=1)
    out = spectral_channel_mi(cov=cov, sigma=sigma)
    exact = 0.5 * math.log2(np.linalg.det(np.eye(d) + cov / sigma**2))
    assert out["i_g_bits"] == pytest.approx(exact, rel=1e-9, abs=1e-9)


def test_per_mode_sums_to_total():
    out = spectral_channel_mi(cov=_random_cov(12, seed=2), sigma=0.5)
    assert float(out["t_i"].sum()) == pytest.approx(out["i_g_bits"], rel=1e-12, abs=1e-9)


def test_monotone_decreasing_in_sigma():
    cov = _random_cov(20, seed=3)
    vals = [spectral_channel_mi(cov=cov, sigma=s)["i_g_bits"] for s in (0.1, 0.3, 1.0, 3.0, 10.0)]
    assert all(a > b for a, b in zip(vals, vals[1:])), vals


def test_large_sigma_goes_to_zero():
    out = spectral_channel_mi(cov=_random_cov(8, seed=4), sigma=1e6)
    assert out["i_g_bits"] < 1e-6


def test_d_eff_counts_modes_above_sigma2():
    # eigenvalues of diag are exactly the diagonal; choose σ² between two of them.
    cov = np.diag([10.0, 4.0, 1.0, 0.25, 0.01])
    out = spectral_channel_mi(cov=cov, sigma=1.0)  # σ²=1 → λ≥1 are {10,4,1} → d_eff=3
    assert out["d_eff"] == 3
    assert np.allclose(np.sort(out["eigenvalues"])[::-1], [10, 4, 1, 0.25, 0.01])


def test_tail_profile_endpoints():
    cov = _random_cov(10, seed=5)
    out = spectral_channel_mi(cov=cov, sigma=0.6, tail_ks=(0, 5, 10))
    assert out["tail"][0] == pytest.approx(out["i_g_bits"], abs=1e-9)   # drop nothing
    assert out["tail"][10] == pytest.approx(0.0, abs=1e-9)              # keep all d modes


def test_ceilings_and_accessible():
    cov = _random_cov(16, seed=6)
    H_X, H_e0 = 50.0, 30.0
    out = spectral_channel_mi(cov=cov, sigma=0.05, H_X=H_X, H_e0=H_e0)  # low σ → I_G large
    # accessible = min{H_e0, I_G}; with tiny σ, I_G > H_e0 → capped at H_e0
    assert out["accessible_bit_ceiling"] == pytest.approx(min(H_e0, out["i_g_bits"]))
    assert out["fano_exact_ceiling"] == pytest.approx(min(1.0, (out["accessible_bit_ceiling"] + 1) / H_X))


def test_sigma_zero_is_vacuous_infinite():
    out = spectral_channel_mi(cov=_random_cov(6, seed=7), sigma=0.0, H_e0=12.0)
    assert math.isinf(out["i_g_bits"])
    assert out["accessible_bit_ceiling"] == 12.0  # discrete cap binds


def test_E0_matrix_path_matches_cov_path():
    rng = np.random.default_rng(8)
    E0 = rng.standard_normal((500, 10))
    cov = np.cov(E0, rowvar=False)  # centered sample covariance (ddof=1)
    a = spectral_channel_mi(E0=E0, sigma=0.4, center=True)["i_g_bits"]
    b = spectral_channel_mi(cov=cov, sigma=0.4)["i_g_bits"]
    assert a == pytest.approx(b, rel=1e-6)


def test_invert_gamma_monotone_and_endpoints():
    V = 256
    assert _invert_gamma(0.0, V) == 0.0
    assert _invert_gamma(math.log2(V) + 1, V) == pytest.approx((V - 1) / V)
    d1, d2 = _invert_gamma(1.0, V), _invert_gamma(3.0, V)
    assert 0 < d1 < d2 < (V - 1) / V
