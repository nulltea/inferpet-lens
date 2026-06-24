"""Geometry-only spectral channel-MI probe for embedding inversion under the
Gaussian mechanism — the matched, attack-independent leakage probe (continuous
analog of :mod:`channel_error_bounds`).

Channel: ``Y = e_0 + N(0, σ²I_d)`` with clean embedding ``e_0`` (deterministic
function of the secret text ``X``), ``Σ = Cov(e_0)``, eigenvalues ``λ_1≥…≥λ_d``.
The **spectral channel mutual information**

    ``I_G(σ) = ½ Σ_i log₂(1 + λ_i/σ²)``

upper-bounds the text leakage ``I(X;Y)=I(e_0;Y)`` (Gaussian max-entropy ceiling;
``I(X;Y) ≤ min{H(e_0), I_G(σ)}``), **localizes** it via the per-mode profile
``t_i = ½log₂(1+λ_i/σ²)`` and ``d_eff = #{i: λ_i ≥ σ²}`` (water-filling), and
**ceilings any attack's recovery** via Fano / rate–distortion. It is computed from
``(Σ, σ)`` ALONE — it never runs an inverter or sees the attack's observations
(the geometry-only invariant; cf. the rejected NNS-PVI). Proof + verification:
``claim:spectral-channel-mi-embedding-inversion`` /
``refine-logs/dp-stronger-attacks/vec2text-pooled/PROOF_PACKAGE.md``.

NumPy in / dict out, dependency-light (mirrors ``channel_error_bounds``); uses Torch
``linalg.eigvalsh`` on GPU for the ``d×d`` covariance eigendecomposition when available,
with a NumPy fallback.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:  # GPU accel optional; the math is identical on CPU
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

_LOG2 = math.log(2.0)


def _cov_eigvals(E0: np.ndarray | None, cov: np.ndarray | None, *, center: bool) -> np.ndarray:
    """Descending, non-negative eigenvalues of the clean-embedding covariance.

    Pass ``cov`` (``d×d``) directly, or ``E0`` (``n×d``) to form the sample
    covariance ``EᵀE/(n−1)`` (centered iff ``center``). GPU eigvalsh when available.
    """
    src = cov if cov is not None else E0
    if src is None:
        raise ValueError("provide either E0 (n×d) or cov (d×d)")
    if np.iscomplexobj(src):
        raise ValueError("complex inputs are not supported")
    if cov is not None:
        C = np.ascontiguousarray(cov, dtype=np.float64)
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError("cov must be a square matrix")
        if not np.allclose(C, C.T, rtol=1e-8, atol=1e-10):
            raise ValueError("cov must be symmetric")
        C = 0.5 * (C + C.T)  # symmetrize away round-off
    else:
        E = np.ascontiguousarray(E0, dtype=np.float64)
        if E.ndim != 2:
            raise ValueError("E0 must be a 2D (n×d) array")
        if E.shape[0] < 2:
            raise ValueError("need at least two samples for the sample covariance")
        if center:
            E = E - E.mean(axis=0, keepdims=True)
        C = (E.T @ E) / (E.shape[0] - 1)
    if not np.all(np.isfinite(C)):
        raise ValueError("covariance contains NaN or Inf")
    if _HAS_TORCH and torch.cuda.is_available():
        w = torch.linalg.eigvalsh(torch.from_numpy(C).cuda()).cpu().numpy()
    else:
        w = np.linalg.eigvalsh(C)
    tol = 1e-10 * max(1.0, float(np.max(np.abs(w))) if w.size else 1.0)
    if w.size and float(np.min(w)) < -tol:
        raise ValueError("covariance has significantly negative eigenvalues (not PSD)")
    return np.maximum(w, 0.0)[::-1].astype(np.float64)  # descending, clip round-off negatives


def _invert_gamma(tau: float, V: int) -> float:
    """Smallest ``D∈[0,(V-1)/V]`` with ``γ(D)=H_b(D)+D·log₂(V−1) ≥ τ`` (bisection).

    ``γ`` is strictly increasing on ``[0,(V-1)/V]`` with ``γ((V-1)/V)=log₂V``; if
    ``τ ≤ 0`` returns 0; if ``τ ≥ log₂V`` returns the max ``(V-1)/V``.
    """
    if not isinstance(V, (int, np.integer)) or V < 2:
        raise ValueError("vocab V must be an integer >= 2")
    Dmax = (V - 1) / V
    if tau <= 0.0:
        return 0.0
    if tau >= math.log2(V):
        return Dmax

    def gamma(D):
        if D <= 0.0:
            return 0.0
        Hb = -D * math.log2(D) - (1 - D) * math.log2(1 - D) if 0 < D < 1 else 0.0
        return Hb + D * math.log2(V - 1)

    lo, hi = 0.0, Dmax
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if gamma(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def spectral_channel_mi(
    E0: np.ndarray | None = None,
    sigma: float = 0.0,
    *,
    cov: np.ndarray | None = None,
    center: bool = True,
    H_X: float | None = None,
    H_e0: float | None = None,
    tail_ks: tuple[int, ...] | None = None,
    n_tokens: int | None = None,
    vocab: int | None = None,
) -> dict[str, Any]:
    """Spectral channel-MI probe of the Gaussian channel ``Y=e_0+N(0,σ²I)``.

    Inputs (geometry + channel only): ``E0`` (``n×d`` clean embeddings) or ``cov``
    (``d×d``); ``sigma>0``. Optional ``H_X`` (bits, secret entropy — e.g.
    ``log₂(#distinct texts)``) and ``H_e0`` (bits, embedding entropy) enable the
    recovery ceilings; ``n_tokens, vocab`` enable the rate–distortion per-token floor.

    Returns ``i_g_bits`` (``I_G``), per-mode ``t_i`` (bits), ``d_eff``, ``eigenvalues``,
    ``tail`` profile (``Σ_{i>k} t_i`` at ``tail_ks``), ``accessible_bit_ceiling``
    (``min{H_e0, I_G}``, a certified UPPER bound on the true ``I(X;Y)`` — not an exact
    count), ``fano_exact_ceiling`` (``(accessible+1)/H_X`` under a uniform prior), and
    ``rd_pertoken_floor`` (Shannon-LB lower bound on the per-token error rate).
    """
    if not math.isfinite(float(sigma)):
        raise ValueError("sigma must be finite")
    if sigma < 0.0:
        raise ValueError("sigma must be nonnegative")
    lam = _cov_eigvals(E0, cov, center=center)
    d = int(lam.size)
    out: dict[str, Any] = {"d": d, "sigma": float(sigma), "eigenvalues": lam}

    if sigma == 0.0:
        # σ→0 (Σ≠0): the channel is noiseless ⇒ I(e_0;Y)→H(e_0)<∞ while I_G→∞.
        # The binding ceiling is the discrete H(e_0); I_G is vacuous here.
        out.update(
            i_g_bits=float("inf"),
            t_i=np.full(d, float("inf")),
            d_eff=d,
            tail={},
            accessible_bit_ceiling=(float(H_e0) if H_e0 is not None else float("inf")),
            fano_exact_ceiling=(min(1.0, ((H_e0 + 1) / H_X)) if (H_X and H_e0 is not None) else None),
            rd_pertoken_floor=None,
            note="sigma<=0: I_G vacuous (→∞); binding ceiling is the discrete H(e0).",
        )
        return out

    s2 = sigma * sigma
    t = 0.5 * np.log1p(lam / s2) / _LOG2  # per-mode bits ½log2(1+λ/σ²)
    i_g = float(t.sum())
    d_eff = int((lam >= s2).sum())

    ks = tuple(tail_ks) if tail_ks is not None else (8, 32, 64, 128, 256, min(512, d), d)
    csum = np.cumsum(t)

    def tail_at(k: int) -> float:
        k = int(k)
        if k <= 0:
            return float(i_g)            # drop nothing
        if k >= d:
            return 0.0                   # keep all d modes (exact, no float residue)
        return float(csum[-1] - csum[k - 1])

    tail = {int(k): tail_at(k) for k in ks}

    accessible = min(float(H_e0), i_g) if H_e0 is not None else i_g
    fano = None
    if H_X is not None and H_X > 0:
        fano = min(1.0, (accessible + 1.0) / float(H_X))
    rd_floor = None
    if H_X is not None and n_tokens and vocab and vocab > 1:
        tau = max(0.0, (float(H_X) - accessible)) / float(n_tokens)
        rd_floor = _invert_gamma(tau, int(vocab))

    out.update(
        i_g_bits=i_g,
        t_i=t,
        d_eff=d_eff,
        tail=tail,
        accessible_bit_ceiling=accessible,
        fano_exact_ceiling=fano,
        rd_pertoken_floor=rd_floor,
    )
    return out


def spectral_channel_mi_diag(
    cov: np.ndarray,
    noise_var: np.ndarray,
    *,
    H_X: float | None = None,
    H_e0: float | None = None,
    n_tokens: int | None = None,
    vocab: int | None = None,
) -> dict[str, Any]:
    """Spectral channel-MI for an **anisotropic (diagonal)** Gaussian channel.

    Generalizes :func:`spectral_channel_mi` from ``N(0,σ²I)`` to ``N(0,D)`` with
    ``D = diag(noise_var)`` (per-coordinate variances ``σ²₁..σ²_d`` in the SAME basis
    as ``cov``). The Gaussian-channel MI ceiling on ``I(e_0;Y)`` is

        ``I_G = ½ log₂ det(I + D^{-1/2} Σ D^{-1/2}) = ½ Σ_i log₂(1 + μ_i)``

    where ``μ_i`` are the eigenvalues of the whitened covariance ``D^{-1/2} Σ D^{-1/2}``.
    Reduces exactly to the isotropic ``½Σ log₂(1+λ_i/σ²)`` when ``noise_var`` is constant.
    Geometry-only / attack-independent: depends solely on ``(Σ, D)``. ``d_eff`` counts
    modes with ``μ_i ≥ 1`` (water-filling: noise-variance-relative signal modes).

    Inputs: ``cov`` (``d×d`` symmetric PSD), ``noise_var`` (length-``d`` strictly positive).
    Optional ``H_X, H_e0, n_tokens, vocab`` enable the same Fano / rate–distortion ceilings.
    """
    C = np.ascontiguousarray(cov, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("cov must be a square d×d matrix")
    if not np.allclose(C, C.T, rtol=1e-8, atol=1e-10):
        raise ValueError("cov must be symmetric")
    C = 0.5 * (C + C.T)
    d = C.shape[0]
    v = np.ascontiguousarray(noise_var, dtype=np.float64).reshape(-1)
    if v.size != d:
        raise ValueError("noise_var must have length d == cov.shape[0]")
    if not np.all(np.isfinite(v)) or float(np.min(v)) <= 0.0:
        raise ValueError("noise_var must be finite and strictly positive")
    if not np.all(np.isfinite(C)):
        raise ValueError("covariance contains NaN or Inf")

    inv_sqrt = 1.0 / np.sqrt(v)
    W = (inv_sqrt[:, None] * C) * inv_sqrt[None, :]  # D^{-1/2} Σ D^{-1/2}
    W = 0.5 * (W + W.T)
    if _HAS_TORCH and torch.cuda.is_available():
        mu = torch.linalg.eigvalsh(torch.from_numpy(W).cuda()).cpu().numpy()
    else:
        mu = np.linalg.eigvalsh(W)
    mu = np.maximum(mu, 0.0)[::-1].astype(np.float64)  # descending, non-negative

    t = 0.5 * np.log1p(mu) / _LOG2  # per-mode bits ½log2(1+μ)
    i_g = float(t.sum())
    d_eff = int((mu >= 1.0).sum())

    accessible = min(float(H_e0), i_g) if H_e0 is not None else i_g
    fano = None
    if H_X is not None and H_X > 0:
        fano = min(1.0, (accessible + 1.0) / float(H_X))
    rd_floor = None
    if H_X is not None and n_tokens and vocab and vocab > 1:
        tau = max(0.0, (float(H_X) - accessible)) / float(n_tokens)
        rd_floor = _invert_gamma(tau, int(vocab))

    return {
        "d": int(d),
        "i_g_bits": i_g,
        "t_i": t,
        "whitened_eigenvalues": mu,
        "d_eff": d_eff,
        "distortion_total": float(v.sum()),
        "accessible_bit_ceiling": accessible,
        "fano_exact_ceiling": fano,
        "rd_pertoken_floor": rd_floor,
    }
