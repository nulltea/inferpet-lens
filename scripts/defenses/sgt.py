"""Stained Glass Transform (SGT) — learned **heteroscedastic** Gaussian noise on a
released embedding, with an MI-budget loss (arXiv 2506.09452).

Scheme-agnostic core stays clean: this is an external defense (callers/tests only).
SGT adds anisotropic Gaussian noise ``N(0, D)`` shaped to the embedding covariance
``Σ`` so that, at a fixed MI budget ``B`` (channel-MI ``I_G`` in bits), it spends
*less* distortion than isotropic DP — better utility at matched privacy.

We build the noise per-mode in the PCA basis of ``Σ`` (eigvals ``λ``, eigvecs ``V``):
per-mode variance ``v_i``; the channel-MI is ``I_G = ½ Σ_i log₂(1+λ_i/v_i)`` (basis-
invariant; matches :func:`talens.measures.spectral_channel_mi.spectral_channel_mi_diag`
with ``cov=diag(λ)``). Raw-space noise is ``n = (z·√v) Vᵀ``, ``z~N(0,I)``.

Three shapes, each constructed to hit a *target* ``I_G = B`` exactly (bisection):
- ``iso``          : ``v_i = σ²`` constant — the isotropic DP baseline.
- ``sgt_opt``      : minimize total distortion ``Σ v_i`` s.t. ``I_G=B``
                     (reverse-water-filling; the SGT MI-budget optimum — more noise on
                     high-λ modes). The "learned" allocation at convergence.
- ``sig_preserve`` : worst-case matched-budget control — noise on the LOW-λ tail, leaving
                     high-signal modes cleaner, still hitting ``I_G=B``. If recovery here
                     ≫ ``sgt_opt`` at the same ``B``, the scalar ``I_G`` is shape-blind.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_LOG2 = math.log(2.0)


def fit_covariance(E0: np.ndarray, *, center: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigendecompose ``Σ = Cov(E0)`` → (``λ`` desc≥0, ``V`` eigvecs cols, ``mean``).

    Estimate from a LARGE clean-embedding pool so ``Σ`` is full-rank / well-conditioned;
    the shape constructors and the matched probe both consume this single ``Σ``.
    """
    E = np.ascontiguousarray(E0, dtype=np.float64)
    mean = E.mean(axis=0) if center else np.zeros(E.shape[1])
    Ec = E - mean[None, :]
    C = (Ec.T @ Ec) / (E.shape[0] - 1)
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)          # ascending
    order = np.argsort(w)[::-1]       # descending
    lam = np.maximum(w[order], 0.0)
    V = V[:, order]
    return lam.astype(np.float64), V.astype(np.float64), mean.astype(np.float64)


def _ig_bits(lam: np.ndarray, v: np.ndarray) -> float:
    """Channel-MI ``I_G = ½ Σ log₂(1+λ_i/v_i)`` (bits) for per-mode variances ``v``."""
    return float(np.sum(0.5 * np.log1p(lam / v) / _LOG2))


def _bisect(f, lo, hi, target, iters=100):
    """Bisect monotone-decreasing ``f`` for ``f(x)=target`` on ``[lo,hi]``."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > target:   # f decreasing in x → too much MI ⇒ raise x
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _floor(lam: np.ndarray) -> float:
    """Tiny positive variance floor (numerical) relative to the signal scale."""
    s = float(lam[lam > 0].mean()) if np.any(lam > 0) else 1.0
    return max(1e-12, 1e-9 * s)


def shape_iso(lam: np.ndarray, B: float) -> np.ndarray:
    """Isotropic ``v_i = σ²``; σ chosen so ``I_G = B``."""
    fl = _floor(lam)
    f = lambda s2: _ig_bits(lam, np.full_like(lam, max(s2, fl)))
    s2 = _bisect(f, fl, float(lam.max()) * 1e6 + 1.0, B)
    return np.full_like(lam, max(s2, fl))


def shape_sgt_opt(lam: np.ndarray, B: float) -> np.ndarray:
    """Distortion-optimal heteroscedastic allocation at ``I_G=B`` (reverse water-filling).

    Lagrangian min ``Σv_i + γ·I_G`` ⇒ per mode ``v_i² + λ_i v_i = a λ_i`` with
    ``a = γ/(2 ln2)`` ⇒ ``v_i = (−λ_i + √(λ_i²+4 a λ_i))/2``. Increasing ``a`` lowers
    ``I_G``; bisect ``a`` to hit ``B``. Null modes (λ_i=0) get the variance floor.
    """
    fl = _floor(lam)

    def v_of_a(a):
        v = 0.5 * (-lam + np.sqrt(lam * lam + 4.0 * a * lam))
        return np.maximum(v, fl)

    f = lambda a: _ig_bits(lam, v_of_a(a))
    a = _bisect(f, 0.0, float(lam.max()) * 1e6 + 1.0, B)
    return v_of_a(a)


def shape_tail_dump(lam: np.ndarray, B: float, *, tail_frac: float = 0.5, dump: float = 100.0) -> np.ndarray:
    """Invariance control: isotropic noise on the head, **huge** extra noise dumped on the
    low-λ tail — all at matched ``I_G=B``.

    The bottom ``tail_frac`` of modes get variance ``dump · λ_max`` (≫ their λ, so they
    contribute ~0 to ``I_G``); the head gets a single isotropic ``σ²`` re-bisected so the
    whole spectrum still hits ``B``. Distortion balloons but ``I_G`` and the head SNR are
    unchanged — if recovery is unchanged vs ``iso``, the probe correctly ignores tail noise
    (the other direction of shape-invariance from ``iso`` vs ``sgt_opt``).
    """
    fl = _floor(lam)
    d = lam.size
    k = max(1, int(round((1.0 - tail_frac) * d)))  # head size
    is_head = np.zeros(d, dtype=bool)
    is_head[:k] = True
    vt = dump * float(lam.max()) + 1.0

    def f(s2):
        v = np.full_like(lam, vt)
        v[is_head] = max(s2, fl)
        return _ig_bits(lam, v)

    s2 = _bisect(f, fl, float(lam.max()) * 1e6 + 1.0, B)
    v = np.full_like(lam, vt)
    v[is_head] = max(s2, fl)
    return v


SHAPES = {"iso": shape_iso, "sgt_opt": shape_sgt_opt, "tail_dump": shape_tail_dump}


@dataclass
class SGT:
    """Apply heteroscedastic Gaussian noise of per-mode variance ``v`` (PCA basis ``V``)
    to clipped raw embeddings: ``e' = clip(e,C) + (z·√v) Vᵀ``, ``z~N(0,I)``.

    ``name`` carries the shape + budget for the report layer. Pure given ``(seed, rows)``.
    """

    lam: np.ndarray
    V: np.ndarray
    v: np.ndarray
    clip_C: float
    shape: str
    budget_bits: float
    seed: int = 0

    @property
    def name(self) -> str:
        return f"sgt[shape={self.shape},B={self.budget_bits:g}b]"

    @property
    def distortion_total(self) -> float:
        return float(self.v.sum())

    def apply(self, emb: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Clip rows to ``clip_C`` then add the shaped noise (NumPy, attack-side)."""
        rng = rng or np.random.default_rng(self.seed)
        scale = np.minimum(1.0, self.clip_C / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9))
        out = (emb * scale).astype(np.float64)
        z = rng.standard_normal((emb.shape[0], self.v.size))
        noise = (z * np.sqrt(self.v)[None, :]) @ self.V.T
        return (out + noise).astype(np.float32)


def build_sgt(lam, V, *, shape: str, budget_bits: float, clip_C: float, seed: int = 0) -> SGT:
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}; pick {list(SHAPES)}")
    v = SHAPES[shape](lam, float(budget_bits))
    return SGT(lam=lam, V=V, v=v, clip_C=clip_C, shape=shape, budget_bits=float(budget_bits), seed=seed)
