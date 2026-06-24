"""Matched, attack-independent probe for the BSS / accumulation channel.

The BSS attacks (:mod:`talens.attacks.bss`) exploit two geometric properties of the
observed operand: (1) JADE recovers sources when the whitened rows are **non-Gaussian**
(ICA separability); (2) JD-across-T recovers a shared demixing when the T observation
covariances **share eigenstructure**. This probe measures both *without ever running the
joint-diagonalization* — it is a function of whitened-row moments and covariance
eigenspectra only, so it could be computed with the attacks deleted (the probe ≠ attack
invariant; cf. :mod:`talens.measures.spectral_channel_mi`).

Two quantities, both canonical in **bits**:

* :func:`negentropy_bits` — JADE separability. Hyvärinen's negentropy approximation
  ``J(y) ≈ (1/12)·E[y³]² + (1/48)·kurt(y)²`` (nats, unit-variance ``y``) summed over the
  whitened rows, in bits (``/ln2``). Higher = more non-Gaussian = more ICA-separable.
* :func:`shared_spectral_capacity_bits` — JD accumulation. Form the averaged row
  covariance over the T-stack, ``C̄ = (1/T)·Σ_t U_t·U_tᵀ/d``; its spectral capacity
  ``½ Σ_i log₂(1 + λ_i/σ²)`` against a noise-floor ``σ² = median(bottom-half eigenvalues)``
  reuses :func:`talens.measures.spectral_channel_mi.spectral_channel_mi`. As T grows the
  shared (anisotropic) subspace concentrates relative to the floor — the capacity tracks how
  much shared structure a JD adversary can lock onto.
"""

from __future__ import annotations

import math

import numpy as np

from ..attacks.bss import _operands, _subsample, _whiten
from ..capture.types import CaptureSet
from ..transforms import Transform
from .spectral_channel_mi import spectral_channel_mi

_LN2 = math.log(2.0)


def _row_negentropy_nats(y: np.ndarray) -> float:
    """Sum over rows of Hyvärinen's negentropy approximation (nats).

    ``y`` is ``(m, T)`` whitened (each row ~ unit variance). For each row:
    ``J ≈ (1/12)·skew² + (1/48)·exkurt²``.
    """
    yc = y - y.mean(axis=1, keepdims=True)
    std = yc.std(axis=1) + 1e-12
    z = yc / std[:, None]
    skew = (z ** 3).mean(axis=1)
    exkurt = (z ** 4).mean(axis=1) - 3.0
    j = (1.0 / 12.0) * skew ** 2 + (1.0 / 48.0) * exkurt ** 2
    return float(np.sum(j))


def negentropy_bits(
    capture: CaptureSet,
    *,
    layer: int,
    kind: str,
    transform: Transform | None = None,
    max_dim: int = 64,
    max_features: int = 256,
    seed: int = 0,
) -> dict:
    """JADE-separability probe: median over prompts of the whitened-row negentropy (bits)."""
    rng = np.random.default_rng(seed)
    bits: list[float] = []
    for h, u in _operands(capture, kind, layer, transform):
        if u.size == 0:
            continue
        u, _h = _subsample(u, h, max_dim, max_features, rng)
        s = u.shape[0]
        if s < 4 or u.shape[1] < 2 * s:
            continue
        try:
            y, _w = _whiten(u, s)
        except np.linalg.LinAlgError:
            continue
        bits.append(_row_negentropy_nats(y) / _LN2)
    if not bits:
        return {"probe": "negentropy", "kind": kind, "layer": layer, "n": 0, "negentropy_bits": None}
    return {
        "probe": "negentropy",
        "kind": kind,
        "layer": layer,
        "n": len(bits),
        "negentropy_bits": float(np.median(bits)),
        "negentropy_bits_per_row": float(np.median(bits)) / float(min(max_dim, s)),
    }


def shared_spectral_capacity_bits(
    capture: CaptureSet,
    *,
    layer: int,
    kind: str,
    transform: Transform | None = None,
    t_values: tuple[int, ...] = (1, 2, 4, 8, 16),
    max_dim: int = 64,
    max_features: int = 256,
    seed: int = 0,
) -> dict:
    """JD-accumulation probe: spectral capacity (bits) of the averaged row covariance, per T.

    Mirrors the JD stacking (same non-overlapping length-T windows) but computes only the
    averaged covariance eigenspectrum + a Gaussian water-filling capacity — no joint-diag.
    """
    rng = np.random.default_rng(seed)
    ops: list[np.ndarray] = []
    for h, u in _operands(capture, kind, layer, transform):
        if u.size == 0:
            continue
        u, _h = _subsample(u, h, max_dim, max_features, rng)
        if u.shape[0] < 4:
            continue
        ops.append(u.astype(np.float64))
    if not ops:
        return {"probe": "shared_spectral_capacity", "kind": kind, "layer": layer, "cap_per_t": {}}
    ref = ops[0].shape
    ops = [o for o in ops if o.shape == ref]
    s, d = ref

    cap_per_t: dict[int, list[float]] = {t: [] for t in t_values}
    deff_per_t: dict[int, list[int]] = {t: [] for t in t_values}
    for t_target in t_values:
        for start in range(0, len(ops) - t_target + 1, t_target):
            window = ops[start : start + t_target]
            # Averaged row covariance over the T observations (the space JD operates in).
            c_avg = np.zeros((s, s), dtype=np.float64)
            for u in window:
                uc = u - u.mean(axis=1, keepdims=True)
                c_avg += (uc @ uc.T) / max(d, 1)
            c_avg /= len(window)
            c_avg = 0.5 * (c_avg + c_avg.T)
            evals = np.linalg.eigvalsh(c_avg)
            evals = np.maximum(evals, 0.0)
            # Noise floor = median of the bottom-half eigenvalues (isotropic residual).
            floor = float(np.median(evals[: max(1, s // 2)]))
            sigma = math.sqrt(floor) if floor > 0 else 0.0
            if sigma <= 0:
                continue
            res = spectral_channel_mi(cov=c_avg, sigma=sigma, center=False)
            cap_per_t[t_target].append(float(res["i_g_bits"]))
            deff_per_t[t_target].append(int(res["d_eff"]))
    return {
        "probe": "shared_spectral_capacity",
        "kind": kind,
        "layer": layer,
        "cap_per_t": {int(t): (float(np.median(v)) if v else None) for t, v in cap_per_t.items()},
        "d_eff_per_t": {int(t): (int(np.median(v)) if v else None) for t, v in deff_per_t.items()},
    }
