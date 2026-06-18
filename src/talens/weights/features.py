"""Obfuscation-invariant row signatures ``φ`` for the τ-recovery family.

Each ``φ`` maps a weight table ``(N, d)`` to per-row feature vectors that
are *invariant* (or nearly so) to whatever the obfuscation does to a row,
while still identifying *which* plaintext row a given obfuscated row came
from. The attack matches on ``φ`` and the measures estimate leakage on
``φ``, so plaintext (width ``d``) and obfuscated (width ``d̃``) tables must
yield comparable signatures.

* :func:`sorted_quantile` — VMA's RowSort signature. Sorting a row's
  values is invariant to any **column permutation** of that row (and is
  dimension-agnostic via fixed quantile positions), so when the
  obfuscation differs from plaintext "almost only by row- and
  column-permutations" (AloePri's low-noise regime, §7.3) the sorted
  signature of an obfuscated row still matches its plaintext partner.
  Noise is what breaks it — the knob the calibration sweeps.
"""

from __future__ import annotations

import numpy as np


def sorted_quantile(W: np.ndarray, *, bins: int = 64) -> np.ndarray:
    """Per-row sorted-value signature: ``bins`` evenly-spaced quantiles of
    each row, mean-centred and L2-normalised. Shape ``(N, bins)``,
    independent of the input width — so plaintext and obfuscated tables of
    different widths produce directly comparable signatures.
    """
    qs = np.linspace(0.0, 1.0, bins)
    feat = np.quantile(W, qs, axis=1).T.astype(np.float64)  # (N, bins)
    feat -= feat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    return (feat / np.clip(norm, 1e-12, None)).astype(np.float32)


def raw(W: np.ndarray) -> np.ndarray:
    """Identity signature — the raw rows. Usable by CLUB (which accepts
    differing ``x``/``y`` widths) and by a trained inverter, but **not** by
    cosine matching across differing widths. Returned L2-normalised so
    cosine matching is well-defined when widths do agree.
    """
    W = W.astype(np.float32)
    norm = np.linalg.norm(W, axis=1, keepdims=True)
    return W / np.clip(norm, 1e-12, None)


_FEATURES = {"sorted_quantile": sorted_quantile, "raw": raw}


def get_feature(name: str):
    """Resolve a feature name to its ``φ``. ``sorted_quantile`` takes a
    ``bins`` kwarg; ``raw`` takes none — callers that need ``bins`` should
    use :func:`sorted_quantile` directly."""
    if name not in _FEATURES:
        raise ValueError(f"unknown feature {name!r}; have {sorted(_FEATURES)}")
    return _FEATURES[name]
