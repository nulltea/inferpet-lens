"""Embedding-utility probes: fidelity of a (pooled) output embedding vs the clean one, and the
denoiser recovery of the clean embedding from a noised one.

These operate on already-captured embedding matrices (N, d) — the caller produces clean / noised /
denoised embeddings under its scheme; the probe only sees the vectors, so the metric is identical
across schemes. For anisotropic LLM hidden states, standardize per-dim by clean stats BEFORE calling
these (raw cosine is dominated by outlier dims) — the caller owns that choice.
"""
from __future__ import annotations

import numpy as np

from .result import UtilityResult, _retention


def _rowwise_cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return float((a * b).sum(1).mean())


def embedding_fidelity(e_def: np.ndarray, e_clean: np.ndarray) -> UtilityResult:
    """Mean row-wise cosine of a defended/recovered embedding to the clean one (retention = clamp(cos,
    0, 1); clean baseline cos = 1). extra.mse = mean squared error."""
    cos = _rowwise_cos(e_def, e_clean)
    mse = float(((e_def - e_clean) ** 2).mean())
    return UtilityResult("embedding_cosine", 1.0, cos, _retention(1.0, cos, higher_is_better=True),
                         higher_is_better=True, extra={"mse": mse})


def embedding_recovery(e_clean: np.ndarray, e_noised: np.ndarray, e_denoised: np.ndarray) -> dict:
    """Denoiser recovery of the clean embedding: cos/MSE of the noised baseline and the denoised
    output vs clean, plus the fraction of the gap the denoiser closes. Returns a dict (composite of
    two :func:`embedding_fidelity` calls + two recovery fractions):

      recovery_cos = (cos_denoised − cos_noised) / (1 − cos_noised)   # share of the cosine gap closed
      recovery_mse = 1 − mse_denoised / mse_noised                    # share of the noised MSE removed
    """
    noised = embedding_fidelity(e_noised, e_clean)
    denoised = embedding_fidelity(e_denoised, e_clean)
    cos_n, cos_d = noised.defended, denoised.defended
    mse_n, mse_d = noised.extra["mse"], denoised.extra["mse"]
    return {
        "cos_noised": cos_n, "cos_denoised": cos_d, "mse_noised": mse_n, "mse_denoised": mse_d,
        "recovery_cos": (cos_d - cos_n) / (1 - cos_n) if (1 - cos_n) > 1e-9 else 0.0,
        "recovery_mse": 1 - mse_d / mse_n if mse_n > 1e-12 else 0.0,
    }
