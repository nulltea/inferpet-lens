#!/usr/bin/env python3
"""SnD utility-recovery sweep — the UTILITY axis of the SnD privacy–utility tradeoff.

Privatize token embeddings with dχ-privacy (DxPrivacy), let the rest of the model produce the
pooled output embedding e_n, denoise it locally with a noise-aware transformer (Denoiser) →
e_d, and measure how much of the clean pooled embedding e_c the denoiser recovers across the
budget η. Also report teacher-forced perplexity/acc degradation under the dχ noise (generation-
utility cost; the denoiser does not touch that surface). Privacy axis stays in dp_leakage_sweep.

η is the dχ budget, NOT the Gaussian ε of dp_leakage_sweep — not interchangeable.

GPU: ONE process at a time; run via scripts/run_in_rocm.sh. Output JSON under refine-logs/snd/.

  scripts/run_in_rocm.sh python3 scripts/evals/snd_utility_sweep.py \
      --etas inf,100,50,10,1 --train-etas 50,10,1 --out refine-logs/snd/snd_utility_sweep.json
"""
from __future__ import annotations

import numpy as np


def recovery_metrics(e_c, e_n, e_d) -> dict:
    """cos/MSE of noised & denoised pooled embeddings vs clean, and the fraction of the gap closed."""
    def _cos(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return float((a * b).sum(1).mean())

    cos_n, cos_d = _cos(e_n, e_c), _cos(e_d, e_c)
    mse_n = float(((e_n - e_c) ** 2).mean())
    mse_d = float(((e_d - e_c) ** 2).mean())
    return {
        "cos_noised": cos_n, "cos_denoised": cos_d,
        "mse_noised": mse_n, "mse_denoised": mse_d,
        "recovery_cos": (cos_d - cos_n) / (1 - cos_n) if (1 - cos_n) > 1e-9 else 0.0,
        "recovery_mse": 1 - mse_d / mse_n if mse_n > 1e-12 else 0.0,
    }
