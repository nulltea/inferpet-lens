"""Local differential privacy — Gaussian mechanism on the input embedding (a defense).

Repo rule: defenses live in scripts/defenses/. evals/spikes import and apply them.

Registered as a forward hook on the embedding layer: clip each row to L2 ≤ C, add N(0, σ²I); the
noise then propagates through the network to whatever depth is captured. σ=0 is clip-only (≈ clean).
Per-row sensitivity is C (add/remove-to-zero adjacency); the caller sets σ = C·z/ε.
"""
from __future__ import annotations

import torch


class LocalDP:
    def __init__(self, C: float, sigma: float):
        self.C, self.sigma = C, sigma

    def __call__(self, mod, inp, out):
        f = out.float()
        n = f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        f = f * (self.C / n).clamp_max(1.0)
        if self.sigma > 0:
            f = f + self.sigma * torch.randn_like(f)
        return f.to(out.dtype)
