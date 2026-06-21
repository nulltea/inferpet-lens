"""Shredder — "Learning Noise Distributions to Protect Inference Privacy"
(Mireshghallah et al., ASPLOS'20, `1905.11814`; code ``mireshghallah/shredder-v3``).

Classic edge/cloud *split inference*: the network is cut at one layer, the edge
adds noise to the intermediate activation ``a`` and ships ``a′ = a + n`` to the
cloud. Unlike AloePri (static, invertible, permutation-based), Shredder is a
**learned, lossy** defense: the noise tensor is trained (weights frozen) to
minimise ``1/SNR`` — a differentiable proxy for ``I(input; a′)`` — subject to a
task-accuracy term, then fit to a **Laplace** distribution and sampled fresh per
input (the descending sort-order of the learned tensor is preserved to keep its
inter-element structure).

Two pieces:

* :class:`ShredderStaticLaplace` — a Transform that adds i.i.d. Laplace noise at a
  split (the *static* proxy: skips the offline training, anchors the lossy-noise
  corner cheaply and honestly).
* :func:`train_shredder_noise` — the faithful learned-noise trainer: optimises a
  per-element noise tensor over a provided differentiable task loss to
  ``min  task_loss(a+n) + λ · (1/SNR)``, then :func:`fit_laplace` recovers the
  scale ``b`` and sort-order for per-input sampling via
  :class:`ShredderLearnedLaplace`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch


class ShredderStaticLaplace:
    """Additive i.i.d. Laplace(0, ``b``) cover at a split activation. ``b`` is the
    privacy knob (larger ``b`` → lower SNR → less leakage, more accuracy loss).
    Deterministic per ``(seed, prompt_index)`` so the Transform stays pure."""

    name = "shredder_static"

    def __init__(self, *, b: float, seed: int = 0):
        self.b = float(b)
        self._seed = seed

    def __call__(self, operand: torch.Tensor, *, prompt_index: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(self._seed + int(prompt_index))
        u = torch.rand(operand.shape, generator=g) - 0.5         # (-0.5, 0.5)
        lap = -self.b * torch.sign(u) * torch.log1p(-2.0 * u.abs())
        return operand + lap.to(operand.device, operand.dtype)


def fit_laplace(noise: np.ndarray) -> dict[str, Any]:
    """MLE Laplace fit: ``μ`` = median, ``b`` = mean absolute deviation from the
    median. Returns ``{loc, scale, order}`` where ``order`` is the descending
    rank-order of the flattened tensor (Shredder preserves it on resampling)."""
    flat = np.asarray(noise, dtype=np.float64).ravel()
    loc = float(np.median(flat))
    b = float(np.mean(np.abs(flat - loc)))
    order = np.argsort(-flat)                                    # descending
    return {"loc": loc, "scale": b, "order": order, "shape": tuple(noise.shape)}


def _snr(signal: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """SNR ``E[a²]/E[n²]`` (signal power / noise power). **Minimising** SNR grows
    the noise → lowers ``I(input; a+n)``; this is the privacy term's right sign."""
    return signal.pow(2).mean() / noise.pow(2).mean().clamp_min(1e-12)


def _inv_snr(signal: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Reverse SNR ``E[n²]/E[a²]`` — reported as the privacy proxy (rises as the
    learned noise grows)."""
    return noise.pow(2).mean() / signal.pow(2).mean().clamp_min(1e-12)


def train_shredder_noise(
    activations: torch.Tensor,
    task_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    lam: float = 1.0,
    steps: int = 300,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a per-element noise tensor (same shape as ``activations``) to
    ``min  task_loss_fn(a + n) + λ·SNR``, weights frozen (only ``n`` learns).

    Minimising ``SNR = E[a²]/E[n²]`` *rewards larger noise* (more privacy) while
    ``task_loss_fn`` (over the frozen downstream net) keeps utility — Shredder's
    accuracy↔leakage trade. (Earlier ``+λ/SNR`` was the wrong sign: it shrinks
    the noise.) Returns the fitted Laplace params + the privacy (1/SNR) and noise
    trajectories — both should *rise* as ``n`` grows."""
    torch.manual_seed(seed)
    a = activations.detach().float()
    n = (1e-3 * torch.randn_like(a)).requires_grad_(True)   # tiny init → finite SNR
    opt = torch.optim.Adam([n], lr=lr)
    inv_snr_hist: list[float] = []
    noise_mad_hist: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = task_loss_fn(a + n) + lam * _snr(a, n)
        loss.backward()
        opt.step()
        inv_snr_hist.append(float(_inv_snr(a, n).detach()))
        noise_mad_hist.append(float(n.detach().abs().mean()))
    laplace = fit_laplace(n.detach().cpu().numpy())
    laplace["inv_snr_hist"] = inv_snr_hist
    laplace["inv_snr_final"] = inv_snr_hist[-1] if inv_snr_hist else 0.0
    laplace["noise_mad_hist"] = noise_mad_hist
    return laplace


class ShredderLearnedLaplace:
    """Sample fresh Laplace noise from a *learned* fit and re-impose its
    descending sort-order (preserving the trained inter-element structure), then
    add to the split activation. The inference-time half of
    :func:`train_shredder_noise`."""

    name = "shredder_learned"

    def __init__(self, fit: dict[str, Any], *, seed: int = 0):
        self.loc, self.b = fit["loc"], fit["scale"]
        self.order = np.asarray(fit["order"])
        self._seed = seed

    def __call__(self, operand: torch.Tensor, *, prompt_index: int) -> torch.Tensor:
        g = np.random.default_rng(self._seed + int(prompt_index))
        m = int(np.prod(operand.shape))
        draw = g.laplace(self.loc, self.b, size=m)
        # re-impose the learned descending order onto a slot-aligned tensor
        ranked = np.empty(m, dtype=np.float64)
        order = self.order[:m] if self.order.shape[0] >= m else np.argsort(-draw)
        ranked[order] = np.sort(draw)[::-1]
        lap = torch.from_numpy(ranked.reshape(operand.shape))
        return operand + lap.to(operand.device, operand.dtype)
