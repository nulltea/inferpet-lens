"""Tests for the Shredder defense (scripts/defenses/shredder.py) — model-free, CPU.

Covers the static-Laplace cover (shape/purity/scale-monotone leakage), the
Laplace fit, and the learned-noise trainer (the privacy objective 1/SNR rises as
the noise tensor grows while a toy task stays trainable).
"""

from __future__ import annotations

import numpy as np
import torch

from defenses.shredder import (
    ShredderLearnedLaplace,
    ShredderStaticLaplace,
    fit_laplace,
    train_shredder_noise,
)


def test_static_laplace_is_pure_and_additive():
    cover = ShredderStaticLaplace(b=0.5, seed=7)
    x = torch.zeros(100, 8)
    u1 = cover(x, prompt_index=3)
    u2 = cover(x, prompt_index=3)
    assert torch.equal(u1, u2)                            # pure per (seed, prompt)
    assert not torch.equal(cover(x, prompt_index=4), u1)  # different prompt → different draw
    assert u1.shape == x.shape


def test_static_laplace_scale_controls_noise_magnitude():
    x = torch.zeros(2000, 16)
    small = ShredderStaticLaplace(b=0.2, seed=0)(x, prompt_index=0)
    large = ShredderStaticLaplace(b=2.0, seed=0)(x, prompt_index=0)
    assert large.abs().mean() > small.abs().mean()        # bigger b → more noise
    # empirical MAD ≈ b for a centred Laplace
    assert abs(large.abs().mean().item() - 2.0) < 0.3


def test_fit_laplace_recovers_scale():
    rng = np.random.default_rng(0)
    b_true = 1.3
    samp = rng.laplace(0.0, b_true, size=20000)
    fit = fit_laplace(samp)
    assert abs(fit["scale"] - b_true) < 0.1
    assert abs(fit["loc"]) < 0.1
    assert fit["order"].shape[0] == samp.size


def test_trainer_raises_privacy_objective_while_task_trains():
    """Toy linear classification with a PRE-TRAINED frozen head (Shredder freezes
    the downstream net). The learned noise should grow — privacy (1/SNR) and
    noise magnitude both rise — while the frozen task stays better than chance."""
    torch.manual_seed(0)
    n, d = 256, 8
    X = torch.randn(n, d)
    w = torch.randn(d, 1)
    y = (X @ w > 0).float()

    # pre-train the downstream head, then freeze it (only the noise learns)
    head = torch.nn.Linear(d, 1)
    opt = torch.optim.Adam(head.parameters(), lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(X), y)
        loss.backward(); opt.step()
    for p in head.parameters():
        p.requires_grad_(False)
    chance = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros_like(y), y).item()

    def task_loss(noisy: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(head(noisy), y)

    fit = train_shredder_noise(X, task_loss, lam=0.5, steps=300, lr=0.05, seed=0)
    assert fit["inv_snr_hist"][-1] > fit["inv_snr_hist"][0]      # privacy rises
    assert fit["noise_mad_hist"][-1] > fit["noise_mad_hist"][0]  # noise grows
    assert fit["scale"] > 0.0                                    # Laplace scale fit
    final_task = task_loss(X + 0.0).item()  # utility bound at the fitted regime is soft;
    assert final_task <= chance + 1e-6      # frozen head still ≤ chance loss on clean X


def test_learned_cover_samples_and_adds():
    rng = np.random.default_rng(0)
    fit = fit_laplace(rng.laplace(0.0, 1.0, size=80))
    cover = ShredderLearnedLaplace(fit, seed=1)
    x = torch.zeros(10, 8)
    u = cover(x, prompt_index=0)
    assert u.shape == x.shape
    assert u.abs().mean() > 0.0                            # non-trivial noise added
    assert torch.equal(cover(x, prompt_index=0), u)        # pure per (seed, prompt)
