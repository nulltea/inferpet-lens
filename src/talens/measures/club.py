"""CLUB — Contrastive Log-ratio Upper Bound of mutual information.

Faithful port of Cheng et al. (ICML 2020), Linear95/CLUB. CLUB gives an
MI *upper* bound (most neural estimators give lower bounds), so pairing
it with PVI/MDL brackets the leakage from above. Here we estimate
``I(representation ; token-embedding)`` — both continuous — which
parallels the ridge attack's ``X → embedding`` map.

A variational network learns ``q(y|x) = N(μ(x), σ²(x))``; the upper
bound is the mean over positive pairs of ``log q(y_i|x_i)`` minus the
mean over all cross pairs ``log q(y_j|x_i)``. The network is trained by
maximising the Gaussian log-likelihood (``learning_loss = −loglikeli``).
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch
from torch import nn

_LN2 = float(np.log(2.0))

# Guards the global-RNG seed + net init so concurrent blocks (the
# orchestrator's thread pool) get reproducible, non-interleaved inits.
_INIT_LOCK = threading.Lock()


class CLUB(nn.Module):
    """Verbatim CLUB from Linear95/CLUB (variable names preserved)."""

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int):
        super().__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh(),
        )

    def get_mu_logvar(self, x_samples):
        return self.p_mu(x_samples), self.p_logvar(x_samples)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        positive = -((mu - y_samples) ** 2) / 2.0 / logvar.exp()
        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)
        negative = -((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.0 / logvar.exp()
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-((mu - y_samples) ** 2) / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)


def _club_estimate(net: CLUB, x: torch.Tensor, y: torch.Tensor) -> float:
    """CLUB MI estimate — mathematically **exact** equal to ``net.forward``
    but O(n·d) memory and O((n+m)·d) compute, with no ``(n, n, d)``
    intermediate.

    The negative term needs ``mean_j (y_j − μ_i)²`` for every ``(i, d)``.
    Expanding the square makes the mean over ``j`` a closed form in the
    first two moments of ``y``::

        mean_j (y_j − μ_i)² = E[y²] − 2·μ_i·E[y] + μ_i²

    so the full mean over all negatives is computed from two ``(d,)``
    moment vectors — no per-pair difference tensor, no chunking, no
    Monte-Carlo subsampling. Pure tensor ops, so it moves to GPU
    unchanged when ``x``/``y``/``net`` are on a device.
    """
    with torch.no_grad():
        mu, logvar = net.get_mu_logvar(x)        # (n, d)
        var = logvar.exp()
        positive = (-((mu - y) ** 2) / 2.0 / var).sum(dim=-1)   # paired, (n,)
        ybar = y.mean(dim=0)                      # E[y]    (d,)
        ysq = (y ** 2).mean(dim=0)                # E[y²]   (d,)
        mean_sq = ysq - 2.0 * mu * ybar + mu ** 2  # mean_j (y_j-μ_i)²  (n, d)
        negative = (-mean_sq / 2.0 / var).sum(dim=-1)           # (n,)
        return float((positive - negative).mean().item())


def _standardize(a: np.ndarray) -> np.ndarray:
    mean = a.mean(axis=0, keepdims=True)
    std = a.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((a - mean) / std).astype(np.float32)


def club_mi_upper_bound(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    hidden_size: int = 128,
    steps: int = 400,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    train_frac: float = 0.7,
    seed: int = 20260615,
    device: str | None = None,
) -> dict[str, Any]:
    """Estimate an MI upper bound between rows of ``X`` (representation)
    and ``Y`` (token embeddings). Both standardised per-feature for
    training stability. Trains ``q(y|x)`` on a train split and evaluates
    the bound on the held-out test split.

    NOTE on interpretation: CLUB is a *loose* upper bound — on a
    closed-form Gaussian it can overshoot the true MI by ~2–4× (verified
    in ``tests/test_analytic.py``). Treat its **magnitude** as an upper
    envelope and rely on its **rank** across layers/conditions for
    calibration; the small ``weight_decay`` curbs variational over-fitting
    that would otherwise inflate the bound further.

    ``device`` controls where the variational net trains — the one part of
    the measure stack that benefits from a GPU (the matmuls at full
    ``d``). Defaults to ``"cuda"`` when a device is available (ROCm
    presents as CUDA), else CPU; pass an explicit string to override.
    PVI/MDL stay on CPU (sklearn) regardless.
    """
    if X.shape[0] < 8:
        return {"club_mi_bits": None, "note": "too few rows"}

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xs = _standardize(X)
    Ys = _standardize(Y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(Xs.shape[0])
    cut = max(4, int(round(perm.size * train_frac)))
    tr, te = perm[:cut], perm[cut:]
    if te.size < 2:
        return {"club_mi_bits": None, "note": "empty test split"}

    xtr = torch.from_numpy(Xs[tr]).to(dev)
    ytr = torch.from_numpy(Ys[tr]).to(dev)
    xte = torch.from_numpy(Xs[te]).to(dev)
    yte = torch.from_numpy(Ys[te]).to(dev)

    # Seed + init under a lock so concurrent blocks stay reproducible
    # (kaiming init draws from the global RNG).
    with _INIT_LOCK:
        torch.manual_seed(seed)
        net = CLUB(Xs.shape[1], Ys.shape[1], hidden_size).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        loss = net.learning_loss(xtr, ytr)
        loss.backward()
        opt.step()

    net.eval()
    mi_nats = _club_estimate(net, xte, yte)
    return {
        "club_mi_nats": mi_nats,
        "club_mi_bits": mi_nats / _LN2,
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "hidden_size": hidden_size,
        "device": dev,
    }
