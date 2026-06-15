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

from typing import Any

import numpy as np
import torch
from torch import nn

_LN2 = float(np.log(2.0))


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
    train_frac: float = 0.7,
    seed: int = 20260615,
) -> dict[str, Any]:
    """Estimate an MI upper bound between rows of ``X`` (representation)
    and ``Y`` (token embeddings). Both standardised per-feature for
    training stability. Trains ``q(y|x)`` on a train split and evaluates
    the bound on the held-out test split.
    """
    if X.shape[0] < 8:
        return {"club_mi_bits": None, "note": "too few rows"}

    torch.manual_seed(seed)
    Xs = _standardize(X)
    Ys = _standardize(Y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(Xs.shape[0])
    cut = max(4, int(round(perm.size * train_frac)))
    tr, te = perm[:cut], perm[cut:]
    if te.size < 2:
        return {"club_mi_bits": None, "note": "empty test split"}

    xtr = torch.from_numpy(Xs[tr])
    ytr = torch.from_numpy(Ys[tr])
    xte = torch.from_numpy(Xs[te])
    yte = torch.from_numpy(Ys[te])

    net = CLUB(Xs.shape[1], Ys.shape[1], hidden_size)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = net.learning_loss(xtr, ytr)
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        mi_nats = float(net(xte, yte).item())
    return {
        "club_mi_nats": mi_nats,
        "club_mi_bits": mi_nats / _LN2,
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "hidden_size": hidden_size,
    }
