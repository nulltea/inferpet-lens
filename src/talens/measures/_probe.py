"""Shared softmax probe for the class-based IT measures (PVI, MDL).

A standardised multinomial-logistic probe ``q(y | x)`` over a fixed
class set, trained by full-batch Adam with L2. Used as the predictive
family ``V`` for V-information and as the coder for MDL online-coding.

Because a softmax classifier can only put mass on classes seen in
training, the class-based measures operate on a *shared* class set
(row-split), unlike the vocab-disjoint inversion attacks. See
``docs/plans/it-leakage-estimation-set.md`` on aligning split regimes
for calibration.
"""

from __future__ import annotations

import numpy as np
import torch


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def train_softmax_probe(
    x_train: np.ndarray,
    y_idx_train: np.ndarray,
    num_classes: int,
    *,
    l2: float = 1e-3,
    steps: int = 300,
    lr: float = 0.1,
    seed: int = 0,
) -> dict:
    """Fit ``q(y|x)``. Returns a dict with the linear weights and the
    standardisation stats. Deterministic given ``seed``.
    """
    torch.manual_seed(seed)
    mean, std = standardize_fit(x_train)
    xs = torch.from_numpy(((x_train - mean) / std).astype(np.float32))
    yt = torch.from_numpy(y_idx_train.astype(np.int64))
    d = xs.shape[1]
    w = torch.zeros((d, num_classes), requires_grad=True)
    b = torch.zeros(num_classes, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = xs @ w + b
        loss = torch.nn.functional.cross_entropy(logits, yt) + l2 * (w * w).sum()
        loss.backward()
        opt.step()
    return {
        "w": w.detach(),
        "b": b.detach(),
        "mean": mean,
        "std": std,
        "num_classes": num_classes,
    }


def probe_log_softmax(probe: dict, x: np.ndarray) -> np.ndarray:
    """Return natural-log class probabilities ``log q(y|x)``, shape
    ``(n, num_classes)``.
    """
    xs = torch.from_numpy(((x - probe["mean"]) / probe["std"]).astype(np.float32))
    logits = xs @ probe["w"] + probe["b"]
    return torch.log_softmax(logits, dim=1).numpy()


def to_class_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map token ids to contiguous class indices ``0..C-1``. Returns
    ``(y_idx, classes)`` where ``classes[y_idx] == y``."""
    classes = np.unique(y)
    y_idx = np.searchsorted(classes, y)
    return y_idx.astype(np.int64), classes


def row_split(n: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic row-shuffle train/test index split over a shared
    class set (so the softmax probe can score test rows)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = max(1, min(n - 1, int(round(n * train_frac))))
    return perm[:cut], perm[cut:]


def class_log_prior(y_idx_train: np.ndarray, num_classes: int) -> np.ndarray:
    """Smoothed natural-log class prior from training frequencies — the
    null model ``q(y | ∅)``.
    """
    counts = np.bincount(y_idx_train, minlength=num_classes).astype(np.float64)
    probs = (counts + 1.0) / (counts.sum() + num_classes)
    return np.log(probs)
