"""Shared softmax probe for the class-based IT measures (PVI, MDL).

The predictive family ``V`` is a multinomial-logistic probe ``q(y | x)``.
Production backend is **torch on GPU** (Adam, fixed steps): the fit is
``(N×d)@(d×C)`` matmuls, which the GPU eats — sklearn on CPU is
GIL-bound under threading and too slow at d≈2560 / C≈2500. Adam at a
fixed step budget replaced an earlier LBFGS+strong_wolfe fit, whose line
search thrashed to ~1.9k closure evals on the (hard, non-separable)
real activation→token problem (~68 s/fit, ~4 h/run); fixed-step Adam
converges to the same held-out cross-entropy in a deterministic
~300 fwd/bwd passes. Trust is kept by validating the torch probe against
**scikit-learn as an oracle** (``tests/test_probe_oracle.py``): the
sklearn implementation lives here too, and the test asserts the two agree
on held-out log-likelihood.

The ridge penalty is **mean**-reduced (``l2 * mean(W²)``) to match the
mean-reduced cross-entropy — a raw ``sum`` over the d·C ≈ 6.4 M weights
made the loss scale depend on d/C and shifted with the MDL prefix size,
which ill-conditioned the old line search.

Auto-device like CLUB: uses ``cuda`` (ROCm) when available, else CPU
(so the venv tests run on CPU unchanged). Because a softmax classifier
can only score classes seen in training, the class-based measures use a
shared class set (row-split) — see ``docs/research/attacks_setting.md``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _resolve_device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def train_softmax_probe(
    x_train: np.ndarray,
    y_idx_train: np.ndarray,
    num_classes: int,
    *,
    l2: float = 1e-4,
    max_iter: int = 300,
    lr: float = 0.2,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    """Fit ``q(y|x)`` — multinomial logistic via fixed-step torch Adam on
    the GPU. ``max_iter`` is the number of full-batch Adam steps; ``l2`` is
    the (mean-CE-relative) ridge penalty on the weights.
    """
    dev = _resolve_device(device)
    torch.manual_seed(seed)
    mean, std = standardize_fit(x_train)
    xs = torch.from_numpy(((x_train - mean) / std).astype(np.float32)).to(dev)
    yt = torch.from_numpy(y_idx_train.astype(np.int64)).to(dev)
    d = xs.shape[1]
    W = torch.zeros((d, num_classes), device=dev, requires_grad=True)
    b = torch.zeros(num_classes, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)

    for _ in range(max_iter):
        opt.zero_grad()
        loss = F.cross_entropy(xs @ W + b, yt) + l2 * (W * W).mean()
        loss.backward()
        opt.step()
    return {
        "W": W.detach(),
        "b": b.detach(),
        "mean": mean,
        "std": std,
        "num_classes": num_classes,
        "device": dev,
    }


def probe_log_softmax(probe: dict, x: np.ndarray) -> np.ndarray:
    """Natural-log class probabilities ``log q(y|x)``, ``(n, num_classes)``."""
    dev = probe["device"]
    xs = torch.from_numpy(((x - probe["mean"]) / probe["std"]).astype(np.float32)).to(dev)
    with torch.no_grad():
        logits = xs @ probe["W"] + probe["b"]
        return torch.log_softmax(logits, dim=1).cpu().numpy()


# --- scikit-learn oracle (correctness reference; used by the oracle test) ---

def sklearn_train_softmax_probe(
    x_train: np.ndarray,
    y_idx_train: np.ndarray,
    num_classes: int,
    *,
    C: float = 1.0,
    max_iter: int = 200,
    seed: int = 0,
) -> dict:
    """Trusted CPU reference: sklearn multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression

    mean, std = standardize_fit(x_train)
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, random_state=seed)
    clf.fit((x_train - mean) / std, y_idx_train)
    return {"clf": clf, "mean": mean, "std": std, "num_classes": num_classes}


def sklearn_probe_log_softmax(probe: dict, x: np.ndarray) -> np.ndarray:
    clf = probe["clf"]
    lp_seen = clf.predict_log_proba((x - probe["mean"]) / probe["std"])
    seen = clf.classes_
    C = probe["num_classes"]
    if seen.size == C and np.array_equal(seen, np.arange(C)):
        return lp_seen
    out = np.full((x.shape[0], C), np.log(1e-12), dtype=np.float64)
    out[:, seen] = lp_seen
    return out


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
