"""Shared softmax probe for the class-based IT measures (PVI, MDL).

The predictive family ``V`` is a multinomial-logistic probe ``q(y | x)``.
Backed by **scikit-learn's** ``LogisticRegression`` (a trusted,
deterministic solver) rather than a hand-rolled optimiser, so the bits
PVI and MDL report don't depend on our own convergence behaviour. CLUB
is the only measure that keeps a torch-trained net (it must — CLUB *is*
a variational neural estimator).

Because a softmax classifier can only place mass on classes seen in
training, the class-based measures operate on a *shared* class set
(row-split), unlike the vocab-disjoint inversion attacks. See
``docs/plans/it-leakage-estimation-set.md`` on aligning split regimes.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


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
    C: float = 1.0,
    max_iter: int = 200,
    seed: int = 0,
) -> dict:
    """Fit ``q(y|x)`` with multinomial logistic regression. ``C`` is the
    inverse L2-regularisation strength (sklearn convention). Returns the
    fitted classifier plus standardisation stats and the full class count
    (so :func:`probe_log_softmax` can return a dense ``num_classes`` row
    even when training missed some classes).
    """
    mean, std = standardize_fit(x_train)
    xs = (x_train - mean) / std
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
    )
    clf.fit(xs, y_idx_train)
    return {"clf": clf, "mean": mean, "std": std, "num_classes": num_classes}


def probe_log_softmax(probe: dict, x: np.ndarray) -> np.ndarray:
    """Natural-log class probabilities ``log q(y|x)``, shape
    ``(n, num_classes)``. Classes absent from training get a small
    floor probability so downstream cross-entropy stays finite.
    """
    xs = (x - probe["mean"]) / probe["std"]
    clf = probe["clf"]
    seen = clf.classes_
    lp_seen = clf.predict_log_proba(xs)  # (n, len(seen))
    n = xs.shape[0]
    C = probe["num_classes"]
    if seen.size == C and np.array_equal(seen, np.arange(C)):
        return lp_seen
    # Dense over all classes; unseen classes get a tiny floor.
    out = np.full((n, C), np.log(1e-12), dtype=np.float64)
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
