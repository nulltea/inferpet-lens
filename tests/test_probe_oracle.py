"""Oracle test: the production torch-GPU softmax probe must agree with
the trusted scikit-learn reference.

The torch probe (LBFGS) is what runs in production (GPU, no GIL); sklearn
is the correctness oracle. On a fixed dataset they should reach
near-identical held-out cross-entropy (hence near-identical PVI/MDL),
which is what lets us trust the GPU probe's leakage numbers. Runs on CPU
in the venv (torch auto-resolves to CPU when no GPU is present).
"""

from __future__ import annotations

import numpy as np

from talens.measures._probe import (
    probe_log_softmax,
    sklearn_probe_log_softmax,
    sklearn_train_softmax_probe,
    train_softmax_probe,
)


def _blobs(n_per: int, d: int, n_classes: int, sep: float, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((n_classes, d)) * sep
    Xs, ys = [], []
    for c in range(n_classes):
        Xs.append(means[c] + noise * rng.standard_normal((n_per, d)))
        ys.append(np.full(n_per, c))
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def _mean_true_ce_nats(logq: np.ndarray, y: np.ndarray) -> float:
    return float(-logq[np.arange(y.size), y].mean())


def test_torch_probe_matches_sklearn_oracle():
    X, y = _blobs(n_per=200, d=16, n_classes=5, sep=3.0, noise=1.0, seed=0)
    n = X.shape[0]
    tr, te = np.arange(int(0.7 * n)), np.arange(int(0.7 * n), n)
    C = int(np.unique(y).size)

    torch_probe = train_softmax_probe(X[tr], y[tr], C, l2=1e-4, max_iter=200, device="cpu")
    skl_probe = sklearn_train_softmax_probe(X[tr], y[tr], C, C=1e4, max_iter=500)

    ce_torch = _mean_true_ce_nats(probe_log_softmax(torch_probe, X[te]), y[te])
    ce_skl = _mean_true_ce_nats(sklearn_probe_log_softmax(skl_probe, X[te]), y[te])

    # Both solve the same (lightly-regularised) multinomial logistic →
    # near-identical held-out cross-entropy, hence near-identical PVI.
    assert abs(ce_torch - ce_skl) < 0.05


def test_torch_probe_separates_perfectly_when_easy():
    X, y = _blobs(n_per=200, d=8, n_classes=4, sep=8.0, noise=0.3, seed=1)
    n = X.shape[0]
    tr, te = np.arange(int(0.7 * n)), np.arange(int(0.7 * n), n)
    probe = train_softmax_probe(X[tr], y[tr], 4, device="cpu")
    ce = _mean_true_ce_nats(probe_log_softmax(probe, X[te]), y[te])
    assert ce < 0.1  # well-separated → near-zero cross-entropy
