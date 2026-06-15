"""Train / val / test splitters — row-shuffle and vocab-disjoint.

Ported from the aloepri harness ``common.py``. The **vocab-disjoint**
split is the paper-faithful methodology: train and test never share a
token id, forcing the inverter / probe to generalise rather than
memorise a per-token bias. The same splits drive both the attacks and
the IT measures so their numbers describe the same generalisation
regime.
"""

from __future__ import annotations

import numpy as np


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic row-shuffle split. Auto-scales to 70/15/15 of
    whatever's available when the corpus is smaller than the request.
    """
    rng = np.random.default_rng(seed)
    n_total = X.shape[0]
    if n_total == 0:
        ex = np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=X.dtype)
        ey = np.zeros((0,), dtype=y.dtype)
        return ex, ey, ex, ey, ex, ey
    perm = rng.permutation(n_total)
    if n_train + n_val + n_test > n_total:
        n_train_eff = max(int(n_total * 0.7), min(4, n_total - 2))
        n_val_eff = max(int(n_total * 0.15), 1)
        n_test_eff = max(n_total - n_train_eff - n_val_eff, 1)
    else:
        n_train_eff, n_val_eff, n_test_eff = n_train, n_val, n_test
    tr = perm[:n_train_eff]
    va = perm[n_train_eff : n_train_eff + n_val_eff]
    te = perm[n_train_eff + n_val_eff : n_train_eff + n_val_eff + n_test_eff]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def vocab_disjoint_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vocab-disjoint split — partition the *distinct token ids* into
    three disjoint sets; every ``(X[i], y[i])`` goes to whichever set
    holds ``y[i]``. ``n_train``/``n_val``/``n_test`` count distinct ids,
    not rows. Auto-scales to a 70/15/15 vocab split when there are too
    few distinct ids.
    """
    rng = np.random.default_rng(seed)
    if X.shape[0] == 0:
        ex = np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=X.dtype)
        ey = np.zeros((0,), dtype=y.dtype)
        return ex, ey, ex, ey, ex, ey
    unique_ids = np.unique(y)
    if unique_ids.size == 0:
        ex = np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=X.dtype)
        ey = np.zeros((0,), dtype=y.dtype)
        return ex, ey, ex, ey, ex, ey

    n_ids = unique_ids.size
    if n_train + n_val + n_test > n_ids:
        n_tr = max(int(n_ids * 0.70), min(2, n_ids - 2))
        n_va = max(int(n_ids * 0.15), 1)
        n_te = max(n_ids - n_tr - n_va, 1)
    else:
        n_tr, n_va, n_te = n_train, n_val, n_test

    perm = rng.permutation(n_ids)
    train_ids = set(unique_ids[perm[:n_tr]].tolist())
    val_ids = set(unique_ids[perm[n_tr : n_tr + n_va]].tolist())
    test_ids = set(unique_ids[perm[n_tr + n_va : n_tr + n_va + n_te]].tolist())

    m_tr = np.isin(y, list(train_ids))
    m_va = np.isin(y, list(val_ids))
    m_te = np.isin(y, list(test_ids))
    return X[m_tr], y[m_tr], X[m_va], y[m_va], X[m_te], y[m_te]
