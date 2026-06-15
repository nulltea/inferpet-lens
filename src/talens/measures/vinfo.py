"""V-usable information and pointwise V-information (PVI).

Ethayarajh, Choi & Swayamdipta (ICML 2022); Xu et al. (ICLR 2020).
For representation ``X`` (the exposed operand) and secret ``Y`` (the
token id), with predictive family ``V`` = the softmax probe:

    PVI(x → y) = log₂ q[X](y | x) − log₂ q[∅](y)
    I_V(X → Y) = mean PVI = H_V(Y | ∅) − H_V(Y | X)   (bits)

``q[X]`` is the probe trained on the representation; ``q[∅]`` is the
class prior (the null model). Higher I_V means more token-identity
information a *bounded* adversary can use — the quantity hypothesised to
predict inversion-attack recovery.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._probe import (
    class_log_prior,
    probe_log_softmax,
    row_split,
    to_class_indices,
    train_softmax_probe,
)

_LN2 = np.log(2.0)


def v_information(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_frac: float = 0.7,
    max_classes: int = 4096,
    C: float = 1.0,
    max_iter: int = 200,
    seed: int = 20260615,
    return_pvi: bool = False,
) -> dict[str, Any]:
    """Estimate I_V(X→Y) in bits with a softmax-probe family. If the
    corpus has more than ``max_classes`` distinct ids, restricts to the
    most frequent ``max_classes`` (rows of rarer ids are dropped).
    """
    if X.shape[0] < 4:
        return {"v_information_bits": None, "note": "too few rows"}

    y_idx_all, classes = to_class_indices(y)
    if classes.size > max_classes:
        counts = np.bincount(y_idx_all, minlength=classes.size)
        keep = np.argsort(counts)[::-1][:max_classes]
        keep_mask = np.isin(y_idx_all, keep)
        X, y = X[keep_mask], y[keep_mask]
        y_idx_all, classes = to_class_indices(y)
    num_classes = int(classes.size)

    tr, te = row_split(X.shape[0], train_frac, seed)
    if tr.size == 0 or te.size == 0:
        return {"v_information_bits": None, "note": "empty split"}

    probe = train_softmax_probe(
        X[tr], y_idx_all[tr], num_classes, C=C, max_iter=max_iter, seed=seed
    )
    logq = probe_log_softmax(probe, X[te])          # (n_te, C) natural log
    log_prior = class_log_prior(y_idx_all[tr], num_classes)  # (C,)

    yte = y_idx_all[te]
    cond_nats = logq[np.arange(yte.size), yte]      # log q[X](y|x)
    prior_nats = log_prior[yte]                      # log q[∅](y)
    pvi_bits = (cond_nats - prior_nats) / _LN2

    out: dict[str, Any] = {
        "v_information_bits": float(pvi_bits.mean()),
        "h_y_given_x_bits": float(-cond_nats.mean() / _LN2),
        "h_y_prior_bits": float(-prior_nats.mean() / _LN2),
        "num_classes": num_classes,
        "n_train": int(tr.size),
        "n_test": int(te.size),
    }
    if return_pvi:
        out["pvi_bits"] = pvi_bits
    return out
