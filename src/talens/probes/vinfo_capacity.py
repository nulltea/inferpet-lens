"""Capacity-matched V-information — fixing class-PVI's ``d>n_val`` overfit.

The independent token-id V-family (``vinfo.v_information``) overfits in the
high-d regime: a free ``d→C`` softmax with ``d > n_val`` memorises the
validation split, so early-stopping is fooled, held-out log-loss → −∞, the
shuffle-control floor sits at ≈ −48 bits and PVI is non-monotonic under noise
(diagnosis: ``docs/handoffs/2026-06-18-independent-vfamily-attack-correlation.md``).

The fix keeps the **same independent target** (token-*id* classes, class-prior
null — NOT the attack's ridge→embedding map) but makes the estimator **well-posed
in ``d>n``** by bounding capacity, and **cheaper** than the full-d free softmax
(the hard cost constraint: PVI is already ~56–59% of every block). Four families:

* ``pca_softmax``     — PCA ``X→k`` (``k<n_val``) then linear softmax.
* ``randproj_softmax``— fixed Gaussian random projection ``X→k`` then softmax.
* ``gauss``           — Gaussian class-conditional (per-class mean + pooled
  diagonal variance; LDA-diagonal). Closed-form, **no iterative fit**.
* ``knn``             — k-NN soft class vote. Non-parametric, **no fit**.

All four return ``PVI = mean_test[ log₂ q(y|x) − log₂ q[∅](y) ]`` in bits with
``q[∅]`` the class prior, mirroring ``vinfo.v_information`` so they are drop-in
for the diag loop and the DP runner's ``panel()``. ``control="shuffle"`` is the
Hewitt–Liang control task (a healthy estimator ⇒ shuffle floor ≈ 0).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._probe import (
    class_log_prior,
    probe_log_softmax,
    row_split,
    standardize_fit,
    to_class_indices,
    train_softmax_probe,
)

_LN2 = np.log(2.0)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def _pca_basis(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` PCA basis from ``x`` (already standardised). Returns
    ``(mean, components)`` where ``components`` is ``(d, k)`` — project with
    ``(x - mean) @ components``. Economy SVD on the GPU (ROCm) when available —
    the full-``d`` SVD is the cost driver and the iGPU is ~50× the host CPU here
    — with a NumPy fallback. Deterministic."""
    mean = x.mean(axis=0, keepdims=True)
    xc = (x - mean).astype(np.float32)
    # centered data has rank ≤ n-1; clamp so we never return a zero-variance axis
    k = min(int(k), xc.shape[1], max(1, xc.shape[0] - 1))
    # Only the top-k RIGHT singular vectors are needed — a full SVD wastes effort
    # on the (n×d) U and all d singular vectors. Eigendecompose the (d×d)
    # covariance instead (d<n here): one matmul + a small eigh. The principal
    # axes are the top eigenvectors of Xᵀ X.
    try:
        import torch

        if torch.cuda.is_available():
            t = torch.from_numpy(np.ascontiguousarray(xc)).cuda()
            cov = t.T @ t                              # (d, d)
            evals, evecs = torch.linalg.eigh(cov)      # ascending eigenvalues
            comp = evecs[:, -k:].flip(1).contiguous().cpu().numpy().astype(np.float32)
            del t, cov, evals, evecs
            return mean, comp
    except Exception:
        pass
    cov = xc.T @ xc
    evals, evecs = np.linalg.eigh(cov)
    return mean, evecs[:, ::-1][:, :k].copy().astype(np.float32)


def _gaussian_logpost(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, num_classes: int, *, var_floor: float = 1e-3
) -> np.ndarray:
    """Log-posterior ``log q(c|x)`` for a Gaussian class-conditional with a
    **pooled diagonal** covariance (LDA-diagonal) + empirical class prior.
    Closed-form, ``O(n·d)``, well-posed for ``d>n``. Returns ``(n_te, C)``."""
    d = Xtr.shape[1]
    means = np.zeros((num_classes, d), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.float64)
    # pooled within-class variance (diagonal)
    sq_resid = np.zeros(d, dtype=np.float64)
    for c in range(num_classes):
        m = ytr == c
        nc = int(m.sum())
        counts[c] = nc
        if nc == 0:
            continue
        mu = Xtr[m].mean(axis=0)
        means[c] = mu
        sq_resid += ((Xtr[m] - mu) ** 2).sum(axis=0)
    pooled = sq_resid / max(1.0, (Xtr.shape[0] - np.count_nonzero(counts)))
    var = np.maximum(pooled, var_floor)
    log_prior = np.log((counts + 1.0) / (counts.sum() + num_classes))
    # log N(x; mu_c, diag(var)) up to the x-only constant (cancels in softmax)
    inv = 1.0 / var
    # ||x-mu||^2 weighted: expand to avoid (n_te, C, d)
    xte = Xte.astype(np.float64)
    x2 = (xte ** 2 * inv).sum(axis=1, keepdims=True)            # (n_te,1)
    mu2 = (means ** 2 * inv).sum(axis=1)                         # (C,)
    cross = xte @ (means * inv).T                               # (n_te,C)
    quad = x2 - 2.0 * cross + mu2                                # (n_te,C) = sum inv*(x-mu)^2
    loglik = -0.5 * quad - 0.5 * np.log(var).sum()
    logpost_un = loglik + log_prior[None, :]
    return logpost_un - _logsumexp(logpost_un, axis=1)[:, None]


def _knn_logpost(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, num_classes: int, *, k: int, alpha: float = 1.0
) -> np.ndarray:
    """Soft k-NN class posterior with Laplace smoothing. Returns ``(n_te, C)``
    log-probabilities. Euclidean distance in the (already reduced) space."""
    # squared euclidean via (a-b)^2 = a^2 - 2ab + b^2
    tr2 = (Xtr ** 2).sum(axis=1)                                 # (n_tr,)
    te2 = (Xte ** 2).sum(axis=1)                                 # (n_te,)
    d2 = te2[:, None] - 2.0 * (Xte @ Xtr.T) + tr2[None, :]       # (n_te, n_tr)
    k = min(k, Xtr.shape[0])
    nn = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]           # (n_te, k)
    votes = ytr[nn]                                             # (n_te, k)
    counts = np.zeros((Xte.shape[0], num_classes), dtype=np.float64)
    for c in range(num_classes):
        counts[:, c] = (votes == c).sum(axis=1)
    probs = (counts + alpha) / (k + alpha * num_classes)
    return np.log(probs)


def v_information_capacity(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fit_X: np.ndarray | None = None,
    fit_y: np.ndarray | None = None,
    family: str = "gauss",
    dim: int = 128,
    n_neighbors: int = 15,
    train_frac: float = 0.7,
    max_classes: int = 256,
    max_rows: int | None = None,
    l2: float = 1e-1,
    max_iter: int = 500,
    seed: int = 20260615,
    control: str = "none",
    control_seed: int = 20260616,
    return_pvi: bool = False,
) -> dict[str, Any]:
    """Capacity-matched class-PVI in bits. ``family`` ∈ {``pca_softmax``,
    ``randproj_softmax``, ``gauss``, ``knn``}. ``dim`` is the reduced
    dimensionality for the projection families (choose ``< n_val`` to keep the
    estimator well-posed). Mirrors ``vinfo.v_information``'s class selection,
    row-split, shuffle control and null (class prior).

    **Threat-model fit pairs.** V-information is the information a predictor can
    extract *given the data it is allowed to train on*. By default the reader is
    fit on a row-split of the released ``X`` itself — the correct in-model choice
    when the release distribution is attacker-reproducible (a public mechanism:
    e.g. differential privacy, where the adversary self-generates noised pairs).
    For a **secret-key** scheme the adversary cannot reproduce the released
    distribution, so it must train on its own accessible reps and transfer: pass
    ``fit_X`` / ``fit_y`` (the attack-accessible (representation, label) pairs,
    e.g. synthetic own-key reps) and the reader is fit on those and *scored on the
    released* ``(X, y)``. The probe then never sees a deployment-basis true-label
    pair the attack could not also obtain. ``classes`` are always defined by the
    released labels ``y``; ``fit_y`` is mapped onto them (rows whose class is
    absent from the released set are dropped)."""
    if X.shape[0] < 4:
        return {"v_information_bits": None, "note": "too few rows"}
    if dim < 1 or n_neighbors < 1:
        raise ValueError("dim and n_neighbors must be ≥ 1")
    if (fit_X is None) != (fit_y is None):
        raise ValueError("pass fit_X and fit_y together, or neither")

    y_idx_all, classes = to_class_indices(y)
    if classes.size > max_classes:
        counts = np.bincount(y_idx_all, minlength=classes.size)
        keep = np.argsort(counts)[::-1][:max_classes]
        keep_mask = np.isin(y_idx_all, keep)
        X, y = X[keep_mask], y[keep_mask]
        y_idx_all, classes = to_class_indices(y)
    num_classes = int(classes.size)

    if max_rows is not None and X.shape[0] > max_rows:
        sel = np.random.default_rng(seed).choice(X.shape[0], size=max_rows, replace=False)
        X, y_idx_all = X[sel], y_idx_all[sel]

    # Fit pairs (what the predictor may train on) vs eval pairs (the released reps we score).
    if fit_X is None:
        # in-model when the release is attacker-reproducible: row-split X against itself
        tr, te = row_split(X.shape[0], train_frac, seed)
        if tr.size == 0 or te.size == 0:
            return {"v_information_bits": None, "note": "empty split"}
        Xtr_raw, ytr = X[tr], y_idx_all[tr]
        Xte_raw, yte = X[te], y_idx_all[te]
    else:
        # explicit attack-accessible fit pairs; evaluate on the whole released (X, y)
        cls_to_idx = {int(c): i for i, c in enumerate(classes)}
        fyi = np.fromiter((cls_to_idx.get(int(c), -1) for c in np.asarray(fit_y)), dtype=np.int64,
                          count=len(fit_y))
        keep = fyi >= 0
        if not keep.any():
            return {"v_information_bits": None, "note": "no fit rows in released class set"}
        Xtr_raw, ytr = np.asarray(fit_X)[keep], fyi[keep]
        if max_rows is not None and Xtr_raw.shape[0] > max_rows:
            s = np.random.default_rng(seed).choice(Xtr_raw.shape[0], size=max_rows, replace=False)
            Xtr_raw, ytr = Xtr_raw[s], ytr[s]
        Xte_raw, yte = X, y_idx_all

    if control == "shuffle":                              # Hewitt–Liang control: corrupt the FIT labels
        ytr = ytr[np.random.default_rng(control_seed).permutation(ytr.size)]

    # standardise on fit stats (shared by every family)
    mean, std = standardize_fit(Xtr_raw)
    Xtr = ((Xtr_raw - mean) / std).astype(np.float32)
    Xte = ((Xte_raw - mean) / std).astype(np.float32)

    # Every family is capacity-matched by reducing to ``dim`` first: a free
    # softmax / Gaussian / kNN at full ``d`` (≫ rows/class) is overconfident and
    # un-calibrated (a naive full-d Gaussian floors at ≈ −log prior on shuffle).
    # ``randproj`` uses a fixed Gaussian sketch; the others use (unsupervised) PCA.
    if family == "randproj_softmax":
        rng = np.random.default_rng(seed + 101)
        # clamp the sketch dim so it stays a genuine reduction, well-posed in d>n
        k_eff = min(int(dim), Xtr.shape[1], max(1, Xtr.shape[0] - 1))
        R = (rng.standard_normal((Xtr.shape[1], k_eff)) / np.sqrt(k_eff)).astype(np.float32)
        Ztr, Zte = (Xtr @ R).astype(np.float32), (Xte @ R).astype(np.float32)
    else:
        pmean, comp = _pca_basis(Xtr, dim)
        Ztr = ((Xtr - pmean) @ comp).astype(np.float32)
        Zte = ((Xte - pmean) @ comp).astype(np.float32)
    eff_dim = Ztr.shape[1]

    if family in ("pca_softmax", "randproj_softmax"):
        probe = train_softmax_probe(Ztr, ytr, num_classes, l2=l2, max_iter=max_iter, seed=seed)
        logq = probe_log_softmax(probe, Zte)
    elif family == "gauss":
        logq = _gaussian_logpost(Ztr, ytr, Zte, num_classes)
    elif family == "knn":
        logq = _knn_logpost(Ztr, ytr, Zte, num_classes, k=n_neighbors)
    else:
        raise ValueError(f"unknown family {family!r}")

    log_prior = class_log_prior(ytr, num_classes)
    cond_nats = logq[np.arange(yte.size), yte]
    prior_nats = log_prior[yte]
    pvi_bits = (cond_nats - prior_nats) / _LN2
    top1_acc = float((logq.argmax(axis=1) == yte).mean())  # reader's token-id accuracy

    out: dict[str, Any] = {
        "v_information_bits": float(pvi_bits.mean()),
        "reader_top1_acc": top1_acc,
        "family": family,
        "eff_dim": int(eff_dim),
        "num_classes": num_classes,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "fit": "external" if fit_X is not None else "row_split",
        "control": control,
    }
    if family in ("pca_softmax", "randproj_softmax"):
        out["dim"] = int(dim)
    if family == "knn":
        out["n_neighbors"] = int(min(n_neighbors, tr.size))
    if return_pvi:
        out["pvi_bits"] = pvi_bits
    return out
