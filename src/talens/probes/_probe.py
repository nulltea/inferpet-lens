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

Regularisation is **AdamW decoupled weight decay + early stopping** on an
internal validation split (``l2`` is the weight decay). An in-loss ridge
term proved fragile: scale-correct tuning across blocks of very different
feature width (dense ``resid_post`` d=2560 vs wide zero-padded
``attn_score``) is hard, and an under-regularised fixed-step fit *diverges*
on under-determined blocks — assigning ~0 probability to held-out tokens
and yielding nonsensical (−1000s of bits) PVI. Early stopping at the
best-held-out-CE iterate bounds the held-out loss regardless of scale.

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
    l2: float = 1e-1,
    max_iter: int = 500,
    lr: float = 0.05,
    device: str | None = None,
    seed: int = 0,
    val_frac: float = 0.15,
    eval_every: int = 10,
) -> dict:
    """Fit ``q(y|x)`` — multinomial logistic via fixed-step torch **AdamW**
    with **early stopping** on an internal validation split.

    ``l2`` is AdamW's decoupled weight decay; ``max_iter`` the max Adam
    steps. Early stopping (keep the weights with the best held-out CE seen)
    is the real guard: an un-/under-regularised fixed-step fit *diverges* on
    under-determined blocks (wide ``attn_score`` features, few rows/class),
    assigning ~0 probability to held-out tokens and producing nonsensical
    (hugely negative) PVI. Stopping at the best-generalising iterate bounds
    the held-out CE regardless of feature scale. Falls back to plain
    fixed-step when there are too few rows to carve a val split.

    Defaults ``lr=0.05`` + ``l2(weight_decay)=0.1`` were tuned against the
    sklearn oracle on real deep-layer blocks: the earlier ``lr=0.2`` jumped
    straight from underfit to *overconfident* logits (high accuracy, huge
    held-out CE, PVI ≪ 0) with no good iterate for early stopping to keep.
    Lower lr converges; stronger decay caps logit magnitude → calibrated
    probabilities. This recovered PVI ≈ +5.5 bits on resid L18 (matching
    sklearn), vs −13 before. See docs/dev/perf_assumptions.md.
    """
    dev = _resolve_device(device)
    torch.manual_seed(seed)
    mean, std = standardize_fit(x_train)
    xn = ((x_train - mean) / std).astype(np.float32)
    yi = y_idx_train.astype(np.int64)

    n = xn.shape[0]
    n_val = int(round(val_frac * n))
    use_es = n_val >= 1 and (n - n_val) >= 1
    perm = np.random.default_rng(seed).permutation(n) if use_es else np.arange(n)
    tr_idx, va_idx = perm[n_val:], perm[:n_val]

    xtr = torch.from_numpy(xn[tr_idx]).to(dev)
    ytr = torch.from_numpy(yi[tr_idx]).to(dev)
    if use_es:
        xva = torch.from_numpy(xn[va_idx]).to(dev)
        yva = torch.from_numpy(yi[va_idx]).to(dev)

    d = xn.shape[1]
    W = torch.zeros((d, num_classes), device=dev, requires_grad=True)
    b = torch.zeros(num_classes, device=dev, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=l2)

    best_val, best_W, best_b = float("inf"), None, None
    for step in range(max_iter):
        opt.zero_grad()
        F.cross_entropy(xtr @ W + b, ytr).backward()
        opt.step()
        if use_es and (step % eval_every == 0 or step == max_iter - 1):
            with torch.no_grad():
                vce = F.cross_entropy(xva @ W + b, yva).item()
            if vce < best_val:
                best_val, best_W, best_b = vce, W.detach().clone(), b.detach().clone()

    Wf, bf = (best_W, best_b) if best_W is not None else (W.detach(), b.detach())
    return {
        "W": Wf,
        "b": bf,
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
