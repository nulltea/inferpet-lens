"""Blind-source-separation / accumulation attacks on the KV/QKV surface.

Faithful array-math ports of the aloepri BSS drivers (``run_gram_error``,
``run_jade``, ``run_jd``) adapted to consume a :class:`~talens.capture.types.CaptureSet`
through the agnostic :class:`~talens.transforms.Transform` seam. No vendored
``sys.path``, no snapshot loader — the attacks read per-prompt ``(H, U)`` operand
pairs (``H`` = plaintext ground truth, ``U`` = exposed observation; under
:class:`~talens.transforms.Identity`, ``U == H``).

Three attacks, all over a ``(kind, layer)`` slice:

* :func:`gram_error` — row-Gram fingerprint side channel. cos-normalised Frobenius
  distance between ``G_U = U·Uᵀ`` and ``G_H = H·Hᵀ`` (range [0,√2]; lower = more
  fingerprintable). On plaintext ``U == H`` ⇒ ≈ 0. *Not* a per-token recovery.
* :func:`jade` — JADE ICA single-observation source recovery (Cardoso & Souloumiac
  1993): joint-diagonalise the 4th-order cumulant matrices of the whitened
  observation. Metric = p95 of the Hungarian-aligned |cosine| between recovered
  sources and true rows (max benefit-of-the-doubt to the attacker).
* :func:`jd` — joint-diagonalisation ACROSS a stack of T observation covariances
  (Belouchrani et al. 1997, SOBI-style), different prompts at one ``(kind, layer)``.
  Recovers a single shared demixing; metric = p95 cosine at each ``T``. The
  accumulation curve: does recovery climb with the observation count?

The row axis is the operand's first dim (query/token positions for kq/kqv_out,
sequence positions for resid_post); the feature axis (d) is treated as the BSS
"sample" axis. Costs are bounded by ``max_dim`` (row cap, JADE cumulant is O(m⁴·T))
and ``max_features`` (feature/sample cap).
"""

from __future__ import annotations

import numpy as np

from ..capture.types import CaptureSet
from ..transforms import Identity, Transform

# ---------------------------------------------------------------------------
# Joint-diagonalisation primitive (numba-JIT if available, numpy fallback).
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba optional
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def _decorator(fn):
            return fn

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _decorator

    def prange(*args):  # type: ignore[no-redef]
        return range(*args)


@njit(cache=True, fastmath=True)
def _joint_diag_one(q: np.ndarray, max_sweeps: int, tol: float) -> np.ndarray:
    """Cardoso Jacobi joint-diagonalisation of one cumulant stack.

    ``q`` is ``(nbcm, m, m)`` (modified in place). Returns the ``(m, m)``
    rotation ``u`` such that ``uᵀ·q[k]·u`` are jointly as diagonal as possible.
    """
    nbcm = q.shape[0]
    m = q.shape[1]
    u = np.zeros((m, m), dtype=q.dtype)
    for i in range(m):
        u[i, i] = 1.0
    for _sweep in range(max_sweeps):
        max_angle = 0.0
        for p in range(m - 1):
            for r in range(p + 1, m):
                gpp = 0.0
                grr = 0.0
                gpr = 0.0
                for k in range(nbcm):
                    v1 = q[k, p, p] - q[k, r, r]
                    v2 = q[k, p, r] + q[k, r, p]
                    gpp += v1 * v1
                    grr += v2 * v2
                    gpr += v1 * v2
                tau = (grr - gpp) / 2.0
                if abs(gpr) < 1e-30 and abs(tau) < 1e-30:
                    continue
                tau_sign = 1.0 if tau >= 0.0 else -1.0
                denom = tau + tau_sign * (tau * tau + gpr * gpr + 1e-30) ** 0.5
                t = gpr / denom
                cos_t = 1.0 / (1.0 + t * t) ** 0.5
                sin_t = t * cos_t
                angle = abs(sin_t)
                if angle < tol:
                    continue
                if angle > max_angle:
                    max_angle = angle
                for i in range(m):
                    u_p_i = u[i, p]
                    u_r_i = u[i, r]
                    u[i, p] = cos_t * u_p_i - sin_t * u_r_i
                    u[i, r] = sin_t * u_p_i + cos_t * u_r_i
                for k in range(nbcm):
                    for i in range(m):
                        qpi = q[k, i, p]
                        qri = q[k, i, r]
                        q[k, i, p] = cos_t * qpi - sin_t * qri
                        q[k, i, r] = sin_t * qpi + cos_t * qri
                for k in range(nbcm):
                    for j in range(m):
                        qpj = q[k, p, j]
                        qrj = q[k, r, j]
                        q[k, p, j] = cos_t * qpj - sin_t * qrj
                        q[k, r, j] = sin_t * qpj + cos_t * qrj
        if max_angle < tol:
            break
    return u


def _joint_diag(q: np.ndarray, max_sweeps: int = 50, tol: float = 1e-6) -> np.ndarray:
    return _joint_diag_one(q.astype(np.float64, copy=True), max_sweeps, tol).astype(q.dtype, copy=False)


# ---------------------------------------------------------------------------
# Whitening + cumulants (JADE) and stack covariances (JD).
# ---------------------------------------------------------------------------
def _whiten(x: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Center on the feature axis + PCA-whiten ``X`` (s×d) to ``Y`` (m×d).

    Returns ``(Y, W)`` with ``W`` the ``(m, s)`` whitener and ``Y = W·X_c``.
    """
    x_centered = x - x.mean(axis=1, keepdims=True)
    cov = (x_centered @ x_centered.T) / max(x.shape[1], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    eigvals = np.maximum(eigvals[order][:m], 1e-12)
    eigvecs = eigvecs[:, order][:, :m]
    w = (eigvecs / np.sqrt(eigvals)[None, :]).T
    return w @ x_centered, w


def _build_cumulants(y: np.ndarray) -> np.ndarray:
    """Upper-triangular JADE 4th-order cumulant stack ``(nbcm, m, m)`` of whitened ``Y`` (m×T)."""
    m, T = y.shape
    mom4 = np.einsum("it,jt,kt,lt->ijkl", y, y, y, y, optimize=True) / T
    eye = np.eye(m, dtype=y.dtype)
    cum = (
        mom4
        - eye[:, :, None, None] * eye[None, None, :, :]
        - eye[:, None, :, None] * eye[None, :, None, :]
        - eye[:, None, None, :] * eye[None, :, :, None]
    )
    triu_ij = [(i, j) for i in range(m) for j in range(i, m)]
    return np.stack([cum[:, :, i, j] for i, j in triu_ij], axis=0)


def _p95_cosine_with_hungarian(s_hat: np.ndarray, h: np.ndarray) -> float:
    """Hungarian-match Ŝ rows to H rows by |cosine|, return the p95 of matched |cosine|.

    Resolves JADE/JD's per-row permutation+sign ambiguity in the attacker's favour.
    """
    from scipy.optimize import linear_sum_assignment

    eps = 1e-12
    n_s, d_s = s_hat.shape
    n_h, d_h = h.shape
    if d_s != d_h or n_s == 0 or n_h == 0:
        return float("nan")
    s_norm = s_hat / (np.linalg.norm(s_hat, axis=1, keepdims=True) + eps)
    h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + eps)
    cos_abs = np.abs(s_norm @ h_norm.T)
    row_idx, col_idx = linear_sum_assignment(-cos_abs)
    matched = cos_abs[row_idx, col_idx]
    if matched.size == 0:
        return float("nan")
    return float(np.quantile(matched, 0.95))


def _operands(
    capture: CaptureSet, kind: str, layer: int, transform: Transform | None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-prompt ``(H, U)`` pairs (float32) for one (kind, layer)."""
    transform = transform or Identity()
    return [(h, u) for _pi, h, u in capture.per_prompt_matrices(kind, layer, transform=transform)]


def _subsample(
    u: np.ndarray, h: np.ndarray, max_dim: int, max_features: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    s = u.shape[0]
    if s > max_dim:
        sel = np.sort(rng.choice(s, size=max_dim, replace=False))
        u, h = u[sel], h[sel]
    if max_features is not None and u.shape[1] > max_features:
        fsel = np.sort(rng.choice(u.shape[1], size=max_features, replace=False))
        u, h = u[:, fsel], h[:, fsel]
    return u, h


# ---------------------------------------------------------------------------
# Attack 1: gram_error (fingerprint side channel).
# ---------------------------------------------------------------------------
def gram_error(
    capture: CaptureSet,
    *,
    layer: int,
    kind: str,
    transform: Transform | None = None,
    max_features: int = 256,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    cos_norms: list[float] = []
    spec_errs: list[float] = []
    eps = 1e-12
    for h, u in _operands(capture, kind, layer, transform):
        if h.size == 0 or u.size == 0 or u.shape != h.shape:
            continue
        if max_features is not None and u.shape[1] > max_features:
            fsel = np.sort(rng.choice(u.shape[1], size=max_features, replace=False))
            u, h = u[:, fsel], h[:, fsel]
        g_u, g_h = u @ u.T, h @ h.T
        n_u, n_h = np.linalg.norm(g_u), np.linalg.norm(g_h)
        if n_u < eps or n_h < eps:
            continue
        cos_norms.append(float(np.linalg.norm(g_u / n_u - g_h / n_h)))
        sv_u = np.linalg.svd(u, compute_uv=False) ** 2
        sv_h = np.linalg.svd(h, compute_uv=False) ** 2
        nlen = max(sv_u.size, sv_h.size)
        eu = np.pad(np.sort(sv_u), (nlen - sv_u.size, 0))
        eh = np.pad(np.sort(sv_h), (nlen - sv_h.size, 0))
        den = np.linalg.norm(eh)
        spec_errs.append(float(np.linalg.norm(eu - eh) / den) if den > 0 else float("nan"))
    if not cos_norms:
        return {"attack": "gram_error", "kind": kind, "layer": layer, "n": 0,
                "cos_norm_distance": None, "row_gram_spectrum_error": None}
    return {
        "attack": "gram_error",
        "kind": kind,
        "layer": layer,
        "n": len(cos_norms),
        "cos_norm_distance": float(np.nanmedian(cos_norms)),
        "row_gram_spectrum_error": float(np.nanmedian(spec_errs)),
    }


# ---------------------------------------------------------------------------
# Attack 2: JADE (single-observation ICA recovery).
# ---------------------------------------------------------------------------
def jade(
    capture: CaptureSet,
    *,
    layer: int,
    kind: str,
    transform: Transform | None = None,
    max_dim: int = 64,
    max_features: int = 256,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    p95s: list[float] = []
    n_skip = 0
    for h, u in _operands(capture, kind, layer, transform):
        if u.shape != h.shape or u.size == 0:
            n_skip += 1
            continue
        u, h = _subsample(u, h, max_dim, max_features, rng)
        s = u.shape[0]
        if s < 4 or u.shape[1] < 2 * s:
            n_skip += 1
            continue
        try:
            y, w = _whiten(u, s)
            q = _build_cumulants(y)
            rot = _joint_diag(q)
        except np.linalg.LinAlgError:
            n_skip += 1
            continue
        s_hat = (rot.T @ w) @ u
        p95 = _p95_cosine_with_hungarian(s_hat, h)
        if p95 == p95:
            p95s.append(p95)
    if not p95s:
        return {"attack": "jade", "kind": kind, "layer": layer, "n": 0,
                "jade_p95_cosine": None, "n_skipped": n_skip}
    return {
        "attack": "jade",
        "kind": kind,
        "layer": layer,
        "n": len(p95s),
        "jade_p95_cosine": float(np.nanmedian(p95s)),
        "n_skipped": n_skip,
    }


# ---------------------------------------------------------------------------
# Attack 3: JD (accumulation across T observations).
# ---------------------------------------------------------------------------
def _whiten_stack(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Whiten a (T, s, d) stack with the SUMMED covariance so one whitener fits all t."""
    t_count, s, d = stack.shape
    centered = stack - stack.mean(axis=2, keepdims=True)
    cov = np.zeros((s, s), dtype=np.float64)
    for t in range(t_count):
        cov += centered[t] @ centered[t].T
    cov /= max(t_count * d, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    eigvals = np.maximum(eigvals[order][:s], 1e-12)
    eigvecs = eigvecs[:, order][:, :s]
    w = (eigvecs / np.sqrt(eigvals)[None, :]).T
    y_stack = np.stack([(w @ centered[t]).astype(np.float64) for t in range(t_count)], axis=0)
    return y_stack, w


def _jd_demix(stack: np.ndarray) -> np.ndarray | None:
    t_count, s, d = stack.shape
    if t_count == 0 or s < 2 or d < 2 * s:
        return None
    try:
        y_stack, w = _whiten_stack(stack)
        q = np.stack([(y_stack[t] @ y_stack[t].T) / max(d, 1) for t in range(t_count)], axis=0)
        q = 0.5 * (q + q.transpose(0, 2, 1))
        rot = _joint_diag(q, max_sweeps=50, tol=1e-7)
        return rot.T @ w
    except np.linalg.LinAlgError:
        return None


def jd(
    capture: CaptureSet,
    *,
    layer: int,
    kind: str,
    transform: Transform | None = None,
    t_values: tuple[int, ...] = (1, 2, 4, 8, 16),
    max_dim: int = 64,
    max_features: int = 256,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for h, u in _operands(capture, kind, layer, transform):
        if u.shape != h.shape or u.size == 0:
            continue
        u, h = _subsample(u, h, max_dim, max_features, rng)
        if u.shape[0] < 4:
            continue
        pairs.append((u, h))
    if not pairs:
        return {"attack": "jd", "kind": kind, "layer": layer, "p95_per_t": {}, "n_per_t": {}}
    # Keep a common (s, d) so observations stack.
    ref = pairs[0][0].shape
    pairs = [p for p in pairs if p[0].shape == ref]

    p95_per_t: dict[int, list[float]] = {t: [] for t in t_values}
    for t_target in t_values:
        for start in range(0, len(pairs) - t_target + 1, t_target):
            sl = pairs[start : start + t_target]
            u_stack = np.stack([p[0] for p in sl], axis=0)
            h_stack = np.stack([p[1] for p in sl], axis=0)
            b = _jd_demix(u_stack)
            if b is None:
                continue
            for t in range(u_stack.shape[0]):
                p95 = _p95_cosine_with_hungarian(b @ u_stack[t], h_stack[t])
                if p95 == p95:
                    p95_per_t[t_target].append(p95)
    return {
        "attack": "jd",
        "kind": kind,
        "layer": layer,
        "p95_per_t": {int(t): (float(np.nanmedian(v)) if v else None) for t, v in p95_per_t.items()},
        "n_per_t": {int(t): len(v) for t, v in p95_per_t.items()},
    }


def jd_floor(
    capture: CaptureSet,
    *,
    layer: int,
    kind: str,
    t_values: tuple[int, ...] = (1, 2, 4, 8, 16),
    max_dim: int = 64,
    max_features: int = 256,
    n_seeds: int = 3,
    seed: int = 0,
) -> dict:
    """Hungarian-alignment chance floor: replace each observation row block with
    i.i.d. Gaussian rows of the SAME shape, run the identical JD pipeline. The p95
    cosine this produces is pure alignment-luck — recovery claims must clear it.
    """
    base = _operands(capture, kind, layer, None)
    if not base:
        return {"attack": "jd_floor", "kind": kind, "layer": layer, "p95_per_t": {}}
    rng0 = np.random.default_rng(seed)
    # Determine a representative (s, d) from the real operands after subsample.
    u0, h0 = _subsample(base[0][1], base[0][0], max_dim, max_features, rng0)
    s, d = u0.shape
    per_t: dict[int, list[float]] = {t: [] for t in t_values}
    for sd in range(n_seeds):
        rng = np.random.default_rng(seed + 1 + sd)
        for t_target in t_values:
            u_stack = rng.standard_normal((t_target, s, d)).astype(np.float64)
            h_stack = rng.standard_normal((t_target, s, d)).astype(np.float64)
            b = _jd_demix(u_stack)
            if b is None:
                continue
            for t in range(t_target):
                p95 = _p95_cosine_with_hungarian(b @ u_stack[t], h_stack[t])
                if p95 == p95:
                    per_t[t_target].append(p95)
    return {
        "attack": "jd_floor",
        "kind": kind,
        "layer": layer,
        "p95_per_t": {int(t): (float(np.nanmedian(v)) if v else None) for t, v in per_t.items()},
    }
