"""Geometry-only two-sided bounds on the MAP (=BNN) error of the L0
embedding-DP channel — the probe matched to the BNN/NNS attack.

The L0 channel ``Y = clip(e_V) + N(0, σ²I)`` with a known codebook
``{e_v}`` and the nearest-neighbour (MAP) decoder is textbook M-ary
Gaussian signaling. This module computes two bounds on the MAP error
``P_e*`` **from the codebook geometry + σ alone** — they never see the
attack's observations ``{Y_i}`` (the independence property the rejected
NNS-PVI design lacked; see ``claim:bnn-error-bounds-bhattacharyya-fano``):

* :func:`union_bhattacharyya` — upper bound (exact-pairwise union of
  Gaussian pairwise errors, plus its looser Bhattacharyya/Chernoff
  relaxation). Function of the codebook self-distance multiset only.
* :func:`fano_equivocation` — lower bound via Fano on the channel
  equivocation ``H(V|Y)``, estimated by Monte-Carlo over **fresh
  synthetic noise** around each codeword (NOT the attack's observations).

Proof: ``refine-logs/PROOF_PACKAGE.md`` (5 theorems, Codex-verified).
Headline scope: uniform prior over the pool, ``K ≥ 3``. Both bounds can
be loose at the extremes (upper vacuous at low SNR, Fano vacuous when
``H(V|Y) ≤ 1`` bit) — the gap between them is itself a diagnostic.

Dependency-light: NumPy in / dict out, mirroring the other measures;
uses Torch (ROCm/CUDA) for the codeword Gram and the MC log-sum-exp when
a GPU is available, with a NumPy fallback.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:  # GPU accel is optional; the math is identical on CPU
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

_LOG2 = math.log(2.0)


def _pairwise_sq_dists(E: np.ndarray) -> np.ndarray:
    """``(K, K)`` matrix of squared Euclidean distances ``‖e_v − e_u‖²``.
    Uses the GPU when available (the ``K×K`` Gram is the cost driver)."""
    E = np.ascontiguousarray(E, dtype=np.float32)
    if _HAS_TORCH and torch.cuda.is_available():
        t = torch.from_numpy(E).cuda()
        sq = (t * t).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (t @ t.T)
        out = d2.clamp_min(0.0).cpu().numpy()
        del t, d2, sq
        return out
    sq = (E * E).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (E @ E.T)
    return np.maximum(d2, 0.0)


def _q(x: np.ndarray) -> np.ndarray:
    """Gaussian tail ``Q(x) = P(N(0,1) > x) = ½ erfc(x/√2)``."""
    from scipy.special import erfc

    return 0.5 * erfc(x / math.sqrt(2.0))


def union_bhattacharyya(
    pool_emb: np.ndarray,
    sigma: float,
    *,
    sq_dists: np.ndarray | None = None,
) -> dict[str, Any]:
    """Upper bound on the uniform-prior MAP (=BNN) error of the Gaussian
    channel with codebook ``pool_emb`` (the **clipped** pool embeddings)
    and isotropic noise ``σ``.

    ``P_e^ub = (1/K) Σ_v Σ_{u≠v} Q(‖Δ_vu‖/2σ)``  (exact-pairwise union)

    ``    ≤  (1/2K) Σ_v Σ_{u≠v} exp(−‖Δ_vu‖²/8σ²)``  (Bhattacharyya).

    Independent of any channel observation — a function of the codebook
    self-distance multiset and ``σ`` only. Returns both forms (raw and
    clamped to ``[0,1]``), the minimum inter-codeword distance, and ``K``.
    Pass a precomputed ``sq_dists`` (``K×K``) to amortise the Gram across σ.
    At ``σ = 0`` distinct codewords give error 0; if codewords collide the branch
    overrides to the exact deterministic MAP error ``(K − G)/K`` (``G`` distinct
    groups) rather than the false 0 the distinctness assumption would give.
    """
    E = np.ascontiguousarray(pool_emb, dtype=np.float64)
    K = E.shape[0]
    if K < 2:
        return {"p_e_ub": 0.0, "p_e_ub_bhat": 0.0, "min_dist": float("nan"), "K": K}

    d2 = _pairwise_sq_dists(E) if sq_dists is None else np.asarray(sq_dists, dtype=np.float64)
    iu = ~np.eye(K, dtype=bool)
    off = d2[iu]                       # K*(K-1) off-diagonal squared distances
    min_dist = float(math.sqrt(max(0.0, off.min())))

    if sigma <= 0.0:
        # σ→0: no noise can flip a decision among DISTINCT codewords ⇒ error 0.
        # If codewords collide (min_dist=0) the MAP decoder cannot resolve the tie:
        # the exact deterministic error is (K − G)/K for G distinct groups — a valid
        # (and tight) upper bound, not the false 0 the distinctness assumption gives.
        # Reporting it keeps the paired recovery floor honest (1 − p_e_ub ≠ 1 under ties).
        G = int(np.unique(E, axis=0).shape[0])
        pe = (K - G) / K
        return {
            "p_e_ub": pe, "p_e_ub_raw": pe,
            "p_e_ub_bhat": pe, "p_e_ub_bhat_raw": pe,
            "min_dist": min_dist, "K": K,
        }

    dist = np.sqrt(off)
    ub_raw = float(_q(dist / (2.0 * sigma)).sum() / K)
    bhat_raw = float(0.5 * np.exp(-off / (8.0 * sigma ** 2)).sum() / K)
    return {
        "p_e_ub": min(1.0, ub_raw), "p_e_ub_raw": ub_raw,
        "p_e_ub_bhat": min(1.0, bhat_raw), "p_e_ub_bhat_raw": bhat_raw,
        "min_dist": min_dist, "K": K,
    }


def fano_equivocation(
    pool_emb: np.ndarray,
    sigma: float,
    *,
    M: int = 64,
    seed: int = 0,
    alpha: float = 0.05,
    chunk: int = 64,
) -> dict[str, Any]:
    """Lower bound on the uniform-prior MAP error via Fano on the channel
    equivocation ``H(V|Y)``, with ``H(V|Y)`` estimated by **fresh-noise**
    Monte-Carlo around each codeword (independent of the attack's data).

    For each codeword ``v`` draw ``M`` synthetic ``ε ~ N(0, σ²I)``, form
    ``Y = e_v + ε`` and accumulate ``−log₂ p(v|Y)`` where ``p(·|Y)`` is the
    uniform-prior Gaussian posterior over the pool. ``Ĥ_M`` is unbiased for
    ``H(V|Y)`` (stratified per-codeword SLLN); the stratified standard error
    is ``se² = (1/K²) Σ_v s_v²/M``. Fano (``K ≥ 3``):

    ``P_e^lb = (Ĥ_M − 1) / log₂(K−1)``  (population, as M→∞)

    and a one-sided ``1−α`` lower-confidence variant uses
    ``H_lcb = Ĥ_M − z_{1−α}·se`` (CLT coverage, not a finite-M certificate).

    Returns equivocation (bits), the channel-MI estimate ``i_channel_bits =
    log₂K − H(V|Y)``, its SE, and both lower bounds (clamped to ``[0,1]`` and raw).
    ``σ = 0``: for DISTINCT codewords ``H(V|Y)=0`` (full leakage ``log₂K``, vacuous
    error lower bound 0); if codewords collide the equivocation is the duplicate-group
    entropy ``Σ_g (n_g/K)·log₂ n_g`` and ``i_channel_bits`` drops accordingly.
    """
    E = np.ascontiguousarray(pool_emb, dtype=np.float32)
    K = E.shape[0]
    out_base = {"h_cond_bits": 0.0, "se": 0.0, "K": K}
    if K < 3:
        # Fano denominator undefined; the channel-MI is likewise not estimable here.
        return {**out_base, "i_channel_bits": None,
                "p_e_lb": 0.0, "p_e_lb_raw": 0.0,
                "p_e_lb_lcb": 0.0, "p_e_lb_lcb_raw": 0.0,
                "note": "K<3: Fano denominator undefined"}
    if sigma <= 0.0:
        # No noise ⇒ Y = e_V deterministic ⇒ I(V;Y) = H(Y), H(V|Y) = within-group
        # entropy. DISTINCT codewords give H(Y) = log₂K and H(V|Y) = 0; if codewords
        # collide (e.g. two tokens clip to the same vector) Y is no longer injective,
        # so I drops and H(V|Y) rises — computed honestly from the duplicate-group
        # sizes rather than overstating leakage as log₂K.
        _, counts = np.unique(E, axis=0, return_counts=True)
        c = counts.astype(np.float64)
        h_cond_det = float((c / K * np.log2(c)).sum())  # Σ_g (n_g/K) log₂ n_g
        return {"h_cond_bits": h_cond_det, "i_channel_bits": math.log2(K) - h_cond_det,
                "se": 0.0, "K": K,
                "p_e_lb": 0.0, "p_e_lb_raw": 0.0,
                "p_e_lb_lcb": 0.0, "p_e_lb_lcb_raw": 0.0}

    inv2s2 = 1.0 / (2.0 * sigma ** 2)
    logK1 = math.log2(K - 1)
    rng = np.random.default_rng(seed)

    use_gpu = _HAS_TORCH and torch.cuda.is_available()
    if use_gpu:
        Et = torch.from_numpy(E).cuda()                 # (K,d)
        sqE = (Et * Et).sum(1)                           # (K,)

    per_v_mean = np.empty(K, dtype=np.float64)
    per_v_var = np.empty(K, dtype=np.float64)

    for s in range(0, K, chunk):
        idx = np.arange(s, min(s + chunk, K))
        B = idx.size
        eps = rng.standard_normal((B, M, E.shape[1])).astype(np.float32) * sigma
        Y = E[idx][:, None, :] + eps                     # (B,M,d)
        Yf = Y.reshape(B * M, -1)
        if use_gpu:
            Yt = torch.from_numpy(Yf).cuda()             # (B*M,d)
            sqY = (Yt * Yt).sum(1, keepdim=True)         # (B*M,1)
            d2 = sqY + sqE[None, :] - 2.0 * (Yt @ Et.T)  # (B*M,K)
            logits = (-d2 * inv2s2)                       # ∝ log p(·|Y)
            lse = torch.logsumexp(logits, dim=1)          # (B*M,)
            true_logit = logits[torch.arange(B * M, device=logits.device),
                                 torch.from_numpy(np.repeat(idx, M)).cuda()]
            neg_log2_p = ((lse - true_logit) / _LOG2).cpu().numpy()  # = g(v,ε)
            del Yt, d2, logits, lse, true_logit
        else:
            sqY = (Yf * Yf).sum(1, keepdims=True)
            d2 = sqY + (E * E).sum(1)[None, :] - 2.0 * (Yf @ E.T)
            logits = -d2 * inv2s2
            m = logits.max(1, keepdims=True)
            lse = (m[:, 0] + np.log(np.exp(logits - m).sum(1)))
            true_logit = logits[np.arange(B * M), np.repeat(idx, M)]
            neg_log2_p = (lse - true_logit) / _LOG2
        g = neg_log2_p.reshape(B, M)                      # (B,M) ≥ 0
        per_v_mean[idx] = g.mean(1)
        per_v_var[idx] = g.var(1, ddof=1) if M >= 2 else 0.0

    if use_gpu:
        del Et, sqE

    h_cond = float(per_v_mean.mean())                     # Ĥ_M (bits)
    se = float(math.sqrt(per_v_var.sum() / (K * K * M)))  # stratified SE
    z = _z_value(1.0 - alpha)
    h_lcb = h_cond - z * se

    lb_raw = (h_cond - 1.0) / logK1
    lb_lcb_raw = (h_lcb - 1.0) / logK1
    clamp01 = lambda x: min(1.0, max(0.0, x))  # noqa: E731
    # Channel MI estimate (uniform prior): I(V;Y) = H(V) − H(V|Y) = log₂K − Ĥ_M.
    # Ĥ_M is unbiased for H(V|Y), so this is an unbiased estimate of I(V;Y) — the
    # sign-consistent leakage scalar (more bits = more leakage), unlike equivocation.
    i_channel = math.log2(K) - h_cond
    return {
        "h_cond_bits": h_cond, "i_channel_bits": i_channel, "se": se, "K": K,
        "p_e_lb": clamp01(lb_raw), "p_e_lb_raw": lb_raw,
        "p_e_lb_lcb": clamp01(lb_lcb_raw), "p_e_lb_lcb_raw": lb_lcb_raw,
    }


def _z_value(p: float) -> float:
    """One-sided normal quantile ``z_p`` (e.g. p=0.95 → 1.645)."""
    try:
        from scipy.special import ndtri

        return float(ndtri(p))
    except Exception:  # pragma: no cover
        # Acklam rational approximation fallback
        return float(math.sqrt(2) * _erfinv(2 * p - 1))


def _erfinv(x: float) -> float:  # pragma: no cover
    a = 0.147
    ln = math.log(1 - x * x)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)
