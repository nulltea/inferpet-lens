"""Cover-break attack (anchor-based recovery) — aloepri §4.3.3.

Threat model: the attacker observes the exposed operand ``U = T(H)`` and
knows ``K`` row-paired *anchors* — for ``i`` in an anchor set the true
plaintext row ``H[i]`` is known (adversary-controlled or universally
frequent tokens). The attack recovers ``H[j]`` for non-anchor ``j`` and
reports the p95 absolute cosine between recovered and true rows.

Two variants in the aloepri original:

* **ridge** — fit a linear map ``M: U → H`` on the anchor pairs, apply to
  non-anchor rows. Ported faithfully below. Under :class:`Identity`
  (``U = H``) this trivially recovers (cosine ≈ 1) — the plaintext
  sanity baseline. Under an orthogonal-cover transform it tests whether
  the cover's row-mixing leaves linearly-recoverable structure.
* **fastica_anchor** — FastICA row-unmixing + Hungarian anchor
  assignment. Only meaningful under an actual (non-identity) orthogonal
  cover; **deferred to pass 2** (faithful port from
  ``attack_drivers/run_anchor_ica.py`` once a real cover Transform is
  injected). Requesting it raises rather than shipping unverified ICA.
"""

from __future__ import annotations

import numpy as np

from ..capture.types import CaptureSet
from ..metrics import AttackResult, classify_risk_level
from ..transforms import Identity, Transform

DEFAULT_ANCHOR_COUNTS: tuple[int, ...] = (1, 4, 16)


def _p95_cosine(pred: np.ndarray, true: np.ndarray) -> float:
    eps = 1e-12
    pn = np.linalg.norm(pred, axis=1) + eps
    tn = np.linalg.norm(true, axis=1) + eps
    cos = np.abs(np.einsum("ij,ij->i", pred, true) / (pn * tn))
    return float(np.quantile(cos, 0.95)) if cos.size else float("nan")


def _ridge_recover(
    u: np.ndarray,
    h: np.ndarray,
    anchor_idx: np.ndarray,
    non_anchor_idx: np.ndarray,
    lam: float = 1.0,
) -> np.ndarray:
    """Fit ``M`` minimising ``Σ_{i∈anchors} ‖M·U[i] − H[i]‖² + λ‖M‖²``
    and apply to non-anchor rows. Closed form
    ``Mᵀ = (U_aᵀU_a + λI)⁻¹ U_aᵀ H_a``; handles non-square ``U→H``.
    """
    u_a, h_a = u[anchor_idx], h[anchor_idx]
    d = u.shape[1]
    gram = u_a.T @ u_a + lam * np.eye(d, dtype=u.dtype)
    m_t = np.linalg.solve(gram, u_a.T @ h_a)
    return u[non_anchor_idx] @ m_t


def run(
    capture: CaptureSet,
    *,
    layer: int,
    kind: str = "resid_post",
    transform: Transform | None = None,
    anchor_counts: tuple[int, ...] = DEFAULT_ANCHOR_COUNTS,
    ridge_lambda: float = 1.0,
    seed: int = 20260615,
    fastica: bool = False,
) -> AttackResult:
    if fastica:
        raise NotImplementedError(
            "fastica_anchor variant is deferred to pass 2 — faithful port "
            "from attack_drivers/run_anchor_ica.py pending a non-identity "
            "cover Transform. Use the ridge variant (fastica=False)."
        )
    transform = transform or Identity()
    mats = capture.per_prompt_matrices(kind, layer, transform=transform)
    # median p95-cosine across prompts, per anchor count.
    per_k: dict[int, list[float]] = {k: [] for k in anchor_counts}
    for pi, h, u in mats:
        n = h.shape[0]
        if n < 3:
            continue
        rng = np.random.default_rng(seed + pi)
        for k in anchor_counts:
            if k >= n:
                continue
            anchor_idx = rng.choice(n, size=k, replace=False)
            non_anchor_idx = np.setdiff1d(np.arange(n), anchor_idx)
            pred = _ridge_recover(u, h, anchor_idx, non_anchor_idx, lam=ridge_lambda)
            per_k[k].append(_p95_cosine(pred, h[non_anchor_idx]))

    medians = {
        k: float(np.median(v)) for k, v in per_k.items() if v
    }
    primary = max(medians.values()) if medians else None
    return AttackResult(
        attack="cover_break",
        transform=transform.name,
        model_id=capture.model_id,
        kind=kind,
        layer=layer,
        n_prompts=capture.n_prompts(),
        n_train=0,
        n_test=int(sum(len(v) for v in per_k.values())),
        ttrsr_top1=None,
        ttrsr_top10=None,
        risk_level=classify_risk_level(primary),
        primary_metric_value=primary,
        primary_metric_name="anchor_p95_cosine",
        extra={
            "variant": "ridge",
            "anchor_counts": list(anchor_counts),
            "median_p95_cosine_by_k": medians,
        },
    )
