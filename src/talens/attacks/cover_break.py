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
    and apply to non-anchor rows.

    The primal closed form ``Mᵀ = (U_aᵀU_a + λI_d)⁻¹ U_aᵀ H_a`` requires a
    ``d×d`` solve (d≈2560), even though only ``k`` anchors (k∈{1,4,16})
    constrain it — an O(d³) cost per prompt that dominated the whole pass.
    The **dual (push-through) identity** gives the *same* map from a ``k×k``
    solve::

        Mᵀ = U_aᵀ (U_a U_aᵀ + λI_k)⁻¹ H_a

    so ``pred = U_naᵀM = (U_na U_aᵀ)·(U_a U_aᵀ + λI_k)⁻¹·H_a`` is O(d·k² + k³)
    — ~700× faster at k≤16, identical to float tolerance.
    """
    u_a, h_a = u[anchor_idx], h[anchor_idx]
    k = u_a.shape[0]
    gram_k = u_a @ u_a.T + lam * np.eye(k, dtype=u.dtype)  # (k, k), not (d, d)
    coef = np.linalg.solve(gram_k, h_a)                    # (k, dh)
    return (u[non_anchor_idx] @ u_a.T) @ coef


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
