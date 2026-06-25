"""Partial Information Decomposition (PID) for the attention QK/OV channel.

The attention surface exposes two operands that both carry token information:
the pre-softmax scores ``QK`` (``kq``) and the per-head attention output
``OV`` (``kqv_out``). A defense that scrambles one need not scramble the other,
so the *matched probe* for this channel must say **which** operand carries the
leak and whether the two share it (redundant), split it (unique), or only
reveal it jointly (synergistic). That is exactly a bivariate PID of
``I(tokens ; {QK, OV})``.

**This is an *operational reader-PID*, not a Shannon PID.** Each information is
estimated as reader-dependent **V-usable information** (a lower bound that can
violate the data-processing inequality), so the MMI lattice identities are not
guaranteed — the atoms are *bounded-adversary* quantities, useful for "which
operand does a bounded attacker read," not certified information atoms. We
therefore (a) report the raw, unclamped ``I_V`` estimates, (b) flag any lattice
violation (``I_joint < max(I_qk, I_ov)``), and (c) also report the **conditional
increments** ``I_joint − I_other`` — the soundest "additional usable leakage from
this operand beyond the other" read, which does not rely on the MMI heuristic.

The **MMI (minimum-mutual-information) redundancy** lattice (Barrett 2015,
`1411.2832`) needs only the three marginal informations:

    R         = min( I(T;QK), I(T;OV) )            # redundancy (MMI heuristic)
    U_qk      = I(T;QK) − R                         # unique to QK
    U_ov      = I(T;OV) − R                         # unique to OV
    S         = I(T;QK,OV) − I(T;QK) − I(T;OV) + R  # synergy

Each ``I(T;·)`` is estimated as **capacity-matched V-usable information** (the
same independent token-id reader the rest of the pipeline uses,
:func:`~talens.probes.vinfo_capacity.v_information_capacity`), so PID inherits
its well-posedness in ``d>n`` and never touches the embedding table — it is an
*independent* probe, not the ISA attack. ``U_qk`` is the matched predictor for
the ``kq`` attack surface; ``U_ov`` for ``kqv_out``.

**Reader family and synergy.** The default ``pca_softmax`` reader is *linear*, so
the atoms measure *linear* usable information — exactly right for the matched
``U_qk`` / ``U_ov`` predictors (the attack surfaces leak token-id linearly), but
it cannot see synergy whose label is a nonlinear (e.g. XOR) combination of the
operands. Pass ``family="knn"`` for a non-parametric reader when nonlinear synergy
must be detected (the joint reader then represents arbitrary interactions).

The decomposition is non-negative for ``R, U_qk, U_ov`` by construction (``R`` is
a min); ``S`` is computed as a residual and can pick up small negative values
from estimator noise — it is clamped at 0 and the raw value reported alongside.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .vinfo_capacity import v_information_capacity


def pid_mmi(
    X_qk: np.ndarray,
    X_ov: np.ndarray,
    y: np.ndarray,
    *,
    family: str = "pca_softmax",
    dim: int = 64,
    max_classes: int = 256,
    l2: float = 1e-1,
    seed: int = 20260620,
    **vinfo_kwargs: Any,
) -> dict[str, Any]:
    """Bivariate MMI-PID of ``I(y ; {X_qk, X_ov})`` in bits.

    ``X_qk`` ``(n, d_qk)`` and ``X_ov`` ``(n, d_ov)`` are the two attention
    operands (row-aligned to token ids ``y``). Returns the four PID atoms plus
    the three marginal informations, all in bits. Each information is a
    capacity-matched V-info estimate (token-id reader, class-prior null) — so
    the atoms are *usable*-information atoms, the bounded-adversary reading.
    """
    if X_qk.shape[0] != X_ov.shape[0] or X_qk.shape[0] != y.shape[0]:
        raise ValueError("X_qk, X_ov, y must share the row axis")

    def _info(X: np.ndarray) -> float:
        out = v_information_capacity(
            X, y, family=family, dim=dim, max_classes=max_classes, l2=l2,
            seed=seed, **vinfo_kwargs,
        )
        v = out.get("v_information_bits")
        return float(v) if v is not None else 0.0

    # raw, unclamped reader informations (kept for diagnostics / CIs)
    i_qk_raw, i_ov_raw, i_joint_raw = _info(X_qk), _info(X_ov), _info(np.concatenate([X_qk, X_ov], axis=1))
    # clamp at 0 only for the lattice atoms (a negative I_V is finite-sample noise
    # around 0; the family is capacity-matched so it is small) — but expose the raw.
    i_qk, i_ov, i_joint = max(0.0, i_qk_raw), max(0.0, i_ov_raw), max(0.0, i_joint_raw)

    redundancy = min(i_qk, i_ov)
    synergy_raw = i_joint - i_qk - i_ov + redundancy
    # lattice sanity: a valid PID needs I_joint ≥ max(I_qk, I_ov); a V-info
    # estimate can violate this (DPI is not guaranteed for reader bounds).
    lattice_ok = i_joint_raw >= max(i_qk_raw, i_ov_raw) - 1e-6

    return {
        # marginal reader informations (V-usable info, bits)
        "i_qk_bits": i_qk, "i_ov_bits": i_ov, "i_joint_bits": i_joint,
        "i_qk_raw_bits": i_qk_raw, "i_ov_raw_bits": i_ov_raw, "i_joint_raw_bits": i_joint_raw,
        # MMI-heuristic reader atoms (NOT certified Shannon atoms — see module docstring)
        "redundancy_bits": redundancy,
        "unique_qk_bits": i_qk - redundancy,
        "unique_ov_bits": i_ov - redundancy,
        "synergy_bits": max(0.0, synergy_raw),
        "synergy_raw_bits": synergy_raw,
        # conditional increments — the sound "additional usable leakage" read
        "cond_increment_qk_bits": i_joint - i_ov,   # leakage from QK beyond OV
        "cond_increment_ov_bits": i_joint - i_qk,   # leakage from OV beyond QK
        "lattice_ok": bool(lattice_ok),
        "family": family, "dim": int(dim), "n_rows": int(y.shape[0]),
    }
