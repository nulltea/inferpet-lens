"""MDL probing — online (prequential) code-length and Surplus
Description Length.

Voita & Titov (EMNLP 2020); Blier & Ollivier (NeurIPS 2018); Whitney et
al. (2020). The online code transmits the labels ``y`` given ``X`` by
training the probe on a growing prefix and paying the cross-entropy of
the next block:

    L_online = |B₀|·log₂ C + Σ_t CE_bits(block_{t+1} | probe trained on prefix_t)

Reported alongside:

* ``compression`` = uniform code length ``N·log₂ C`` ÷ ``L_online`` —
  higher means the representation makes the labels more compressible.
* ``surplus_description_length_bits`` (Whitney SDL) ≈ ``L_online`` minus
  the floor ``N·H_floor``, where ``H_floor`` is the test cross-entropy of
  a probe trained on all available data. The surplus is the extra code
  length paid for *learning* the map — a complexity-aware leakage signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._probe import probe_log_softmax, row_split, to_class_indices, train_softmax_probe

_LN2 = np.log(2.0)


def _ce_bits(probe: dict, X: np.ndarray, y_idx: np.ndarray) -> float:
    logq = probe_log_softmax(probe, X)
    return float(-(logq[np.arange(y_idx.size), y_idx]).sum() / _LN2)


def online_code_length(
    X: np.ndarray,
    y: np.ndarray,
    *,
    block_fractions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
    max_classes: int = 256,
    l2: float = 1e-1,
    max_iter: int = 500,
    seed: int = 20260615,
    control: str = "none",
    control_seed: int = 20260616,
) -> dict[str, Any]:
    """Prequential online code length (bits) of ``y`` given ``X``, plus
    compression ratio and the Whitney SDL surplus.

    ``control="shuffle"`` permutes the labels over the same kept rows
    (before the prequential ordering) to break the X↔Y pairing — the
    Hewitt–Liang control task. The surplus then reads the probe's
    memorisation floor. See ``docs/dev/control-tasks.md``.
    """
    if X.shape[0] < 8:
        return {"online_code_length_bits": None, "note": "too few rows"}

    y_idx_all, classes = to_class_indices(y)
    if classes.size > max_classes:
        counts = np.bincount(y_idx_all, minlength=classes.size)
        keep = np.argsort(counts)[::-1][:max_classes]
        m = np.isin(y_idx_all, keep)
        X, y = X[m], y[m]
        y_idx_all, classes = to_class_indices(y)
    n_cls = int(classes.size)

    if control == "shuffle":
        y_idx_all = y_idx_all[np.random.default_rng(control_seed).permutation(y_idx_all.size)]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    Xp, yp = X[perm], y_idx_all[perm]
    n = Xp.shape[0]

    cuts = sorted({max(2, int(round(f * n))) for f in block_fractions})
    cuts = [c for c in cuts if c <= n]
    if cuts[-1] != n:
        cuts.append(n)

    # First block: uniform code.
    first = cuts[0]
    code_bits = first * np.log2(n_cls)
    # Subsequent blocks: train on prefix, pay CE of the next block.
    for i in range(len(cuts) - 1):
        lo, hi = cuts[i], cuts[i + 1]
        probe = train_softmax_probe(
            Xp[:lo], yp[:lo], n_cls, l2=l2, max_iter=max_iter, seed=seed + i
        )
        code_bits += _ce_bits(probe, Xp[lo:hi], yp[lo:hi])

    uniform_bits = n * np.log2(n_cls)

    # SDL floor: probe trained on a train split, CE on held-out test,
    # scaled to N. (Whitney surplus = code paid above the achievable floor.)
    tr, te = row_split(n, 0.7, seed)
    floor_probe = train_softmax_probe(
        Xp[tr], yp[tr], n_cls, l2=l2, max_iter=max_iter, seed=seed + 999
    )
    floor_ce_per_row = _ce_bits(floor_probe, Xp[te], yp[te]) / max(1, te.size)
    sdl_bits = float(code_bits - n * floor_ce_per_row)

    return {
        "online_code_length_bits": float(code_bits),
        "uniform_code_length_bits": float(uniform_bits),
        "compression": float(uniform_bits / code_bits) if code_bits > 0 else None,
        "surplus_description_length_bits": sdl_bits,
        "floor_ce_bits_per_row": float(floor_ce_per_row),
        "num_classes": n_cls,
        "n_rows": int(n),
        "control": control,
    }


def online_code_length_retrieval(
    X: np.ndarray,
    y: np.ndarray,
    embed_table: "torch.Tensor",
    *,
    block_fractions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
    ridge_alpha: float = 1e-2,
    candidate_pool_size: int = 2048,
    seed: int = 20260615,
) -> dict[str, Any]:
    """MDL online (prequential) code length under the **retrieval family**
    (resolution B). Each prefix fits the ridge ``X→embedding`` inverter
    and pays the retrieval-softmax cross-entropy of the next block. Unlike
    the class-probe online code, the retrieval likelihood generalises to
    tokens unseen in the prefix. ``τ`` is calibrated once globally (a 70/30
    pre-split) for tractability. See ``docs/research/attacks_setting.md``.
    """
    import torch

    from ..splits import train_val_test_split
    from ._retrieval import (
        build_candidate_pool,
        fit_inverter,
        log_prob_true,
        pick_temperature,
        predict_embeddings,
    )

    if X.shape[0] < 8:
        return {"online_code_length_bits": None, "note": "too few rows"}

    n_cls = int(np.unique(y).size)
    pool = build_candidate_pool(
        y, vocab_size=embed_table.shape[0], pool_size=candidate_pool_size, seed=seed
    )

    # Calibrate τ once on a 70/30 pre-split, and reuse for the floor.
    Xc_tr, yc_tr, _, _, Xc_te, yc_te = train_val_test_split(
        X, y, n_train=max(4, int(0.7 * X.shape[0])), n_val=1, n_test=max(2, int(0.3 * X.shape[0])), seed=seed
    )
    cal_model = fit_inverter(Xc_tr, yc_tr, embed_table, ridge_alpha=ridge_alpha)
    tau = pick_temperature(
        predict_embeddings(cal_model, Xc_te), torch.from_numpy(yc_te), pool, embed_table
    )

    def _ce_bits(model, Xb, yb) -> float:
        lp = log_prob_true(predict_embeddings(model, Xb), torch.from_numpy(yb), pool, embed_table, tau)
        return float(-lp.sum().item() / _LN2)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    Xp, yp = X[perm], y[perm]
    n = Xp.shape[0]
    cuts = sorted({max(2, int(round(f * n))) for f in block_fractions})
    cuts = [c for c in cuts if c <= n]
    if cuts[-1] != n:
        cuts.append(n)

    first = cuts[0]
    code_bits = first * np.log2(n_cls)
    for i in range(len(cuts) - 1):
        lo, hi = cuts[i], cuts[i + 1]
        model = fit_inverter(Xp[:lo], yp[:lo], embed_table, ridge_alpha=ridge_alpha)
        code_bits += _ce_bits(model, Xp[lo:hi], yp[lo:hi])

    uniform_bits = n * np.log2(n_cls)
    floor_ce_per_row = _ce_bits(cal_model, Xc_te, yc_te) / max(1, Xc_te.shape[0])
    sdl_bits = float(code_bits - n * floor_ce_per_row)
    return {
        "online_code_length_bits": float(code_bits),
        "uniform_code_length_bits": float(uniform_bits),
        "compression": float(uniform_bits / code_bits) if code_bits > 0 else None,
        "surplus_description_length_bits": sdl_bits,
        "floor_ce_bits_per_row": float(floor_ce_per_row),
        "family": "retrieval",
        "temperature": float(tau),
        "num_classes": n_cls,
        "n_rows": int(n),
    }
