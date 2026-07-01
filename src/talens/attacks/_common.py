"""Shared array-interface primitives + the surface-agnostic cascade orchestrator.
Imported by the per-surface attack modules (talens.attacks.<surface>.*)."""
from __future__ import annotations

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def nearest_token(pred_emb, pool_emb, pool_ids):
    """Cosine nearest-neighbour decode of predicted embeddings against a candidate pool."""
    p = pred_emb / np.clip(np.linalg.norm(pred_emb, axis=1, keepdims=True), 1e-9, None)
    e = pool_emb / np.clip(np.linalg.norm(pool_emb, axis=1, keepdims=True), 1e-9, None)
    return pool_ids[(p @ e.T).argmax(1)]


def ridge_W(Xtr, Etr, alpha=1.0):
    """Closed-form ridge map X→E (float64 Gram/solve for stability; returned float32 [d_in, d_out])."""
    d = Xtr.shape[1]
    A = (Xtr.T @ Xtr).astype(np.float64) + alpha * np.eye(d, dtype=np.float64)
    return np.linalg.solve(A, (Xtr.T @ Etr).astype(np.float64)).astype(np.float32)


def multikey_ridge_W(G0, H0, Pks, alpha=1.0):
    """Multi-key-synthesis ridge (AloePri ISA-HiddenState blind, paper §F.1). Fits one inverter over
    pooled synthetic obfuscated reps {Xc·Pk}_k WITHOUT materializing the K·n stack:

        X^T X = Σ_k Pk^T G0 Pk,   X^T E = (Σ_k Pk)^T H0,   with G0 = Xc^T Xc, H0 = Xc^T E.

    Algebraically identical to stacking and ridge_W (test_multikey_ridge_matches_stacking); K small
    matmuls instead of a K·n-row solve. Returns the (d+2h)→d_emb map (float32). float64 internally."""
    D1 = Pks[0].shape[1]
    Gram = np.zeros((D1, D1), np.float64)
    Psum = np.zeros_like(Pks[0], dtype=np.float64)
    for Pk in Pks:
        Pk64 = np.asarray(Pk, np.float64)
        Gram += Pk64.T @ np.asarray(G0, np.float64) @ Pk64
        Psum += Pk64
    Gram += alpha * np.eye(D1)
    return np.linalg.solve(Gram, Psum.T @ np.asarray(H0, np.float64)).astype(np.float32)


def orthogonal_procrustes_R(P, D):
    """Orthogonal R ∈ O(d) minimising ‖P·R − D‖_F — closed form R = U Vᵀ from svd(Pᵀ D).
    So P·R ≈ D; map D back into the P basis with D·Rᵀ. float64 internally, returns float32 (d,d).
    This is the least-squares known-plaintext solution for an orthogonal cipher (needs ~d anchors)."""
    M = np.asarray(P, np.float64).T @ np.asarray(D, np.float64)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return (U @ Vt).astype(np.float32)


def blockwise_procrustes_R(P, D, n_heads=12, hd=64):
    """Per-head block variant of orthogonal_procrustes_R for R = head-permutation ∘ blkdiag(Û_vo)
    (AloePri Alg2's value transform). Fit an O(hd) Procrustes per (plaintext head h, deployment head h'),
    assign the head-permutation by min total residual (Hungarian if scipy, else greedy), assemble the
    block-permuted-orthogonal R (P·R ≈ D). Needs only ~hd anchors (each anchor supplies all heads), vs
    ~n_heads·hd for the global solve — the sample-efficient known-plaintext solve when R is block-structured."""
    Pb = np.asarray(P, np.float64).reshape(-1, n_heads, hd)
    Db = np.asarray(D, np.float64).reshape(-1, n_heads, hd)
    res = np.zeros((n_heads, n_heads)); Q = {}
    for h in range(n_heads):
        for hp in range(n_heads):
            U, _, Vt = np.linalg.svd(Pb[:, h].T @ Db[:, hp], full_matrices=False)
            Q[(h, hp)] = U @ Vt
            res[h, hp] = np.linalg.norm(Pb[:, h] @ Q[(h, hp)] - Db[:, hp])
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(res)
    except Exception:                                        # greedy fallback (few heads)
        rows, cols, used = list(range(n_heads)), [], set()
        for h in range(n_heads):
            hp = next(c for c in np.argsort(res[h]) if c not in used); used.add(hp); cols.append(hp)
    R = np.zeros((n_heads * hd, n_heads * hd), np.float32)
    for h, hp in zip(rows, cols):
        R[h * hd:(h + 1) * hd, hp * hd:(hp + 1) * hd] = Q[(h, hp)].astype(np.float32)
    return R


def cascade_attack(attack, X, y, harvested_types, table, pool, *, X_aug=None, y_aug=None, **kw):
    """Two-stage τ-leak cascade. A harvest (e.g. TFMA) reveals the true labels for `harvested_types`
    (a set/array of token ids); train ANY array-interface `attack` on those (deployment-basis rep,
    token) pairs — optionally augmented with blind pairs (`X_aug`, `y_aug`, e.g. multi-key synthetic
    reps) — and score recovery on the HELD-OUT (unharvested) types. Generic over the target
    representation: `X` / `table` / `attack` decide whether it is embeddings, residual, or q/k/v
    (residual ISA-HiddenState, IMA-EmbedRow on the static table, … all reuse this).

    Returns {unharvested (generalization to never-harvested types), harvested (in-set sanity),
    n_harv_types, n_held}. `unharvested` is the bootstrap signal: does knowing k token mappings let the
    inverter read the rest? `**kw` is forwarded to `attack` (alpha / hidden / epochs / seed / …)."""
    H = {int(t) for t in harvested_types}
    inset = np.fromiter((int(t) in H for t in y), bool, len(y))
    tr, te = np.where(inset)[0], np.where(~inset)[0]
    if te.size == 0 or (tr.size == 0 and X_aug is None):
        return {"unharvested": None, "harvested": None, "n_harv_types": len(H), "n_held": int(te.size)}
    Xtr, ytr = X[tr], y[tr]
    if X_aug is not None:
        Xtr = np.concatenate([Xtr, X_aug], 0)
        ytr = np.concatenate([ytr, np.asarray(y_aug)])
    pred = lambda idx: attack(Xtr, table[ytr], X[idx], table[pool], pool, ytr=ytr, full_emb=table, **kw)
    return {"unharvested": float((pred(te) == y[te]).mean()),
            "harvested": (float((pred(tr) == y[tr]).mean()) if tr.size else None),
            "n_harv_types": int(np.unique(y[tr]).size if tr.size else 0), "n_held": int(te.size)}
