"""Invariant Attack (IA) — AloePri §F.1, the static-weight obfuscation surface.

IA recovers the secret token permutation τ from a per-token scalar INVARIANT of the weights that
survives the covariant obfuscation (the keymat P̂Q̂=I cancels, so the quantity for obfuscated row
τ(i) equals the plaintext quantity for token i):

* **Gate-IA**  — ``a_i = mean_k (e_i · W_gateᵀ)_k`` : the row-mean of the gate-projected embedding.
  Survives permutation + scaling of W_gate and the keymat. Defence: αₑ embedding noise drifts it.
* **Attn-IA**  — ``b_i = e_i (W_qᵀ W_q)⁻¹ e_iᵀ`` : a quadratic form invariant under invertible
  right-multiplication of W_q. Defence: head/block permutations (Algorithm 2).

Both reduce to a per-token scalar fingerprint; τ is recovered by matching the obfuscated fingerprints
to the plaintext ones (rank match, same machinery as TFMA). This is an *attack* — it consumes only
the (plaintext-public, obfuscated) weight pair, never per-token labels.
"""
from __future__ import annotations

import numpy as np


def gate_ia_scalar(embed: np.ndarray, w_gate: np.ndarray) -> np.ndarray:
    """Gate-IA per-token invariant: row-mean of the gate projection e·W_gateᵀ. ``w_gate`` is the
    gate/up linear in HF (out, in) convention; ``embed`` is (V, d)."""
    return (np.asarray(embed, float) @ np.asarray(w_gate, float).T).mean(axis=1)


def attn_ia_scalar(embed: np.ndarray, w_q: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Attn-IA per-token invariant: the quadratic form e_i (W_qᵀW_q)⁻¹ e_iᵀ (diagonal only).
    Invariant under invertible right-multiplication of W_q. ``w_q`` is (d, d) (HF out,in)."""
    embed = np.asarray(embed, float)
    Wq = np.asarray(w_q, float)
    G = Wq.T @ Wq
    G = G + ridge * float(np.trace(G) / G.shape[0]) * np.eye(G.shape[0])
    Ginv = np.linalg.inv(G)
    return np.einsum("vi,ij,vj->v", embed, Ginv, embed)             # per-row quadratic form


def recover_by_invariant(plain_scalar, obf_scalar, *, tau_true=None, top_k: int = 100):
    """Recover τ by matching obfuscated per-token invariant scalars to the plaintext ones via
    frequency-free RANK matching (the obf token with the j-th largest invariant is guessed to be the
    plaintext token with the j-th largest invariant). With ``tau_true`` (obf = tau_true[plain]),
    grade Top-`top_k` recovery over the tokens with the most extreme (most identifying) invariants.
    Returns the Top-k recovery rate."""
    plain_scalar = np.asarray(plain_scalar, float)
    obf_scalar = np.asarray(obf_scalar, float)
    V = plain_scalar.shape[0]
    o_sorted = np.argsort(-obf_scalar, kind="stable")
    p_sorted = np.argsort(-plain_scalar, kind="stable")
    tau_hat = np.empty(V, dtype=np.int64)
    tau_hat[o_sorted] = p_sorted                                   # tau_hat[obf row] = guessed true token
    if tau_true is None:
        return tau_hat
    tau_true = np.asarray(tau_true, dtype=np.int64)
    # grade on the top_k most extreme plaintext invariants (the most uniquely identifiable tokens)
    rank = np.argsort(-np.abs(plain_scalar - plain_scalar.mean()), kind="stable")[:top_k]
    correct = tau_hat[tau_true[rank]] == rank
    return float(correct.mean()) if correct.size else 0.0
