"""Weight-surface Invariant Attack (IA) — AloePri §F.1 (Gate-IA + Attn-IA).

The IA recovers the secret token permutation τ from a per-token weight INVARIANT that survives the
covariant obfuscation: because the keymat cancels (P̂Q̂=I) by forward-correctness design, the gate
projection and the attention quadratic form of token i equal those of its obfuscated counterpart
τ(i). So IA recovers τ on keymat-only; the αₑ embedding noise is the only thing that breaks it
(paper: "add sufficient noise αₑ≥1.0 so the row-mean drifts"). Model-free / synthetic here.
"""
from __future__ import annotations

import numpy as np

from talens.weights.invariant_attack import attn_ia_scalar, gate_ia_scalar, recover_by_invariant


def _setup(V=400, d=32, ffn=64, seed=0):
    rng = np.random.default_rng(seed)
    We = rng.standard_normal((V, d)).astype(np.float64)
    W_gate = rng.standard_normal((ffn, d)).astype(np.float64)      # (out, in) HF convention
    W_q = rng.standard_normal((d, d)).astype(np.float64)
    tau = rng.permutation(V)                                       # secret true→obf token map
    return We, W_gate, W_q, tau, rng


def test_gate_ia_recovers_tau_under_lossless_obfuscation():
    """Covariant keymat cancels → the obf gate-projection row-mean of obf-row τ(i) equals the plain
    one of token i → IA recovers τ. Here we model the lossless case directly: obf scalar = plain
    scalar permuted by τ."""
    We, W_gate, _, tau, _ = _setup()
    a_plain = gate_ia_scalar(We, W_gate)                           # per true token
    a_obf = np.empty_like(a_plain)
    a_obf[tau] = a_plain                                           # obf row τ(i) carries token i's invariant
    rec = recover_by_invariant(a_plain, a_obf, tau_true=tau, top_k=100)
    assert rec > 0.8, rec


def test_attn_ia_recovers_tau_under_lossless_obfuscation():
    We, _, W_q, tau, _ = _setup()
    b_plain = attn_ia_scalar(We, W_q)
    b_obf = np.empty_like(b_plain)
    b_obf[tau] = b_plain
    rec = recover_by_invariant(b_plain, b_obf, tau_true=tau, top_k=100)
    assert rec > 0.8, rec


def test_gate_ia_breaks_under_embedding_noise():
    """αₑ embedding noise drifts the invariant → matching collapses far below the lossless case."""
    We, W_gate, _, tau, rng = _setup()
    a_plain = gate_ia_scalar(We, W_gate)
    noisy = We + 1.0 * float(We.std()) * rng.standard_normal(We.shape)   # αₑ=1.0
    a_obf = np.empty_like(a_plain)
    a_obf[tau] = gate_ia_scalar(noisy, W_gate)
    rec_noisy = recover_by_invariant(a_plain, a_obf, tau_true=tau, top_k=100)
    a_clean = np.empty_like(a_plain); a_clean[tau] = a_plain
    rec_clean = recover_by_invariant(a_plain, a_clean, tau_true=tau, top_k=100)
    assert rec_noisy < rec_clean - 0.3, (rec_clean, rec_noisy)
