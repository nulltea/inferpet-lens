"""Surface-B token-id attacks (AloePri, paper §6.2/§7.6): the online token-perturbation
mechanism M1 and the Token-Frequency-Matching Attack (TFMA).

M1 is the exponential mechanism with the paper's transposition metric (Definition 2); on single
tokens d(x,y)∈{0,1}, so it reduces to RANDOMIZED RESPONSE: keep the token with p₀ =
1/(1+(V−1)e^{−ε1}), else substitute a uniform random token. ε1=∞ ⇒ p₀=1 (no perturbation, pure Π).

TFMA recovers the secret token substitution τ by matching the obfuscated stream's unigram
frequencies to a public reference frequency table (the deterministic Π alone is a monoalphabetic
substitution cipher → broken by frequency analysis; M1 is what blurs the fingerprint).

All model-free / GPU-free (token streams are integer arrays) — host pytest.
"""
from __future__ import annotations

import numpy as np

from defenses.aloepri import m1_randomized_response
from talens.attacks.token_frequency import tfma_recover


def _zipf_stream(vocab: int, n: int, seed: int = 0) -> np.ndarray:
    """A Zipf-distributed token stream (distinct, skewed unigram frequencies — the regime TFMA
    exploits; a uniform stream has no frequency fingerprint to match)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / (np.arange(1, vocab + 1) ** 1.07)
    p /= p.sum()
    return rng.choice(vocab, size=n, p=p).astype(np.int64)


def test_m1_keeps_everything_at_eps_inf():
    toks = _zipf_stream(200, 5000)
    out = m1_randomized_response(toks, vocab=200, eps1=float("inf"), seed=0)
    assert np.array_equal(out, toks)                       # p₀=1 → identity


def test_m1_keep_rate_matches_p0():
    V, eps1 = 500, 8.0
    p0 = 1.0 / (1.0 + (V - 1) * np.exp(-eps1))
    toks = _zipf_stream(V, 40000)
    out = m1_randomized_response(toks, vocab=V, eps1=eps1, seed=1)
    kept = float((out == toks).mean())
    assert abs(kept - p0) < 0.03, (kept, p0)               # empirical keep-rate ≈ p₀


def test_tfma_recovers_pure_permutation():
    """No M1 (ε1=∞): obf = τ(true), so obf unigram freqs are the true freqs permuted → TFMA
    recovers τ for the high-frequency tokens (distinct freqs → unique matching)."""
    V = 300
    true = _zipf_stream(V, 60000, seed=2)
    rng = np.random.default_rng(3)
    tau = rng.permutation(V)                                # secret substitution
    released = m1_randomized_response(true, vocab=V, eps1=float("inf"), seed=0)
    obf = tau[released]
    ref_freq = np.bincount(true, minlength=V).astype(float)   # public reference (distribution-aware)
    rec = tfma_recover(obf, ref_freq, tau_true=tau, top_k=100)
    assert rec["recovery_topk"] > 0.8, rec["recovery_topk"]


def test_tfma_collapses_under_heavy_m1():
    """Small ε1 (p₀→1/V): the stream regresses toward uniform, the frequency fingerprint is
    destroyed → TFMA recovery falls far below the pure-Π case."""
    V = 300
    true = _zipf_stream(V, 60000, seed=2)
    rng = np.random.default_rng(3)
    tau = rng.permutation(V)
    ref_freq = np.bincount(true, minlength=V).astype(float)
    obf_clean = tau[m1_randomized_response(true, vocab=V, eps1=float("inf"), seed=0)]
    obf_noisy = tau[m1_randomized_response(true, vocab=V, eps1=2.0, seed=0)]
    rec_clean = tfma_recover(obf_clean, ref_freq, tau_true=tau, top_k=100)["recovery_topk"]
    rec_noisy = tfma_recover(obf_noisy, ref_freq, tau_true=tau, top_k=100)["recovery_topk"]
    assert rec_noisy < rec_clean - 0.3, (rec_clean, rec_noisy)
