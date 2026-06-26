"""Token-Frequency-Matching Attack (TFMA) — AloePri §7.6, the token-id surface.

The deterministic token substitution Π (the secret token mapping τ) is a monoalphabetic
substitution cipher over the vocabulary; its unigram-frequency fingerprint survives unless the
online perturbation M1 (randomized response, ε1) blurs it. TFMA recovers τ by matching the
observed obfuscated stream's unigram frequencies to a PUBLIC reference frequency table.

This is an *attack* (it consumes only the obfuscated stream + a public reference, never the true
labels). The independent leakage probe for this surface — I(released ; true) under M1 in bits —
lives beside the eval and is computed from the ground-truth joint, not from this attack's output
(probe ≠ attack).
"""
from __future__ import annotations

import numpy as np


def tfma_recover(obf_stream, ref_freq, *, tau_true=None, top_k: int = 100):
    """Match obfuscated-token ids to reference-token ids by unigram frequency rank, then (if the
    ground-truth substitution ``tau_true`` is supplied for grading) report the Top-`top_k`
    token-recovery success rate over the most frequent reference tokens.

    ``obf_stream``  — 1-D int array of observed obfuscated token ids.
    ``ref_freq``    — length-V array of public reference counts/frequencies per true token id.
    ``tau_true``    — the secret map true→obf (``obf = tau_true[released]``); only for grading.
    Returns ``{"recovery_topk", "tau_hat"}`` where ``tau_hat[o]`` is the guessed true id for obf id o.
    """
    V = len(ref_freq)
    obf_freq = np.bincount(np.asarray(obf_stream, dtype=np.int64), minlength=V).astype(float)
    # rank-match: the obf id with the j-th highest frequency is guessed to be the reference id with
    # the j-th highest frequency (the frequency-analysis break of a substitution cipher).
    o_sorted = np.argsort(-obf_freq, kind="stable")
    t_sorted = np.argsort(-np.asarray(ref_freq, float), kind="stable")
    tau_hat = np.empty(V, dtype=np.int64)
    tau_hat[o_sorted] = t_sorted                            # tau_hat[obf id] = guessed true id
    rec = None
    if tau_true is not None:
        tau_true = np.asarray(tau_true, dtype=np.int64)
        top_true = t_sorted[:top_k]                         # the top_k most frequent TRUE tokens
        correct = tau_hat[tau_true[top_true]] == top_true   # obf of t guessed back to t?
        rec = float(correct.mean()) if correct.size else 0.0
    return {"recovery_topk": rec, "tau_hat": tau_hat}


def _bigram_counts(stream, top_ids):
    """K×K consecutive-pair counts restricted to the `top_ids` symbols (others dropped)."""
    K = len(top_ids)
    pos = -np.ones(int(top_ids.max()) + 1, dtype=np.int64)
    pos[top_ids] = np.arange(K)
    s = np.asarray(stream, dtype=np.int64)
    idx = np.where(s <= pos.shape[0] - 1, pos[np.clip(s, 0, pos.shape[0] - 1)], -1)
    a, b = idx[:-1], idx[1:]
    m = (a >= 0) & (b >= 0)
    B = np.zeros((K, K), dtype=np.float64)
    np.add.at(B, (a[m], b[m]), 1.0)
    return B


def sda_recover(obf_stream, ref_stream, *, tau_true=None, top_k: int = 100, n_iters: int = 4000,
                seed: int = 0):
    """SDA — Substitution-Deciphering Attack (AloePri §7.6). Refines the TFMA unigram match with
    BIGRAM structure: a hill-climb over the top-`top_k` symbol mapping σ (obf-rank → ref-rank,
    seeded at identity = the TFMA solution) maximizing the bigram log-likelihood of the deciphered
    stream under the reference bigram model. Composes with TFMA; recovers ids that survive frequency
    matching via n-gram regularities. Heavy M1 blurs the bigram structure too → recovery collapses."""
    rng = np.random.default_rng(seed)
    obf_freq = np.bincount(np.asarray(obf_stream, np.int64))
    ref_freq = np.bincount(np.asarray(ref_stream, np.int64))
    K = min(top_k, (obf_freq > 0).sum(), (ref_freq > 0).sum())
    obf_top = np.argsort(-obf_freq, kind="stable")[:K]
    ref_top = np.argsort(-ref_freq, kind="stable")[:K]
    Bo = _bigram_counts(obf_stream, obf_top)
    Br = _bigram_counts(ref_stream, ref_top)
    logBr = np.log((Br + 1.0) / (Br + 1.0).sum())           # smoothed reference bigram log-prob
    sigma = np.arange(K)                                    # seed = TFMA (identity in rank space)

    def score(sig):
        return float((Bo * logBr[np.ix_(sig, sig)]).sum())

    cur = score(sigma)
    for _ in range(n_iters):
        a, b = rng.integers(0, K, size=2)
        if a == b:
            continue
        sigma[a], sigma[b] = sigma[b], sigma[a]
        new = score(sigma)
        if new > cur:
            cur = new
        else:
            sigma[a], sigma[b] = sigma[b], sigma[a]         # revert
    # σ maps obf_top[i] → ref_top[σ[i]] (the guessed plaintext token)
    rec = None
    if tau_true is not None:
        inv_tau = np.argsort(np.asarray(tau_true, np.int64))  # inv_tau[obf id] = true id
        pred_true = ref_top[sigma]                            # guessed true id for each obf_top symbol
        correct = pred_true == inv_tau[obf_top]
        rec = float(correct.mean()) if correct.size else 0.0
    return {"recovery_topk": rec, "sigma": sigma, "obf_top": obf_top, "ref_top": ref_top}
