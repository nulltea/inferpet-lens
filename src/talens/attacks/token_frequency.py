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
