"""Synthetic tests for the weight-pair τ-recovery family
(:mod:`talens.weights`) — model-free, CPU.

The synthetic obfuscation (a *defense*, so it lives here in the test, not
in core per the two-part rule) builds an obfuscated table that differs
from plaintext by a row permutation τ, a column permutation, and Gaussian
noise scaled by ``alpha_e`` — AloePri's low-noise regime where weights
"differ almost only by row- and column-permutations" (§7.3). The VMA
sorted-quantile signature is invariant to the column permutation, so:

* ``alpha_e = 0``  → near-perfect Π recovery (sanity / undefended).
* ``alpha_e`` ↑    → recovery falls toward the 1/N chance floor.

This replicates AloePri Fig 3 (VMA >30% at αe=0.5 → defended at αe=1.0)
and is the ground truth the IT measures will be calibrated against.
"""

from __future__ import annotations

import numpy as np

from talens.weights import measures, vma
from talens.weights.types import WeightPair

N, D = 96, 48


def _plaintext(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Distinctive rows: a Gaussian base plus a few large per-row spikes
    (real embedding rows are heavy-tailed and heterogeneous). Pure
    standard-normal rows are near-exchangeable, so their sorted signatures
    barely differ and any noise collapses matching to the floor with no
    gradient — the spikes give each row a recognisable sorted tail."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, D)).astype(np.float32)
    spike_mask = rng.random((N, D)) < (4.0 / D)            # ~4 spikes/row
    W += (spike_mask * rng.normal(0.0, 8.0, (N, D))).astype(np.float32)
    token_ids = np.arange(N, dtype=np.int64)
    return W, token_ids


def _obfuscate(W: np.ndarray, token_ids: np.ndarray, *, alpha_e: float, seed: int = 1) -> WeightPair:
    rng = np.random.default_rng(seed)
    tau = rng.permutation(N)                       # row permutation τ
    col = rng.permutation(D)                       # column permutation (keymat-lite)
    noisy = W + alpha_e * W.std() * rng.standard_normal(W.shape).astype(np.float32)
    transformed = noisy[:, col]                    # obfuscated partner of plain[i]
    obf = np.empty_like(transformed)
    obf[tau] = transformed                         # place partner of plain[i] at row τ[i]
    return WeightPair(plain=W, obf=obf, perm=tau, token_ids=token_ids, kind="embed")


def test_recovers_permutation_when_undefended():
    W, ids = _plaintext()
    pair = _obfuscate(W, ids, alpha_e=0.0)
    res = vma.run(pair, bins=32, match="hungarian")
    assert res.attack == "vma_tau_recovery"
    assert res.primary_metric_name == "permutation_recovery_rate"
    assert res.ttrsr_top1 > 0.95
    assert res.risk_level == "high"


def test_recovery_decays_with_noise():
    """Monotone defense curve: recovery falls as the noise knob rises."""
    W, ids = _plaintext()
    recs = [
        vma.run(_obfuscate(W, ids, alpha_e=a), bins=32, match="hungarian").ttrsr_top1
        for a in (0.0, 0.5, 1.0, 2.0)
    ]
    assert recs[0] >= recs[1] >= recs[2] >= recs[3]   # monotone defense curve
    assert recs[0] > 0.95          # undefended
    assert recs[1] > 0.30          # αe=0.5 still leaks (cf. AloePri Fig 3 >30%)
    assert recs[-1] < 0.20         # heavy noise -> near chance


def test_nn_and_hungarian_both_recover_undefended():
    W, ids = _plaintext()
    pair = _obfuscate(W, ids, alpha_e=0.0)
    nn = vma.run(pair, bins=32, match="nn")
    hg = vma.run(pair, bins=32, match="hungarian")
    assert nn.ttrsr_top1 > 0.95
    assert hg.ttrsr_top1 >= nn.ttrsr_top1   # global assignment >= independent 1-NN
    assert nn.ttrsr_top10 >= nn.ttrsr_top1


def test_measures_track_the_defense_curve():
    """CLUB I(obf;plain) and targeted retrieval-PVI both fall as the noise
    knob rises — the measure↔recovery link the repo's thesis predicts."""
    W, ids = _plaintext()
    und = _obfuscate(W, ids, alpha_e=0.0)
    dfn = _obfuscate(W, ids, alpha_e=2.5)

    def _club(p):
        return measures.club_mi_weights(p, bins=32, steps=120, hidden_size=64, seed=0)["club_mi_bits"]

    def _pvi(p):
        return measures.v_information_weights(
            p, bins=32, n_train=48, n_val=24, n_test=24, candidate_pool_size=N
        )["v_information_bits"]

    club_und, club_dfn = _club(und), _club(dfn)
    pvi_und, pvi_dfn = _pvi(und), _pvi(dfn)

    assert pvi_und is not None and pvi_und > 0.0     # undefended leaks usable info
    assert pvi_und > pvi_dfn                          # PVI falls under noise
    assert club_und is not None and club_und > club_dfn  # CLUB falls under noise


def test_chance_floor_on_unrelated_tables():
    """Obfuscated table independent of plaintext -> ~1/N recovery."""
    W, ids = _plaintext(seed=0)
    rng = np.random.default_rng(7)
    junk = rng.standard_normal((N, D)).astype(np.float32)
    pair = WeightPair(plain=W, obf=junk, perm=rng.permutation(N), token_ids=ids, kind="embed")
    res = vma.run(pair, bins=32, match="hungarian")
    assert res.ttrsr_top1 < 0.1
