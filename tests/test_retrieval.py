"""Resolution-B tests: the retrieval-family measures run **vocab-disjoint**
(train and test share no token id) and still register leakage, because
recovery flows through the embedding map and generalises to unseen ids —
exactly where the class-probe family collapses. See
``docs/research/attacks_setting.md``.
"""

from __future__ import annotations

import numpy as np
import torch

from talens.measures import (
    online_code_length_retrieval,
    v_information,
    v_information_retrieval,
)

VOCAB, DIM, N = 60, 24, 1500


def _embed() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(VOCAB, DIM, generator=g)


def _data(embed: torch.Tensor, *, informative: bool, seed: int = 1):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, VOCAB, size=N)
    if informative:
        X = (embed[torch.from_numpy(ids)] + 0.05 * torch.randn(N, DIM)).numpy()
    else:
        X = rng.standard_normal((N, DIM)).astype(np.float32)
    return X.astype(np.float32), ids.astype(np.int64)


def test_retrieval_vinfo_generalizes_under_vocab_disjoint():
    emb = _embed()
    Xi, yi = _data(emb, informative=True)
    Xn, yn = _data(emb, informative=False)
    info = v_information_retrieval(Xi, yi, emb, split_mode="vocab")
    noise = v_information_retrieval(Xn, yn, emb, split_mode="vocab")
    # Train/test share no id, yet the retrieval family still recovers
    # leakage on informative reps and ~none on noise.
    assert info["v_information_bits"] > 0.5
    assert info["v_information_bits"] > noise["v_information_bits"]
    assert noise["v_information_bits"] < 0.3


def test_retrieval_mdl_compresses_under_vocab_generalization():
    emb = _embed()
    Xi, yi = _data(emb, informative=True)
    Xn, yn = _data(emb, informative=False)
    info = online_code_length_retrieval(Xi, yi, emb)
    noise = online_code_length_retrieval(Xn, yn, emb)
    assert info["compression"] > 1.5
    assert info["compression"] > noise["compression"]


def test_retrieval_vinfo_reports_temperature_and_pool():
    emb = _embed()
    Xi, yi = _data(emb, informative=True)
    out = v_information_retrieval(Xi, yi, emb, split_mode="vocab")
    assert out["family"] == "retrieval"
    assert out["temperature"] > 0
    assert out["candidate_pool_size"] <= VOCAB


# --- Regression: the class-probe V-info overfits on high-d operands (shuffle
# floor ≪ 0, non-monotonic in noise); the retrieval family is the sane fix.
# Diagnosed in docs/dev/sae-attack.md (PVI/V-info overfit). ---
_HI_VOCAB, _HI_DIM, _HI_N = 150, 512, 1200


def _hi_data(noise: float = 0.1, seed: int = 2):
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(_HI_VOCAB, _HI_DIM, generator=g)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, _HI_VOCAB, size=_HI_N).astype(np.int64)
    X = (emb[torch.from_numpy(ids)] + noise * torch.randn(_HI_N, _HI_DIM)).numpy()
    return X.astype(np.float32), ids, emb


def test_classprobe_overfits_high_d_but_retrieval_floor_is_sane():
    """Shuffle-control floor: class-probe blows far below 0 (overfit); the
    bounded retrieval family stays near 0."""
    X, y, emb = _hi_data()
    cp_floor = v_information(X, y, control="shuffle")["v_information_bits"]
    perm = np.random.default_rng(3).permutation(y.size)
    retr_floor = v_information_retrieval(X, y[perm], emb, split_mode="row")["v_information_bits"]
    assert retr_floor > -2.0, retr_floor          # bounded family ~0
    assert cp_floor < retr_floor - 3.0, (cp_floor, retr_floor)  # class-probe overfits


def test_retrieval_pvi_monotone_under_noise():
    """Healthy estimator: PVI falls with added noise and shows no spurious rise
    at the first noise step (the class-probe's failure mode)."""
    X0, y, emb = _hi_data()
    rms = np.sqrt((X0.astype(np.float64) ** 2).mean(axis=1, keepdims=True))
    rng = np.random.default_rng(5)
    pvis = []
    for sigma in [0.0, 0.5, 1.0, 2.0]:
        Xn = (X0 + sigma * rms * rng.standard_normal(X0.shape)).astype(np.float32)
        pvis.append(v_information_retrieval(Xn, y, emb, split_mode="row")["v_information_bits"])
    assert pvis[0] > pvis[-1] + 0.5, pvis          # clean leaks more than heavy-noise
    assert pvis[1] <= pvis[0] + 0.3, pvis          # no spurious rise at first noise step
