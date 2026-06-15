"""Resolution-B tests: the retrieval-family measures run **vocab-disjoint**
(train and test share no token id) and still register leakage, because
recovery flows through the embedding map and generalises to unseen ids —
exactly where the class-probe family collapses. See
``docs/research/attacks_setting.md``.
"""

from __future__ import annotations

import numpy as np
import torch

from talens.measures import online_code_length_retrieval, v_information_retrieval

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
