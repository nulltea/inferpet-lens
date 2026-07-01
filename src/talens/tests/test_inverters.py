"""Sanity tests for the inverter family (ridge / nn / mlp2) added for the
resid-depth-inversion phase. Model-free: synthetic activations whose linear
map to a small embedding table is recoverable, so ridge/mlp2 generalise and
the nn baseline reads the floor under a vocab-disjoint split."""

from __future__ import annotations

import numpy as np
import torch

from talens.attacks import (
    INVERTERS,
    learned_inversion,
    nn_inversion,
    ridge_inversion,
)


def _synthetic(n=600, d=32, d_emb=16, vocab=40, seed=0):
    """A linear-recoverable surface: X = E[y] @ A + noise, so resid→emb is a
    linear map a ridge/MLP can learn and generalise to held-out tokens."""
    rng = np.random.default_rng(seed)
    embed = rng.standard_normal((vocab, d_emb)).astype(np.float32)
    A = rng.standard_normal((d_emb, d)).astype(np.float32)
    y = rng.integers(0, vocab, size=n).astype(np.int64)
    X = (embed[y] @ A + 0.05 * rng.standard_normal((n, d))).astype(np.float32)
    return X, y, torch.from_numpy(embed)


def test_registry_keys():
    assert set(INVERTERS) == {"ridge", "nn", "mlp2"}


def test_ridge_recovers_and_shuffle_collapses():
    X, y, emb = _synthetic()
    real = ridge_inversion(X, y, emb, split_mode="vocab", candidate_pool_size=40)
    floor = ridge_inversion(X, y, emb, split_mode="vocab", control="shuffle",
                            candidate_pool_size=40)
    assert real is not None and floor is not None
    # genuine generalising recovery above the label-shuffle floor.
    assert real["ttrsr_top1"] > floor["ttrsr_top1"]
    assert set(real) >= {"ttrsr_top1", "ttrsr_top10", "n_train", "n_test", "split_mode"}


def test_mlp2_recovers_above_shuffle():
    X, y, emb = _synthetic()
    real = learned_inversion(X, y, emb, split_mode="vocab", candidate_pool_size=40,
                             hidden=64, epochs=60, patience=15)
    floor = learned_inversion(X, y, emb, split_mode="vocab", control="shuffle",
                              candidate_pool_size=40, hidden=64, epochs=60, patience=15)
    assert real is not None and floor is not None
    assert real["ttrsr_top1"] > floor["ttrsr_top1"]


def test_nn_is_memorization_floor_under_vocab_disjoint():
    """Cosine-NN predicts a train token; under a vocab-disjoint split that token
    is never a test token, so top-1 reads the generalisation floor (≈0)."""
    X, y, emb = _synthetic()
    nn = nn_inversion(X, y, emb, split_mode="vocab", candidate_pool_size=40)
    ridge = ridge_inversion(X, y, emb, split_mode="vocab", candidate_pool_size=40)
    assert nn is not None and ridge is not None
    assert nn["ttrsr_top1"] <= 0.05  # floor
    assert ridge["ttrsr_top1"] > nn["ttrsr_top1"]  # ridge generalises, nn cannot


def test_nn_memorizes_under_row_split():
    """Under a row-split (vocab overlap), the nearest train neighbour shares the
    test token's identity, so nn recovers well — confirming it is the
    memorisation baseline, not broken."""
    X, y, emb = _synthetic(n=800)
    nn = nn_inversion(X, y, emb, split_mode="row", candidate_pool_size=40)
    assert nn is not None and nn["ttrsr_top1"] > 0.3
