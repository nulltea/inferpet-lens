"""Array-interface inversion attacks (src/talens/attacks/dp_inversion.py) — model-free."""
from __future__ import annotations

import numpy as np

from talens.attacks.dp_inversion import nn_attack


def test_nn_recovers_when_obs_is_in_embedding_space():
    """NN cosine-matches a hidden-state row to the nearest embedding row → recovers when the
    observation lives in the embedding space (L0 / plaintext / oracle-de-obfuscated)."""
    rng = np.random.default_rng(0)
    pool_ids = np.arange(50, dtype=np.int64)
    pool_emb = rng.standard_normal((50, 16)).astype(np.float32)
    te_ids = rng.choice(pool_ids, size=20)
    Xte = pool_emb[te_ids] + 0.01 * rng.standard_normal((20, 16)).astype(np.float32)  # near its own row
    yhat = nn_attack(None, None, Xte, pool_emb, pool_ids)
    assert (yhat == te_ids).mean() > 0.9


def test_nn_is_chance_under_basis_width_mismatch():
    """Under AloePri the released residual is in the P̂-basis (wider) → cross-space match is
    undefined; NN degenerates to ~chance (paper: AloePri NN = 0%)."""
    rng = np.random.default_rng(1)
    pool_ids = np.arange(50, dtype=np.int64)
    pool_emb = rng.standard_normal((50, 16)).astype(np.float32)
    Xte = rng.standard_normal((20, 24)).astype(np.float32)        # obf width 24 ≠ emb 16
    yhat = nn_attack(None, None, Xte, pool_emb, pool_ids)
    assert yhat.shape == (20,)
    assert np.all(yhat == pool_ids[0])                            # degenerate constant ⇒ ~chance
