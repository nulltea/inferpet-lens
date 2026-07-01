"""Two-stage τ-leak cascade primitive (talens.attacks.dp_inversion.cascade_attack).

A harvest reveals true labels for k token types; the cascade trains a generic inverter on those
(deployment-basis rep, token) pairs and scores recovery on the HELD-OUT (unharvested) types. The
bootstrap property: a fixed linear obfuscation X = E·A is invertible, so given ENOUGH harvested types
ridge learns A⁻¹ and generalizes to never-harvested tokens; with too few it cannot. Model-free.
"""

from __future__ import annotations

import numpy as np

from talens.attacks.dp_inversion import cascade_attack, ridge_attack


def test_cascade_bootstrap_threshold():
    rng = np.random.default_rng(0)
    V, d = 200, 16
    table = rng.standard_normal((V, d)).astype(np.float32)        # token embeddings (the target space)
    A = rng.standard_normal((d, d)).astype(np.float32)            # fixed invertible linear obfuscation
    y = rng.integers(0, V, 4000)
    X = (table[y] @ A).astype(np.float32)                         # deployment rep of each token
    pool = np.arange(V, dtype=np.int64)

    big = cascade_attack(ridge_attack, X, y, np.arange(0, 150), table, pool)     # 150 harvested types
    small = cascade_attack(ridge_attack, X, y, np.arange(0, 3), table, pool)     # 3 harvested types

    # generalization to UNHARVESTED types rises with the harvest size (the bootstrap)
    assert big["unharvested"] > small["unharvested"]
    # enough pairs => ridge inverts A => reads held-out tokens it never saw a label for
    assert big["unharvested"] > 0.5
    # in-set sanity recovers its own harvested types
    assert big["harvested"] > 0.8
    assert big["n_held"] > 0 and small["n_held"] > 0


def test_cascade_blind_aug_only():
    """With no harvested types but a blind augmentation, the cascade still runs (k=0 baseline)."""
    rng = np.random.default_rng(1)
    V, d = 80, 8
    table = rng.standard_normal((V, d)).astype(np.float32)
    y = rng.integers(0, V, 600)
    X = table[y].astype(np.float32)                               # plaintext-like (aug shares basis)
    pool = np.arange(V, dtype=np.int64)
    out = cascade_attack(ridge_attack, X, y, harvested_types=[], table=table, pool=pool,
                         X_aug=X.copy(), y_aug=y.copy())
    assert out["unharvested"] is not None
