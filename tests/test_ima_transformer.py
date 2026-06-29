"""IMA-EmbedRow-transformer inverter (talens.attacks.dp_inversion.ima_transformer_attack).

Plain control: trained on a SINGLE invertible linear obfuscation (row-split, matched), the 2-layer/
8-head inverter must recover the held-out rows (a common inverse exists). This is the control
private-rag's driver failed at paper hyperparameters; our port passes it, so the architecture/training
are sound and any low number on a real τ-invariant run is the defense, not a broken attack. Model-free.
"""

from __future__ import annotations

import numpy as np
import torch

from talens.attacks.dp_inversion import ima_transformer_attack


def test_ima_transformer_inverts_single_key():
    rng = np.random.default_rng(0); torch.manual_seed(0)
    V, d = 400, 32
    table = rng.standard_normal((V, d)).astype(np.float32)
    A = rng.standard_normal((d, d)).astype(np.float32)                 # one invertible linear obfuscation
    y = rng.integers(0, V, 6000)
    X = (table[y] @ A + 0.05 * rng.standard_normal((len(y), d))).astype(np.float32)
    pool = np.arange(V, dtype=np.int64)
    n = len(y); tr = np.arange(0, int(0.7 * n)); te = np.arange(int(0.7 * n), n)
    acc = float((ima_transformer_attack(X[tr], table[y[tr]], X[te], table[pool], pool,
                                        hidden=32, n_heads=4, epochs=40, batch=512) == y[te]).mean())
    assert acc > 0.6, f"IMA-transformer failed the single-key plain control: {acc}"
