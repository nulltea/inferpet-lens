"""Guard the multi-key-synthesis ridge optimization (talens.attacks.dp_inversion.multikey_ridge_W).

The eval computes the AloePri ISA-HiddenState blind inverter by accumulating the Gram analytically
(Σ_k Pk^T G0 Pk) instead of stacking K copies of the synthetic obfuscated reps. This must equal the
naive stack-then-ridge result exactly; the test fails if the accumulation is ever broken. Model-free.
"""

from __future__ import annotations

import numpy as np

from talens.attacks.dp_inversion import multikey_ridge_W, ridge_W


def test_multikey_ridge_matches_stacking():
    rng = np.random.default_rng(0)
    n, d0, d1, de, K, alpha = 300, 40, 60, 32, 8, 1.0
    Xc = rng.standard_normal((n, d0)).astype(np.float32)
    E = rng.standard_normal((n, de)).astype(np.float32)
    Pks = [rng.standard_normal((d0, d1)).astype(np.float32) for _ in range(K)]

    # naive: build the K·n synthetic stack and fit ridge directly
    Xstack = np.concatenate([Xc @ P for P in Pks], 0)
    Estack = np.concatenate([E] * K, 0)
    W_stack = ridge_W(Xstack, Estack, alpha=alpha)

    # optimized: accumulate the Gram from G0 = Xc^T Xc, H0 = Xc^T E
    G0 = Xc.T.astype(np.float64) @ Xc.astype(np.float64)
    H0 = Xc.T.astype(np.float64) @ E.astype(np.float64)
    W_accum = multikey_ridge_W(G0, H0, Pks, alpha=alpha)

    assert W_stack.shape == W_accum.shape == (d1, de)
    assert np.allclose(W_stack, W_accum, atol=1e-4), np.abs(W_stack - W_accum).max()
