"""Threat-model fit-pairs override for v_information_capacity (PVI).

The PVI reader trains on the supplied (fit_X, fit_y) and is scored on the released (X, y). When the
fit reps live in the SAME basis as the eval reps the reader transfers (high PVI = information is
accessible to a predictor trained on attack-accessible data); when the fit reps live in a DIFFERENT
basis (a secret-key change of basis the attacker cannot reproduce) the reader cannot transfer and PVI
collapses. This is the DP (reproducible release -> high) vs AloePri (secret key -> collapse) distinction
in miniature, and it is the property that makes the same probe valid across both threat models.
"""

from __future__ import annotations

import numpy as np

from talens.probes.vinfo_capacity import v_information_capacity


def _orth(d, rng):
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q.astype(np.float32)


def _draw(centers, m, rng):
    y = rng.integers(0, centers.shape[0], m)
    X = (centers[y] + rng.standard_normal((m, centers.shape[1]))).astype(np.float32)
    return X, y


def test_fit_pairs_basis_access():
    rng = np.random.default_rng(0)
    d, C, n = 32, 6, 1200
    centers = rng.standard_normal((C, d)) * 4.0            # well-separated classes
    Xev, yev = _draw(centers, n, rng)
    Xf, yf = _draw(centers, n, rng)                        # fit draw, shared class structure
    P, Q = _orth(d, rng), _orth(d, rng)                    # eval basis P, a DIFFERENT basis Q

    same = v_information_capacity((Xev @ P).astype(np.float32), yev,
                                  fit_X=(Xf @ P).astype(np.float32), fit_y=yf,
                                  family="pca_softmax", dim=16)
    diff = v_information_capacity((Xev @ P).astype(np.float32), yev,
                                  fit_X=(Xf @ Q).astype(np.float32), fit_y=yf,
                                  family="pca_softmax", dim=16)

    assert same["v_information_bits"] is not None and diff["v_information_bits"] is not None
    # same-basis fit transfers (information accessible to an attack-accessible-trained reader);
    # a different (secret) basis the attacker cannot reproduce degrades transfer substantially.
    assert same["v_information_bits"] > 1.0
    assert diff["v_information_bits"] < 0.6 * same["v_information_bits"]


def test_default_path_still_works():
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((5, 24)) * 4.0
    X, y = _draw(centers, 800, rng)
    out = v_information_capacity(X, y, family="pca_softmax", dim=12)   # fit_X=None -> internal row-split
    assert out["v_information_bits"] is not None and out["v_information_bits"] > 0.5


def test_fit_pairs_must_be_paired():
    rng = np.random.default_rng(2)
    X, y = _draw(rng.standard_normal((4, 8)) * 4.0, 200, rng)
    try:
        v_information_capacity(X, y, fit_X=X, family="pca_softmax", dim=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError when fit_X given without fit_y")
