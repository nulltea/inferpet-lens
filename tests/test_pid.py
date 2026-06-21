"""Synthetic oracle for the MMI-PID attention-channel probe
(:func:`talens.measures.pid.pid_mmi`) — model-free, CPU.

Three canonical bivariate sources with known PID structure, encoded as
Gaussian-separated continuous features so the capacity-matched V-info readers
can pick them up:

* **Redundant** — QK and OV both reveal the label → R≈I, unique/synergy≈0.
* **Unique-to-QK** — only QK reveals the label → U_qk≈I, U_ov≈0, R≈0.
* **Synergistic (XOR)** — label = QK_bit ⊕ OV_bit, each marginal uninformative
  → R≈0, U≈0, synergy > 0.
"""

from __future__ import annotations

import numpy as np

from talens.measures.pid import pid_mmi

N, D = 1200, 16
SEP = 4.0  # class separation in feature space


def _embed_bits(bits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Map a {0,1} vector to a (N, D) Gaussian blob separated by class."""
    centers = np.where(bits[:, None] > 0, SEP, -SEP)
    return (centers + rng.standard_normal((bits.size, D))).astype(np.float32)


def _redundant(seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    return _embed_bits(y, rng), _embed_bits(y, rng), y.astype(np.int64)


def _unique_qk(seed=1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    junk = rng.integers(0, 2, N)
    return _embed_bits(y, rng), _embed_bits(junk, rng), y.astype(np.int64)


def _synergy_xor(seed=2):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, N)
    b = rng.integers(0, 2, N)
    y = (a ^ b).astype(np.int64)
    return _embed_bits(a, rng), _embed_bits(b, rng), y


# A non-parametric reader (knn) is used for the oracle so the joint reader can
# represent XOR — the default ``pca_softmax`` is linear and measures *linear*
# usable-info (it cannot see nonlinear synergy, by design; see pid.py).
KW = dict(family="knn", dim=8, n_neighbors=15)


def test_redundant_source_is_mostly_redundancy():
    Xqk, Xov, y = _redundant()
    r = pid_mmi(Xqk, Xov, y, **KW)
    assert r["redundancy_bits"] > 0.5            # most of the ~1 bit is shared
    assert r["unique_qk_bits"] < 0.2
    assert r["unique_ov_bits"] < 0.2
    assert r["synergy_bits"] < 0.2


def test_unique_source_loads_on_one_operand():
    Xqk, Xov, y = _unique_qk()
    r = pid_mmi(Xqk, Xov, y, **KW)
    assert r["unique_qk_bits"] > 0.5             # QK carries it
    assert r["i_ov_bits"] < 0.2                  # OV is junk
    assert r["unique_qk_bits"] > r["unique_ov_bits"]


def test_xor_source_is_synergistic():
    Xqk, Xov, y = _synergy_xor()
    r = pid_mmi(Xqk, Xov, y, **KW)
    assert r["i_qk_bits"] < 0.2                  # neither marginal reveals XOR
    assert r["i_ov_bits"] < 0.2
    assert r["i_joint_bits"] > 0.5               # but jointly they do
    assert r["synergy_bits"] > 0.4


def test_atoms_reconstruct_the_informations():
    """R+U_qk = I(QK), R+U_ov = I(OV), and joint = R+U_qk+U_ov+S_raw."""
    Xqk, Xov, y = _redundant()
    r = pid_mmi(Xqk, Xov, y, **KW)
    assert abs((r["redundancy_bits"] + r["unique_qk_bits"]) - r["i_qk_bits"]) < 1e-6
    assert abs((r["redundancy_bits"] + r["unique_ov_bits"]) - r["i_ov_bits"]) < 1e-6
    recon = (r["redundancy_bits"] + r["unique_qk_bits"]
             + r["unique_ov_bits"] + r["synergy_raw_bits"])
    assert abs(recon - r["i_joint_bits"]) < 1e-6
