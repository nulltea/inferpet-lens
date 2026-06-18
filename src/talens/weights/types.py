"""``WeightPair`` — the pure-data input for the τ-recovery (weight-pair)
attack family, the weight-surface analog of
:class:`~talens.capture.types.CaptureSet`.

The τ-recovery attacks (VMA / IA / trained EmbedRow inverter) operate on a
**pair of weight matrices** — the public plaintext rows ``W[i]`` and the
deployed obfuscated rows ``W̃[j]`` (e.g. the embedding table of an AloePri
GGUF) — and recover the secret token permutation **Π** that aligns them.
No forward pass, no activations: pure linear algebra over the two tables,
so the whole family is CPU-unit-testable.

This module is **defense-agnostic** (Part 1). Constructing an obfuscated
table (keymat + noise + permutation) is a *defense* and lives in Part 2 /
tests; extracting the real tables from a GGUF is Part 2. Here we only hold
the two tables + the ground-truth permutation (for scoring) + the token
ids (the secret label the measures predict).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

FeatureFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class WeightPair:
    """One plaintext/obfuscated weight-table pair over ``N`` tokens.

    ``perm`` is the ground-truth permutation τ: the obfuscated partner of
    plaintext row ``i`` is ``obf[perm[i]]``. It is used **only for
    scoring** an attack and for aligning the paired views the measures
    consume — an attack never reads it.
    """

    plain: np.ndarray       # (N, d)   public plaintext rows W[i]
    obf: np.ndarray         # (N, d̃)  deployed obfuscated rows W̃[j]
    perm: np.ndarray        # (N,)     τ: obf[perm[i]] matches plain[i]
    token_ids: np.ndarray   # (N,)     token id of plaintext row i (the secret)
    kind: str = "embed"     # which weight surface: embed | head | gate | qk
    model_id: str = "synthetic"

    def n(self) -> int:
        return int(self.plain.shape[0])

    def inverse_perm(self) -> np.ndarray:
        """``inv[j] = i`` such that ``perm[i] = j`` — the true plaintext
        index of obfuscated row ``j`` (the per-row recovery ground truth)."""
        return np.argsort(self.perm)

    def aligned(self, feature_fn: FeatureFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(A, B, token_ids)`` with ``A[i] = φ(plain[i])`` and
        ``B[i] = φ(obf[perm[i]])`` — the paired plaintext/obfuscated row
        signatures, aligned by the true permutation. This is what the
        measures (CLUB ``I(A;B)``, retrieval-PVI) consume; the alignment
        is the eval-time pairing, not information the attack gets.
        """
        A = feature_fn(self.plain)
        B = feature_fn(self.obf)[self.perm]
        return A, B, self.token_ids
