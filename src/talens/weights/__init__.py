"""Weight-pair (τ-recovery) attack family.

The weight-surface analog of the activation pipeline: attacks that recover
the secret token permutation Π from a public/obfuscated weight-table pair
(:class:`~talens.weights.types.WeightPair`), the family AloePri evaluates
against (VMA / IA / trained EmbedRow inverter).

* :mod:`types`    — ``WeightPair`` input.
* :mod:`features` — obfuscation-invariant row signatures ``φ``.
* :mod:`vma`      — Vocabulary-Matching Attack (RowSort + assignment).
* :mod:`measures` — CLUB ``I(obf;plain)`` + targeted retrieval-PVI.

Defense-agnostic (Part 1): constructing an obfuscated table and extracting
real GGUF tables are Part 2 / tests.
"""

from __future__ import annotations

from . import features, measures, vma
from .types import WeightPair

__all__ = ["WeightPair", "features", "measures", "vma"]
