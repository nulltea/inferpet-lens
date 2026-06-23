"""transformer-attacks-lens (``talens``).

Measure the invertibility / information leakage of transformer
representations, and calibrate information-theoretic probes (PVI /
V-usable information, MDL online-coding) against confidential-inference
attacks (hidden-state inversion, cover-break, attention-score inversion).

The library is **scheme-agnostic**: it knows nothing about any particular
confidential-inference defense. A "defense" is an external, pluggable
:class:`~talens.transforms.Transform` applied to a representation before
the attacks/measures see it; the only built-in transform is
:class:`~talens.transforms.Identity` (the plaintext model). See
``docs/plans/it-leakage-estimation-set.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .metrics import AttackResult, classify_risk_level, topk_recovery, ttrsr
from .report import (
    LeakageReport,
    Readout,
    canonical_bits,
    embedding_readout,
    error_band_readout,
    format_bits,
    membership_readout,
    permutation_readout,
    perplexity_from_bits,
    text_readout,
    token_f1,
    token_id_readout,
)
from .transforms import Identity, Transform

__all__ = [
    "AttackResult",
    "classify_risk_level",
    "topk_recovery",
    "ttrsr",
    "Identity",
    "Transform",
    # unified reporting layer (bits canonical + per-secret readout)
    "LeakageReport",
    "Readout",
    "canonical_bits",
    "format_bits",
    "perplexity_from_bits",
    "token_f1",
    "token_id_readout",
    "text_readout",
    "permutation_readout",
    "embedding_readout",
    "membership_readout",
    "error_band_readout",
]
