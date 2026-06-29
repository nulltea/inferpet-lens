"""Utility probes — the *utility* axis of the privacy–utility tradeoff (companion to the leakage
probes in :mod:`talens.probes`).

A defense is studied on two axes: how much it leaks (leakage probes → bits) and how much usefulness
it costs (utility probes → retention). Every utility probe returns a :class:`UtilityResult` whose
``retention`` ∈ [0, 1] (1 = lossless) is the SINGLE comparable scalar, so a Gaussian local-DP scheme,
a dχ-privacy SnD scheme, and any other defense line up on one axis.

Two families:

* **generation utility** (:mod:`.token`) — :func:`teacher_forced_pass` (the shared model-running
  primitive; run clean + defended), then :func:`next_token_accuracy`, :func:`perplexity`,
  :func:`output_agreement` on the two passes.
* **embedding utility** (:mod:`.embedding`) — :func:`embedding_fidelity` and the denoiser
  :func:`embedding_recovery` on captured (pooled) embedding matrices.

:func:`retention_thresholds` locates the privacy-budget value (ε/η) at which retention crosses
−10/−20/−50%.
"""
from __future__ import annotations

from .embedding import embedding_fidelity, embedding_recovery
from .result import UtilityResult, retention_thresholds
from .token import (
    TokenPass,
    next_token_accuracy,
    output_agreement,
    perplexity,
    teacher_forced_pass,
)

__all__ = [
    "UtilityResult",
    "retention_thresholds",
    "TokenPass",
    "teacher_forced_pass",
    "next_token_accuracy",      # next-token accuracy retention (the canonical generation-utility probe)
    "perplexity",               # perplexity + degradation
    "output_agreement",         # self-consistency vs clean argmax
    "embedding_fidelity",       # cosine/MSE of an embedding vs clean
    "embedding_recovery",       # denoiser gap-closing (cos/MSE recovery)
]
