"""IT measures for the weight-pair surface — the same lenses the
activation pipeline uses, reused verbatim on the paired row signatures.

Given a :class:`WeightPair` and a signature ``φ``, ``aligned(φ)`` yields
the paired views ``A[i]=φ(plain[i])`` (public plaintext row) and
``B[i]=φ(obf[τ(i)])`` (its true obfuscated partner), plus the token ids.

* :func:`club_mi_weights` — CLUB ``I(B ; A)``: an MI upper bound between
  the obfuscated and plaintext row views. Unlike the SAE case (where
  ``Z=f(x)`` is deterministic and CLUB is meaningless), ``B`` is a *noisy*
  transform of ``A``, so this MI is a genuine cross-view quantity driven
  by the obfuscation's noise — the defense knob the calibration sweeps.
"""

from __future__ import annotations

from typing import Any

from ..probes import club_mi_upper_bound
from .features import get_feature, sorted_quantile
from .types import WeightPair


def _feature_fn(feature: str, bins: int):
    if feature == "sorted_quantile":
        return lambda W: sorted_quantile(W, bins=bins)
    return get_feature(feature)


def club_mi_weights(
    pair: WeightPair, *, feature: str = "sorted_quantile", bins: int = 64, **club_kwargs: Any
) -> dict[str, Any]:
    """CLUB ``I(obf-feature ; plain-feature)`` — leakage upper bound."""
    A, B, _ = pair.aligned(_feature_fn(feature, bins))
    return club_mi_upper_bound(B, A, **club_kwargs)
