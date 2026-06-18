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

* :func:`v_information_weights` — targeted PVI under the retrieval family:
  the candidate "embedding table" is the plaintext signatures ``A`` and the
  query is the obfuscated signatures ``B``. The probabilistic model (ridge
  ``B→A`` + cosine softmax over the ``A`` pool) **is** the VMA matching
  attack, so PVI is exactly the attack's pointwise usable information, and
  it runs vocab-disjoint (recover τ for tokens held out of the known-pair
  training set — the honest-attacker / IMA-EmbedRow regime).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..measures import club_mi_upper_bound, v_information_retrieval
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


def v_information_weights(
    pair: WeightPair, *, feature: str = "sorted_quantile", bins: int = 64, **vinfo_kwargs: Any
) -> dict[str, Any]:
    """Targeted retrieval-PVI: usable info about token identity carried by
    the obfuscated row, with the plaintext table as the candidate pool.

    The candidate "embedding table" is the plaintext signature table ``A``
    indexed by **row position** (= which plaintext row a query matches);
    the label is the position, not ``pair.token_ids`` (which are arbitrary
    vocab ids and would index out of ``A``). Position↔token is a bijection,
    so recovering the position recovers the token.
    """
    A, B, _ = pair.aligned(_feature_fn(feature, bins))
    ids = np.arange(A.shape[0], dtype=np.int64)
    embed_table = torch.from_numpy(np.ascontiguousarray(A)).to(torch.float32)
    return v_information_retrieval(B, ids, embed_table, **vinfo_kwargs)
