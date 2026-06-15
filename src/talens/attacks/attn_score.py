"""Attention-score inversion attack (ISA-AttnScore).

The attacker observes per-head attention scores ``(n_heads, n_q, n_kv)``
at a layer. :meth:`CaptureSet.stack` flattens each query row to
``(n_heads · max_kv)`` features (ragged key length zero-padded to the
global max), and the same ridge inverter as the hidden-state attack
recovers the token id of each query position.

This is the surface that, under a split TEE↔GPU scheme, is exposed only
if attention is computed off the trusted device. It is also the locus of
the unpublished *cover-invariance* result: ``softmax(QKᵀ)`` is invariant
under a shared orthogonal rotation of Q and K, so an orthogonal-cover
:class:`~talens.transforms.Transform` would leave this attack's input
information untouched — a hypothesis this driver lets us test once a
real cover is injected.
"""

from __future__ import annotations

import torch

from ..capture.types import CaptureSet
from ..metrics import AttackResult
from ..transforms import Transform
from . import hidden_state


def run(
    capture: CaptureSet,
    embed_table: torch.Tensor,
    *,
    layer: int,
    kind: str = "attn_score",
    transform: Transform | None = None,
    n_train: int = 256,
    n_val: int = 64,
    n_test: int = 64,
    **kwargs,
) -> AttackResult:
    """Delegate to the shared ridge inversion with the attention-score
    kind and attack label. Smaller default splits mirror aloepri's
    ``ISABaselineConfig`` (sequence-level capture yields fewer rows).
    """
    return hidden_state.run(
        capture,
        embed_table,
        layer=layer,
        kind=kind,
        transform=transform,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        attack_name="attn_score_inversion",
        **kwargs,
    )
