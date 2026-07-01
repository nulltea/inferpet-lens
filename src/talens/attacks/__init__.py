"""Attacks, organized per surface — one file per attack. The names below are the stable
public API: ``from talens.attacks import <name>``. Surface subpackages hold the sources:

* ``residual/``    — ISA-HiddenState (ridge, skip_decoder, isa_grad, logit_lens, hidden_state,
                     cover_break, inversion)
* ``attn_value/``  — ISA-AttnValue / kqv_out (rotation_recovery)
* ``attn_score/``  — ISA-AttnScore / kq (attn_score)
* ``embed_table/`` — IMA-EmbedRow (ima_transformer, nn)
* ``wire/``        — token-id wire (token_frequency: TFMA/SDA)
* ``weights/``     — VMA (vocab_match)
* ``kv/``          — attention k/v BSS (bss)

Shared primitives + the surface-agnostic cascade orchestrator live in ``_common``.
"""
from __future__ import annotations

from ._common import (nearest_token, ridge_W, multikey_ridge_W, cascade_attack,
                      orthogonal_procrustes_R, blockwise_procrustes_R)
from .residual.ridge import ridge_attack
from .residual.skip_decoder import LinearSkipDecoder, skip_decoder_attack
from .residual.isa_grad import isa_grad_attack
from .residual.logit_lens import LensHead, logit_lens_attack
from .residual.inversion import INVERTERS, ridge_inversion, nn_inversion, learned_inversion
from .residual import hidden_state, cover_break
from .attn_value.rotation_recovery import rotation_recovery_attack
from .attn_score import attn_score
from .embed_table.ima_transformer import IMAInverter, ima_transformer_attack
from .embed_table.nn import nn_attack
from .wire.token_frequency import tfma_recover, sda_recover
from .wire import token_frequency
from .weights import vocab_match
from .kv import bss

__all__ = [
    "nearest_token", "ridge_W", "multikey_ridge_W", "cascade_attack",
    "orthogonal_procrustes_R", "blockwise_procrustes_R",
    "ridge_attack", "LinearSkipDecoder", "skip_decoder_attack", "isa_grad_attack",
    "LensHead", "logit_lens_attack", "INVERTERS", "ridge_inversion", "nn_inversion",
    "learned_inversion", "rotation_recovery_attack", "IMAInverter", "ima_transformer_attack",
    "nn_attack", "tfma_recover", "sda_recover",
    "hidden_state", "cover_break", "attn_score", "token_frequency", "vocab_match", "bss",
]
