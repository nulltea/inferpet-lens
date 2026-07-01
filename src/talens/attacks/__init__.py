"""Attacks, organized per surface — one file per attack. The names below are the stable
public API: ``from talens.attacks import <name>``. Surface subpackages hold the sources:

* ``residual/``    — ISA-HiddenState (ridge, skip_decoder, isa_grad, logit_lens, hidden_state,
                     cover_break, inversion)
* ``attn_qkv/``    — attention q/k/v surfaces: rotation_recovery (kqv_out value), attn_score (kq
                     scores), bss (k/v blind source separation)
* ``embed_table/`` — IMA-EmbedRow (ima_transformer, nn)
* ``vocab/``       — vocabulary-mapping attacks: vocab_match (VMA, weight surface), token_frequency
                     (TFMA/SDA, token-id wire)

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
from .attn_qkv.rotation_recovery import rotation_recovery_attack
from .attn_qkv import attn_score
from .embed_table.ima_transformer import IMAInverter, ima_transformer_attack
from .embed_table.nn import nn_attack
from .vocab.token_frequency import tfma_recover, sda_recover
from .vocab import token_frequency
from .vocab import vocab_match
from .attn_qkv import bss

__all__ = [
    "nearest_token", "ridge_W", "multikey_ridge_W", "cascade_attack",
    "orthogonal_procrustes_R", "blockwise_procrustes_R",
    "ridge_attack", "LinearSkipDecoder", "skip_decoder_attack", "isa_grad_attack",
    "LensHead", "logit_lens_attack", "INVERTERS", "ridge_inversion", "nn_inversion",
    "learned_inversion", "rotation_recovery_attack", "IMAInverter", "ima_transformer_attack",
    "nn_attack", "tfma_recover", "sda_recover",
    "hidden_state", "cover_break", "attn_score", "token_frequency", "vocab_match", "bss",
]
