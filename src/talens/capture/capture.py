"""nnsight-backed capture of Qwen3 representations.

The only module that imports the model stack (nnsight + transformers).
It runs prompts through the faithful HF Qwen3 and returns a
:class:`~talens.capture.types.CaptureSet`. Faithfulness is why we chose
nnsight over a reimplemented architecture: Qwen3's QK-norm / RoPE / GQA
are exactly as shipped, so the activations we measure are the real ones.

Representations captured (pass 1):

* ``resid_post`` — residual stream after each block, read from HF
  ``output_hidden_states`` (``hidden_states[i+1]`` is post-layer ``i``;
  ``hidden_states[0]`` is the embedding output, skipped).
* ``attn_score`` — per-head attention weights after softmax, read from
  HF ``output_attentions`` (``attentions[i]`` is ``(n_heads, n_q, n_kv)``
  for layer ``i``).

NOTE: validate the nnsight trace API + attention-weights availability
against the installed nnsight/transformers versions on the first real
capture (set ``attn_implementation="eager"`` if a kernel backend omits
attentions). This module is written but not yet executed on hardware.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch

from .types import CaptureSet

# --- internal-attention capture (kq / kqv_out) -------------------------------
# HF's output_attentions only exposes the POST-softmax weights, which barely
# leak token identity (position/sink-dominated). The aloepri ISA-AttnScore
# results come from two *other* attention surfaces:
#   kq      = pre-softmax Q·Kᵀ·scaling   (n_heads, n_q, n_kv)   ~48% recovery
#   kqv_out = per-head attention output, pre-W_o  (n_q, heads·head_dim) ~97%
# Both are locals inside the functional ``eager_attention_forward``, so we grab
# them with a drop-in replacement that computes *identically* (faithful — the
# model output is unchanged) and stashes the two tensors per layer.
_ATTN_BUF: dict | None = None


def _capturing_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Byte-faithful copy of qwen3 ``eager_attention_forward`` that also
    stashes the pre-softmax scores (kq) and the pre-W_o output (kqv_out)."""
    import torch.nn.functional as F
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    if _ATTN_BUF is not None and (_ATTN_BUF.get("want_k") or _ATTN_BUF.get("want_v")):
        # raw stored KV-cache, pre-repeat_kv (per KV-head): (1, n_kv_heads, seq, head_dim).
        # Store flattened to 2-D (seq, n_kv_heads*head_dim) so rows == tokens; the
        # n_kv_heads / head_dim split is recoverable from shape (head_dim known to callers).
        if _ATTN_BUF.get("want_k"):
            k0 = key.detach()[0].permute(1, 0, 2).contiguous()  # (seq, n_kv_heads, head_dim)
            sq, nh, hd = k0.shape
            _ATTN_BUF["k"][module.layer_idx] = k0.reshape(sq, nh * hd).to(torch.float32).cpu()
        if _ATTN_BUF.get("want_v"):
            v0 = value.detach()[0].permute(1, 0, 2).contiguous()
            sq, nh, hd = v0.shape
            _ATTN_BUF["v"][module.layer_idx] = v0.reshape(sq, nh * hd).to(torch.float32).cpu()

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if _ATTN_BUF is not None and _ATTN_BUF["want_kq"]:
        # pre-mask, pre-softmax scores: (1, n_heads, n_q, n_kv) -> drop batch
        _ATTN_BUF["kq"][module.layer_idx] = attn_weights.detach()[0].to(torch.float32).cpu()
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()  # (1, n_q, n_heads, head_dim)
    if _ATTN_BUF is not None and _ATTN_BUF["want_kqv"]:
        _, n_q, n_h, h_d = attn_output.shape
        _ATTN_BUF["kqv_out"][module.layer_idx] = (
            attn_output.reshape(n_q, n_h * h_d).detach().to(torch.float32).cpu()
        )
    return attn_output, attn_weights


@contextlib.contextmanager
def _patched_eager_attention():
    """Swap qwen3's module-level ``eager_attention_forward`` for the capturing
    variant for the duration of the block. Safe because ``eager`` is not in
    ``ALL_ATTENTION_FUNCTIONS`` — ``forward`` passes the module-level function
    as the get_interface default, so rebinding it is what's actually called."""
    import transformers.models.qwen3.modeling_qwen3 as mq

    orig = mq.eager_attention_forward
    mq.eager_attention_forward = _capturing_eager
    try:
        yield
    finally:
        mq.eager_attention_forward = orig


def load_model(model_id: str = "Qwen/Qwen3-4B", *, dtype: torch.dtype = torch.bfloat16):
    """Load the faithful HF model under nnsight. Native dtype (bf16);
    we never upcast the weights — only small captured slices are cast to
    f32 downstream for ridge/probe numerics.
    """
    from nnsight import LanguageModel  # lazy: keeps the dep optional

    return LanguageModel(
        model_id,
        dtype=dtype,                   # transformers 5.x (was torch_dtype)
        attn_implementation="eager",   # ensures attention weights are returned
        device_map="auto",             # place on the GPU (gfx1151 via ROCm)
        dispatch=True,
    )


def embed_table(model) -> torch.Tensor:
    """The input-embedding table as ``(vocab, d)`` float32 — the
    candidate-matching basis for the inversion attacks.
    """
    w = model.model.embed_tokens.weight
    return w.detach().to(torch.float32).cpu().clone()


def capture_representations(
    model,
    prompts: list[str],
    *,
    layers: list[int] | None = None,
    kinds: tuple[str, ...] = ("resid_post", "kq", "kqv_out"),
) -> CaptureSet:
    """Run ``prompts`` and collect the requested representation kinds at
    the requested layers (default: all layers). Returns a CaptureSet.

    Kinds: ``resid_post`` (residual stream, via HF hidden_states),
    ``kq`` (pre-softmax Q·Kᵀ scores, per head) and ``kqv_out`` (per-head
    attention output before W_o) — the latter two via the faithful
    eager-attention patch. (Legacy ``attn_score`` = post-softmax weights is
    still accepted for back-compat.)
    """
    global _ATTN_BUF
    want_resid = "resid_post" in kinds
    want_attn = "attn_score" in kinds
    want_kq = "kq" in kinds
    want_kqv = "kqv_out" in kinds
    want_k = "k" in kinds
    want_v = "v" in kinds
    want_internal = want_kq or want_kqv or want_k or want_v
    tokenizer = model.tokenizer
    model_id = getattr(model.config, "_name_or_path", "unknown")

    prompt_token_ids: list[list[int]] = []
    # operands[(kind, layer)] grows one entry per prompt, in order.
    operands: dict[tuple[str, int], list[torch.Tensor]] = {}

    patch = _patched_eager_attention() if want_internal else contextlib.nullcontext()
    with patch:
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
            prompt_token_ids.append(ids.tolist())

            if want_internal:
                _ATTN_BUF = {
                    "kq": {}, "kqv_out": {}, "k": {}, "v": {},
                    "want_kq": want_kq, "want_kqv": want_kqv,
                    "want_k": want_k, "want_v": want_v,
                }

            with model.trace(
                prompt,
                output_hidden_states=want_resid,
                output_attentions=want_attn,
            ):
                hs = model.output.hidden_states.save() if want_resid else None
                att = model.output.attentions.save() if want_attn else None

            buf = _ATTN_BUF if want_internal else None
            if want_resid:
                n_layers = len(hs) - 1
            elif want_attn:
                n_layers = len(att)
            else:
                n_layers = len(buf["kq"] or buf["kqv_out"] or buf["k"] or buf["v"])
            layer_set = layers if layers is not None else list(range(n_layers))

            for li in layer_set:
                if want_resid:
                    # hidden_states[li+1]: post-block li; drop the batch dim.
                    op = hs[li + 1][0].detach().to(torch.float32).cpu()
                    operands.setdefault(("resid_post", li), []).append(op)
                if want_attn:
                    op = att[li][0].detach().to(torch.float32).cpu()
                    operands.setdefault(("attn_score", li), []).append(op)
                if want_kq:
                    operands.setdefault(("kq", li), []).append(buf["kq"][li])
                if want_kqv:
                    operands.setdefault(("kqv_out", li), []).append(buf["kqv_out"][li])
                if want_k:
                    operands.setdefault(("k", li), []).append(buf["k"][li])
                if want_v:
                    operands.setdefault(("v", li), []).append(buf["v"][li])

    _ATTN_BUF = None
    return CaptureSet(
        model_id=model_id,
        prompt_token_ids=prompt_token_ids,
        operands=operands,
    )


def load_or_capture(
    model_id: str,
    prompts: list[str],
    *,
    capture_layers: list[int] | None = None,
    kinds: tuple[str, ...] = ("resid_post", "kq", "kqv_out"),
    cache_dir: Path | str | None = None,
    refresh: bool = False,
) -> tuple[CaptureSet, torch.Tensor, str]:
    """Return ``(capture, embed_table, source)`` for ``prompts``, reusing
    a disk cache when one covers the requested ``capture_layers``.

    On a hit the model is **not** loaded (both the capture and the
    embedding table are read from disk); ``source`` is ``"cache"``. On a
    miss the model is loaded, the requested layers are captured and the
    cache is (re)written; ``source`` is ``"captured"``. ``refresh=True``
    forces a recapture even when a usable cache exists.
    """
    from .cache import (
        DEFAULT_CACHE_DIR,
        can_reuse,
        capture_cache_path,
        embed_cache_path,
        load_capture,
        load_embed,
        present_layers,
        save_capture,
        save_embed,
    )

    cdir = DEFAULT_CACHE_DIR if cache_dir is None else Path(cache_dir)
    cpath = capture_cache_path(cdir, model_id, prompts, kinds)
    epath = embed_cache_path(cdir, model_id)

    if not refresh and cpath.exists() and epath.exists():
        cap, cached_spec = load_capture(cpath)
        if can_reuse(present_layers(cap), cached_spec, capture_layers):
            return cap, load_embed(epath), "cache"

    model = load_model(model_id)
    emb = embed_table(model)
    cap = capture_representations(model, prompts, layers=capture_layers, kinds=kinds)
    save_capture(cap, cpath, capture_layers=capture_layers)
    save_embed(emb, epath)
    # Capture is done; free the model's GPU memory so the downstream
    # GPU measures (probe/CLUB/wide kqv_out ridge solve) aren't starved.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cap, emb, "captured"
