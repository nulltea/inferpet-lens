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

from pathlib import Path

import torch

from .types import CaptureSet


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
    kinds: tuple[str, ...] = ("resid_post", "attn_score"),
) -> CaptureSet:
    """Run ``prompts`` and collect the requested representation kinds at
    the requested layers (default: all layers). Returns a CaptureSet.
    """
    want_resid = "resid_post" in kinds
    want_attn = "attn_score" in kinds
    tokenizer = model.tokenizer
    model_id = getattr(model.config, "_name_or_path", "unknown")

    prompt_token_ids: list[list[int]] = []
    # operands[(kind, layer)] grows one entry per prompt, in order.
    operands: dict[tuple[str, int], list[torch.Tensor]] = {}

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        prompt_token_ids.append(ids.tolist())

        with model.trace(
            prompt,
            output_hidden_states=want_resid,
            output_attentions=want_attn,
        ):
            hs = model.output.hidden_states.save() if want_resid else None
            att = model.output.attentions.save() if want_attn else None

        n_layers = (len(hs) - 1) if want_resid else len(att)
        layer_set = layers if layers is not None else list(range(n_layers))

        for li in layer_set:
            if want_resid:
                # hidden_states[li+1]: post-block li; drop the batch dim.
                op = hs[li + 1][0].detach().to(torch.float32).cpu()
                operands.setdefault(("resid_post", li), []).append(op)
            if want_attn:
                # attentions[li]: (1, n_heads, n_q, n_kv) -> (n_heads, n_q, n_kv)
                op = att[li][0].detach().to(torch.float32).cpu()
                operands.setdefault(("attn_score", li), []).append(op)

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
    kinds: tuple[str, ...] = ("resid_post", "attn_score"),
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
    return cap, emb, "captured"
