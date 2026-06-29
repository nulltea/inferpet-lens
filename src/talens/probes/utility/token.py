"""Generation-utility probes: teacher-forced next-token accuracy, perplexity, output agreement.

The shared model-running primitive is :func:`teacher_forced_pass` — run it ONCE clean (``hook=None``)
and ONCE with the defense's embedding-layer hook, then feed the two passes to the pure probes below.
This is the part every scheme's utility eval was reimplementing; centralising it is what makes the
numbers comparable. Conventions follow the DP-LLM literature (perplexity + ground-truth top-1
accuracy referenced to the clean baseline; agreement = self-consistency vs the clean argmax).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .result import UtilityResult, _retention


@dataclass
class TokenPass:
    """Flattened teacher-forced outputs over all real target positions (one row per predicted token)."""

    ce: np.ndarray          # (N,) per-token cross-entropy, nats
    pred_ids: np.ndarray    # (N,) argmax next-token prediction
    target_ids: np.ndarray  # (N,) ground-truth next token


@torch.no_grad()
def teacher_forced_pass(model, tok, prompts, hook=None, *, max_tokens=64, batch_size=32,
                        device=None) -> TokenPass:
    """One teacher-forced forward sweep. ``hook`` (e.g. a defense applied at the embedding layer) is
    registered for the duration; ``hook=None`` is the clean baseline. Prompts are right-padded and
    truncated to ``max_tokens`` (None ⇒ no truncation; pads masked out). Returns flattened per-token CE / preds / targets in
    a deterministic order, so a clean and a defended pass align position-for-position."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    handle = model.get_input_embeddings().register_forward_hook(hook) if hook is not None else None
    trunc = {"truncation": True, "max_length": max_tokens} if max_tokens else {"truncation": False}
    ce, preds, tgts = [], [], []
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, **trunc)
            ids, mask = enc.input_ids.to(dev), enc.attention_mask.to(dev)
            logits = model(ids, attention_mask=mask, use_cache=False).logits.float()
            pred, tgt, m = logits[:, :-1], ids[:, 1:], mask[:, 1:].bool()   # predict t+1 from ≤t
            lp = torch.log_softmax(pred, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            ce.append((-lp[m]).cpu().numpy())
            preds.append(pred.argmax(-1)[m].cpu().numpy())
            tgts.append(tgt[m].cpu().numpy())
    finally:
        if handle is not None:
            handle.remove()
    return TokenPass(np.concatenate(ce), np.concatenate(preds), np.concatenate(tgts))


def _check_aligned(defended: TokenPass, clean: TokenPass) -> None:
    if not np.array_equal(defended.target_ids, clean.target_ids):
        raise ValueError("defended and clean passes are not aligned (different prompts/order/truncation)")


def next_token_accuracy(defended: TokenPass, clean: TokenPass) -> UtilityResult:
    """Top-1 next-token accuracy vs ground truth, and its retention = acc(defended)/acc(clean)."""
    _check_aligned(defended, clean)
    acc_def = float((defended.pred_ids == defended.target_ids).mean())
    acc_clean = float((clean.pred_ids == clean.target_ids).mean())
    return UtilityResult("next_token_accuracy", acc_clean, acc_def,
                         _retention(acc_clean, acc_def, higher_is_better=True),
                         higher_is_better=True, extra={"n_tokens": int(defended.target_ids.size)})


def perplexity(defended: TokenPass, clean: TokenPass) -> UtilityResult:
    """Teacher-forced perplexity and its retention = ppl(clean)/ppl(defended). extra.degradation =
    ppl(defended)/ppl(clean) − 1 (the DP-LLM generation-utility convention)."""
    ppl_def = float(np.exp(defended.ce.mean()))
    ppl_clean = float(np.exp(clean.ce.mean()))
    return UtilityResult("perplexity", ppl_clean, ppl_def,
                         _retention(ppl_clean, ppl_def, higher_is_better=False),
                         higher_is_better=False,
                         extra={"degradation": ppl_def / ppl_clean - 1.0 if ppl_clean else None})


def output_agreement(defended: TokenPass, clean: TokenPass) -> UtilityResult:
    """Self-consistency: fraction of positions where the defended argmax equals the clean argmax
    (quantization-eval style). Clean baseline is 1.0 by definition, so retention == agreement."""
    _check_aligned(defended, clean)
    agree = float((defended.pred_ids == clean.pred_ids).mean())
    return UtilityResult("output_agreement", 1.0, agree, _retention(1.0, agree, higher_is_better=True),
                         higher_is_better=True, extra={"n_tokens": int(defended.pred_ids.size)})
