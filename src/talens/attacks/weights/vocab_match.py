"""Hidden No More vocab-matching cover-break attack (ICML'25,
`2505.18332 <https://arxiv.org/abs/2505.18332>`_).

This is the *published* permutation-cover attack — distinct from the
anchor-ridge baseline in :mod:`cover_break`. Method: a greedy,
per-position **vocabulary search**. At position ``n``, holding the first
``n-1`` already-recovered tokens fixed as the prefix, the attacker runs a
forward pass over every candidate length-``n`` sequence and keeps the
vocab token whose hidden state is closest (L1) to the observed target.
``O(V·N)`` forward passes, no training.

Why it sits apart from every other attack here: it is the **first attack
that must run the model**. The rest consume a :class:`CaptureSet` (pure
arrays) and never import the model stack. To preserve that invariant this
driver takes a ``forward_fn`` callback rather than importing a model —

    ``forward_fn(token_ids: (B, L) int64) -> (B, d) float32``

returning the hidden state at the **last position** of each candidate
sequence, at the attack's target ``(kind, layer)``, in the **same space
as the observed targets**. The caller (Part-2 defense-eval / capture)
binds this to the real plaintext *or* obfuscated Qwen3 on GPU; tests bind
a synthetic forward map on CPU.

Cover-agnosticism falls out for free: if the targets were captured from an
obfuscated model, the attacker's ``forward_fn`` runs the *same* obfuscated
weights, so candidate and observed hidden states live in the same covered
space and the L1 match needs no knowledge of the cover. (This holds for a
*deterministic* / static-weight cover such as the AloePri GGUF; a
per-forward random cover would break the shared space.)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ...capture.types import CaptureSet
from ...metrics import AttackResult, classify_risk_level, topk_recovery, ttrsr
from ...transforms import Identity, Transform

# (B, L) int64 token-ids -> (B, d) float32 hidden at each sequence's last
# position, at the target (kind, layer), in the observed targets' space.
ForwardFn = Callable[[np.ndarray], np.ndarray]


def vocab_match_prompt(
    target_hidden: np.ndarray,
    forward_fn: ForwardFn,
    candidate_ids: np.ndarray,
    *,
    prefix: list[int] | None = None,
    p_norm: int = 1,
    topk: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily recover one prompt's token ids from its observed hidden
    states ``target_hidden`` ``(N, d)``.

    Returns ``(recovered (N,), topk_ids (N, k))`` where ``recovered`` is
    the top-1 id per position (also used as the growing prefix) and
    ``topk_ids`` are the ``k`` nearest candidates per position (for
    top-k recovery reporting). ``k = min(topk, len(candidate_ids))``.
    """
    n = int(target_hidden.shape[0])
    cand = np.asarray(candidate_ids, dtype=np.int64)
    v = int(cand.shape[0])
    k = min(topk, v)
    ctx = list(prefix or [])
    recovered = np.empty(n, dtype=np.int64)
    topk_ids = np.empty((n, k), dtype=np.int64)
    for pos in range(n):
        seqs = np.empty((v, len(ctx) + 1), dtype=np.int64)
        if ctx:
            seqs[:, : len(ctx)] = np.asarray(ctx, dtype=np.int64)
        seqs[:, -1] = cand
        hs = np.asarray(forward_fn(seqs), dtype=np.float32)  # (V, d)
        dist = np.linalg.norm(hs - target_hidden[pos], ord=p_norm, axis=1)
        order = np.argpartition(dist, k - 1)[:k]
        order = order[np.argsort(dist[order])]  # k nearest, ascending
        topk_ids[pos] = cand[order]
        best = int(cand[int(order[0])])
        recovered[pos] = best
        ctx.append(best)
    return recovered, topk_ids


def _default_candidate_pool(
    capture: CaptureSet, pool_size: int, vocab_size: int | None, seed: int
) -> np.ndarray:
    """Candidate ids = the token ids that actually appear in the corpus
    (the plausible-vocab restriction every search attack needs to stay
    tractable), random-filled to ``pool_size`` from ``[0, vocab_size)`` if
    a vocab size is given. Capped at ``pool_size``.
    """
    seen = np.unique(
        np.concatenate([np.asarray(ids, dtype=np.int64) for ids in capture.prompt_token_ids])
        if capture.prompt_token_ids
        else np.zeros(0, dtype=np.int64)
    )
    if vocab_size and pool_size > seen.shape[0]:
        rng = np.random.default_rng(seed)
        extra = rng.choice(vocab_size, size=pool_size, replace=False)
        seen = np.unique(np.concatenate([seen, extra.astype(np.int64)]))
    if seen.shape[0] > pool_size:
        rng = np.random.default_rng(seed + 1)
        keep = rng.choice(seen.shape[0], size=pool_size, replace=False)
        seen = np.sort(seen[keep])
    return seen


def run(
    capture: CaptureSet,
    forward_fn: ForwardFn,
    *,
    layer: int,
    kind: str = "resid_post",
    transform: Transform | None = None,
    candidate_ids: np.ndarray | None = None,
    candidate_pool_size: int = 2048,
    vocab_size: int | None = None,
    max_positions: int | None = None,
    prefix: list[int] | None = None,
    p_norm: int = 1,
    topk: int = 10,
    seed: int = 20260618,
) -> AttackResult:
    """Run the vocab-matching attack over every prompt and report TTRSR.

    Targets are read from ``capture`` under ``transform`` (default
    :class:`Identity` — the capture is already in whatever space the model
    that produced it used, including an obfuscated one). ``forward_fn``
    must produce candidate hidden states in that *same* space.
    ``max_positions`` caps the recovered prefix length per prompt (cost is
    ``O(pool · positions)`` forward calls). ``candidate_ids`` overrides the
    default corpus-derived pool (pass the full vocab for the strict attack).
    """
    transform = transform or Identity()
    mats = capture.per_prompt_matrices(kind, layer, transform=transform)
    if not mats:
        return AttackResult(
            attack="vocab_match_inversion", transform=transform.name,
            model_id=capture.model_id, kind=kind, layer=layer,
            n_prompts=capture.n_prompts(), n_train=0, n_test=0,
            ttrsr_top1=None, ttrsr_top10=None, risk_level="unknown",
            primary_metric_value=None,
            extra={"note": "no operands for this (kind, layer)"},
        )
    pool = (
        np.asarray(candidate_ids, dtype=np.int64)
        if candidate_ids is not None
        else _default_candidate_pool(capture, candidate_pool_size, vocab_size, seed)
    )

    pred_all: list[np.ndarray] = []
    truth_all: list[np.ndarray] = []
    topk_all: list[np.ndarray] = []
    for pi, _h, u in mats:
        ids = np.asarray(capture.prompt_token_ids[pi], dtype=np.int64)
        n = min(u.shape[0], ids.shape[0])
        if max_positions is not None:
            n = min(n, max_positions)
        if n == 0:
            continue
        rec, tk = vocab_match_prompt(
            u[:n], forward_fn, pool, prefix=prefix, p_norm=p_norm, topk=topk
        )
        pred_all.append(rec)
        topk_all.append(tk)
        truth_all.append(ids[:n])

    if not pred_all:
        return AttackResult(
            attack="vocab_match_inversion", transform=transform.name,
            model_id=capture.model_id, kind=kind, layer=layer,
            n_prompts=capture.n_prompts(), n_train=0, n_test=0,
            ttrsr_top1=None, ttrsr_top10=None, risk_level="unknown",
            primary_metric_value=None,
            extra={"note": "no positions to recover"},
        )

    pred = np.concatenate(pred_all)
    truth = np.concatenate(truth_all)
    top1 = ttrsr(pred, truth)
    top10 = topk_recovery(np.concatenate(topk_all), truth)
    return AttackResult(
        attack="vocab_match_inversion", transform=transform.name,
        model_id=capture.model_id, kind=kind, layer=layer,
        n_prompts=capture.n_prompts(), n_train=0, n_test=int(truth.shape[0]),
        ttrsr_top1=float(top1), ttrsr_top10=float(top10),
        risk_level=classify_risk_level(float(top1)),
        primary_metric_value=float(top1),
        primary_metric_name="token_top1_recovery_rate",
        extra={
            "method": "hidden_no_more_vocab_match",
            "candidate_pool_size": int(pool.shape[0]),
            "p_norm": p_norm,
            "max_positions": max_positions,
        },
    )
