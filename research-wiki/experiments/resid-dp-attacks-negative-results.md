---
type: experiment
node_id: exp:resid-dp-attacks-negative-results
title: "Negative results — residual-stream DP attacks (consolidated dead-ends)"
idea_id: "idea:info-efficient-attacks"
verdict: no
confidence: high
date: "2026-06-23"
hardware: "gemma-2-2b (prior GPU runs)"
duration: "consolidation only"
provenance: "results/b2_lpos_decoder.json; results/b6_strong_decoder.json; refine-logs/dp-stronger-attacks/EXPERIMENT_RESULTS.md (B7); .aris/traces/result-to-claim/2026-06-23_run01/"
added: 2026-06-23T15:38:41Z
tags: ["negative-result", "first-class", "resid-dp-attacks"]
---

# Negative results — residual-stream DP attacks (consolidated dead-ends)

**verdict:** `no`  ·  **confidence:** `high`  ·  tests `idea:info-efficient-attacks`

First-class negative results from the resid-dp-attacks consolidation. None is a failure of the
thesis — each **bounds** the headline claim [[restore-correlation]] and is recorded so the open
frontier is not re-attempted blindly.

## DE-1 — A plain MLP decoder does NOT beat ridge at depth under AT-LAYER noise
`results/b2_lpos_decoder.json`. A 250-epoch channel-aware MLP loses to ridge at every depth/level
(uplift-selectivity −0.01…−0.30, L5/12/20). **Why it fails:** at depth the clean embedding is not
directly observable; ridge's closed-form linear map already captures resid→embedding geometry, and a
vanilla MLP does not improve on it. **What supersedes it:** beating ridge at depth requires noise
*propagation* (not at-layer noise) AND a stronger decoder — demonstrated under propagated DP in
`exp:b6-strong-decoder`. **Constraint for future work:** do not re-test plain MLP-vs-ridge under
at-layer Gaussian noise; the regime has no gap (both track MI ρ=1.0).

## DE-2 — Embedding-space Vec2Text iteration is NULL (greedy plateau at t=1)
`results/b6_strong_decoder.json` (C7). `iter_T3 − iter_T1 = +0.000`; `iter − deep = +0.023`. A
one-shot-trained per-position corrector is a deterministic fixed function of Y → reaches its fixed
point after one application; multi-round embedding-space iteration adds nothing. Morris's gains come
from sequence-level **beam** + training on the iterative hypothesis distribution, neither implemented
(budget). **Constraint:** "iterative corrector" is not a free lever here — the lever is noise-aware
training, not re-embedding feedback.

## DE-3 — Vec2Text is the WRONG attack for the per-position resid surface (category error)
`refine-logs/dp-stronger-attacks/EXPERIMENT_RESULTS.md` §B7/B7-analysis. Faithful Vec2Text inverts a
**single pooled bottleneck** vector (mean-pooled sentence embedding); its iterative re-embed + residual
feedback exists to resolve per-token under-determination from that compression. The per-position
`resid_post` surface gives each token its OWN d-dim observation (near-linearly readable, logit-lens;
cf. injectivity arXiv 2510.15511) → **no bottleneck under-determination for feedback to resolve**, so
feedback is structurally moot (measured null/slightly negative). **Resolution:** Vec2Text moved to its
native pooled-embedding surface (Task 4 `embed-vec2text`); per-position resid is a probing/logit-lens
problem, attacked by the (noise-aware) per-position decoder above. **Constraint:** do not build a
Vec2Text iterative-feedback attack on per-position activations.

## What remains open (the live frontier, queued — not this phase)
No single attack dominates the full noise range (forward-model best at low noise +0.53; trained decoder
best at high noise). The named optimum is a **noise-aware FMV** (match E[Y|candidate] under the noise
model, not a clean reference) ⊕ a learned denoiser, with multi-seed dense-ε CIs. Tracked in
[[restore-correlation]] §Open.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
