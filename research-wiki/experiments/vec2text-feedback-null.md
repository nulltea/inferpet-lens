---
type: experiment
node_id: exp:vec2text-feedback-null
title: "B7 — Vec2Text iterative-corrector feedback is null on the per-position residual stream (negative result)"
idea_id: "idea:info-efficient-attacks"
verdict: negative
confidence: high
date: "2026-06-22"
hardware: "AMD Radeon 8060S iGPU (gfx1151, ROCm container)"
duration: "~4.5 min (T=4 × 3 ε)"
provenance: "results/b7_vec2text.json; scripts/spikes/b7_vec2text_corrector.py"
added: 2026-06-23T00:00:00Z
tags: ["vec2text", "iterative-corrector", "negative-result", "per-position-resid", "l20", "dp", "surface-mismatch", "rep2text"]
---

# B7 — Vec2Text iterative feedback is null on the per-position residual stream

**verdict:** `negative` (first-class) · **confidence:** `high` · tests `idea:info-efficient-attacks`

## What was tried

Implemented the **full** Morris et al. 2023 Vec2Text loop that the one-shot B6c forward-model
([[exp:b6c-forward-model-vec2text]]) lacked: a T-step hypothesis-refinement corrector
`c(e, ê^(t), e−ê^(t), h^(t))` that re-embeds the current hypothesis each round and accepts iff cosine to
the observed embedding improves. Repo mapping: φ = clip-only forward to `resid_post @L20` (WEIGHTS-PUB),
e = Y_obs (DP-noised), corrector = per-position MLP on the 4-block input. No full-prefix teacher
forcing of test tokens (honest full-sequence recovery); the train-token context positions are held
fixed at their true values and acceptance/recovery is scored on **test positions only**. T=4,
ε∈{∞,512,256} @L20. Two disjoint token pools (train-only corrector, test-only eval) after a bug where
a shared pool let memorized train predictions crush test recovery.

## Metrics (token-recovery readout @L20)

```
   ε      σ │ ridge  base_dec  FMV_tf │ V2T-feedback(final)  V2T-nofeedback(final) │ Δ(fb−nf)
   ∞   0.00 │ 0.407   0.270    0.528  │      0.273                0.307            │  −0.034
 512   1.88 │ 0.374   0.302    0.358  │      0.307                0.347            │  −0.040
 256   3.77 │ 0.126   0.121    0.020  │      0.111                0.136            │  −0.025
```

Recorrelation vs capPVI: V2T-feedback Spearman +1.0 but that is degenerate (the feedback variant barely
moves); the informative number is the **feedback delta**.

## Result — feedback does NOT transfer (contradicts Morris Fig 3)

V2T-with-feedback ≤ V2T-without-feedback at **every** ε (Δ = −0.034 / −0.040 / −0.025). The iterative
hypothesis-embedding feedback that drives Morris's gains is **null/harmful** here. Greedy iteration
reaches a fixed point at t=1 (one-shot corrector → fixed point). Positive side note: the noise-aware
*no-feedback* corrector recovers the high-σ regime FMV loses (ε=256: V2T-nf 0.136 vs FMV 0.020 vs ridge
0.126), reproducing B6c regime-dependence — the high-σ lever is noise-aware refinement, not feedback.

## Why it fails (two independent causes; lit-supported, mechanism not yet ablation-proven here)

> The direct empirical finding is the feedback-null itself (feedback ≤ no-feedback at every ε). The
> bottleneck explanation below is supported by the cited literature and the surface geometry but is not
> yet confirmed by a controlled bottleneck ablation on *our* data — that ablation (compress the residual
> / vary dimensionality / strip logit-lens-readable components, check whether feedback starts helping)
> is the confirmatory hook.


1. **Surface mismatch.** Vec2Text inverts a single *pooled* bottleneck vector (whole sequence → one
   d-dim, severe under-determination → feedback resolves by search). The per-position residual stream
   gives each token its *own* observation — near-linearly readable (logit-lens); "LMs are Injective &
   Hence Invertible" (arXiv:2510.15511) shows clean hidden states are per-position invertible → **no
   bottleneck for feedback to resolve** → feedback is moot.
2. **Implementation is not Vec2Text.** Real Vec2Text = T5-base 220M autoregressive seq2seq, MSMARCO 8.8M
   docs, sequence beam — ours is a per-position MLP regressor (trains in seconds), no generation/beam →
   hollow loop, fixed point at t=1.

## Conclusion (supersedes / pivot)

Vec2Text is the **wrong attack** for the per-position residual surface — that is a probing/logit-lens
problem (matched attack = the base decoder / lens; under DP = a noise-aware decoder, which B7's
no-feedback corrector already is). Vec2Text belongs on a **single-vector** surface: the pooled
RAG/retrieval embedding (private-RAG motivating surface) or a single last-token rep (Rep2Text,
arXiv:2511.06571, which itself *dropped* the iterative corrector for an adapter+AR-decoder). The pivot
to the pooled-GTR surface is where Vec2Text succeeds and the matched spectral probe is validated —
see [[spectral-mi-probe-eval]] and [[claim:spectral-channel-mi-embedding-inversion]].

## Connections

- supersedes-attack-on → per-position resid surface (Vec2Text → logit-lens / noise-aware decoder)
- pivots-to → [[spectral-mi-probe-eval]] (pooled-GTR surface where Vec2Text + matched probe work)
- refines → [[exp:b6c-forward-model-vec2text]] (one-shot FMV; this adds the iterative loop and finds it null)
