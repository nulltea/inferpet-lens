---
type: experiment
node_id: exp:b6c-forward-model-vec2text
title: "Experiment b6c-forward-model-vec2text"
idea_id: "idea:info-efficient-attacks"
verdict: partial
confidence: high
date: ""
hardware: ""
duration: ""
provenance: "results/b6c_forward_model.json; scripts/spikes/b6c_forward_model_vec2text.py"
added: 2026-06-21T14:19:11Z
tags: []
---

# Experiment b6c-forward-model-vec2text

**verdict:** `partial`  ·  **confidence:** `high`  ·  tests `idea:info-efficient-attacks`

## Metrics
Forward-model-in-loop Vec2Text @L20: FMV recovers 0.738 at clean vs ridge 0.212/dec 0.380 (+0.53/+0.36) — closes the low-noise gap. But noise-fragile: collapses to 0.025 at ε=256 (matches clean forward to single noisy draw → swamped). Mirror image of decoder (wins high-noise). No single attack dominates noise range.

## Reasoning
Built the faithful forward-model attack the embedding-space corrector (B6) could not be: re-embed decoder-seeded top-k candidates through the actual model (clip-only), match to observed noised resid. Dominant at low noise (the model-in-loop extracts what ridge/decoder miss, +0.53) confirming the thesis's strongest form; noise-fragile at high noise (clean-reference vs single noisy draw). Optimal attack is regime-dependent; the named next step is a NOISE-AWARE FMV (denoise Y or match E[Y|cand]) combining FMV low-noise power + decoder high-noise robustness.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

