---
type: experiment
node_id: exp:b6-strong-decoder
title: "Experiment b6-strong-decoder"
idea_id: "idea:info-efficient-attacks"
verdict: partial
confidence: high
date: ""
hardware: ""
duration: ""
provenance: "results/b6_strong_decoder.json; scripts/spikes/b6_strong_decoder.py"
added: 2026-06-21T13:57:52Z
tags: []
---

# Experiment b6-strong-decoder

**verdict:** `partial`  ·  **confidence:** `high`  ·  tests `idea:info-efficient-attacks`

## Metrics
Propagated-DP L20, 6 ε: trained decoder selectivity RE-CORRELATES with MI (deep+0.83, iter+0.71 vs capPVI) where ridge ANTI-correlates (−0.09). Uplift crossover: decoder>ridge at high noise (ε≤384, Δ+0.07), ridge>decoder clean. Iteration null: iterT3−T1=+0.00, iter−deep=+0.02 (embedding-space iteration adds nothing; faithful Vec2Text needs forward-model-in-loop).

## Reasoning
Vec2Text-style iterative corrector + deep capacity control vs ridge/MLP under propagated input-DP. HEADLINE (C6): the stronger trained decoder restores the MI↔recovery correlation ridge breaks (+0.83 vs −0.09) — confirms the objective. C5 uplift is a noise crossover (decoder wins high-noise). C7: pure embedding-space iteration = fixed decoder (T3=T1), capacity doesn't help → faithful Vec2Text requires re-embedding through the model (forward-model-in-loop), the open frontier exceeding 10-20min budget.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

