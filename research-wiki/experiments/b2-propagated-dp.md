---
type: experiment
node_id: exp:b2-propagated-dp
title: "Experiment b2-propagated-dp"
idea_id: "idea:info-efficient-attacks"
verdict: partial
confidence: medium
date: ""
hardware: ""
duration: ""
provenance: "results/b2_propagated_dp.json; scripts/spikes/b2_propagated_dp.py"
added: 2026-06-21T13:34:36Z
tags: []
---

# Experiment b2-propagated-dp

**verdict:** `partial`  ·  **confidence:** `medium`  ·  tests `idea:info-efficient-attacks`

## Metrics
Propagated input-DP L20: decoder uplift-sel grows with noise −0.07→+0.14 (beats ridge as ε falls); re-correlation decSel↔capPVI 0.80(L12)/0.40(L20) vs ridge 0.40/0.20. Reverses at-layer-noise negative — decoder advantage is propagation-specific.

## Reasoning
Under PROPAGATED input-DP (noise at embedding, reshaped through depth) ridge's linear obs->emb map breaks AND decorrelates from MI (B3 L20); a channel-aware MLP trained on propagated-noised resid increasingly beats ridge as noise grows and tracks capPVI better. Confirms the thesis in the open regime: probes faithful, recovery gap is attack-strength, even at depth. Suggestive (4 eps,1 seed); needs denser sweep + stronger decoder (iterative/MAP) to firm.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

