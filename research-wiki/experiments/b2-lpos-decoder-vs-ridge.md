---
type: experiment
node_id: exp:b2-lpos-decoder-vs-ridge
title: "Experiment b2-lpos-decoder-vs-ridge"
idea_id: "idea:info-efficient-attacks"
verdict: no
confidence: high
date: ""
hardware: ""
duration: ""
provenance: "results/b2_lpos_decoder.json + results/b2_lpos_run.log; scripts/spikes/b2_lpos_decoder.py"
added: 2026-06-21T13:22:57Z
tags: []
---

# Experiment b2-lpos-decoder-vs-ridge

**verdict:** `no`  ·  **confidence:** `high`  ·  tests `idea:info-efficient-attacks`

## Metrics
MLP decoder LOSES to ridge at all depths (uplift-selectivity −0.01..−0.30, L5/12/20). Shuffle floor≈chance (sel≈recovery). At-layer Gaussian noise: BOTH ridge & decoder selectivity ↔ capPVI/CLUB Spearman=1.00 (no decorrelation).

## Reasoning
L>0 channel-aware MLP decoder vs ridge under at-layer Gaussian noise, cached gemma-2-2b resid L5/12/20, vocab-disjoint + shuffle control, WEIGHTS-PUB. NEGATIVE for the MLP: the L0 uplift is observation-layer-specific (clean embedding directly observable); at depth ridge's closed-form linear map already captures resid->embedding and a 250-epoch MLP doesn't beat it — beating ridge at depth needs iterative/Vec2Text or MAP+LM-prior. KEY NUANCE: under at-layer noise even ridge tracks MI perfectly (rho=1.0); the B3 ridge-decorrelation was specific to input-DP noise PROPAGATION through depth, not generic. So probes are faithful except under propagation geometry, where attack strength limits.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

