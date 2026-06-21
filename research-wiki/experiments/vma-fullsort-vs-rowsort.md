---
type: experiment
node_id: exp:vma-fullsort-vs-rowsort
title: "Experiment vma-fullsort-vs-rowsort"
idea_id: "idea:info-efficient-attacks"
verdict: yes
confidence: high
date: ""
hardware: ""
duration: ""
provenance: "results/vma_stronger.json; scripts/spikes/vma_stronger.py"
added: 2026-06-21T13:42:42Z
tags: []
---

# Experiment vma-fullsort-vs-rowsort

**verdict:** `yes`  ·  **confidence:** `high`  ·  tests `idea:info-efficient-attacks`

## Metrics
AloePri perm-core, gemma embed N=1000, 3 seeds: full-sorted-row matcher beats RowSort-64 by +0.43 @α_e=0.2 (τ 0.999 vs 0.565), +0.60 @0.35. CLUB-on-φ barely moves (245→235) → info preserved, RowSort's 64-quantile binning is lossy. Both track CLUB ρ≈1.0; at fixed small noise RowSort under-reports leakage.

## Reasoning
Permutation-channel analog of the L0 Bayes result. RowSort-64 (VMA baseline) is information-inefficient: 64-quantile binning discards recoverable info, collapses under small noise. The FULL sorted row is the sufficient statistic for the column-perm+Gaussian channel and recovers the leakage CLUB-φ said was preserved. Confirms thesis on BOTH channels: weak-attack collapse = attack weakness, not leakage loss; probes faithful.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

