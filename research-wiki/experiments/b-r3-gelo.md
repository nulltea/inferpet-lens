---
type: experiment
node_id: exp:b-r3-gelo
title: "Experiment b-r3-gelo"
idea_id: ""
verdict: partial
confidence: medium
date: "2026-06-24"
hardware: ""
duration: ""
provenance: "refine-logs/resid-gelo/"
added: 2026-06-24T06:36:01Z
tags: []
---

# Experiment b-r3-gelo

**verdict:** `partial`  ·  **confidence:** `medium`

## Metrics
feat-Gram relerr 2.5e-16@κ=1→1780@κ=100; JADE p95~0.70 vs floor~0.66; ridge 0.288<floor 0.667; Spearman 0.507/0.293

## Reasoning
C0 verified, C1 partial→yes, C2 no (ρ<0.6); GELO row-mix defeats per-row BSS+ridge, feature-Gram is load-bearing leak

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

