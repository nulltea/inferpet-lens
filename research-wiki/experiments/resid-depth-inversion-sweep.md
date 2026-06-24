---
type: experiment
node_id: exp:resid-depth-inversion-sweep
title: "Depth hidden-state inversion sweep (Qwen3-4B resid_post, L0..L32 every-4, nn/ridge/mlp2 + cap-PVI + CLUB)"
idea_id: ""
verdict: partial
confidence: medium
date: "2026-06-24"
hardware: "ROCm gfx1151 iGPU"
duration: "205s"
provenance: "refine-logs/resid-depth-inversion/runs/full/depth_sweep.json"
added: 2026-06-24T05:19:13Z
tags: ["resid-depth-inversion", "inversion", "depth", "measurement-loop"]
---

# Depth hidden-state inversion sweep (Qwen3-4B resid_post, L0..L32 every-4, nn/ridge/mlp2 + cap-PVI + CLUB)

**verdict:** `partial`  ·  **confidence:** `medium`

## Metrics
best-inverter sel 0.39-0.69 all depths; nn floor 0.000; L32 mlp2 0.542 vs ridge 0.390 (disjoint CIs); Spearman(cap,rec)+0.85 CLUB+0.78

## Reasoning
C1 depth-irreversibility falsified; learned>linear at deepest layer; C2 positive probe-attack tracking across depth

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

