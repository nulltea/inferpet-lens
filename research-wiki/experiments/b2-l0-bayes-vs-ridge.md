---
type: experiment
node_id: exp:b2-l0-bayes-vs-ridge
title: "Experiment b2-l0-bayes-vs-ridge"
idea_id: "idea:info-efficient-attacks"
verdict: yes
confidence: high
date: ""
hardware: ""
duration: ""
provenance: "results/l0_fast.txt + results/b2_l0_bayes.json; scripts/spikes/b2_l0_bayes_attack.py"
added: 2026-06-21T12:52:35Z
tags: []
---

# Experiment b2-l0-bayes-vs-ridge

**verdict:** `yes`  ·  **confidence:** `high`  ·  tests `idea:info-efficient-attacks`

## Metrics
uplift +0.98 @ε128 (r=1.82): Bayes-NN TTRSR 1.000 vs ridge 0.020; clean both 1.000; CLUB 3084→1912b, capacity-PVI 0.98→0.74 (slow decay)

## Reasoning
L0 exact-Bayes (closed-form NN-to-known-table) vs ridge under input-DP, GPU-free gemma-2-2b N=7000 vocab-disjoint pool=2048. Confirms C1 (uplift grows with noise, +0.98) and C2 (strong recoverers Bayes-NN + capacity-PVI track the MI-preserved info while ridge decorrelates/crashes 50x). In d=2304 isotropic DP noise is ~orthogonal to inter-embedding directions so NN-to-table is geometrically noise-robust — the info is preserved, ridge just can't use it. HONEST LIMIT: L0 is the easiest layer (obs≈noised embedding); L>0 needs a learned channel-aware denoiser. Bug fixed: pool sort-truncation dropped large true ids (clean 0.616→1.000).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

