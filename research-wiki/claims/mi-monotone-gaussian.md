---
type: claim
node_id: claim:mi-monotone-gaussian
name: "Optimal attack recovery is monotone in MI (Gaussian arm)"
description: ""
node_type: claim
status: drafted
provenance: ""
tags: ["backbone", "provable", "gaussian-only"]
date: 2026-06-21
added: 2026-06-21T11:48:34Z
---

# Optimal attack recovery is monotone in MI (Gaussian arm)

**status:** `drafted`

## Statement
Along the Gaussian SNR path, I(S;Y) and -MMSE(S|Y) are both monotone in snr (I-MMSE: dI/dsnr=1/2 mmse), so the Bayes-optimal attack's recovery is a monotone function of MI by construction; weak attacks carry a non-monotone approximation-gap term and need not track MI.

## Honest scope
GAUSSIAN/DP ONLY. Laplace/Shredder uses the degradation-order DPI argument (monotone but no 1/2-MMSE identity). Finite variance required.

## Evidence chain
_TODO: proof obligations, jury verdicts, provenance pointers._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

