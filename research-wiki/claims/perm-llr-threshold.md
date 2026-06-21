---
type: claim
node_id: claim:perm-llr-threshold
name: "Full-row Gaussian-LLR matcher attains the MI alignment threshold; RowSort is DPI-dominated"
description: ""
node_type: claim
status: drafted
provenance: ""
tags: ["backbone", "provable", "permutation"]
date: 2026-06-21
added: 2026-06-21T11:48:34Z
---

# Full-row Gaussian-LLR matcher attains the MI alignment threshold; RowSort is DPI-dominated

**status:** `drafted`

## Statement
Noise-aware Mahalanobis/Gaussian-LLR + Hungarian is the MAP permutation matcher (sufficient statistic), with exact-recovery threshold stated in MI: I_XY >= 2 log n (Dai-Cullina-Kiyavash). RowSort sorted-quantile phi is a lossy MLE-relaxation, dominated by DPI, so it cannot reach the threshold the LLR matcher attains.

## Honest scope
Shared-width (embedding) surface; internal d+2h surfaces need robust/partial GW. Jointly-Gaussian feature model for the sharp 2 log n constant.

## Evidence chain
_TODO: proof obligations, jury verdicts, provenance pointers._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

