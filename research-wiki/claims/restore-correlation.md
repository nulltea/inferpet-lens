---
type: claim
node_id: claim:restore-correlation
name: "Info-efficient attacks restore MI-recovery correlation (hypothesis to test)"
description: ""
node_type: claim
status: drafted
provenance: ""
tags: ["hypothesis", "headline"]
date: 2026-06-21
added: 2026-06-21T11:48:34Z
---

# Info-efficient attacks restore MI-recovery correlation (hypothesis to test)

**status:** `drafted`

## Statement
Replacing ridge with denoise-then-invert MMSE/MAP (a la BeamClean: 77% vs ridge 17% @eps~9.6) and RowSort with Gaussian-LLR Hungarian makes attack-recovery climb toward the MI ceiling and re-correlate with CLUB/PVI across the defence knob, where the weak attacks decorrelate.

## Honest scope
Empirical claim to be tested on our DP and AloePri sweeps; external empirical support from BeamClean, no MI-tracking proof exists yet.

## Evidence chain
_TODO: proof obligations, jury verdicts, provenance pointers._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

