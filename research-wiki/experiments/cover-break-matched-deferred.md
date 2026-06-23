---
type: experiment
node_id: exp:cover-break-matched-deferred
title: "Negative/deferred: matched anchor-ICA cover-break against a non-identity cover"
idea_id: "idea:info-efficient-attacks"
verdict: no
confidence: high
date: "2026-06-23"
hardware: "n/a (not run)"
duration: "n/a"
provenance: "src/talens/attacks/cover_break.py; results/fullcheck-L0-10.json; refine-logs/perm-cover/RESULTS_STANDARDIZED.md (R3)"
tags: ["perm-cover", "cover_break", "negative-result", "deferred"]
added: 2026-06-23T00:00:00Z
---

# Negative/deferred: matched anchor-ICA cover-break against a non-identity cover

**verdict:** `no` (deferred — first-class negative result, recorded, not a manufactured claim)

## What was tried / what exists
`cover_break` (anchor-ridge, `src/talens/attacks/cover_break.py`) reconstructs non-anchor rows from K
known anchor pairs `(U[i], H[i])` and reports p95 |cosine| to the true row. The **executed** result
is the *plaintext baseline* under `Identity` (`U=H`): on **Qwen3-4B residual-stream activations**
(`results/fullcheck-L0-10.json`, K∈{1,4,16}) p95-cosine = 0.917 @L0 decaying to ~0.78 @L8 — residual
rows are linearly reconstructable from a handful of anchors at plaintext.

## Why this is not a claim
1. **Surface mismatch.** The plaintext baseline is on Qwen3-4B *activations*, a different surface
   from the gemma embedding-table permutation channel of `exp:vma-fullsort-vs-rowsort`. It situates
   the anchor-recovery threat; it is not an AloePri-cover result.
2. **The matched attack is unimplemented.** The *matched* cover-break — anchor recovery against a
   genuine non-identity orthogonal/permutation cover — is the `fastica_anchor` variant, which
   **raises `NotImplementedError`** (`cover_break.py:81`); the faithful port from
   `attack_drivers/run_anchor_ica.py` is pending a real cover Transform. On the permutation cover the
   relevant executed attack is VMA (see `exp:vma-fullsort-vs-rowsort`), not ridge-anchor cover_break.

## What supersedes / next
Implement `fastica_anchor` (FastICA row-unmixing + Hungarian anchor assignment) and run it against an
injected non-identity cover Transform, graded vs Identity and chance. Until then the matched
cover-break is an open gap, deliberately not claimed. Integrity audit `refine-logs/perm-cover/
EXPERIMENT_AUDIT.md` (Codex `019ef64a`) flagged the deferred branch as disclosed dead code — WARN,
not FAIL.

## Connections
Negative companion of `exp:vma-fullsort-vs-rowsort`. _Edges in `graph/edges.jsonl`._
