---
type: experiment
node_id: exp:vma-fullsort-vs-rowsort
title: "Experiment vma-fullsort-vs-rowsort"
idea_id: "idea:info-efficient-attacks"
verdict: yes
confidence: high
date: "2026-06-23"
hardware: "CPU (model-free table algebra)"
duration: "<2 min"
provenance: "results/vma_stronger.json; results/aloepri_vma_sweep.json; scripts/spikes/vma_stronger.py; scripts/spikes/aloepri_vma_sweep.py; refine-logs/perm-cover/RESULTS_STANDARDIZED.md"
tags: ["perm-cover", "permutation", "VMA", "consolidated"]
added: 2026-06-21T13:42:42Z
updated: 2026-06-23
---

# Experiment vma-fullsort-vs-rowsort

**verdict:** `yes`  ·  **confidence:** `high`  ·  tests `idea:info-efficient-attacks` · supports `claim:perm-llr-threshold`

Consolidated 2026-06-23 (Task a5-perm-cover). Jury: result-to-claim PARTIAL/scoped
(`019ef646`), experiment-audit WARN/no-FAIL (`019ef64a`), proof-checker PASS (`019ef653`).

## Metrics (bits canonical + per-secret readout)
AloePri column-permutation embedding cover (gemma-2 family table, d=2304). Secret = row
correspondence Π; readout = VMA recovery rate; canonical bits = **independent CLUB-on-φ**.

- **Attack strength** (`vma_stronger.json`, N=1000, 3 seeds): full-sorted-row matcher beats RowSort-64
  by **+0.434 @α_e=0.2** (0.999 vs 0.565) and **+0.600 @α_e=0.35** (0.804 vs 0.204), while CLUB-on-φ
  moves only 245→235 bits (−4%). RowSort-64's 64-quantile binning is lossy; the full sorted row
  recovers the leakage the probe says is preserved.
- **Probe tracking** (`aloepri_vma_sweep.json`, N=1200, 1 seed): Spearman(**independent** CLUB-on-φ,
  VMA recovery) = **+0.976** across α_e; recorrelation in `vma_stronger.json` +1.00 (RowSort) / +0.99
  (full-sort). The keymat cover drives VMA → 0.0 (chance≈8.3e-4) **and** CLUB-on-φ → ≈0 (estimator
  floor) together — clean negative control.
- **Integrity correction applied:** retrieval-PVI (ρ=1.0) is the *attack-in-bits* (dependent), not an
  independent probe — the thesis rests on CLUB-on-φ only.

## Reasoning
Permutation-channel analog of the L0 Bayes-NN info-efficiency result. RowSort-64 (the VMA baseline)
collapses under small noise where the leakage has NOT disappeared; the full sorted row is the
maximal invariant under the column-permutation cover and recovers it. Confirms the thesis on the
permutation channel: weak-attack collapse = attack weakness, not leakage loss; the independent probe
stays faithful, both directions (sweep + keymat control). Theory: `claim:perm-llr-threshold`
(Lemma 1 DPI domination of RowSort-64 by the full sorted row; Lemma 2 per-row maximal-invariant
profile-MLE; shared-P scope Remark; 2 log n cited).

## Scope
Single embedding table; `aloepri_vma_sweep` single-seed (`vma_stronger` 3-seed). L2 exact-MAP
optimality is the per-row-permutation channel; the AloePri shared-P channel is a DPI-dominating
relaxation. Firm-ups: multi-seed `aloepri_vma_sweep`; second embedding table; a Gram-/cross-row-aware
matcher to test the shared-P gap.

## Connections
Supports `claim:perm-llr-threshold`. Negative companion `exp:cover-break-matched-deferred`.
_Edges recorded in `graph/edges.jsonl`._
