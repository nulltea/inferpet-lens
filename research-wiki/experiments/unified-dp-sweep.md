---
type: experiment
node_id: exp:unified-dp-sweep
title: "Unified DP ε-sweep: ridge × BNN × FMV at L0+L20, 3 probes, ε∈{∞,1024,512,256,64}"
idea_id: "idea:info-efficient-attacks"
verdict: partial
confidence: high
date: "2026-06-22"
hardware: "AMD Radeon 8060S iGPU (gfx1151, ROCm container)"
duration: "~3.5 hours"
provenance: "results/unified_dp_sweep.json; scripts/spikes/unified_dp_sweep.py"
added: 2026-06-22T00:00:00Z
tags: ["bnn", "fmv", "ridge", "club", "pvi", "mdl", "l0", "l20", "dp-sweep", "c-runtime-fix"]
---

# Unified DP ε-sweep: ridge × BNN × FMV at L0+L20, 3 probes, ε∈{∞,1024,512,256,64}

**verdict:** `partial`  ·  **confidence:** `high`  ·  tests `idea:info-efficient-attacks`

## Setup
gemma-2-2b, 256 prompts (corpora/release-gate-512.txt), vocab-disjoint split, pool=2048, seed=20260622.
C_raw=4.147 (embedding table norms, for L0); C_runtime=198.952 (runtime output norms, for L20 hook).
δ=1e-5, z_dp=4.845. Attacks: ridge (L0+L20, alpha-sweep / fixed alpha=1.0), BNN (L0 only, exact MAP),
FMV (L20 only, MLP decoder top-k=16 + clip-only forward rerank, max_pos=300).
Probes: CLUB, CapPVI (pca_softmax dim=64), MDL-SDL — all with shuffle control (selectivity=real−floor).

## Critical Bug Fixed: C_runtime vs C_raw
**Gemma-2 scales embeddings by √d=√2304≈48 at runtime** (hidden_states multiplied by sqrt(hidden_size)
before layer 0). Prior runs computed C from the raw embedding table (C_raw≈4.147) and applied it to the
L20 hook. This clipped all runtime embeddings (norm≈199) down to 4.147 — a 48× reduction — completely
destroying the L20 signal. Fix: calibrate C_runtime from actual forward-pass output norms. ridge@L20@ε=∞
recovered from 0.080 → 0.528 after fix.

## Results Table

```
     ε    r │ ridge@L0   BNN@L0  ↑BNN │ ridge@L20 FMV@L20  ↑FMV │ CLUB@L0 PVI@L0 MDL@L0 │ CLUB@L20 PVI@L20 MDL@L20
     ∞ 0.00 │    1.000    0.994 -0.006│     0.528   0.607 +0.078 │    3833  0.935  15665  │     3478   0.743   26213
  1024 0.23 │    0.950    1.000 +0.050│     0.483   0.577 +0.094 │    3752  0.940   1365  │     3333   0.766    3592
   512 0.45 │    0.874    1.000 +0.126│     0.473   0.447 -0.027 │    3614  0.919   3759  │     3456   0.816    2251
   256 0.91 │    0.489    1.000 +0.511│     0.074   0.010 -0.064 │    3086  0.758   4219  │     2435   0.583    2441
    64 3.63 │    0.038    0.969 +0.931│     0.017   0.013 -0.003 │    1312  0.048   2426  │     1142   0.040    2770

Spearman ρ(attack TTRSR, probe selectivity):
               CLUB@L0  PVI@L0  MDL@L0  CLUB@L20  PVI@L20  MDL@L20
  ridge@L0      +1.00   +0.90   +0.30    +0.90    +0.60    +0.60
  BNN@L0        +0.22   +0.45   -0.11    +0.22    +0.67    -0.45
  ridge@L20     +1.00   +0.90   +0.30    +0.90    +0.60    +0.60
  FMV@L20       +0.90   +0.80   +0.10    +0.80    +0.50    +0.70
```

## Key Findings

**F1 — BNN proof of concept at high noise (C1 replicated at scale):**
BNN stays 1.000 at ε=256,512,1024 and 0.969 at ε=64 (r=3.63). PVI=0.048 at r=3.63 is NOT MI absence
— it reflects the incapacity of a linear PCA-64 classifier. BNN's 9.35σ decision margin (see
claim:bnn-nns-high-d-geometry) explains near-perfect recovery despite seemingly high noise ratio.
BNN ρ(BNN, CLUB@L0)=+0.22 — BNN extracts information that the probes say is dwindling.

**F2 — L20 noise cliff (r=0.45 → r=0.91):**
ridge@L20 drops 0.473 → 0.074 (6×) between ε=512 and ε=256.
FMV@L20 drops 0.447 → 0.010 (45×) over the same step.
But CLUB@L20 only drops 3456 → 2435 (30%) and PVI@L20 stays at 0.583.
Attack-probe decorrelation at L20 is dramatic: MI is preserved but linear/FMV attacks can't extract it.
This is the propagated-DP nonlinear amplification effect. The cliff makes ε=512 (r=0.45) the
practical boundary for L20 attack effectiveness under these conditions.

**F3 — FMV vs ridge regime dependence:**
FMV beats ridge only at low noise (ε=∞,1024: +0.078,+0.094). At ε≥512, ridge wins.
The MLP decoder trained on clean data generates bad top-k seeds at moderate noise (r≥0.45),
and FMV's clean-reference matching is swamped. Matches B6c findings (partial verdict: FMV
closes low-noise gap but is noise-fragile).

**F4 — PVI@L20 non-monotone:**
PVI@L20: 0.743 → 0.766 → 0.816 → 0.583 → 0.040. Higher at ε=512 than ε=∞.
Likely a capacity-matching artifact (pca_softmax dim=64 captures a cleaner signal subspace
in the moderately noisy representation than in the clean one). Flag as probe artefact; not
interpretable as "more MI at ε=512 than clean."

**F5 — MDL high variance persists:**
MDL@L0@ε=∞=15665, @ε=1024=1365, @ε=512=3759 — highly non-monotone. Known overfit instability.
MDL should be reported as auxiliary only; CLUB/PVI are primary.

**F6 — Spearman ρ no longer trivially +1.00:**
Under correct C_runtime, the ε sweep no longer forces trivial +1.00 confound at L20.
ridge@L20 has ρ=+0.90 vs CLUB@L20 (not +1.00) because the cliff at ε=256 breaks monotone rank.
FMV@L20 ρ=+0.80 vs CLUB@L20. BNN@L0 ρ=+0.22 (consistent with prior findings — the channel-
specific non-monotone nature of BNN's advantage is real, not confounded by noise axis).

## Limitations / Open Questions
- FMV evaluated on 300 sampled test positions (max_pos=300) vs ridge on all 1437 — sampling noise in FMV numbers, especially at small TTRSR.
- Single seed; no confidence intervals on ρ.
- "Noise-aware FMV" (denoise Y_obs before forward-model match) not implemented; expected to close the ε=256+ gap for FMV.
- BNN@L20 (exhaustive NNS with forward-model references, topk=pool_size=2048) not run — would give the true MAP upper bound at L20 but costs ~3M forward passes.
- PVI non-monotone at L20 (F4) not fully explained; may need varying dim or family.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

Supports [[claim:bnn-nns-high-d-geometry]], [[claim:restore-correlation]].
Extends exp:b2-l0-bayes-vs-ridge (denser ε, adds MDL, FMV@L20), exp:b6c-forward-model-vec2text (adds BNN, all probes, ε=256/64).
