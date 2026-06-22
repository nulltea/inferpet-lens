---
type: experiment
node_id: exp:bnn-error-bounds-validation
title: "Geometry-only error bounds (union-Bhattacharyya ⊕ Fano) vs BNN@L0 on gemma-2-2b"
idea_id: "idea:info-efficient-attacks"
verdict: yes
confidence: high
date: "2026-06-22"
hardware: "AMD Radeon 8060S iGPU (gfx1151, ROCm container)"
duration: "~2 min (probe ⊥ test-set size n)"
provenance: "results/bnn_error_bounds_validation.json; results/bnn_error_bounds_validation_dense.json; scripts/spikes/bnn_error_bounds_validation.py; src/talens/measures/channel_error_bounds.py"
added: 2026-06-22T00:00:00Z
tags: ["bnn", "union-bhattacharyya", "fano-equivocation", "dp-l0", "matched-probe", "bracketing", "geometry-only"]
---

# Geometry-only error bounds (union-Bhattacharyya ⊕ Fano) vs BNN@L0 on gemma-2-2b

**verdict:** `yes` · **confidence:** `high` · tests [[claim:bnn-error-bounds-bhattacharyya-fano]]

## Objective
A *formal MI probe that correlates with embedding-distance-geometry attacks (BNN@L0)* and is
**independent of the attack**. Validates the Codex-verified proof (`refine-logs/PROOF_PACKAGE.md`).

## Setup
gemma-2-2b embedding table (vocab=256000, d=2304), C_raw=4.147, z_dp=4.845, pool=2048 (515
vocab-disjoint test tokens + fillers), seed=20260622. Channel V~Unif(pool), Y=clip(e_V,C_raw)+N(0,σ²I),
σ=C_raw·z_dp/ε. BNN = nearest-neighbour (uniform-prior MAP). Bounds from `channel_error_bounds.py`
(union_bhattacharyya M-free; fano_equivocation M=64 fresh-noise draws/codeword). Run in ROCm container.

## Result (dense grid ε∈{128…16}, the varying-BNN regime)
```
 eps     r |  BNN_err |  P_e^lb  P_e^ub | in? | H(V|Y)
 128  1.82 |   0.0001 |  0.0000  0.0001 |  Y  | 0.00
  96  2.42 |   0.0006 |  0.0000  0.0007 |  Y  | 0.00
  80  2.91 |   0.0058 |  0.0000  0.0125 |  Y  | 0.03
  64  3.63 |   0.0628 |  0.0000  0.3460 |  Y  | 0.35
  56  4.15 |   0.1638 |  0.0000  1.0000 |  Y  | 0.96
  48  4.84 |   0.3409 |  0.1052  1.0000 |  Y  | 2.16
  40  5.81 |   0.5676 |  0.2717  1.0000 |  Y  | 3.99
  32  7.27 |   0.7781 |  0.4689  1.0000 |  Y  | 6.16
  24  9.69 |   0.9142 |  0.6531  1.0000 |  Y  | 8.18
  16 14.53 |   0.9765 |  0.7939  1.0000 |  Y  | 9.73
```
**Bracketing 10/10 ε.  ρ(P_e^ub, BNN_err)=+0.888.  ρ(P_e^lb, BNN_err)=+0.937.**
(Coarse grid ε∈{∞,1024,512,256,64}: 5/5 bracketed, but BNN~0 until r=3.63 → degenerate range.)

## Findings
- **C1 bracketing — CONFIRMED (10/10).** Measured BNN error always in [P_e^lb, P_e^ub].
- **C4 correlation — CONFIRMED, complementary regimes.** The union-Bhattacharyya upper bound is **tight at
  low noise** (ε=80: 0.0058 vs 0.0125; ε=96: 0.0006 vs 0.0007) then saturates; the **Fano-equivocation
  lower bound tracks at high noise** (ε≤48: BNN 0.34→0.98 vs lb 0.11→0.79). The two-sided bound brackets
  throughout; the "which side is informative" crossover at r≈3.6 is a SNR-regime diagnostic.
- **C2 morphological floor — CONFIRMED.** Top union-bound pairs at ε=16 are case/space/number neighbours:
  `'Hardware'~' Hardware'`, `' pasta'~' Pasta'`, `' six'~' seven'`, `' differential'~'Differential'`,
  `'3'~'5'`, `' global'~' Global'` — exactly the tokens BNN confuses, from geometry alone.
- **C3 independence — CONFIRMED.** Probe takes no observation argument; depends only on (codebook, σ, RNG).

## Correlation caveat
ρ over an ε-sweep is partly a **monotonicity confound** (BNN error and both bounds are provably monotone
in σ, proof T5 — cf. [[claim:bayes-gap-diagnosis]]). Load-bearing evidence is the **bracketing** (value
containment, 10/10), the **upper-bound tightness at low noise** (ε=80: 0.0058 vs 0.0125), and the **C2
morphological-floor attribution** (predicts *which* tokens confuse — no σ-confound). ρ is supportive only.

## Limitations
Mid-transition (r≈3.6) bound gap widest (BNN 0.063 ∈ [0, 0.346]); α-information-Fano / min-distance
union refinement deferred. Uniform-prior headline scope (empirical-prior Remark-N variant not run). L0 only.

## Connections
Supports [[claim:bnn-error-bounds-bhattacharyya-fano]], brackets [[claim:bnn-nns-high-d-geometry]].
Edges in `graph/edges.jsonl`.
