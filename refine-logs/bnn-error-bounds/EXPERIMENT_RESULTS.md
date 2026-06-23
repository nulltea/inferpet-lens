---
type: dev-log
status: current
created: 2026-06-22
updated: 2026-06-22
tags: [experiment-results, bhattacharyya-fano-bounds, bnn, dp-l0, validation]
companion: [EXPERIMENT_PLAN, PROOF_PACKAGE]
---

# Experiment Results — geometry-only error bounds vs BNN@L0

**Date**: 2026-06-22
**Plan**: `refine-logs/EXPERIMENT_PLAN.md` · **Proof**: `refine-logs/PROOF_PACKAGE.md` (PASS)
**Objective**: a *formal MI probe that correlates with embedding-distance-geometry attacks (BNN@L0)*.

## Verdict: OBJECTIVE MET

The **Fano-equivocation lower bound** — literally `H(V|Y) = log K − I(V;Y)`, a formal MI
quantity — correlates with the BNN@L0 attack error at **ρ = +0.937** across a 10-point ε-sweep,
**computed independently of the attack** (codebook geometry + fresh synthetic noise only). The
**union-Bhattacharyya upper bound** correlates at **ρ = +0.888**. Together the two-sided bound
**brackets the measured BNN error at 10/10 ε**.

## Setup
gemma-2-2b embedding table (vocab=256000, d=2304), C_raw=4.147, z_dp=4.845, pool=2048
(515 true vocab-disjoint test tokens + fillers), seed=20260622. Channel: V~Unif(pool),
Y=clip(e_V,C_raw)+N(0,σ²I), σ=C_raw·z_dp/ε. BNN = nearest-neighbour (uniform-prior MAP) decode.
Bounds: `union_bhattacharyya` (M-free) + `fano_equivocation` (M=64 fresh-noise draws/codeword).
Implementation `src/talens/measures/channel_error_bounds.py`; validation
`scripts/spikes/bnn_error_bounds_validation.py` (ROCm container, ~1 min, ⊥ test-set size).

## M0 — sanity (CPU, host .venv): PASSED
`tests/test_channel_error_bounds.py` 8/8: σ→0 limits, orthonormal closed-form, exact-Q≤Bhattacharyya,
Ĥ_M unbiased vs brute-force H(V|Y), independence (no observation arg), σ-monotonicity, end-to-end
bracketing on a synthetic codebook. Codex code review: no CRITICAL; MAJOR fix applied (use the
certified LCB on H(V|Y) for the bracketing pass/fail, not the raw point estimate).

## M1 — coarse sweep ε∈{∞,1024,512,256,64} (results/bnn_error_bounds_validation.json)
```
 eps     r   sigma |  BNN_err     ± |  P_e^lb  P_e^ub  P_e^ubB | in? | H(V|Y)
 inf  0.00  0.0000 |   0.0000 .0000 |  0.0000  0.0000   0.0000 |  Y  | 0.00
1024  0.23  0.0196 |   0.0000 .0038 |  0.0000  0.0000   0.0000 |  Y  | 0.00
 512  0.45  0.0392 |   0.0000 .0038 |  0.0000  0.0000   0.0000 |  Y  | 0.00
 256  0.91  0.0785 |   0.0000 .0038 |  0.0000  0.0000   0.0000 |  Y  | 0.00
  64  3.63  0.3139 |   0.0628 .0038 |  0.0000  0.3460   1.0000 |  Y  | 0.35
```
Bracketing 5/5. But BNN@L0 is ~0 error until r≈3.63 (full-d near-orthogonal embeddings are highly
separable — confirms `claim:bnn-nns-high-d-geometry`), so the dynamic range is degenerate and the
default ε-grid is uninformative for the *correlation* question. → dense low-ε grid.

## M1-dense — ε∈{128…16} through the transition (results/bnn_error_bounds_validation_dense.json)
```
 eps     r   sigma |  BNN_err     ± |  P_e^lb  P_e^ub  P_e^ubB | in? | H(V|Y)
 128  1.82  0.1570 |   0.0001 .0038 |  0.0000  0.0001   0.0003 |  Y  | 0.00
  96  2.42  0.2093 |   0.0006 .0038 |  0.0000  0.0007   0.0030 |  Y  | 0.00
  80  2.91  0.2511 |   0.0058 .0038 |  0.0000  0.0125   0.0653 |  Y  | 0.03
  64  3.63  0.3139 |   0.0628 .0038 |  0.0000  0.3460   1.0000 |  Y  | 0.35
  56  4.15  0.3588 |   0.1638 .0038 |  0.0000  1.0000   1.0000 |  Y  | 0.96
  48  4.84  0.4186 |   0.3409 .0038 |  0.1052  1.0000   1.0000 |  Y  | 2.16
  40  5.81  0.5023 |   0.5676 .0038 |  0.2717  1.0000   1.0000 |  Y  | 3.99
  32  7.27  0.6278 |   0.7781 .0038 |  0.4689  1.0000   1.0000 |  Y  | 6.16
  24  9.69  0.8371 |   0.9142 .0038 |  0.6531  1.0000   1.0000 |  Y  | 8.18
  16 14.53  1.2557 |   0.9765 .0038 |  0.7939  1.0000   1.0000 |  Y  | 9.73
```
**Bracketing 10/10.  ρ(P_e^ub, BNN_err)=+0.888.  ρ(P_e^lb, BNN_err)=+0.937.**

## Claim verdicts
- **C1 (bracketing) — CONFIRMED.** 10/10 ε: `P_e^lb ≤ BNN_err ≤ P_e^ub` (Hoeffding hw=0.0038 on BNN; certified LCB on the lower).
- **C2 (morphological floor) — CONFIRMED.** Top union-bound pairs at ε=16 (σ=1.26) are exactly case/space/number neighbours: `'Hardware'~' Hardware'`, `' pasta'~' Pasta'`, `' six'~' seven'`, `' differential'~'Differential'`, `' states'~'states'`, `'3'~'5'`, `' global'~' Global'`. The geometry-only bound "knows" which tokens BNN confuses.
- **C3 (independence) — CONFIRMED by construction + numerically.** The probe takes no observation argument (unit test `test_no_observation_argument`); values depend only on (codebook, σ, synthetic RNG).
- **C4 (correlation) — CONFIRMED, with a complementarity finding.** The two bounds track BNN in complementary regimes:
  - **low noise (ε≥80, r≤2.9): union-Bhattacharyya upper bound is TIGHT** (ε=80: BNN 0.0058 vs ub 0.0125; ε=96: 0.0006 vs 0.0007) and tracks; Fano lower is vacuous (0).
  - **high noise (ε≤48, r≥4.8): Fano lower bound tracks** (BNN 0.34→0.98 vs lb 0.11→0.79, near-constant gap); union upper saturates at 1.0 (vacuous).
  - **mid (ε=64, r=3.63): widest gap** — BNN 0.063 ∈ [0, 0.346]; neither side tight (union over-counts ~5×, Fano vacuous since H(V|Y)=0.35<1 bit). This is the honest weak spot, flagged in the proof scope.

## Interpretation
At L0 the leakage **is** the embedding-table distance geometry, and the M-ary-Gaussian-channel bounds
capture it: the equivocation `H(V|Y)` (an MI quantity) and the Chernoff/union error exponent bracket the
optimal (BNN/MAP) attack from a route that never runs the attack. The "which bound is informative"
crossover at r≈3.6 is itself a diagnostic of the SNR regime. This is the matched, formal, attack-
independent MI probe the objective asked for — and it does what NNS-PVI could not (NNS-PVI was the attack
re-scored; these bounds never touch the attack's observations).

## Correlation caveat (intellectual honesty)
ρ over an ε-sweep carries a **monotonicity confound** — both BNN error and the bounds are *provably*
monotone in σ (proof T5), so a positive ρ is partly built-in (the same caution as `claim:bayes-gap-diagnosis`
about trivially-monotone ε-sweep correlations). The **load-bearing evidence is therefore NOT ρ** but:
(1) **bracketing** — the bound *values contain* the BNN error at 10/10 ε (a quantitative containment, far
stronger than co-ranking); (2) **upper-bound tightness at low noise** where it is not saturated (ε=80:
0.0058 vs 0.0125; ε=96: 0.0006 vs 0.0007) — a sharp value match, not a rank; (3) **C2 morphological-floor
attribution** — the bound predicts *which specific tokens* BNN confuses (`'3'~'5'`, `'Hardware'~' Hardware'`),
a non-monotone, geometry-specific prediction with no σ-confound. ρ is reported as supportive only.

## Limitations / open
- Mid-transition (r≈3.6) bound gap is wide; tightening the upper via α-information Fano (Rioul 2021) or
  a min-distance-dominant union refinement is future work (proof Remark, deferred).
- Uniform-prior over the pool (headline scope). Empirical-prior corpus BNN (prior-aware bounds, Remark N)
  not run.
- Validation is the embedding channel (L0) only; at depth BNN/this channel model does not apply.

## Reproduce
```
pytest tests/test_channel_error_bounds.py                      # M0
scripts/run_in_rocm.sh python3 scripts/spikes/bnn_error_bounds_validation.py            # M1
scripts/run_in_rocm.sh python3 scripts/spikes/bnn_error_bounds_validation.py \
    --epsilons 128,96,80,64,56,48,40,32,24,16 \
    --out results/bnn_error_bounds_validation_dense.json       # M1-dense
```
