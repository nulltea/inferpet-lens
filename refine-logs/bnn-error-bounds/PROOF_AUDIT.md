# Proof Audit — Union-Bhattacharyya & Fano error bounds

**Source**: `refine-logs/PROOF_PACKAGE.md`  
**Date**: 2026-06-22  
**Reviewer**: gpt-5.5 via Codex MCP, xhigh  
**Thread**: 019eefac-891a-7a52-82d7-4287c4ddb47b

---

## Round 1 — findings (20 issues) and fixes

The unifying defect: the package wavered between uniform and non-uniform prior. The
union-bound proof, the Bhattacharyya bound, and the equivocation estimator are all
**uniform-prior** objects; the setup's "empirical p̂" aside induced ~half the issues.
Fix: scope the theorems to **uniform prior + K≥3 + fixed declared (σ,{e_v},p,K)**, and
separate the **population** Fano lower bound from the **random finite-M estimator**.

| id | Issue | Resolution |
|----|-------|-----------|
| T1-01, T1-02, T1-04, H-01 | Union/Bhattacharyya/estimator assume uniform prior; package claimed general p | Scope all theorems to uniform prior; move non-uniform to a clearly-flagged remark with prior-aware formulas |
| T1-03 | Fano denominator log₂(K−1)=0 at K=2 | Assume K≥3 (pool=2048); note binary case uses inverse-binary-entropy separately |
| T1-05, D-01, T2-02, T5-02 | Finite-M Ĥ_M is random, can exceed H(V|Y) ⟹ plug-in not a certified lower bound | Distinguish population P_e^lb (M→∞, exact) from MC estimate; certified finite bound via one-sided lower confidence bound on H(V|Y) |
| T4-01 | Summands independent but NOT identically distributed across v | SLLN per-v then average over finite K (or sample V_j~Unif(P)) |
| T4-02 | Var formula wrong | Var(Ĥ_M) = (1/(K²M))Σ_v Var(g_v) = O(1/(KM)) for fixed K |
| T3-01, T3-02, T3-03 | Independence needs (σ,p,{e_v}) not estimated from {Y_i}; "geometry alone" overstated | Add explicit fixed-inputs assumption; state conditional independence precisely; "codebook geometry + declared prior + noise level" |
| T5-01 | Strict monotonicity of H(V|Y) needs more than DPI | State non-decreasing; strictness via I-MMSE noted as remark |
| T5-03 | Co-monotonicity with BNN error not proven for the error itself | Add explicit degradation/simulation argument that P_e* is non-decreasing |
| T1-06 | Δ_vu only defined via its norm | Define Δ_vu := e_v − e_u (vector) |
| T2-01 | O(1/√n) mode unstated | State iid test, Hoeffding high-prob bound |
| T4-03, T4-04 | Integrability constants depend on σ,K; max-term bound needs a_v=0 wrapping | Fix σ, finite codebook; clean log-sum-exp bound with a_v=0 |

All fixes applied in `PROOF_PACKAGE.md` (round 2).

## Round 2 — re-review (8 LOCAL issues)

All Round-1 issues discharged under the uniform/K≥3/A0 scope. 8 new LOCAL issues, all
wording/precision on the finite-M certification (R2-01..R2-08). R2-03 (INVALID-LOCAL):
`g` is unbounded under Gaussian noise → Hoeffding inapplicable; fixed by noting `g` is
sub-exponential and using Bernstein-type concentration. Others = OVERSTATED/UNCLEAR/
UNDERSTATED-LOCAL: CLT-coverage wording (liminf), stratified se² definition, Fano-functional
monotonicity wording, T4 proof order (integrability first), variance constant non-uniformity,
full-transcript independence, Remark-N prior caveat. All applied.

## Round 3 — closure CONFIRMED

Reviewer verdict (thread 019eefac, xhigh): **acceptance gate met.**
- Zero open FATAL/CRITICAL (Round 1 + Round 2).
- Theorems scoped to uniform prior, K≥3, A0.
- Population vs finite-M lower bounds separated; CLT coverage correctly asymptotic.
- Stratified variance/SE fixed; T3 independence covers full transcript; T4 integrability-first.
- Big-O dependence declared non-uniform in K, geometry, σ.
- Only residual: wording hygiene (α∈(0,½), M≥2) — applied; non-blocking.

**VERDICT: PASS.** All five theorems PROVABLE AS STATED. The probe rests on textbook
bounds (Proakis union; Cover-Thomas Fano) with honest scope; the independence property
(T3) is established by construction under the declared-inputs assumption A0 — the property
NNS-PVI lacked.

**Maps to wiki claim status**: `verified` (proof closes, bounds are imported textbook
results flagged as such → arguably `sound-modulo-imports` for the bound steps, `verified`
for the novel T3/T4 independence+consistency).
