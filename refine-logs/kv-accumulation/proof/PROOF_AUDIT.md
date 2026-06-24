# Proof audit — claim:kv-bss-subspace-floor-and-negentropy-probe

Reviewer: Codex gpt-5.5 (xhigh), thread 019ef6e2-7536-7eb3-895e-d07e431e0d75. 2 rounds.
Target: Lemma L1 (parts 1-3) and Observation C1 in
research-wiki/claims/kv-bss-subspace-floor-and-negentropy-probe.md.

## Round 1 — verdict FAIL
12 issues. Load-bearing defects:
- L1-04/05/06 (INVALID, GLOBAL): the analytic membership-floor lower bound `c_in >= 1/sqrt(s)` rested
  on three false steps — true rows treated as orthonormal, "Hungarian assignment >= per-row max", and
  "p95 >= mean". Counterexamples given for each.
- L1-02/03 (INVALID, GLOBAL): claim `S0 subset of rowspan(H)` is false for uncentered H (centering
  subtracts multiples of 1). Counterexample s=1, H=(1,2).
- L1-01 (UNJUSTIFIED): rank(Ũ)=s not stated as hypothesis (s=1 constant row centers to 0).
- L1-07/08 (UNDERSTATED/OVERSTATED): F_in depends on H-geometry not just (s,T); c_out=Θ(1/√T)≪c_in
  not theorem-level (1/sqrt(64)=0.125 < 0.155).
- L1-09 (UNJUSTIFIED): "isolates cumulant-alignment signal" overclaims JD identifiability.
- C1-01/02/03 (UNCLEAR/UNJUSTIFIED/OVERSTATED): zero slope not provable from stated hypotheses; T
  overloaded (matrix width vs stacked-observation index).

## Fixes (Round 2)
- DELETED the analytic 1/sqrt(s) bound entirely; floor magnitude now reported as empirical measurement
  (F_in ≈ 0.708, F_out ≈ 0.155) with a scope note recording the rejected attempt. (L1-04/05/06)
- DROPPED "S0 subset rowspan(H)"; added explicit centering caveat — the additive-mean offset is common
  to B_jd and Haar-B targets and cancels in `margin`. (L1-02/03)
- Added hypothesis: centered Ũ has full row rank s; T >= 2s. (L1-01)
- F_in redefined as functional F_in(S0,H;p95cos) of (S0,H) + fixed metric convention; F_in ≫ F_out is
  empirical only. (L1-07/08)
- L1.3 weakened: Haar-B is the matched null (holds fixed all nuisances, randomizes only the in-S0
  rotation); margin = raw − F_in differences the membership baseline out BY CONSTRUCTION (estimand
  definition, not population ICA identifiability). (L1-09)
- C1 retitled "Observation C1 ... empirical, with identifiability rationale (NOT a theorem)"; statement
  explicitly empirical; rationale labeled heuristic. (C1-01/02/03)

## Round 2 — verdict PASS
All 12 prior issues PASS. One residual minor (L1-new-01, COSMETIC): note the metric convention +
Haar left-invariance in the F_in definition — addressed (F_in(S0,H;p95cos) + invariance clause).

Overall: PASS. L1 parts (1)-(3) prove a valid structural / null-mismatch statement (not an analytic
magnitude bound); C1's empirical framing is acceptable.
