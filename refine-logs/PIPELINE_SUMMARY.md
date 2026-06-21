---
type: reference
status: current
created: 2026-06-20
updated: 2026-06-20
tags: [pipeline-summary, matched-probe]
companion: [FINAL_PROPOSAL, EXPERIMENT_PLAN]
---

# Pipeline Summary

**Problem**: Confidential-inference leakage is scored as one conflated scalar
(ridge-attack TTRSR) + one fitted measure, yet defences protect different
*targets* at different depths (the established L20×input-DP token-id↔embedding
decoupling). Need a per-channel account: what leaks, through which surface, which
independent probe predicts it.

**Final Method Thesis**: Leakage decomposes into (target × surface) channels;
each has a *matched independent* IT probe that calibratedly predicts its attack,
and **mismatched probe↔target pairs provably decouple** (a falsifiable law; the
L20 divergence is its first datum).

**Final Verdict**: READY (build B0–B3). Headline framing and Π-probe choice are
**intentionally deferred to experimental data** (user decision, 2026-06-20).

**Date**: 2026-06-20

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- (prior thread archived under `refine-logs/archive-capacity-pvi/`)

## Contribution Snapshot
- **Dominant**: matched-probe taxonomy of leakage channels + the decoupling law.
- **Supporting**: cross-scheme calibration over 6 defence families (DP, split-depth,
  AloePri perm-core, AloePri full Alg1, Shredder static, Shredder learned).
- **Rejected**: membership channel (probe=attack), Vec2Text decoder (CLUB suffices),
  MoE-routing (no MoE), eager formal bound (conditional only).

## Must-Prove Claims
- C1 — each channel has a matched **independent** probe (ρ≥0.9, collinearity<0.9).
- C2 — the decoupling law: matrix diagonal dominates off-diagonal; ≥1 sign-flip.
- C4 — select the independent Π-probe (3-way bake-off).
- C5 — declare the framing (F-A/B/C) from the matrix.

## First Runs to Launch
1. **B0** — implement + unit-test: AloePri Alg1 obf-table generator (`P̂Q̂=I`),
   4 defence Transforms/trainers, PID probe, independent Π-probe.
2. **B1** — per-channel matched calibration on cached capture (DP + split-depth).
3. **B2** — Π bake-off + AloePri α_e sweep → select the Π-probe (C4).
4. **B3** — the decoupling matrix (C2) → **then `/auto-review-loop`**.

## Main Risks
- **Forward-consistency under obfuscation** (R1): prefer VMA on `WeightPair` for the
  Π-calibration (pure algebra); vocab-match only as a static-cover cross-check.
- **AloePri-full logit preservation** (R2): B0 logit-fidelity gate before sweeps.
- **Matrix breadth** (R4): cheap cached-capture phases (M0–M4) decide the framing
  before the expensive new-defence sweeps (M5).
- **No independent Π-probe may exist** (R5): that is itself a result (weakens F-A).

## Next Action
- Proceed to `/run-experiment` (or the spike scripts) for **B0**, then B1–B3.
- Run `/auto-review-loop` after **B3** (paste matrix tables inline — Codex sandbox
  can't read repo files).
