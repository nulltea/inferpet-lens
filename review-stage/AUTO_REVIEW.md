# Auto Review — CE logit-lens attack (implementation + run-plan)

Target: `logit_lens_attack`/`LensHead` (src/talens/attacks/dp_inversion.py), eval wiring
(scripts/evals/dp_leakage_sweep.py), doc (docs/research/ce-logit-lens-attack.md), test
(tests/test_dp_inversion.py). Two gates: correctness + perf (scripts/harness/perf_gate.md).
Reviewer: codex (gpt-5.5, xhigh), difficulty medium.

## Round 1 (2026-06-25)

### Assessment (Summary)
- Score: 5/10 (correctness 8/10, run-plan 3/10)
- Verdict: NOT READY (to launch the full GPU grid)
- Correctness: CLEAN — reviewer found no target-index bug, no negative-collision bug, no
  labelled-test-token leak, no probe/attack circularity, no scheme-core violation. Ridge warm-start
  shape correct; train/val split disjoint; floor shuffles Etr+ytr with the same permutation; eval
  argmax over the test pool only; head has no per-token params (open-set preserved).
- Blockers are all PERFORMANCE / run-plan:
  1. Run not optimal scope + no saturation pass → run a scoped lens,declens pilot first.
  2. CE path may exceed the 10-min gate without evidence → benchmark one CE fit at scale.
  3. Redundant ridge_W recompute across ridge/lens/declens → cache or drop ridge from this run.
  4. CPU→GPU per-epoch candidate gather unproven at scale → benchmark; fix doc "≈ cosine decoder".
  5. Add invariant guards (0<=ytr<V; positives in denominator).

### Actions Taken (Phase C)
- Fix #5: added `assert ytr aligned + 0<=ytr<V` in logit_lens_attack; documented that train_toks ⊆ cand
  guarantees exact searchsorted indices (positives always in the CE denominator).
- Fix #4 (doc): replaced the unmeasured "cost ≈ cosine decoder" with "measured in a scoped pilot
  before the full grid".
- Fix #1/#2/#4 (evidence): launched a scoped saturation pilot — L12 × {inf,256} × {lens,declens} ×
  512 prompts (production n_train) → measure per-fit wall-time + GPU saturation, extrapolate to the
  full grid; results below.
- Fix #3: pilot drops `ridge` (already have ridge from dp-decoder-r3-fixed); ridge_W redundancy is a
  sub-10s CPU cost vs CE training (will quantify), so no interface refactor — decided after pilot.
- Tests still pass (3/3).

---

# Topic: MI-probe configuration note (docs/research/2026-06-26-mi-probe-configuration.md)

threadId: 019f040f-44d5-73a3-b92f-73abc93cfb23 · reviewer: codex/gpt-5.5 xhigh · difficulty: medium

## Round 1 (2026-06-26) — Score 5→ **6/10**, Verdict: not ready
Key criticisms: central "L20 ≠ token-id" inference overclaimed; Voita extrapolation needs external-validity
caveat; vocab-disjoint split impossible for closed-set class PVI/MDL; protocol ≠ current eval; **attack
split is layer-dependent (RNG advances inside layer loop) → depth confound**; PAC bound stated as absolute
(it is one-sided); monotonicity overread; citations swapped (MDL=2003.12298, Bottom-up=1909.01380); CLUB
param count 590k→~198k; MDL "no tuning" overgeneralized.

### Actions taken
Fixed all 10: softened central inference to "unlikely, not ruled out → adjudicate"; added external-validity
paragraph; corrected split guidance (row-split + type-control for class probes, disjoint only for
CLUB/retrieval); marked protocol as target-not-current; **fixed the layer-dependent split bug in
dp_leakage_sweep.py (split/pool/shuffle computed once, with assert)**; one-sided PAC; monotonicity caveat;
citations corrected; CLUB ≈198k; MDL softened to "verify by size sweep".

## Round 2 (2026-06-26) — **7/10**, Verdict: **Almost** → STOP (score≥6 AND verdict∈{ready,almost})
Reviewer confirmed fixes are real (verified the split fix in code). Remaining minimum fixes (all applied
post-round): type-control vs row-shuffle distinction (type-control can subtract token-id signal); split
PVI into PVI_class(→token-id, softmax) vs V_Gauss(→embedding, R²/Prop.1.5); disclose probe
controls/selectivity not wired; soften MDL table language; require class-coverage reporting; disambiguate
"L20" as the Gemma observation (Pythia=analogous mid/deep peak).

### Status
Loop complete at Round 2 (7/10, Almost). All round-2 minimum fixes folded into the note. Verdict reflects an
internal methodology note, not a submission.
