---
type: plan
status: current
created: 2026-06-20
updated: 2026-06-20
tags: [experiment-plan, class-PVI, capacity-matched, leakage-measure, calibration, independence]
companion: [idea-stage/IDEA_REPORT.md]
---

# Experiment Plan — fix (or formally retire) the independent leakage family

**Problem**: We need a leakage measure that is **both** (a) *independent* of the inversion
attack (not a reparameterisation of its ridge `X→embedding` map/target) **and** (b) a
*faithful* predictor of token-recovery (TTRSR). The only independent+cheap family we have,
**class-PVI**, overfits in the `d>n_val` regime (free `d=2304→256` softmax; shuffle floor
≈ −48 b, non-monotonic under noise). retrieval-PVI is faithful only because it *is* the
attack; CLUB is independent-estimator but upper-bound-only and shares the embedding target.

**Method Thesis**: class-PVI's failure is an **estimator-validity regime** (`capacity ≫
data`), not the token-id V-family being wrong — a **capacity-matched** member of the *same*
family is well-posed *and* cheaper, and tracks the attack; if no such member can, we deliver
a **formal impossibility verdict** for the family. Either outcome closes the standing question.

**Non-negotiable constraints**: probe cost ≤ current class-PVI cost (PVI is already 56–59% of
every block); heavy runs only via `scripts/run_in_rocm.sh`; favour the model-free cached-capture
loop. This is **not** an LLM-component paper — the frontier-necessity block is explicitly skipped.

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Linked blocks |
|-------|----------------|-----------------------------|---------------|
| **C1 (primary)** — class-PVI's failure is the `d>n_val` regime; a capacity-matched member of the same independent token-id family is well-posed, faithful, and cheaper. | This is the whole standing objective: an independent **and** faithful **and** cheap predictor either exists or it doesn't. | ≥1 variant with shuffle floor ∈ ~[−1, +1] b, PVI monotone↓ under post-hoc noise, Spearman(measure, TTRSR) ≥ 0.9 across blocks, wall-clock/block ≤ class-PVI — while class-PVI fails ≥1 of these (known: floor −48 b, non-monotonic). | B1, B2, B3 |
| **C1′ (fallback, mutually exclusive with C1)** — if no variant qualifies, the unconstrained token-id V-family is formally the wrong methodology in `d>n`. | A clean negative is publishable and is *required* before abandoning the family. | An estimator-validity / identifiability condition + a bias-floor argument that predicts the observed failure; consistent with McAllester–Stratos O(ln N). | B5 |
| **C2 (supporting, gated on C1)** — the working independent measure *calibrates across defense schemes*: one measure→TTRSR map holds across DP-ε, static obfuscation, and split-depth. | This is the edge over PAF / Jacobian-Rank-Recovery (correlation, one scheme) and FSInfo (obfuscation-only): IT + **calibrated** + **cross-scheme**. | A single monotone calibration curve whose fit transfers (held-out-scheme Spearman ≥ 0.85) across ≥2 (ideally 3) defense families. | B4 |

**Anti-claims to rule out**
- **A1 — "it secretly became the attack."** The capacity-matched variant tracks TTRSR only
  by re-deriving the embedding map. → Ruled out in **B3**: variant uses token-*id* classes,
  never the embedding table; its per-instance PVI is **not** collinear with retrieval-PVI.
- **A2 — "it's just shrinkage of the same softmax."** Plain l2 on the full `d=2304` softmax
  is enough, capacity/dim is irrelevant. → Ruled out in **B3**: l2-only sweep (known: floor
  −48→−9.5 at l2=10, degenerates at l2≥100) does *not* reach floor≈0; reducing dim < n_val
  does. Dim is the operative lever.
- **A3 — "it's row-split memorisation."** → shuffle-control floor ≈ 0 and a vocab-disjoint
  check where the family admits it.

## Paper Storyline
- **Main paper must prove**: C1 (B1–B3) — the independent family, fixed and cheap, tracks the
  attack — *or* C1′ (B5) if it can't. Plus C2 (B4) for the cross-scheme calibration headline.
- **Appendix can support**: MDL/SDL same-family robustness check (B6a); two-sided bracket (B6b).
- **Intentionally cut**: PML/α-leakage on activations (Idea 6), SAE effective-DoF (Idea 8) —
  parked until the standing question resolves. No frontier-necessity block (non-frontier method).

## Experiment Blocks

### Block 1 — Capacity-matched class-PVI: well-posedness + cost (model-free fast loop) — MUST-RUN
- **Claim tested**: C1 (well-posedness + cost half).
- **Why**: the standing objective; cheapest decisive screen before any GPU sweep.
- **Data/split**: cached gemma-2-2b capture (`results/capture_cache/capture-4ca8a33e16bfbec9.pt`),
  `resid_post` L12 (primary), L5 + L20 (confirm). Same `(X,y)` as the attack.
- **Compared systems** (new `v_information` variants in `src/talens/measures/`, same token-id
  target + class-prior null + shuffle control): (a) **PCA→dim k<n_val** then linear softmax
  (sweep k∈{64,128,256,384}); (b) **control-anchored l2/dim** (pick the knob by *shuffle floor≈0*,
  not val-CE); (c) **non-parametric** kNN and Gaussian class-conditional (no iterative fit);
  (d) **random-projection→dim k** + linear readout. Reference: current class-PVI (free `d→256`).
- **Metrics**: shuffle-control floor (bits), monotonicity of PVI vs post-hoc Gaussian noise
  σ∈{0,.25,.5,1,2} (existing `diag_pvi.py` sweep), wall-clock/fit. 3 seeds (report mean±sd).
- **Setup**: extend `scripts/spikes/diag_pvi.py`; model-free, runs on host `.venv` (CPU) or ROCm.
- **Success**: ≥1 variant with floor ∈ ~[−1,+1] b, monotone↓, wall-clock ≤ class-PVI.
- **Failure interpretation**: if *no* variant is well-posed at ≤ cost → strong prior for C1′; go to B5.
- **Table/figure**: Table 1 (variant × {floor, monotone?, cost}); Fig 1 (PVI vs σ curves).
- **Priority**: MUST-RUN.

### Block 2 — Faithfulness: does the survivor track TTRSR across sweeps — MUST-RUN
- **Claim tested**: C1 (faithfulness half).
- **Why**: well-posed ≠ predictive; this is the headline correlation.
- **Data/split**: (i) DP ε-sweep via `scripts/spikes/localdp_runner.py` (swap `panel()`'s measure
  to the B1 survivor), ε∈{∞,8192,4096,2048,1024,512,256}, layers {5,12,20}, vocab split; (ii) the
  36-layer control sweep (108 blocks: resid_post/kqv_out/kq) for cross-block rank.
- **Compared systems**: survivor variant vs class-PVI (expected: fails), retrieval-PVI (mechanical
  reference), CLUB (independent upper-bound reference). Ground truth = TTRSR.
- **Metrics**: Spearman & r² (measure ↔ TTRSR) across blocks/ε; monotonicity vs ε. Decisive =
  Spearman across the 108-block sweep (compare to CLUB's 0.987, class-PVI's 0.891 from 2026-06-17).
- **Setup**: ROCm container; capture cache hit (no recapture). ~10–23 min/sweep.
- **Success**: survivor Spearman ≥ 0.9 across blocks **and** monotone over ε (where class-PVI is flat).
- **Failure interpretation**: well-posed but not faithful → the independent family carries
  bounded-reader info that the attack's geometry doesn't → evidence for C1′ (bias-floor).
- **Table/figure**: Table 2 (measure × {Spearman, r², monotone-ε}); Fig 2 (calibration scatter).
- **Priority**: MUST-RUN.

### Block 3 — Independence isolation (rule out A1/A2) — MUST-RUN
- **Claim tested**: C1's *independence* (the half that distinguishes us from retrieval-PVI).
- **Why**: a faithful tracker that secretly is the attack is worthless; this block is what makes
  the result "independent," not "another attack."
- **Compared systems**: survivor vs retrieval-PVI (per-instance PVI vectors), and a **deletion
  study**: l2-only on full `d` (no dim reduction) vs dim<n_val.
- **Metrics**: (i) per-instance Spearman(survivor-PVI, retrieval-PVI) — want **moderate, not ≈1**
  (collinearity ≈1 ⇒ it's the attack); (ii) confirm the measure never reads the embedding table;
  (iii) floor≈0 reached by dim-reduction but **not** by l2-only (rules out A2).
- **Success**: survivor preserves floor≈0 + faithfulness while *not* collinear with retrieval-PVI,
  and dim (not shrinkage) is the operative lever.
- **Failure interpretation**: if faithfulness coincides with retrieval-PVI collinearity → the only
  faithful capacity-matched member is a disguised attack → independence is unattainable in-family → C1′.
- **Table/figure**: Table 3 (collinearity + deletion study).
- **Priority**: MUST-RUN. *(Doubles as the simplicity/elegance check: prefer the cheapest variant
  that passes; show l2-only is insufficient and the non-parametric family is not needed if PCA suffices.)*

### Block 4 — Cross-scheme calibration (supporting claim C2) — NICE-TO-HAVE (gated on C1)
- **Claim tested**: C2.
- **Why**: turns a metric note into a paper; the white-space contribution (G2) and the anti-scoop edge.
- **Data/split**: same measure→TTRSR map fit on DP-ε; tested on **static obfuscation** and
  **split-depth** schemes. Obfuscation + split-depth runners **do not exist** — build minimal ones
  modelled on `localdp_runner.py` (an obfuscation Transform; a layer-skip/cut harness).
- **Compared systems**: survivor measure; report PAF/FSInfo-style baselines if cheaply reproducible.
- **Metrics**: held-out-scheme calibration transfer (fit on one scheme, predict TTRSR on another;
  Spearman ≥ 0.85); single-curve overlap across schemes.
- **Success**: one calibration curve transfers across ≥2 (ideally 3) schemes.
- **Failure interpretation**: scheme-specific calibration → report as a scoped (per-scheme) result.
- **Table/figure**: Table 4 (cross-scheme transfer matrix); Fig 3 (overlaid calibration curves).
- **Priority**: NICE-TO-HAVE; gated on C1 passing B1–B3.

### Block 5 — Formal verdict on the token-id V-family (C1′) — CONDITIONAL MUST-RUN
- **Claim tested**: C1′ (only if B1/B2/B3 fail to yield an independent+faithful+cheap variant).
- **Why**: the "decisively conclude wrong methodology" obligation; prevents an unjustified pivot.
- **Deliverable**: (i) estimator-validity / identifiability condition for `I_V` with an
  unconstrained family when `d>n` (when is the plug-in estimate consistent?); (ii) a **bias-floor**
  argument for why a token-id classifier cannot track an embedding-geometry attack; (iii) numerical
  corroboration on the cached capture. Anchors: McAllester–Stratos O(ln N); de Chérisey et al.
  MI↔success-rate; Pimentel & Cotterell Bayesian-probing estimand.
- **Success**: a condition that *predicts* the observed B1/B2 failures (e.g. floor magnitude scaling
  with `d/n_val`).
- **Table/figure**: Theorem + Fig 4 (predicted vs observed floor vs `d/n_val`).
- **Priority**: MUST-RUN **iff** the decision gate after M3 is "fix failed."

### Block 6 — Appendix robustness — NICE-TO-HAVE
- **6a (MDL/SDL same-family check)**: confirm `online_code_length` inherits the overfit (small-prefix
  fits) and costs 6–7× — justifying its exclusion from the critical path. Reuses `mdl.py`.
- **6b (two-sided bracket)**: add a MINE/InfoNCE lower bound to pair with CLUB (upper); report
  bracket width as a confidence signal (Idea 3, G1b).
- **Priority**: NICE-TO-HAVE / appendix.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| **M0** sanity | implement 4 variants + unit tests; floor≈0 on a synthetic separable toy | host `.venv` pytest | variants pass oracle + toy floor≈0 | <10 min CPU | impl bug |
| **M1** fast screen | B1 on cached capture (L12, then L5/L20), 3 seeds | extended `diag_pvi.py` | ≥1 variant: floor∈[−1,1], monotone, ≤ cost | ~10–20 min CPU/ROCm | no variant well-posed → M3 gate toward B5 |
| **M2** faithfulness | B2: DP ε-sweep + 108-block sweep with survivor | `localdp_runner.py`, `--control all` sweep | survivor Spearman≥0.9, monotone-ε | ~30–45 min ROCm | well-posed but unfaithful |
| **M3** independence | B3 collinearity + deletion study | model-free on cached capture | not collinear w/ retrieval-PVI; dim is lever | ~15 min | faithful⇒collinear ⇒ C1′ |
| **M4** decision | branch: **PASS**→B4 (cross-scheme) / **FAIL**→B5 (formal verdict) | new obfuscation+split runners *or* theory | — | B4: ~1–2 GPU-hr (new runners) | new-runner code cost |
| **M5** polish | B6a/B6b, figures | `mdl.py`, MINE | — | ~30 min | — |

## Compute and Data Budget
- **Total estimated**: ~2–4 GPU-hours if C1 passes through B4 (most spent building the
  obfuscation/split-depth runners); ~1 GPU-hour if the run stops at the formal-verdict branch.
- **Data prep**: none new — cached capture + existing corpora. B4 needs an obfuscation Transform
  and a split-depth/layer-skip harness (new code, modelled on `localdp_runner.py`).
- **Human eval**: none.
- **Biggest bottleneck**: M4-B4 new runners. Everything up to the C1 verdict is the cheap
  model-free loop + two existing sweeps.

## Risks and Mitigations
- **R1 — capacity reduction kills the signal** (dim<n_val too small to carry token info): sweep k;
  fall back to kNN/Gaussian which regularise without a hard linear bottleneck.
- **R2 — faithful-but-collinear** (the only tracker is a disguised attack): B3 is designed to detect
  this; if so it is itself the evidence for C1′, not a failure of the plan.
- **R3 — new-runner cost for B4**: gate B4 behind C1; reuse `localdp_runner.py` scaffolding; start
  with split-depth (cheapest — just change capture layer) before obfuscation.
- **R4 — 256-class cap limits resolution**: documented design choice; note in limitations; the cap is
  the same across all compared measures so rank comparisons are fair.
- **R5 — scoop (PAF / Rank-Recovery)**: keep the IT + calibration + cross-scheme + formal-independence
  framing front-and-centre; cite and out-position, don't ignore.

## Final Checklist
- [x] Main paper tables covered (T1 well-posedness, T2 faithfulness, T3 independence, T4 cross-scheme)
- [x] Novelty isolated (B3 independence + deletion study)
- [x] Simplicity defended (B3 prefers cheapest passing variant; l2-only deletion study)
- [x] Frontier contribution explicitly **not** claimed (non-frontier method; block skipped)
- [x] Nice-to-have (B4 partial, B6) separated from must-run (B1–B3, conditional B5)
