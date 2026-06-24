# Auto Review Loop — campaign-C Task 3 (capacity-pvi restructure)

Target: `docs/plans/campaign-C-report-hardening.md` Task 3 — split `resid-capacity-pvi.html` along the probe/surface boundary.
Reviewer backend: Codex (gpt-5.5, xhigh). Difficulty: medium. Started 2026-06-24.

## Round 1 (2026-06-24)

### Assessment (Summary)
- Score: 8/10
- Verdict: almost
- Key criticisms: (1) L0 cap-accuracy ρ source cell is "+0.96 / +0.99"; R7 kept only +0.96 — preserve the source form. (2/3/4) minor: label normalization, future-page link, source-vs-handoff citation distinction — verified already satisfied in the live file.

STOP CONDITION met at round 1: score 8 ≥ 6 AND verdict "almost" ∈ {ready, almost}.

### Reviewer Raw Response

<details>
<summary>Full reviewer response (round 1)</summary>

**Score: 8/10**

The restructure is largely faithful: the measured input-DP material moved to `resid-dp-attacks.html` as R7, the old page is retired rather than left as a fake surface, the V_cap methodology is staged for Task 4, and the depth-decoupling claim is cited from the new surface home.

**Ranked Weaknesses**

1. R7 has one numerical fidelity ambiguity at L0. Source has `+0.96 / +0.99` for L0 cap-accuracy tracking; R7 keeps only `+0.96`. Minimum fix: preserve the source form, or add a note.
2. Some source labels slightly normalized (`cap reader accuracy` → `V_cap reader accuracy`; `isotropic hidden-state noise` → `isotropic noise`). Minimum fix: use `isotropic hidden-state noise`.
3. Stub names the future probe page `probe-vcap.html` — must not be an actual hyperlink until created.
4. R7 should cite `RESULTS_STANDARDIZED.md` as measurement source; the staged handoff is interpretation/later-page, not surface evidence.

**Numerical Check** — No mismatch in R7 Tables 1/2/3 except the L0 `+0.96 / +0.99` issue. `3.85e3`→`3850` etc. numerically identical, not a fidelity problem. The R7↔R5 reconciliation is honest: complementary cuts, both showing deepest-layer decorrelation, not a contradiction.

**Verdict: almost ready** — after the small L0 source-fidelity fix, ready for the documentation-restructure phase.

</details>

### Actions Taken (round 1, before closing)
- Fix #1: added the source's tie-handling form to the R7 by-depth caption ("The L0 accuracy↔recovery ρ is +0.96 (+0.99 under alternative tie-handling) in the source.").
- Fix #2: non-issue — the live file already labels both rows "isotropic hidden-state noise" (the abbreviation existed only in the paste to the reviewer).
- Fix #3: already satisfied — `probe-vcap.html` appears inside `<code>`, not as an `<a href>`.
- Fix #4: already satisfied — R7 heading + footer cite `refine-logs/resid-capacity-pvi/RESULTS_STANDARDIZED.md` as source; the staged handoff is not cited as surface evidence.

### Status
- Stopping at round 1 (positive assessment). Difficulty: medium.

## Method Description
Documentation restructure (no GPU, no new research claim). The page `resid-capacity-pvi.html`, which conflated the `V_cap` probe's methodology with one surface's measured input-DP results under a false "surface" identity, was split: measured input-DP tables + the depth-decoupling finding relocated to the surface page `resid-dp-attacks.html` (block R7, numbers preserved verbatim from `RESULTS_STANDARDIZED.md`); the `V_cap` estimator-repair methodology + Lemma 1 / Prop 2 / Prop 3 staged to `refine-logs/capacity-pvi-restructure/VCAP-PROBE-PAGE-CONTENT.md` for Task 4's dedicated probe page; the old page retired to an off-nav stub linking both destinations; `claim:depth-decoupling-input-dp` cited from its new home; the synthesis V_cap overview row repointed.
