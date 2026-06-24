# Auto Review — embed-sgt (Task 7)

## Round 1 (2026-06-24)
### Assessment
- Score: 7/10
- Verdict: almost
- Reviewer: gpt-5.5 xhigh, read-only, thread 019ef8f1-4c27-72a3-9c5b-5bcf0cdd28b5
### Key criticisms
1. "across all 13 cells" wording inaccurate — Spearman computed over 12 noisy finite-budget settings (script filters budget_bits is not None); adding clean anchor → ~0.59 not 0.48. [FIXED]
2. No uncertainty estimates (per-cell CIs / correlation p-values). [deferred — needs GPU reruns]
3. Single seed / single noise realization. [deferred — needs GPU reruns]
4. Single surface + fixed attack — add explicit Limitations paragraph. [FIXED: added to Discussion]
5. tail_dump is an extreme construction — add continuum or justify as worst-case. [justified as worst-case in text]
### Faithfulness: PASS (numbers match sgt_eval.json). Scope: honest, weak-attack arm left open. Theory/probe: consistent, probe attack-independent.
### Actions taken (round 1 → 2)
- Corrected "13 cells"/"all settings" → "12 noisy settings" + clean-anchor 0.59 note, in claim, experiment log, and HTML (table header + findings).
- Added explicit Limitations paragraph to the report Discussion.
### Status: re-reviewing (round 2)

## Round 2 (2026-06-24)
- Score 7.5/8, verdict almost. Confirmed 13->12 fix in all three artifacts + limitations paragraph. One residual: masthead/intro still said "full/whole sweep".

## Round 3 (2026-06-24)
- Fixed masthead + intro "full/whole sweep" -> "twelve noisy settings".
- FINAL: Score 8/10, verdict READY as a scoped negative-result report. Residual weaknesses (no CIs, single seed, single surface/attack, extreme tail construction) disclosed, non-blocking; carried to spawn-depth-1 follow-up Task 11.

---

# Auto-Review Session — docs/html/synthesis.html (2026-06-24)

Target: the cross-surface quantitative synthesis page. Focus: structure, coverage, results comparability/representation. Reviewer: Codex gpt-5.5 xhigh (medium difficulty). Thread 019ef9de.

## Round 1 (2026-06-24)
### Assessment (Summary)
- Score: 7/10 — Verdict: Almost
- No fabricated tables found; most spot-checks matched source. Critical issues:
  1. Definite transcription error: depth-inversion L32 cosine-NN floor reported 0.390, source says 0.000.
  2. KV-accumulation & Shredder tables carried recovery only, no bits axis / not marked recovery-only.
  3. GELO verdict "probe not matched" stated definitively; source frames it as a hypothesis with weak-attack arm open.
  4. Capacity-PVI: §02 mentions at-layer but §03 only tabled L0 input-DP.
  5. STYLE: visible run-IDs (R3/C2) and em dashes in prose; inline styles; div-in-span nav.

## Round 2 (2026-06-24)
### Assessment (Summary)
- Score: 8/10 — Verdict: Almost (threshold met: score>=6 AND verdict in {ready,almost})
- All Round-1 critical issues verified fixed: L32 floor=0.000; recovery-only tables marked; GELO softened; at-layer capacity table added (rows match source); run-IDs removed. GELO/Stained-Glass distinction judged scientifically fair.
- Remaining (minor, addressed after this round): (1) KV overview implied a T-sweep → reworded to "plaintext baseline (T-axis flat)"; (2) Shredder missing b=0.109 row → added (0.977); (3) STYLE surface-page-sequence exception → documented inline + here; (4) utility TODO → disclosed in §05 (honesty OK).

### Actions Taken (post Round 2)
- Reworded KV-accumulation overview cell; added Shredder b=0.109/0.977 row; added inline HTML comment documenting synthesis.html as an intentional cross-cutting-page exception to STYLE.md's surface-page section sequence.
- Final structure re-validated: 20 tables all cell-consistent, all tag pairs balanced.

### Status
- STOPPED at Round 2 (positive assessment 8/10 "almost"; remaining items cosmetic and applied directly). Difficulty: medium.

## Method Description
synthesis.html aggregates the measured sweep results of the leakage-measurement program across 13 surfaces onto one page: a one-row-per-surface overview (attack, defense parameter, probe, Spearman ρ, verdict) followed by per-surface quantitative tables that reduce each multi-axis sweep to a representative 1-D slice along the load-bearing parameter (held-fixed axes named in-caption), with columns for measured probe bits, every attack's recovery, and utility where recorded, and a derived ρ beneath each. Verdicts: tracks / attack-limited / probe-not-matched / vacuous. Numbers are quoted from each surface's recorded results file.

---

# Auto-Review Session — docs/html/DIAGRAM-STYLE.md (2026-06-24)

Target: the method-diagram style reference. Focus: style/accuracy, depth-requirement enforceability,
interactivity eagerness, enough examples. Reviewer: Codex gpt-5.5 xhigh (medium). Thread 019efa92.

## Round 1 — Score 7/10, Almost
Strong intent/structure; depth rules explicit; a shallow 3-box diagram is rejectable from the doc.
Adoption blockers (all verified against files):
1. Cited baseline (vec2text FIG.01) didn't fully satisfy R1 — unlabeled arrows, missing DP sublabel, `release` label on the wrong boundary.
2. D3 skeleton not runnable: `js/d3.v7.min.js` path absent; `.diagram-controls`/`.is-active`/`.is-dim` not in site.css.
3. `--warn` described as a box class but it is only a CSS variable; vec2text used inline `style`.
4. Interactivity "only where it improves understanding" too escapable — no trigger rules.
5. Accessibility stated but not embodied in the skeleton.

## Round 2 — Score 8.5/10, Almost (threshold met; loop terminated)
Reviewer confirmed: the named CSS classes now exist in site.css; FIG.01 matches the reference claims;
the skeleton is runnable (CDN) and accessible. Remaining items were minor and fixed after this round:
the x̂ recovery arrow set to `arrow-warn`/`warn`; the final checklist updated off the stale `--warn`
wording; the no-inline rule scoped to "color only (geometry/opacity inline ok)"; a vendor-D3 checklist
item added.

### Actions taken
- `css/site.css`: appended `.box-warn`, `.box-label.warn`/`.box-sub.warn`, `.arrow-warn`,
  `.arrow-label.warn`, `.diagram-controls` (+ `button:disabled`), `[data-step].is-active`/`.is-dim`,
  and a `prefers-reduced-motion` block.
- `vec2text.html` FIG.01: full R1 compliance — DP zone sublabel; `release` moved to the e′ boundary;
  4 arrow labels added (X, e₀, h₀, Σ,σ); inline `--warn` styles → `.box-warn`/`.warn`; x̂ arrow → `arrow-warn`.
- `DIAGRAM-STYLE.md`: anatomy now states FIG.01 is the compliant baseline; CSS vocabulary + rule #6
  name the warn classes and scope the inline-style rule; interactivity **trigger rules** added (eager
  but bounded); runnable + accessible D3 skeleton (CDN, type=button, aria-controls, aria-live, disabled,
  scoped keydown); checklist updated (+ vendor-D3 item).

### Status
- STOPPED at Round 2 (8.5/10 "almost"; all listed items applied). Difficulty: medium.

## Method Description
`docs/html/DIAGRAM-STYLE.md` is the depth standard for method diagrams on the research-report site:
research the source paper's figure first (R0); satisfy mandatory content (R1: trust-zone bands +
boundary, numbered train/infer sequence, both measurement-loop arms, defense-parameter locus, typed
labeled arrows) using the house `site.css` palette; add D3 interactivity (R2) per explicit trigger
rules with a runnable accessible step-through skeleton; anti-patterns + a checklist gate review; a
per-surface backlog lists the 7 pages still missing a method diagram. vec2text FIG.01 is the compliant
baseline.

---

## Plan review — campaign-C-report-hardening.md (2026-06-24)

Target: `docs/plans/campaign-C-report-hardening.md` (13-task ralphex campaign plan). Objective: verify each task is self-contained + executable headless, dependencies stated and acyclic, file parses as a ralphex plan. Reviewer: Codex gpt-5.5 xhigh (sandbox could not read repo — plan pasted inline with verified context anchors).

### Round 1 — Score: 8/10 — Verdict: almost (POSITIVE, stop condition met)
Parseability OK (13 tasks each end `- [ ] run-phase:`, unique run_ids/surfaces, only Task 7 gpu:true). Dependency order correct (Task 2 gates naming; Task 3 stages V_cap → Task 4; Tasks 4/6 queue → Task 7; Task 7 → Task 12), no hard cycle. Tasks 2,3,5,9,10,11,12 called especially well-scoped.

Six weaknesses, all fixed inline this round:
1. Task 4 completion ambiguity (plaintext "across layers" vs queued placeholders) → added explicit COMPLETION RULE: done with real existing baselines + queued placeholders; Task 7 owns backfill.
2. Task 6 possible hidden circular dep on Task 7 → added COMPLETION RULE: real readout OR explicit placeholder; phase must not wait on Task 7.
3. Task 7 too large / unordered → restructured into 3 priority-ordered subdeliverables (DATA-GPU first, then synthesis backfill, then placeholder backfill).
4. Task 7 output contract unnamed for Task 12 → named `refine-logs/utility-tradeoff/leakage_utility.json` with explicit row schema + `queue_results.json`; Task 12 now reads that path.
5. Task 1 acceptance relied on un-checkable "/auto-review-loop confirms" → replaced with file artifacts (`assembled/*.txt`, `instruction-diff.txt`) + grep/diff checks; verdict left to the gate.
6. Task 8 / Task 13 vague escape hatches → Task 8 requires per-claim `research-wiki/claims/<slug>.proof-check.md` artifact; Task 13 "skip if time-bound" replaced with B2-mandatory / B1-optional completion rule.

Status: COMPLETED at round 1 (positive). HTML render + result-to-claim termination steps skipped (not applicable to a plan-document review).
