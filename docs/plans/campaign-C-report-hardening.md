<!-- ralphex-native plan (parsed by `### Task N:` sections); exempt from docs frontmatter.
     BLOCK C — report hardening. Run supervised or unattended:
       scripts/harness/run_campaign.sh docs/plans/campaign-C-report-hardening.md
     Decisions grilled 2026-06-24 (companions: docs/handoffs/2026-06-24-html-synthesis-and-consistency-review.md;
     refine-logs/visual-reporting-research/RESEARCH.md; memory ralphex-warm-resume-and-campaign-c).
     Threat anchor WEIGHTS-PUB.
     PREREQS already DONE this session (NOT phases — see "Completed this session" below):
       harness warm-resume drop-off fix; docs/html/DIAGRAM-STYLE.md diagram standard.
     Each task carries its full spec inline (objective + problem + decision + steps + acceptance) on purpose
     — thin task descriptions hurt earlier campaigns. Only the `- [ ] run-phase:` line is parsed for completion.
     EXEC ORDER = FILE ORDER. Dependencies are noted inline; do not reorder without re-checking them. -->

# Plan: Block C — report hardening (registry, restructure, readouts, diagrams, utility, proofs, visual reporting)

## Overview

The cross-surface synthesis (`docs/html/synthesis.html`) and the 2026-06-24 consistency review exposed
report gaps; a 2026-06-24 grilling + literature pass (`refine-logs/visual-reporting-research/RESEARCH.md`)
extended the scope. This block closes, in order:

1. **Harness prompt split** (Task 1) — recipe-keyed task prompts so each phase loads only its own recipe.
2. **Probe registry** (Task 2) — canonical probe names+symbols used verbatim everywhere; gates naming below.
3. **capacity-pvi restructure** (Task 3) — `resid-capacity-pvi.html` is a *probe* (`V_cap`) wearing a
   surface costume and duplicating the input-DP surface; split it.
4. **Per-probe pages + plaintext reference** (Task 4) — one page per registered probe, each carrying a
   plaintext (no-attack/no-defense) reference measurement.
5. **Method diagrams** (Task 5) — redraw shallow/missing method diagrams to DIAGRAM-STYLE.md.
6. **Readout metrics** (Task 6) — render a legible per-secret readout beside every bits value.
7. **Leakage–utility** (Task 7) — the only GPU phase; measures the unrecorded utility axis and serves as
   the single GPU run that backfills the model-required gaps Tasks 4 and 6 queue onto it.
8. **Cross-cutting proofs** (Task 8) — prove the four synthesis claim stubs.
9. **Visual reporting** (Tasks 9–13) — the reports are table-only on the results axis; add the five
   Group-A result plots (harness-first, then by visualization family). Distinct from Task 5: Task 5 is
   *method/architecture* diagrams (DIAGRAM-STYLE.md); Tasks 9–13 are *result/data* plots (PLOT-STYLE.md).

Recipe `consolidate` for documentation/proof phases (no model runs), `full` only for Task 7 (utility).
Task 2 gates the naming used by every later task; run strictly in file order. The diagram standard is
already written (`docs/html/DIAGRAM-STYLE.md`); the harness drop-off fix is already shipped.

---

### Task 2: probe registry — canonical names + symbols, used verbatim everywhere
recipe: consolidate
gpu: false
surface: probe-registry
run_id: c2-probe-registry
gate: review refine-logs/probe-registry/REVIEW_STATE.json
objective: define a single probe registry (quantity-led formal name + short symbol) and apply it verbatim across every HTML page, so probe names are self-descriptive and unambiguous. Gates the naming used by Tasks 3–13.

problem: names drift and mislead across the site — "spectral channel-MI" reads like an umbrella term, and "spectral capacity" (a distinct KV diagnostic) is confused with `I_G`. synthesis.html has an ad-hoc probe glossary that must become the canonical source.

decision — the canonical registry (write it once, then use verbatim):

| symbol | canonical name | quantity it reports |
|---|---|---|
| `I_G` | Gaussian channel-capacity MI ceiling | MI ceiling (converse, geometry-only) |
| `CLUB` | variational MI upper bound | MI upper bound (learned) |
| `V_cap` | capacity-matched predictive V-information | V-information (capacity-bounded reader accuracy) |
| `J` | whitened-row negentropy separability | ICA separability resource |
| — | Bhattacharyya–Fano error bounds | two-sided decode-error bounds |
| `SDL` | surplus description length | MDL/SDL |
| — | shared spectral capacity | KV secondary diagnostic (NOT `I_G` — call out the distinction) |

NAV TAXONOMY (set by the 2026-06-24 restructure): the topnav is now `index · synthesis · residual · embedding · KV/QKV · defenses`, where the surface groups (residual/embedding/KV) hold ATTACKS ONLY and `defenses` is a flat cross-surface group. `metric-std.html` and `defenses-existing.html` were REMOVED; the bits+readout convention now lives only in `src/talens/report.py`. Probe/registry pages therefore get their OWN new nav group `probes` (parallel to `defenses`), NOT a slot under a surface group and NOT a revived metric page.

steps: (a) write the registry as a dedicated `probes`-registry page (do NOT recreate metric-std.html) — each row = symbol, canonical name, quantity, one-line "what it bounds / why attack-independent", and the source `src/talens/measures/*` module; add a new `probes` navgroup to the canonical topnav and put the registry page in it; (b) sweep every `docs/html/*.html` and replace each probe mention with the canonical name + symbol (lead with the name, keep the symbol for table headers); (c) reconcile synthesis.html's existing probe-glossary paragraph to point at / match the registry (the `metric-std` link was already removed in the restructure).
acceptance: every probe mention in `docs/html/` uses a registry name+symbol verbatim (grep audit); the dedicated registry page exists under a new `probes` navgroup + index; synthesis.html glossary defers to the registry; no page still says "spectral channel-MI" as an umbrella or conflates `I_G` with shared-spectral-capacity.
- [ ] run-phase: c2-probe-registry

### Task 3: restructure resid-capacity-pvi — it is the V_cap probe, not a surface
recipe: consolidate
gpu: false
surface: capacity-pvi-restructure
run_id: c3-capacity-pvi-restructure
gate: review refine-logs/capacity-pvi-restructure/REVIEW_STATE.json
objective: stop `resid-capacity-pvi.html` from masquerading as a surface page and duplicating the input-DP surface. Split its content along the probe/surface boundary: probe methodology → the `V_cap` per-probe page (built in Task 4); measured DP results → `resid-dp-attacks.html` (the actual surface).

problem: `resid-capacity-pvi.html` mixes two unrelated things under a fake "surface" identity: (1) the `V_cap` probe's estimator-repair methodology (capacity-matching fixes the shuffle-control floor −49 → −1.5 bits; the accuracy-primary / bits-auxiliary rationale; Lemma 1 / Prop 2 / Prop 3), which is PROBE documentation; and (2) measured input-DP sweep tables + the depth-decoupling finding (probe tracks attack +0.96 at L0, decouples to −0.21 at L20), which duplicates and belongs on the input-DP SURFACE page. The site already has `resid-dp-attacks.html` as that surface, so the DP tables appear twice.

decision — SPLIT (grilled 2026-06-24):
- The estimator-repair methodology and the accuracy-vs-bits rationale move to the `V_cap` probe page (created in Task 4 — this task hands that content over; do NOT build the probe page here, just stage the content and cross-link).
- The measured input-DP sweep tables + the depth-decoupling finding move into `resid-dp-attacks.html`. The claim node `claim:depth-decoupling-input-dp` stays as-is and is cited from the DP surface page (and from the `V_cap` probe page's Rationale as the canonical decoupling example).
- `resid-capacity-pvi.html` is RETIRED as a surface. NAV/INDEX DROP IS ALREADY DONE (2026-06-24 restructure): it was removed from the residual navgroup and the index card+footer; the file stays on disk, off-nav. This task's remaining work is the CONTENT split + stub (below); when the `V_cap` probe page exists (Task 4), point the stub at it. Keep the old file reachable as a stub linking the two destinations (do not 404 the synthesis.html row that still references it).

steps: (a) move the input-DP measured tables (the at-layer ablation table, the input-DP-by-depth table, the per-ε table) + the depth-decoupling prose into `resid-dp-attacks.html`, reconciling with whatever DP content already lives there (de-duplicate, single source of truth); (b) stage the estimator-repair + rationale + the three propositions for the `V_cap` probe page (write to refine-logs/capacity-pvi-restructure/ as the handoff content Task 4 consumes); (c) retire `resid-capacity-pvi.html` (stub + nav/index edits using the Task-2 `V_cap` name); (d) verify no measured number is lost in the move (diff the relocated tables against the source) and the claim node is cited from its new home; (e) cleanup pass (/humanize → /proofread → /term-audit) on the edited `resid-dp-attacks.html`.
acceptance: the input-DP tables + depth-decoupling finding appear exactly once, on `resid-dp-attacks.html`, with no number changed (diff shown); `resid-capacity-pvi.html` no longer presents itself as a surface (retired/stub) and is gone from the residual navgroup + index (already true since the 2026-06-24 restructure); the V_cap methodology content is staged for Task 4; `claim:depth-decoupling-input-dp` is cited from its new home; cleanup pass run.
- [ ] run-phase: c3-capacity-pvi-restructure

### Task 4: per-probe pages — Algorithm / Method / Rationale + plaintext reference (one per registered probe)
recipe: consolidate
gpu: false
surface: probe-pages
run_id: c4-probe-pages
gate: review refine-logs/probe-pages/REVIEW_STATE.json
objective: give every registered probe its own HTML page so a reader can understand exactly how it is computed, what information property it estimates, on which surface, why it is the matched probe, and what it reads on a clean plaintext model across layers — the descriptions the site currently lacks. COMPLETION RULE: this phase is DONE when every page carries a real plaintext baseline for the probes that have on-disk data PLUS an explicit "queued onto Task 7" placeholder for the two that do not (SDL, shared-spectral-capacity); final placeholder backfill is owned by Task 7, not this phase. Do NOT block on the missing data.

scope: one page per Task-2 registry entry — `I_G`, `CLUB`, `V_cap`, `J`, Bhattacharyya–Fano error bounds, `SDL`, shared-spectral-capacity (7 pages). Use the registry name in the title. The `V_cap` page absorbs the estimator-repair methodology staged by Task 3. Keep flat `docs/html/` naming (e.g. `probe-vcap.html`) — a `probes/` subdir would break the relative `css/site.css` link.

each page has exactly these sections:
- Algorithm — step-by-step how it is computed (inputs → transforms → output in bits), with a detailed diagram of the computation (per DIAGRAM-STYLE.md). Cite the implementing `src/talens/measures/*` module and the model-free test.
- Method — what information-theoretic property it estimates and of what KIND (MI ceiling / MI upper bound / V-information / two-sided error bound / separability); WHICH surface(s) it accepts as input (residual / pooled-embedding / KV / permutation table), with a diagram per surface where more than one applies; and what it bounds (e.g. Fano recovery ceiling, BSS margin).
- Rationale — why this probe is the attack-INDEPENDENT matched measure for its attack/defense (what channel it reads, why it is geometry-only or channel-matched, where it tracks recovery and where it is provably vacuous/shape-blind — cross-link the relevant claim node and the synthesis row).
- Plaintext reference (issue 2) — one measured reference on a CLEAN model (no attack, no defense), across layers where the probe is layer-defined, so the reader sees the probe's baseline reading. Source order (grilled 2026-06-24, keeps Task 4 GPU-free): (1) use the on-disk plaintext-across-layers number where present — `CLUB` (36-layer control sweep, results-chronicles 2026-06-17), `V_cap` (depth sweep), `J` (KV plaintext baseline, layer×kind), `I_G` (embedding), Bhattacharyya–Fano (bnn pool); (2) if none exists, DO NOT run GPU here — queue a small plaintext-probe-bits emission onto Task 7's GPU run (likely needed for `SDL` — MDL was off in the control sweep — and `shared-spectral-capacity`) and backfill the page after Task 7.

diagrams: MUST follow docs/html/DIAGRAM-STYLE.md (research the source paper figure first; trust-zone/surface boundaries; the computation sequence; D3 interactivity per the R2 triggers). depends on Task 2 (canonical names), Task 3 (V_cap staged content), and the shipped DIAGRAM-STYLE.md.
acceptance: 7 probe pages exist, each with Algorithm/Method/Rationale + a plaintext-reference block (real number from disk, or an explicit "queued onto Task 7" placeholder for SDL / shared-spectral-capacity) + at least one compliant `.diagram-frame`; all wired into the new `probes` navgroup (created in Task 2, parallel to `defenses`) + index; each cross-links its claim node and its synthesis.html row; the V_cap page carries the relocated estimator-repair methodology; the list of queued-onto-Task-7 emissions is written to refine-logs/probe-pages/queued-for-utility.md; cleanup pass (/humanize → /proofread → /term-audit) run.
- [ ] run-phase: c4-probe-pages

### Task 5: redo the shallow/missing method diagrams to DIAGRAM-STYLE.md
recipe: consolidate
gpu: false
surface: method-diagrams
run_id: c5-method-diagrams
gate: review refine-logs/method-diagrams/REVIEW_STATE.json
objective: redraw every page that is missing a method diagram or has a shallow one to the depth standard in docs/html/DIAGRAM-STYLE.md, so each surface page shows its trust boundaries, algorithmic/training sequence, and both measurement-loop arms.

backlog (7 pages) with what each diagram MUST show:
- resid-rep2text — last-token residual @L10 → adapter → frozen decoder; length axis; probe `I_G` path (vacuous-capacity annotation).
- resid-gelo — residual rows → fresh row-mix A (κ locus, shield rows) → exposed U; BSS attack path + feature-Gram leak; row-negentropy `J` probe path.
- resid-split — split-layer boundary → PriPert sparsify+perturb (β locus) → released activation; ridge/mlp2 inverter; `I_G` + Fano probe path.
- resid-dp-attacks — input-embedding DP (ε locus) → propagation to depth L; ridge vs Bayes/decoder paths; `CLUB`/`V_cap` probe path. (Note: this page gained the relocated input-DP tables in Task 3 — diagram the surface it now fully owns.)
- kv-cloak — KV rows → channels (feature-mix M / token-mix S·P̂ / mask A) as separate transforms → BSS path; `J` probe; show M as the sole load-bearing channel.
- embed-sgt — pooled embedding → SGT heteroscedastic noise (budget B × shape) → Vec2Text path; `I_G(D)` probe; the shape axis.
- bnn-attack — embedding table → DP release → nearest-neighbour decode; Bhattacharyya upper + Fano lower bound annotations; ALSO restyle this page onto css/site.css (it currently uses its own inline styling and is the one page off the shared stylesheet/nav).

rules (all in DIAGRAM-STYLE.md): R0 research the source paper figure first and adapt it; R1 mandatory depth (trust-zone bands + boundary, numbered train/infer sequence, both loop arms attack→recovery + probe→bits, defense-parameter locus, typed labeled arrows); R2 add D3 interactivity per the trigger rules via the /d3-viz skill. Vendor `docs/html/js/d3.v7.min.js` at the first interactive diagram. Use canonical probe names from Task 2.
acceptance: each of the 7 pages has a compliant `.diagram-frame` passing the DIAGRAM-STYLE.md checklist (no 3-box flow, no unlabeled arrows, both loop arms present); bnn-attack.html on css/site.css with the canonical topnav; cleanup pass run.
- [ ] run-phase: c5-method-diagrams

### Task 6: readout metrics — a legible per-secret readout beside every bits value
recipe: consolidate
gpu: false
surface: readout-metrics
run_id: c6-readout-metrics
gate: review refine-logs/readout-metrics/REVIEW_STATE.json
objective: make the HTML tables reasoning-legible by rendering a per-secret human readout (perplexity / token-F1 / cosine / recovery-rate / AUC) beside every bits value, on synthesis.html and every per-report table — closing the gap where bits move by 1/10 or 1/100 and the reader cannot tell what that means.

problem: the bits + per-secret readout convention ALREADY exists in `src/talens/report.py` (incl. the sub-0.1-bit millibit fix) — note the metric-std.html page that documented it was REMOVED in the 2026-06-24 restructure — but most surface tables still show bare bits. Differences of fractions of a bit between sweep points are illegible without the paired native-units readout.

decision — no-GPU render pass (grilled 2026-06-24). Source each readout in this order: (1) already in `refine-logs/<surface>/runs/*.json` → use it; (2) derivable by the model-free `talens.report` layer from stored measure dicts → recompute (no GPU); (3) genuinely needs the model and is not stored (e.g. a probe's predictive-distribution perplexity) → DO NOT run GPU here; queue it onto Task 7's GPU run and backfill. Apply the metric-std legibility contract verbatim: bits canonical (millibits below 0.1 bit), `bits_kind` tag preserved, `n/a` distinct from measured zero. COMPLETION RULE (no hidden circular dependency on Task 7): this phase is DONE when every bits column carries EITHER a real readout from source-order (1)/(2) OR an explicit "queued onto Task 7" placeholder for source-order (3); final replacement of placeholders is owned by Task 7. This phase must not wait on Task 7.

steps: (a) audit every `.spec` table on every `docs/html/*.html` for bits columns lacking a paired readout; (b) backfill the readout from source-order (1)/(2); (c) write the list of source-order-(3) items to refine-logs/readout-metrics/queued-for-utility.md (merge with Task 4's queue) so Task 7 emits them once; (d) update synthesis.html so every probe-bits and recovery cell carries its native-units companion; (e) cleanup pass on edited pages.
acceptance: every bits column on synthesis.html and the per-surface pages has a paired per-secret readout (real value, or an explicit "queued onto Task 7" placeholder); the millibit/`n/a` legibility contract holds; the Task-7 queue file lists all model-required-missing readouts; no GPU run launched by this task; cleanup pass run.
- [ ] run-phase: c6-readout-metrics

### Task 7: leakage–utility — measure downstream utility + emit the queued model-required gaps (ONLY GPU phase)
recipe: full
gpu: true
surface: utility-tradeoff
run_id: c7-utility
gate: review refine-logs/utility-tradeoff/REVIEW_STATE.json
objective: measure the leakage–utility tradeoff axis that no surface currently records, backfill the TODO utility columns in synthesis.html, AND — as the campaign's single GPU run (models already loaded) — emit the model-required gaps that Tasks 4 and 6 queued, so each defense sweep shows (bits, recovery, utility) and the legibility/plaintext-reference gaps close.

decision — utility is defense-class-aware:
- LOSSY defenses (input-DP, Shredder, PriPert, SGT): measure the protected model's task metric vs the privacy parameter on the SAME sweep points as recovery — perplexity / a downstream-benchmark accuracy for the residual + KV surfaces (Qwen3-4B), retrieval nDCG@10 / Recall@k for the embedding surfaces (GTR). Each sweep row becomes `(bits, recovery, utility)`.
- INVERTIBLE-in-TEE defenses (KV-Cloak, orthogonal GELO, AloePri keymat): utility is lossless by construction — instead report the un-mix reconstruction error (≈0) + the compute/latency overhead.

piggybacked emissions (consume the queue files; one GPU load, no extra phase):
- refine-logs/probe-pages/queued-for-utility.md — plaintext-across-layers probe references missing from disk (expected: `SDL`/MDL, `shared-spectral-capacity`). Emit on a clean model, no attack/defense.
- refine-logs/readout-metrics/queued-for-utility.md — model-required readouts not stored (e.g. probe-distribution perplexity). Emit and hand back for backfill.

required subdeliverables, in priority order (this phase mixes one GPU run with broad report edits — do the GPU/data work first so a mid-phase interruption still leaves the durable artifacts):
  1. DATA (GPU): for each lossy defense, re-run/extend its sweep to also emit the task metric at every parameter value, aligned to the recovery sweep; for each invertible defense, measure recon-error + overhead; consume the two queue files and emit their items in the SAME capture where possible. Write all of it to the canonical dataset `refine-logs/utility-tradeoff/leakage_utility.json`, one row per sweep point with schema: `{surface, defense, param_name, param_value, leakage_bits, bits_kind, recovery, recovery_metric, utility_metric, utility_value, provenance}` (invertible rows set `utility_metric="recon_error"`/`"overhead_ms"`). Write the consumed-queue results to `refine-logs/utility-tradeoff/queue_results.json`.
  2. BACKFILL synthesis.html: populate the utility column per row from `leakage_utility.json` (or explicit TODO + reason where a cell is genuinely unmeasurable).
  3. BACKFILL the placeholders Tasks 4 and 6 left: the probe-page plaintext-reference rows (SDL, shared-spectral-capacity) and the queued model-required readouts.
perf gate: this is the only GPU phase — pass scripts/harness/perf_gate.md (optimal scope, max GPU utilization) before launch; ONE GPU process at a time via run_step.sh; estimate wall-time and trim to representative layers/params first; batch the piggybacked emissions into the same capture where possible.
acceptance: `refine-logs/utility-tradeoff/leakage_utility.json` exists with the schema above (it is Task 12's input contract); `queue_results.json` records both consumed queue files (items emitted + backfilled, or re-marked TODO with a reason); synthesis.html utility column populated (or explicit TODO + reason) for every defense sweep, lossy rows a real task metric, invertible rows recon-error + overhead; the Task-4/Task-6 placeholders are replaced (or re-marked TODO + reason); perf gate passed before launch.
- [ ] run-phase: c7-utility

### Task 8: prove the 4 cross-cutting claims (currently proofs-TODO stubs)
recipe: consolidate
gpu: false
surface: crosscutting-proofs
run_id: c8-crosscutting-proofs
gate: review refine-logs/crosscutting-proofs/REVIEW_STATE.json
objective: write and verify the proofs for the four cross-cutting claim nodes that back synthesis.html, which are currently registered as proof-TODO stubs, and fold each verified proof in full into its claim file.

the four claims (research-wiki/claims/*.md):
- claim:cross-surface-matched-probe-tracks-recovery — a channel-matched attack-independent probe predicts recovery on the majority of surfaces; scope/limits (within-sweep, within-family; calibration non-transfer).
- claim:probe-failure-dichotomy-matched-or-vacuous — every probe-vs-recovery non-correlation resolves to probe-not-matched or vacuous-capacity, not a refutation.
- claim:attack-strength-governs-realized-leakage — a weak attack can decorrelate where a stronger admissible attack re-correlates (the Bayes-optimality / information-efficiency line).
- claim:defense-privacy-is-single-channel-localizable — deployed defenses concentrate privacy in one load-bearing parameter a matched probe localizes.

method: /proof-writer → /proof-checker inline (cross-model jury, never self-certify); fold each verified proof IN FULL into its claim file (remove the `proof: TODO`), per the project rule that verified proofs live in research-wiki/claims/*.md. Each claim's checker run writes a per-claim artifact `research-wiki/claims/<slug>.proof-check.md` recording the checker identity, the PASS/FAIL verdict, and the final (possibly weakened) statement — this is the checkable evidence, not a self-assertion. Where a claim cannot be proved as stated, weaken it to the provable form and bound the gap (do not overclaim). Re-register with /research-wiki.
acceptance: each of the 4 claim files contains a full proof-checked proof or a corrected/weakened statement with its proof, AND a paired `research-wiki/claims/<slug>.proof-check.md` artifact with checker identity + PASS verdict (or a recorded weakening); `proof: TODO` removed from all four; synthesis.html §04 footnotes updated to drop "proof pending" where proven.
- [ ] run-phase: c8-crosscutting-proofs

### Task 9: visual reporting Phase 0 — plotting harness + PLOT-STYLE.md + the A1 baseline
recipe: consolidate
gpu: false
surface: viz-harness
run_id: c9-viz-harness
gate: review refine-logs/viz-harness/REVIEW_STATE.json
objective: build the one shared result-plot idiom (inline SVG: axes, scales, rects, polyline, colorbar) the whole site reuses, lock the visual language in a short PLOT-STYLE.md, and ship the first A1 scatter as the copyable baseline. Everything in Tasks 10–13 depends on this; fan out only after the primitive is locked. (Source: refine-logs/visual-reporting-research/RESEARCH.md, "Group-A rendering recipes".)

problem: of 16 report pages, every SVG is a method `diagram-frame`; NONE renders the bits-vs-recovery relationship — the most load-bearing object in the thesis (the measurement loop's verdict) is shown only as a number in a table. Splitting plot work by surface would re-derive the SVG primitives six times with inconsistent axes; build the primitive once.

decision — split axis = visualization-family, harness-first (grilled 2026-06-24). Distinct from Task 5: Task 5 = method/architecture diagrams (DIAGRAM-STYLE.md); this family = result/data plots (PLOT-STYLE.md). Plots are inline SVG, complete without JS (D3 only where the R2 interactivity triggers fire), and carry `source: results/<file>.json` provenance like the tables. Use canonical probe names (Task 2) and the legible readouts (Task 6) on the axes.

steps: (a) add a results-plot CSS block to `css/site.css` — `.plot-frame` (mirrors `.diagram-frame`), `.plot-cap`, `.axis`, `.tick`, `.gridline`, `.heatmap`/`.cell`, `.colorbar`, plus a sequential color-ramp var; (b) write `docs/html/PLOT-STYLE.md` (linked from STYLE.md) — the plot idiom, the five A-plot types, provenance + accessibility (titles, reduced-motion) rules, and when D3 is warranted; (c) implement the A1 bits-vs-recovery scatter (x=bits, y=recovery, one point per sweep param, polyline in sweep order, Spearman ρ annotated) on `vec2text.html` (its `(bits, recovery)` data is already on disk) as the reference implementation; (d) cleanup pass.
acceptance: `css/site.css` carries the `.plot-frame` block; `docs/html/PLOT-STYLE.md` exists and is linked from STYLE.md; `vec2text.html` shows a compliant A1 scatter with ρ annotated, rendering without JS, with JSON provenance; cleanup pass run.
- [ ] run-phase: c9-viz-harness

### Task 10: visual reporting Phase 1 — A1 bits-vs-recovery scatter across all sweep pages
recipe: consolidate
gpu: false
surface: viz-scatter
run_id: c10-viz-scatter
gate: review refine-logs/viz-scatter/REVIEW_STATE.json
objective: roll the A1 scatter (built in Task 9) across every page that has a `(bits, recovery)` sweep — the thesis-spine plot; highest impact, pure reuse of the Phase-0 primitive.

scope (pages with paired sweep data): resid-split, resid-depth-inversion, resid-dp-attacks (the relocated input-DP sweeps from Task 3), embed-sgt, kv-cloak. Plus synthesis.html where a cross-surface ρ-summary scatter is warranted. Encoding per RESEARCH.md A1: x=bits (canonical, legible per Task 6), y=recovery (native readout), points=sweep params, polyline in sweep order, ρ in a corner; for the multi-attack pages (resid-dp-attacks ridge-vs-learned-vs-Bayes, resid-split ridge-vs-mlp2) plot the attack series together so the re-correlation under a stronger attack is visible.

steps: (a) for each page, extract the `(bits, recovery)` pairs from its results JSON; (b) render the A1 scatter into the Results/Measures section; (c) where multiple attacks exist, overlay series; (d) cross-check each plotted ρ against the number already stated in the page's prose (they must match); (e) cleanup pass.
acceptance: each in-scope page carries a compliant A1 scatter whose annotated ρ matches the page's stated correlation; multi-attack pages overlay their attack series; all render without JS with JSON provenance; cleanup pass run.
- [ ] run-phase: c10-viz-scatter

### Task 11: visual reporting Phase 2 — A2 layer×param heatmap + A5 eigenspectrum/noise-floor
recipe: consolidate
gpu: false
surface: viz-spectral
run_id: c11-viz-spectral
gate: review refine-logs/viz-spectral/REVIEW_STATE.json
objective: add the two "defense moves the floor" plots — the layer×defense-parameter leakage heatmap (A2, Voita regularity pattern) and the eigenspectrum + noise-floor / waterfilling plot (A5) that makes `I_G` legible and localizes where leakage lives.

scope:
- A2 (rows=layer, cols=defense parameter e.g. ε, color=recovery, plaintext column) on resid-depth-inversion (its 9-depth grid is on disk) and resid-dp-attacks (depth×ε from `localdp_depth_*`). Render from EXISTING measured layers — no GPU. If a denser 36-layer grid is wanted later, queue it as an OPTIONAL piggyback, do not block this task.
- A5 (sorted λ_i log-y + σ² line + per-direction ½log₂(1+λ_i/σ²) bits) from `anisotropic_geometry_diagnostic.json`, on vec2text, metric-std (as the `I_G` worked example), and resid-split.

steps: (a) build the A2 `<rect>` grid (reuse the heatmap idiom from kv-accumulation/defenses-existing + the Phase-0 colorbar) from existing per-layer recovery; (b) build the A5 spectrum plot from the eigenspectrum JSON, marking the noise floor and shading the directions above/below it; (c) caption each with the reading ("leakage concentrates in the top-k directions", "depth does not buy privacy"); (d) cleanup pass.
acceptance: A2 heatmap on the two depth pages and A5 spectrum on the three spectral pages, all from on-disk data (no GPU launched), compliant with PLOT-STYLE.md, with provenance + captions; cleanup pass run.
- [ ] run-phase: c11-viz-spectral

### Task 12: visual reporting Phase 3 — A3 Gram/interference heatmap + A4 privacy–utility Pareto
recipe: consolidate
gpu: false
surface: viz-channel-tradeoff
run_id: c12-viz-channel-tradeoff
gate: review refine-logs/viz-channel-tradeoff/REVIEW_STATE.json
objective: add the "which channel leaks / what does it cost" pair — the Gram/interference heatmap (A3, Toy-Models WᵀW) that shows the load-bearing channel, and the privacy–utility Pareto curve (A4) that turns the leakage–utility data from Task 7 into the tradeoff frontier the prose currently only describes.

scope:
- A3 (feature×feature diverging heatmap, before/after defense pair; off-diagonal = interference) on resid-gelo (feature-Gram leak at κ=1) and kv-cloak (feature-mix M as the sole load-bearing channel).
- A4 (x=defense parameter, leakage series + utility series, knee/operating-window marked) on vec2text, embed-sgt, resid-split, and a cross-surface panel on synthesis. DEPENDS ON Task 7's canonical dataset `refine-logs/utility-tradeoff/leakage_utility.json` (schema named in Task 7) — read `{surface, defense, param_value, leakage_bits, utility_value}` rows from it; if a surface's `utility_value` is still TODO after Task 7, plot leakage alone and mark the utility series pending.

steps: (a) build the A3 diverging heatmap from the feature-Gram / feature-mix matrices, showing a clean-vs-defended pair so the channel that collapses is visible; (b) build the A4 dual-series Pareto from Task 7's `(parameter, leakage, utility)` rows, marking the operating window (e.g. vec2text ε≈256–512); (c) caption each with the reading; (d) cleanup pass.
acceptance: A3 heatmaps on gelo + kv-cloak showing the load-bearing channel; A4 Pareto on the three (+synthesis) pages from Task 7 data (or leakage-only with utility-pending noted where Task 7 left a TODO); PLOT-STYLE.md compliant with provenance; cleanup pass run.
- [ ] run-phase: c12-viz-channel-tradeoff

### Task 13: visual reporting Phase 4 (optional) — Group-B selectives
recipe: consolidate
gpu: false
surface: viz-groupb
run_id: c13-viz-groupb
gate: review refine-logs/viz-groupb/REVIEW_STATE.json
objective: add the highest-value Group-B plot now that its data is ready — the cross-family probe×defense matrix (B2, mandatory, data is on disk) — and the per-example difficulty histogram (B1, optional, ONLY if per-example logging already exists). COMPLETION RULE: B2 is required for this phase to pass; B1 is optional and is satisfied either by rendering it (if per-example data is present) or by a one-line deferral note with the reason (data absent). There is no "skip if time-bound" — the phase is small and deterministic.

scope:
- B2 (ρ-annotated cross-family matrix: probe-family × defense, diagonal = matched) from `b3_decoupling_matrix.json` (already on disk) on synthesis and resid-split — directly the matched-vs-vacuous-probe story (cf. Xu et al. 2020 Fig 3b cross-family matrix).
- B1 (pointwise PVI / per-token recovery histogram, frequency-stratified) ONLY if per-example logging is already present; otherwise mark deferred, do not add logging in this task.
- Group-C techniques (t-SNE/UMAP, polytope geometry, PWCCA drift, heavy dashboards) are explicitly OUT per RESEARCH.md.

steps: (a) build the B2 matrix heatmap from the decoupling JSON with ρ cell labels and diagonal highlight; (b) if per-example data exists, build B1; else write a one-line deferral note; (c) cleanup pass.
acceptance: B2 cross-family matrix on synthesis + resid-split from on-disk data; B1 present or explicitly deferred with reason; no Group-C technique added; PLOT-STYLE.md compliant; cleanup pass run.
- [ ] run-phase: c13-viz-groupb

---

## Completed this session (context, NOT ralphex phases)

**HTML nav restructure — DONE manually 2026-06-24 (pre-campaign-C).** Topnav rewritten across all 14
pages to `index · synthesis · residual · embedding · KV/QKV · defenses`: surface groups now hold
ATTACKS ONLY (residual = Rep2Text inversion, depth inversion; embedding = Vec2Text, Bayes-NN; KV/QKV =
source separation), `defenses` is a new flat cross-surface group (differential privacy, GELO, split
inference (PriPert), Stained Glass, KV-Cloak, permutation cover). REMOVED `metric-std.html` (no
dedicated metric page — convention lives in `src/talens/report.py`) and `defenses-existing.html`
(redundant with synthesis.html); fixed all inbound body links in index/synthesis. `perm-cover` moved
under defenses. `resid-capacity-pvi.html` dropped from nav + index (off-nav, file kept for Task 3);
its only remaining link is the synthesis.html overview row. This shifts the nav taxonomy Tasks 2/3/4/6
build on: probe pages get a NEW `probes` navgroup (parallel to defenses), the registry is a dedicated
page (not metric-std), and the capacity-pvi nav-drop is already done.

**Task 1 (recipe-keyed task prompts) — DONE manually 2026-06-24 (verified).** Split
`prompts/task.txt` (the ~150-line monolith) into a shared `_preamble.txt` + four recipe bodies
(`consolidate.txt`, `full.txt`, `experiment.txt`, `theory.txt`); the GPU/DURABLE-RUNS/PERF-GATE
mechanics block now lives ONLY in `full.txt`+`experiment.txt`. `task.txt` is now a tiny DISPATCHER
that ralphex feeds every iteration: it reads the next unchecked phase's `recipe:` and routes the
executor to `_preamble.txt` + `<recipe>.txt` (resolving {{VARS}} via a table it passes through), so
the executor never loads the other three recipe bodies. Chosen over a ralphex binary change because
the Go toolchain isn't on PATH here and a broken binary breaks every campaign; ralphex loads prompts
by fixed filename so the extra files are ignored (no further config needed). Verified: instruction
preservation (no net loss — 4 re-wrap diffs confirmed present via fragment search,
`refine-logs/harness-prompts/instruction-diff.txt`), GPU block absent from consolidate/theory
assemblies and present in full/experiment (`refine-logs/harness-prompts/assembled/*.txt`). Monolith
preserved at `prompts/task.txt.monolith.bak`. Config documented in `ralphex-config/config`.

**Grilling + decisions (2026-06-24).** Five issues folded into this plan via /grill-me: (1) capacity-pvi
split → V_cap probe page + DP relocation (Task 3); (2) per-probe plaintext reference, existing-first +
piggyback (Task 4/7); (3) visual reporting folded as Tasks 9–13, all Group-A, harness-first→viz-family;
(4) readouts as a no-GPU render pass + piggyback (Task 6/7); (5) recipe-keyed task-prompt split (Task 1).

**Visual-reporting literature pass — DONE.** `refine-logs/visual-reporting-research/RESEARCH.md`: read
Voita 2019/2020, Xu 2020, Toy-Models-of-Superposition; identified 5 Group-A result-plot techniques
(A1 bits-vs-recovery scatter, A2 layer×param heatmap, A3 Gram/interference heatmap, A4 privacy–utility
Pareto, A5 eigenspectrum/noise-floor), mapped each to our surfaces + on-disk data, proposed the
harness-first / visualization-family phasing now encoded as Tasks 9–13.

**Harness session drop-off on wait — DONE + installed.** Root cause: ralphex is fresh-per-iteration
(`--print`); the agent ending its turn while a detached run is in flight exits claude and ralphex
cold-restarts the phase memory-less. Fix in the talens ralphex fork (`~/.local/src/ralphex-v1.5.1-talens`):
capture claude `session_id` into `executor.Result.SessionID`; `ClaudeExecutor.SetResumeSession` injects
`--resume <id>`; the task loop warm-RESUMEs the same session when an iteration ends cleanly-but-incomplete
AND a `run_step.sh` `run.pid` is live (`liveRunExists`, signal-0 probe), capped at `resumeMaxConsecutive=6`,
env-gated `RALPHEX_RESUME_ON_LIVE_RUN` (default off). Built, all `./pkg/...` tests pass (+3 new), installed
(old binary backed up `ralphex.bak-pre-resume-*`), enabled in `run_campaign.sh`; `idle_timeout` 45m→90m;
`task.txt` notes the net. Optional follow-up: a phase-level unit test for the resume-arm path; upstream to
the fork's suite.

**Diagram standard — DONE.** `docs/html/DIAGRAM-STYLE.md` (linked from `STYLE.md`), auto-reviewed 8.5/10.
Interactive diagrams via the installed `/d3-viz` skill (vanilla-JS direct-DOM), D3 v7 vendored at
`docs/html/js/d3.v7.min.js` for offline robustness; figure-spec SVG baseline; Plotly skipped. `vec2text.html`
FIG.01 is the compliant exemplar; shared `.box-warn`/`.arrow-warn`/`.diagram-controls`/`.is-active`/
`.is-dim` + reduced-motion CSS already added to `css/site.css`.
