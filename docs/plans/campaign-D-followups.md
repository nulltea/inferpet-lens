<!-- ralphex-native plan (parsed by `### Task N:` sections); exempt from docs frontmatter.
     BLOCK D — campaign-C intent-miss follow-ups. Run supervised or unattended:
       scripts/harness/run_campaign.sh docs/plans/campaign-D-followups.md
     Inherits all campaign-C prereqs (recipe-keyed prompt split; HTML nav restructure incl. the
     `probes` navgroup; DIAGRAM-STYLE.md + PLOT-STYLE.md; the gate-path reconcile in _preamble STEP 4).
     EXEC ORDER = FILE ORDER. Each task carries its full spec inline; only the `- [ ] run-phase:` line
     is parsed for completion.

     WHY THIS CAMPAIGN EXISTS — a campaign-C failure pattern: two phases passed their cross-model gate
     GREEN while missing the user's actual intent, because the gate validates SPEC-conformance, not
     INTENT-conformance. Each task below therefore carries a CONCRETE, OBSERVABLE (grep-checkable)
     acceptance and explicitly names the "audit-says-done" escape the C-phase took, so the gate cannot
     pass on a reframing. -->

# Plan: Block D — campaign-C intent-miss follow-ups

## Overview

Campaign-C completed all 12 phases, but two deliverables did not match what was asked:

1. **Interpretable probe-bit readouts (Task 1)** — issue-4's real intent was to make the PROBE BITS
   legible (perplexity / effective-candidates), since bits move by 1/10–1/100 of a bit and "3850 bits"
   means nothing to a reader. Campaign-C Task 6 instead ran as an AUDIT, concluded the bits↔recovery
   pairing "already exists" (recovery in an adjacent column), made only a millibit fix on bnn + an I_G
   column on rep2text, and passed its gate 9/10. The interpretable translation of the bits themselves
   was never added; perplexity appears only as a Task-7 protected-model UTILITY metric, not as a
   probe-bits readout. This task does the real pass.

2. **Cross-surface DP page overhaul (Tasks 2 + 3)** — `resid-dp-attacks.html` is mislabeled and
   residual-only. Decisions grilled 2026-06-25: (a) it should AGGREGATE all surfaces' DP results, not
   just residual; (b) the only real DP scheme is LOCAL DP (Gaussian mechanism on the input embedding),
   with OBSERVATION DEPTH as the axis (L0 = released noised embedding, L>0 = same noise propagated) —
   the "at-layer noise" regime is NOT DP and is DROPPED from the page (relocated to a control note);
   (c) the headline table must be ONE cross-layer table, probe columns FIRST then attack columns
   (probes precede + predict the attack), with both a linear (ridge) and a non-linear (decoder) attack
   across L0/L5/L12/L20. The non-linear decoder under local DP exists only at L20 today, so Task 2 is a
   GPU experiment to fill L0/L5/L12; Task 3 rewrites the page.

Known additional follow-up (NOT yet a task — add on request): the **CCG-style per-category separability
heatmap** (Voita `regularity-min.png`). Campaign-C Task 10/11 rendered a layer×ε *recovery* heatmap (A2)
but dropped the per-category SEPARABILITY semantics that was the seed example; see
`refine-logs/visual-reporting-research/RESEARCH.md` lines 34 / 113–122.

---

### Task 1: interpretable probe-bit readouts (translate the bits, not just pair recovery)
recipe: consolidate
gpu: false
surface: readout-interpretable
run_id: d1-readout-interpretable
gate: review refine-logs/readout-interpretable/REVIEW_STATE.json
objective: beside every PROBE bits value in `docs/html/*.html`, render a companion that makes the BITS themselves legible to a reader (an interpretable translation of the leakage magnitude), so a "3850 bits" or a "0.3 → 0.4 bit" sweep step is understandable at a glance. This is issue-4's actual intent, which campaign-C Task 6 did not deliver.
pointers — consume: the bits + recovery + reader-accuracy values ALREADY on disk and in-page (no new measurement); `src/talens/report.py` (the millibit / per-secret-readout convention); campaign-C Task 6's `refine-logs/readout-metrics/AUDIT.md` (the table-by-table inventory of every bits-bearing table — reuse it as the worklist). produce: an inline interpretable companion on each probe-bits cell + `refine-logs/readout-interpretable/MAPPING.md` (per probe family: which companion, how derived). do-not-redo: do NOT re-audit and conclude "recovery column already pairs the bits → done" — that is exactly the campaign-C Task-6 escape this task exists to fix; the adjacent recovery column does NOT count as the bits-translation companion. No GPU. recipe-fit: `consolidate` = edit pages + cleanup + auto-review; no research claim.

problem: campaign-C Task 6 treated "bits has a same-row recovery column" as satisfying issue 4 and made ~no visible change (4 files, tiny diff). The reader still sees bare bits whose magnitude and fractional differences are illegible — the precise pain issue 4 named ("differences are often in 1/10 or 1/100 of a bit; this hurts understanding"). A recovery number (token-F1, recovery-rate) is the ATTACK's result, not a translation of the probe's bits.

decision — add a bits-TRANSLATION companion, derived no-GPU, per probe family:
- token-identity probes (`V_cap`, `CLUB` on a token-id channel, `SDL`): companion = effective candidates remaining `K_eff = 2^(H_prior − I)` and/or perplexity, where `I` is the reported leakage bits and `H_prior = log2(pool size)`. Render e.g. `3.5 bits (≈11 of 2048 tokens remain)`. Reader top-1 accuracy may sit beside it but is not the companion.
- large-magnitude MI probes (`I_G`, `CLUB` in the thousands of bits): raw `2^I` is meaningless — companion = the leakage FRACTION `I / H_ceiling` (0–1) and/or the effective independent-dimension count (λ_i above the noise floor). Render e.g. `3850 bits (0.62 of channel capacity; ≈190 eff. dims)`.
- `J` (negentropy separability) and Bhattacharyya–Fano already carry an interpretable readout (margin-vs-floor; recovery error-band) — leave them, but verify they read legibly.
- FRACTIONAL-BIT LEGIBILITY: anywhere consecutive sweep cells differ by < 0.1 bit, render the value or the step in millibits (per the `report.py` contract) so the change is visible rather than collapsing to two decimals.
The companion is derived from already-stored quantities (bits, pool size, eigenspectrum, accuracy) — model-free. If any companion genuinely needs the model and is not derivable, render the derivable `K_eff`/fraction instead and note the gap in MAPPING.md (do NOT open a GPU run for it).

steps: (a) take the worklist from `refine-logs/readout-metrics/AUDIT.md` (every bits-bearing `.spec` table across `docs/html/*.html`, incl. synthesis §03, the surface pages, and the 7 `probe-*.html` pages); (b) for each, add the family-appropriate bits-translation companion inline in the bits cell (or an explicit adjacent companion column clearly labeled as a bits-translation, distinct from recovery); (c) apply the sub-0.1-bit millibit rule to sweep tables; (d) write `refine-logs/readout-interpretable/MAPPING.md` enumerating, per probe family, the companion used + the no-GPU derivation; (e) cleanup pass (/humanize → /proofread → /term-audit) on edited pages.
acceptance (CONCRETE + grep-checkable — an audit concluding "already paired" does NOT pass):
- every `probe-*.html` page and every probe-bits `.spec` table on synthesis.html + the surface pages shows an inline bits-translation companion (`K_eff` / perplexity / leakage-fraction / eff-dims) that is NOT the recovery column — verify by grep that the companion token (e.g. `eff.`/`of channel capacity`/`tokens remain`/`perplexity`/`ppl`) appears in the bits-bearing tables on each updated page, and report the per-page count;
- at least one sweep table whose adjacent bits differ by < 0.1 bit now renders the step in millibits;
- `refine-logs/readout-interpretable/MAPPING.md` exists and names the companion + derivation per probe family;
- the diff is substantive (not the ~4-file no-op of Task 6) — list the pages changed;
- cleanup pass run.
- [ ] run-phase: d1-readout-interpretable

### Task 2: DP non-linear decoder grid — fill the decoder × {L0,L5,L12} under local DP (GPU)
recipe: experiment
gpu: true
surface: dp-decoder-grid
run_id: d2-dp-decoder-grid
gate: review refine-logs/dp-decoder-grid/REVIEW_STATE.json
objective: train a non-linear, noise-aware decoder under LOCAL DP at layers L0, L5, L12 (L20 already exists in `results/b6_strong_decoder.json`), on the SAME ε sweep as the existing ridge+probe local-DP depth sweep, so Task 3's unified cross-layer table has a real non-linear-attack column at every depth. This is the "live frontier — stronger depth decoder for propagated-DP, not yet built" named in `refine-logs/resid-dp-attacks/RESULTS_STANDARDIZED.md`.
pointers — consume: the existing local-DP depth sweep (`results/localdp_depth_L0_5_12_20.json` — ε grid, ridge recovery, CLUB, V_cap) and the L20 decoder recipe (`scripts/spikes/b6_strong_decoder.py`). produce: `refine-logs/dp-decoder-grid/decoder_by_layer.json` — rows `{layer, epsilon, decoder_recovery, decoder_selectivity}` for L0/L5/L12, ALIGNED to the existing ridge+probe ε grid, plus per-layer `ρ(decoder_selectivity, V_cap)` and `ρ(decoder_selectivity, CLUB)`. do-not-redo: do NOT recompute L20 (reuse b6); do NOT touch at-layer noise (dropped — local DP only); ONE GPU process via run_step.sh, perf-gate first. recipe-fit: `experiment` = pure data; no page edit here (Task 3 owns the page); /result-to-claim + /experiment-audit then /auto-review-loop.
decision: decoder family = the same deep / noise-aware decoder that re-correlated at L20 (b6), trained on σ-matched (noised-representation → token) pairs at each layer; ε grid = the 7 points of `localdp_depth`; single seed (matches the existing sweep; multi-seed CIs remain the named firm-up, not this task); LOCAL DP only (noise at the input embedding, propagated to the observed layer).
steps: (a) PERF GATE (scripts/harness/perf_gate.md) — trim to representative ε first, estimate wall-time; (b) via `TALENS_SURFACE=dp-decoder-grid run_step.sh ...` run the decoder at L0/L5/L12 over the ε grid, GPU-wrapped, serial; (c) emit `decoder_by_layer.json` aligned to the ridge+probe sweep; (d) compute the per-layer ρ(decoder, V_cap/CLUB); (e) /result-to-claim + /experiment-audit (probe≠attack: the decoder is the ATTACK, V_cap/CLUB are the independent probes); (f) /auto-review-loop.
acceptance: `refine-logs/dp-decoder-grid/decoder_by_layer.json` exists with decoder recovery + selectivity at L0/L5/L12 across the ε grid, aligned to the existing ridge+probe sweep; per-layer ρ(decoder, probe) reported; integrity audit confirms the decoder output is not fed into the probes; perf gate passed; gate verdict written.
- [ ] run-phase: d2-dp-decoder-grid

### Task 3: overhaul resid-dp-attacks.html → cross-surface "Differential privacy" page
recipe: consolidate
gpu: false
surface: dp-page-overhaul
run_id: d3-dp-overhaul
gate: review refine-logs/dp-page-overhaul/REVIEW_STATE.json
objective: rewrite `docs/html/resid-dp-attacks.html` into the cross-surface Differential-Privacy page per the 2026-06-25 grill: local DP as the single scheme across observation depth, one probe-first cross-layer table with linear + non-linear attacks, all surfaces' DP results aggregated, probe pages linked.
pointers — consume: Task 2's `decoder_by_layer.json`; the existing ridge + CLUB + V_cap local-DP depth sweep; the embedding-DP data on disk (BNN: `refine-logs/bnn-error-bounds/`; Vec2Text: `results/v2t_dp_sweep.json` / `results/spectral_mi_probe_eval.json`); the probe pages `probe-*.html` (Task-C registry). produce: the overhauled page (keep the filename `resid-dp-attacks.html` so the `defenses` navgroup link still resolves; update the H1/title). do-not-redo: do NOT reintroduce "at-layer noise" as a DP variant; do NOT duplicate the full BNN/Vec2Text analyses (summarize + link). recipe-fit: `consolidate` = edit page + cleanup + auto-review; no new claim (the depth-decoupling claim already exists).
decisions (grilled 2026-06-25):
1. TITLE → "Differential privacy" (cross-surface). The masthead/H1 drop the residual-only framing.
2. CONCEPTUAL SPINE: LOCAL DP (Gaussian mechanism on the input embedding) is THE scheme; OBSERVATION DEPTH is the axis — L0 = the released noised embedding, L>0 = the same noise propagated through L blocks (one scheme, not two). Central DP = out of scope (one line). DROP the at-layer regime entirely: remove R2, R3, the §03 at-layer rows, the at-layer/PCA-ablation/isotropic-noise table in R7, and all "at-layer" glossary/overview/method/conclusion text. RELOCATE that representation-space-noise control material to `research-wiki/experiments/representation-space-noise-control.md` (it is the evidence that the decorrelation is propagation-specific — preserve it, do NOT delete) and link it as a one-line "see also (control, not DP)".
3. HEADLINE RESULTS = ONE unified cross-layer table, columns PROBE-FIRST then attacks: `layer | CLUB (bits) | V_cap | ridge recovery | non-linear decoder recovery` (+ the per-secret readout per the metric convention), rows L0 / L5 / L12 / L20. Probe columns come first to show the probes precede and predict the attack; the table shows probes stay high while ridge (linear) collapses/decorrelates and the non-linear decoder (Task 2 data) accesses more of the preserved information across depth. SDL stays auxiliary (footnote or a muted aux column).
4. L0 ROW: ridge + non-linear decoder (the Task-2 L0 data). Bayes-NN MOVES OUT of the residual L0 table into the embedding section — it is the embedding-surface channel optimum, not a residual decoder (per the grill: "BNN is separate, it operates on embeddings").
5. EMBEDDING SECTION (Gaussian DP): a compact DP-cut summary table each for BNN bounds (I_G / Bhattacharyya–Fano + error-band vs ε) and Vec2Text (I_G + token-F1 vs ε), each LINKING its dedicated page (`bnn-attack.html`, `vec2text.html`) — no duplication of their full analysis.
6. PROBES SECTION: every probe name links its `probe-*.html` page (CLUB→probe-club, V_cap→probe-vcap, SDL→probe-sdl, I_G→probe-ig, Bhattacharyya–Fano→probe-bhattacharyya-fano).
7. KEEP the depth-decoupling finding (V_cap tracks ridge at L0, attenuates/decouples with depth; `claim:depth-decoupling-input-dp`), reframed WITHOUT the at-layer contrast — it now reads as a property of local DP across depth.
8. KV / permutation: one line that they use non-DP defenses and are out of scope.
9. cleanup pass (/humanize → /proofread → /term-audit); re-open to confirm it renders; keep the Method diagram but strip its at-layer/"three regimes" content.
steps: (a) restructure per decisions 1–8; (b) build the unified table from the ridge+probe sweep + Task-2 decoder_by_layer.json; (c) write the relocated control note; (d) wire probe-page + embedding-page links; (e) cleanup + render check.
acceptance: H1/title is "Differential privacy" (cross-surface); grep shows NO "at-layer" framed as a DP scheme remains on the page; ONE unified cross-layer table exists with probe columns BEFORE attack columns and a populated non-linear decoder column at L0/L5/L12/L20 (from Task 2); the embedding section carries BNN + Vec2Text DP summaries that link their pages; every probe name on the page links its `probe-*.html`; the at-layer material exists at `research-wiki/experiments/representation-space-noise-control.md` (relocated, not deleted) and is linked as a control; cleanup pass run; page renders.
- [ ] run-phase: d3-dp-overhaul
