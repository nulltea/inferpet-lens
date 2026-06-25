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
objective: run the cross-layer local-DP sweep (ridge + non-linear decoder + CLUB + V_cap at L0/L5/L12/L20) so Task 3's unified table has a real non-linear-attack column at every depth. IMPLEMENTATION IS DONE: the reusable, /ponytail-clean, /auto-review-loop-passed (codex 8.5/10 "ready") eval `scripts/evals/dp_leakage_sweep.py` (promoted from the retired spikes b2_propagated_dp + b2_lpos_decoder) produces the WHOLE grid self-consistently in one sweep — so Task 2's remaining work is the perf-gated RUN, not new code. Use the eval, not a spike.
pointers — consume: `scripts/evals/dp_leakage_sweep.py` (the reusable sweep; ATTACKS/PROBES registries, CLI lists). produce: `refine-logs/dp-decoder-grid/dp_leakage_sweep.json` — per `{layer, epsilon}` record with ridge + decoder top-1 recovery + selectivity, CLUB bits, V_cap bits + reader_top1_acc + perplexity readout, plus per-(layer,attack) `ρ(sel, probe)`; ALL self-consistent (one tool, one ε grid). do-not-redo: do NOT re-run the spikes; do NOT touch at-layer noise (dropped — local DP only); ONE GPU process via run_step.sh, perf-gate (fast-iterate 1 layer/1 ε pilot) FIRST. recipe-fit: `experiment` = pure data; no page edit here (Task 3 owns the page); /result-to-claim + /experiment-audit then /auto-review-loop.
decision: decoder family = the same deep / noise-aware decoder that re-correlated at L20 (b6), trained on σ-matched (noised-representation → token) pairs at each layer; ε grid = the 7 points of `localdp_depth`; single seed (matches the existing sweep; multi-seed CIs remain the named firm-up, not this task); LOCAL DP only (noise at the input embedding, propagated to the observed layer).
steps: (a) PERF GATE — fast-iterate pilot first: `run_step.sh dp-pilot -- scripts/run_in_rocm.sh python3 scripts/evals/dp_leakage_sweep.py --layers 12 --epsilons inf,256 --out refine-logs/dp-decoder-grid/pilot.json`, confirm finite bits + pool size + sigma_convention in the JSON and estimate wall-time; (b) full run `TALENS_SURFACE=dp-decoder-grid run_step.sh dp-sweep -- scripts/run_in_rocm.sh python3 scripts/evals/dp_leakage_sweep.py --layers 0,5,12,20 --epsilons inf,1024,512,256 --attacks ridge,decoder --probes club,vcap --out refine-logs/dp-decoder-grid/dp_leakage_sweep.json` (ONE GPU process, serial); (c) /result-to-claim + /experiment-audit (probe≠attack: ridge/decoder are ATTACKS, CLUB/V_cap independent probes — the eval already separates them); (d) /auto-review-loop.
acceptance: `refine-logs/dp-decoder-grid/dp_leakage_sweep.json` exists with ridge + decoder recovery+selectivity and CLUB + V_cap (bits + perplexity) at L0/L5/L12/L20 across the ε grid, plus per-(layer,attack) ρ(sel, probe); integrity audit confirms probes do not consume attack output; perf gate passed; gate verdict written.
- [ ] run-phase: d2-dp-decoder-grid

### Task 3: overhaul resid-dp-attacks.html → cross-surface "Differential privacy" page
recipe: consolidate
gpu: false
surface: dp-page-overhaul
run_id: d3-dp-overhaul
gate: review refine-logs/dp-page-overhaul/REVIEW_STATE.json
objective: rewrite `docs/html/resid-dp-attacks.html` into the cross-surface Differential-Privacy page per the 2026-06-25 grill: local DP as the single scheme across observation depth, one probe-first cross-layer table with linear + non-linear attacks, all surfaces' DP results aggregated, probe pages linked.
pointers — consume: Task 2's `refine-logs/dp-decoder-grid/dp_leakage_sweep.json` (ridge+decoder+CLUB+V_cap+perplexity, self-consistent across layers); the embedding-DP data on disk (BNN: `refine-logs/bnn-error-bounds/`; Vec2Text: `results/v2t_dp_sweep.json` / `results/spectral_mi_probe_eval.json`); the probe pages `probe-*.html` (Task-C registry). produce: the overhauled page (keep the filename `resid-dp-attacks.html` so the `defenses` navgroup link still resolves; update the H1/title). do-not-redo: do NOT reintroduce "at-layer noise" as a DP variant; do NOT duplicate the full BNN/Vec2Text analyses (summarize + link). recipe-fit: `consolidate` = edit page + cleanup + auto-review; no new claim (the depth-decoupling claim already exists).
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

### Task 4: re-run the DP decoder grid on full data with the R2 decoder + refresh the page (GPU)
recipe: full
gpu: true
surface: dp-decoder-r2
run_id: d4-dp-decoder-r2
gate: review refine-logs/dp-decoder-r2/REVIEW_STATE.json
objective: re-run `scripts/evals/dp_leakage_sweep.py` on the FULL 512-prompt corpus (≈3× the 160-prompt data — the binding constraint per the decoder research) with the R2 decoder now in the eval (GELU, gated zero-init skip, FROZEN ridge-warm-started linear path, narrow h≈384, early-stop on a disjoint val split), grid L0/5/12/20/25 × ε, then refresh FIG.02 + the table on resid-dp-attacks.html with the new numbers.
pointers — consume: the updated `scripts/evals/dp_leakage_sweep.py` (R2 decoder + warm-start, implemented 2026-06-25). produce: `refine-logs/dp-decoder-r2/dp_leakage_sweep.json` and the refreshed FIG.02 data + collapsed table on `docs/html/resid-dp-attacks.html`. do-not-redo: do NOT widen h (narrow is correct); do NOT re-derive the decoder (it's done); one GPU process via run_step.sh; perf-gate (fast-iterate pilot) FIRST; this is the full version of the data already partially shown (160 prompts) on the page. recipe-fit: `full` GPU mechanics + report refresh; no new claim (the affine-saturation claim already exists).
steps: (a) perf-gate fast-iterate pilot (1 layer / 2 ε); (b) full run `--layers 0,5,12,20,25 --epsilons inf,1024,512,256 --attacks ridge,decoder --probes club,vcap --max-prompts 512 --out refine-logs/dp-decoder-r2/dp_leakage_sweep.json`; (c) update the FIG.02 embedded data + the collapsed table values + the L25 row; (d) /result-to-claim + /experiment-audit; (e) cleanup + render check.
acceptance: new JSON exists with decoder ≥ ridge at every (layer, ε) (the frozen-warm-start guarantee — verify), L25 included, on 512 prompts; FIG.02 + table on resid-dp-attacks.html refreshed with the new numbers + provenance; the L12-valley / L20-rebound depth shape and the L25 point are visible; perf gate passed.
- [ ] run-phase: d4-dp-decoder-r2

### Task 5: BeamClean — LM-prior beam-decode attack (the real stronger attack under DP)
recipe: full
gpu: true
surface: dp-beamclean
run_id: d5-dp-beamclean
gate: review refine-logs/dp-beamclean/REVIEW_STATE.json
objective: implement and evaluate a BeamClean-style attack — a beam decode over token positions that fuses (i) the per-token affine likelihood from the ridge / tuned-lens map `E·(A r + b)` with (ii) a frozen small-LM prior over the token sequence and (iii) the published DP noise model (C, σ). Per the decoder research this is the genuinely stronger attack under DP — its advantage over nearest-neighbour GROWS with noise (BeamClean arXiv:2505.13758: 86% vs 18% at high Laplace noise). Per-vector regressors saturate; the LM prior is where recovery comes from.
pointers — consume: the affine/ridge per-token likelihood (`_ridge_W` / tuned-lens), a frozen LM prior (gemma-2-2b itself or a small prior LM), the DP noise model. produce: a NEW reusable attack module under `scripts/evals/` (NOT a spike) — ideally an `--attacks beamclean` option in `dp_leakage_sweep.py` or a sibling `scripts/evals/beamclean_decode.py`; plus `refine-logs/dp-beamclean/*.json` results. do-not-redo: NOT a spike — clean reusable code; /ponytail + /auto-review-loop BEFORE running (standing rule); one GPU, perf-gate first. recipe-fit: `full` — implement → review → run → claim → page.
method: /ponytail implement (reuse the affine likelihood + a frozen LM logits call as the prior; beam over positions; fuse log p_affine + λ·log p_LM, calibrate λ; incorporate the Gaussian/Laplace noise model in the likelihood) → /auto-review-loop → perf-gate → run a DP ε-sweep at a representative layer → /result-to-claim + /experiment-audit (it IS an attack; keep probes independent) → add to resid-dp-attacks.html (a third attack series) + write the claim.
acceptance: a reviewed, reusable BeamClean attack exists under scripts/evals/ (passed /auto-review-loop); a DP ε-sweep shows it beats ridge and the R2 decoder, with the gap WIDENING as ε shrinks (high noise); results in refine-logs/dp-beamclean/; resid-dp-attacks.html gains the LM-prior attack series + a claim; perf gate passed.
- [ ] run-phase: d5-dp-beamclean
