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
