<!-- ralphex-native plan (parsed by `### Task N:` sections); exempt from docs frontmatter.
     BLOCK A — consolidation PILOT. Run SUPERVISED, FOREGROUND:
       scripts/harness/run_campaign.sh docs/plans/campaign-A-consolidate.md
     Task 1 doubles as the harness shakedown. When all 8 are [x] and the site looks right,
     enable the systemd service on campaign-B-expand.md (unattended).
     Design: memory autonomous-campaign-decisions; scripts/harness/README.md. -->

# Plan: Block A — consolidate existing leakage results

## Overview

Harvest the existing ~25 spikes / 8 experiment logs into standardized
claims+proofs+experiment-logs+HTML on the bits-canonical + per-secret-readout metric, surface by
surface. Recipe `consolidate` throughout. Run **supervised, foreground** — Task 1 is the harness
shakedown (validates gate paths, ARIS output locations, Codex auth, GPU teardown). Dead-ends are
first-class jury-gated negative results, not failures. Threat anchor: `WEIGHTS-PUB`. When all
phases are `[x]` and `serve_docs.sh` shows a coherent site, proceed to `campaign-B-expand.md`.

---

### Task 1: metric-standardization + reporting layer (SHAKEDOWN)
recipe: consolidate
gpu: false
surface: metric-std
run_id: a0-metric-std
gate: review refine-logs/metric-std/REVIEW_STATE.json
objective: build/extend the src/talens reporting layer so every probe emits bits canonical + a per-secret human readout (perplexity/token-F1/recovery-rate/cosine/AUC); retrofit existing measures; this is the supervised harness shakedown.
- [x] run-phase: a0-metric-std

### Task 2: residual-stream — capacity-matched PVI
recipe: consolidate
gpu: true
surface: resid-capacity-pvi
run_id: a1-capacity-pvi
gate: review refine-logs/resid-capacity-pvi/REVIEW_STATE.json
objective: consolidate the capacity-matched class-PVI work into claim(s)+proof+experiment-log+HTML; standardize metrics; record where capPVI tracks attack recovery across depth and where it diverges (L20 under input-DP).
- [x] run-phase: a1-capacity-pvi

### Task 3: residual-stream — DP-stronger / info-efficient attacks
recipe: consolidate
gpu: true
surface: resid-dp-attacks
run_id: a2-dp-attacks
gate: review refine-logs/resid-dp-attacks/REVIEW_STATE.json
objective: consolidate the b2 family (L0 Bayes-NN, lpos decoder, propagated-DP) + info-efficient-attacks findings into claims+proofs+log+HTML; standardize metrics; document the MI-decorrelation-is-propagated-DP-specific result and the open stronger-depth-decoder gap.
- [x] run-phase: a2-dp-attacks

### Task 4: pooled-embedding — Vec2Text + spectral channel-MI probe
recipe: consolidate
gpu: true
surface: embed-vec2text
run_id: a3-vec2text
gate: review refine-logs/embed-vec2text/REVIEW_STATE.json
objective: consolidate Vec2Text (forward-model + corrector) and the spectral-channel-MI probe into claim+proof+log+HTML; record the per-position-fails / pooled-succeeds result and I_G(σ) as the matched converse probe under DP.
- [x] run-phase: a3-vec2text

### Task 5: pooled-embedding — BNN error bounds
recipe: consolidate
gpu: true
surface: embed-bnn
run_id: a4-bnn
gate: review refine-logs/embed-bnn/REVIEW_STATE.json
objective: consolidate the geometry-only union-Bhattacharyya + Fano L0 error-bound probe; run/validate bnn_error_bounds_validation; fold the verified proof into the claim file; HTML page.
- [x] run-phase: a4-bnn

### Task 6: permutation-cover — VMA + cover-break
recipe: consolidate
gpu: true
surface: perm-cover
run_id: a5-perm-cover
gate: review refine-logs/perm-cover/REVIEW_STATE.json
objective: consolidate VMA full-sorted-matcher >> RowSort-64 and the anchor cover-break into claim(s)+proof+log+HTML; standardize metrics; frame the permutation channel as the thesis confirmation.
- [x] run-phase: a5-perm-cover

### Task 7: defenses — aloepri + shredder (existing)
recipe: consolidate
gpu: true
surface: defenses-existing
run_id: a6-defenses
gate: review refine-logs/defenses-existing/REVIEW_STATE.json
objective: consolidate the implemented aloepri + shredder defense evals into a leakage–utility story (bits + readout vs parameter), claim(s)+log+HTML; note which surfaces each touches.
- [ ] run-phase: a6-defenses

### Task 8: site assembly + index + zero-leakage reference
recipe: consolidate
gpu: false
surface: site-assembly
run_id: a7-site
gate: review refine-logs/site-assembly/REVIEW_STATE.json
objective: assemble docs/html/index.html (one card/row per surface page from Tasks 1–7, shared topnav) per docs/html/STYLE.md, including the cryptographic zero-leakage reference box (Euston/Fision/TwinShield-full, documented not swept); verify serve_docs.sh renders the full site.
- [ ] run-phase: a7-site
