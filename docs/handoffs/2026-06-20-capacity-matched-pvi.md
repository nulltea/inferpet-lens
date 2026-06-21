---
type: handoff
status: current
created: 2026-06-20
updated: 2026-06-20
tags: [PVI, capacity-matched, leakage-measure, calibration, DP-sweep, depth-decoupling, auto-review]
companion: [2026-06-18-independent-vfamily-attack-correlation]
supersedes: [2026-06-18-independent-vfamily-attack-correlation]
---

# Handoff: capacity-matched class-PVI — the independent family, fixed (scoped)

## The question (from the 2026-06-18 handoff) and the answer

**Q:** find a V-family that is (a) independent of the inversion attack, (b) well-behaved
(no overfit), (c) tracks attack success, (d) explanatory — class-PVI had (a) but failed (b)/(c).

**A (achieved, scoped + honest):** class-PVI's failure was the **estimator regime**
(`d>n_val`), not the token-id family. Capacity-matching fixes it; the robust fixed measure
is the **reader's token-id accuracy** (PVI-in-bits is only partially rescued); and where it
diverges from the attack that divergence is a **measurable depth-localised quantity**. Full
numbers + conclusions are in `docs/dev/results-chronicles.md` (2026-06-20 entry) and
`refine-logs/EXPERIMENT_RESULTS.md` — **do not duplicate; read those.**

## What was built (branch `capacity-matched-pvi`, 2 commits, UNPUSHED)

- `src/talens/measures/vinfo_capacity.py` — `v_information_capacity(X, y, family, dim, l2)`,
  families {pca_softmax, randproj_softmax, gauss, knn}; dim-reduce (GPU covariance-eigh) then
  cheap reader over token-id. Returns PVI bits **and** `reader_top1_acc`. Never touches the
  embedding table → independent by construction. Exported in `measures/__init__.py`.
- `tests/test_vinfo_capacity.py` (9/9; full suite 60/60).
- Spikes: `scripts/spikes/{diag_capacity,diag_nondp,analyze_faithfulness}.py`;
  `localdp_runner.py` extended (`--capacity-family/-dim/-l2`, `--every-n`, logs `dp_cap_acc`).
- Process artifacts: `idea-stage/IDEA_REPORT.md`, `refine-logs/{EXPERIMENT_PLAN,RESULTS,TRACKER}.md`,
  `review-stage/AUTO_REVIEW.md` (+REVIEW_STATE.json). Auto-review (gpt-5.5 xhigh) 3 rounds: **5→6.5→7/10, scoped-established**.
- Raw outputs (gitignored): `results/{capacity_screen,capacity_screen_dims,nondp_intervention*,
  localdp_m2_*,localdp_depth_L0_5_12_20}.json`.

## Headline results (one-liners; tables in the chronicle / EXPERIMENT_RESULTS)

1. Capacity-matching removes the catastrophe: shuffle floor **−49 → −1.5 b**
   (**dim-anchored, NOT l2-anchored** — l2 only trades signal), monotone, **~0.3× class-PVI cost**.
   gauss reader fails; **pca_softmax** is the pick.
2. **Readout matters**: reader **accuracy** (bounded) tracks the attack at ρ **0.82–1.0** under
   PCA-ablation / iso-noise (all layers); **PVI-bits** is fragile (the −48 floor & "rise under
   noise" are unbounded-log-loss artifacts). Report accuracy primary, bits auxiliary; **do NOT
   call accuracy "V-information"** (reviewer's main framing fix).
3. **Depth-resolved decoupling under input-DP** (the explanatory result): ρ(fixed-PVI, TTRSR) =
   **+0.99 (L0) → +0.68 (L5) → +0.43 (L12) → −0.21 (L20)**; CLUB shows the same gradient
   (0.96/0.96/0.89/0.29). DP noise is injected at the embedding; propagation depth decouples
   token-id decodability from embedding-reconstruction. Input-DP protects embedding geometry
   (attack target) before token-identity (measure target). Unfixed class-PVI is within-layer
   anti-correlated everywhere.

## Non-blocking next steps (reviewer R3 polish list — none change the verdict)

1. **Calibration diagnostic** to *prove* the log-loss-artifact story: report NLL / confidence /
   ECE / reliability on the bad cells (iso-noise L5/L12; input-DP L20). Cheap, model-free on
   cached capture.
2. **Stats framing**: report within-layer / macro-average ρ as PRIMARY; partial-ρ|r as
   secondary (class-PVI's +0.613 partial|r shows it can mislead under layer×ε structure).
3. **Cross-model replication** (currently gemma-2-2b only): cheapest subset — one mid layer,
   pca_ablate + iso_noise, accuracy readout, on another cached model.
4. **Second/third defense (plan B4)**: obfuscation Transform + split-depth runner (the depth
   sweep already doubles as split-depth; obfuscation is the missing family).
5. **dim16 sensitivity** (floor −1.10 ∈ band vs dim64 −1.53 just outside) — confirm tracking
   holds at dim16 to claim band compliance.

## Gotchas / discipline (also in auto-memory)

- **Heavy runs via `scripts/run_in_rocm.sh` only.** Validate GPU saturation before launching;
  if a run exceeds ~10 min, inspect (CPU% vs `rocm-smi --showuse`).
- **PCA = covariance-eigh on GPU, never a full SVD** (`vinfo_capacity._pca_basis`): a full SVD
  CPU-thrashed 26 min at idle GPU; cov-eigh is 11s→0.44s. Kill leftover containers by **explicit
  ID** (`docker kill <id>`) — `--filter ancestor` + `pkill` left a zombie stealing the GPU once.
- Fast iteration: `--layers 12 --every-n 2`. Model-free fast loop on the cached capture
  (`results/capture_cache/capture-4ca8a33e16bfbec9.pt`, layers {5,12,20}; L0/20 etc. need a fresh
  capture via `localdp_runner.py`).
- Codex MCP sandbox here **cannot read repo files** — paste artifacts inline when reviewing.

## Suggested skills for next session

- **`/run-experiment`** or the spike scripts directly for the polish runs (1)–(4).
- **`/auto-review-loop`** to push 7→8+ once the calibration diagnostic + cross-model + 2nd
  defense land (resume: `review-stage/REVIEW_STATE.json` is `completed`; start fresh).
- **`/paper-plan`** / `/paper-writing` — the scoped claim is established enough to outline
  (frame around the capacity-matched token-id reader + the depth-decoupling, per the reviewer).
- Decide whether to **push `capacity-matched-pvi` + open a PR** (currently local only).
