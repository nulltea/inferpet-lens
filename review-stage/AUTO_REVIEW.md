# Auto-review log — Task 10 (c11-viz-spectral): A2 leakage heatmaps + A5 eigenspectrum/noise-floor

## Round 1 (2026-06-25)

### Assessment (Summary)
- Score: 8.5/10
- Verdict: READY (with one wording hardening, applied)
- Backend: codex (gpt-5.5, xhigh). Reviewer sandbox could not read files (bwrap loopback
  denied); artifacts + verbatim cross-checks were pasted inline for verification.

### What was reviewed
Two A2 layer×parameter leakage heatmaps and one shared A5 eigenspectrum worked example,
all rendered from on-disk data (no GPU) by `refine-logs/viz-spectral/gen_plots.py`:
- A2 `resid-dp-attacks.html` — layer {L0,L5,L12,L20} × input-DP ε {plain,4k,1k,768,512,384,256},
  fill = token-id top-1 recovery (`results/localdp_depth_L0_5_12_20.json`).
- A2 `resid-depth-inversion.html` — {ridge,mlp2,NN} × 9 depths, fill = selectivity
  (`runs/full/depth_sweep.json`).
- A5 on `vec2text.html`, `probes-registry.html`, `resid-split.html` — eigenspectrum of the
  error-weighted scatter S (`results/anisotropic_geometry_diagnostic.json`, gemma-2-2b, ε=128),
  σ²=0.0246 noise floor, per-mode bits ½log₂(1+λ/σ²), eff_rank≈7 of 2304, top-10 = 81% energy.

### Faithfulness (verified by reviewer from pasted artifacts)
- A2 dp L0 row matches the page's per-point L0 table within plot rounding: 0.809→0.81, 0.661→0.66,
  0.428→0.43, 0.140→0.14.
- A2 depth matches the per-layer table: ridge L0 0.685→0.69, L32 0.390→0.39; mlp2 L32 0.542→0.54; NN 0.000.
- A5 matches the JSON ε=128 row: eff_rank_S=7.10→"≈7", top10_eval_frac=0.8139→"81%", σ²=0.02464→"0.025", d=2304.

### Reviewer raw response (verbatim)
Score: 8.5/10. Verdict: READY, with one wording hardening I would still make.
Numeric faithfulness checks pass. Weaknesses ranked:
1. A5 honesty acceptable but slightly easy to misread on vec2text — add an explicit parenthetical
   that this is not the GTR Cov(e₀) spectrum and does not resolve the deferred empirical localization.
2. Keep A5 formula language illustrative ("worked example", "representative anisotropic geometry").
3. A2 DP x-axis: clarify the `plain` column is not an ε value (label `plain / ε=∞` or note clean_top1).
No blocking faithfulness issue. Plot-style/accessibility/on-disk-only/inline-SVG validity sufficient.

### Actions taken (post-review hardening — all applied)
1. A5 Read (all 3 pages): appended "It is not the GTR embedding Cov(e₀) spectrum and does not resolve
   vec2text's deferred empirical-localization claim."
2. A2 dp Read: added "The plaintext column is the no-noise baseline (clean_top1, ε=∞), not an ε value."
3. (#2 already satisfied: "worked example" + "gemma-2-2b codebook scatter S" present in the visible text.)
All edits mirrored into the generator `gen_plots.py`; regeneration reproduces the live pages exactly.

### Status
- Stopping: score 8.5 ≥ 6 AND verdict ∈ {ready, almost}. Gate met.
- Difficulty: medium.
