# Auto Review — Block-C visual-reporting Phase 0 (plotting harness)

Surface: viz-harness · run_id c9-viz-harness · backend codex (xhigh) · difficulty medium

## Round 1 (2026-06-25)

### Assessment (Summary)
- Score: 8/10
- Verdict: almost
- Key criticisms: no data-fidelity defect (reviewer independently re-derived all five A1 SVG
  coordinates + the 479-bit cap line and confirmed they match the JSON and the B8 table; ρ=+1.00
  matches `c1_spearman` and B9). Two stale documentation cross-references would weaken reuse:
  (1) STYLE.md `.diagram-frame` bullet still allowed a `/paper-figure` bits-vs-recovery plot,
  conflicting with the new `.plot-frame` rule; (2) DIAGRAM-STYLE.md still said data plots follow
  `paper-figure`, not PLOT-STYLE.md.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Findings**
No blocking data-fidelity defect found. The A1 SVG maps the displayed B8 values correctly:

- `1597 / 0.80` → `(620.13, 92.8)` → SVG `(620,93)`
- `312 / 0.45` → `(172.65, 220.2)` → SVG `(173,220)`
- `220 / 0.30` → `(140.61, 274.8)` → SVG `(141,275)`
- `135 / 0.16` → `(111.01, 325.76)` → SVG `(111,326)`
- `68 / 0.09` → `(87.68, 351.24)` → SVG `(88,351)`

The 479-bit accessible-cap line maps to `x=230.86`, rendered as `231`, so that is correct. The
sweep polyline is in order `∞ → 1024 → 512 → 256 → 128`. JSON `c1_spearman.token_f1.i_g` is
`0.9999999999999999`, matching B9 and the plot’s `ρ = +1.00`.

Main remaining weakness: STYLE.md:111 still says `.diagram-frame` may contain a `/paper-figure`
plot with `.diagram-cap`, which conflicts with the new `.plot-frame` rule at STYLE.md:116.
Minimum fix: rewrite the `.diagram-frame` bullet to method diagrams only, and point all
result/data plots to `.plot-frame`.

Minor cleanup: DIAGRAM-STYLE.md:13 still says data plots follow `paper-figure`, not
`PLOT-STYLE.md`. Minimum fix: update that row/line to say data plots follow `PLOT-STYLE.md`.

**Score**
8/10. The harness and reference plot satisfy the acceptance criteria, and the arithmetic checks
out. The only real problem is documentation ambiguity that could weaken reuse.

**Verdict**
Almost. Data and SVG are ready; clean up the two stale doc references before calling Phase 0
fully ready.

</details>

### Actions Taken
- STYLE.md: rewrote the `.diagram-frame` bullet to method/architecture diagrams only; result plots
  (bits-vs-recovery, sweeps, tradeoff frontiers) now explicitly route to `.plot-frame`.
- DIAGRAM-STYLE.md: the "Data plot" row and the governing-scope line now point to `PLOT-STYLE.md`
  (the `.plot-frame` idiom), not `paper-figure`.

### Results
- Both flagged weaknesses (the only items above the passing bar) addressed; no data/SVG change was
  required (the reviewer certified the plot arithmetic and ρ exact).

### Status
- Stopping: score 8/10 ≥ 6 AND verdict "almost" ∈ {ready, almost} → positive threshold met at
  round 1; the two doc fixes were implemented post-verdict to close the noted ambiguity.
- Difficulty: medium

## Method Description

Phase 0 builds the shared result-plot harness reused by Block-C Tasks 9–12. It adds a `.plot-frame`
CSS vocabulary to `css/site.css` (mirroring `.diagram-frame`): margined axes (`.axis`/`.tick`/
`.gridline`), data series (`.plot-line`/`.plot-point`, with a `.b` contrast variant), reference
lines (`.ref-line`), a rank-correlation badge (`.stat-badge`), heatmap cells (`.cell`) and a
`.colorbar`, plus `:root` color ramps (sequential `--ramp-0..5`, diverging `--div-neg`/`--div-pos`,
and a trust-zone-neutral `--plot-series`). The idiom is documented in `docs/html/PLOT-STYLE.md`
(linked from `STYLE.md`): the five Group-A plot types, axis rules (x = attack-independent quantity,
y = consequence; canonical probe names; legible per-secret readouts), the no-JS / D3-only-on-R2-
trigger policy, and accessibility. The reference implementation is the A1 bits-vs-recovery scatter
on `vec2text.html` §07: inline SVG, x = `I_G` (Gaussian channel-capacity MI ceiling, bits), y =
token-F1 recovery, one point per DP ε, polyline in sweep order, Spearman ρ=+1.00 annotated,
accessible-cap reference line at 479 bits, provenance `results/spectral_mi_probe_eval.json`. No GPU;
renders existing on-disk measurement.
