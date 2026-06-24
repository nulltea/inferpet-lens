# Result-plot style guide — `docs/html/`

Companion to `STYLE.md` (page structure) and `DIAGRAM-STYLE.md` (method/architecture
diagrams). This file governs **result/data plots**: the inline-SVG figures that render a
*measured* relationship (bits vs recovery, leakage vs depth, leakage vs utility), as opposed
to a `DIAGRAM-STYLE.md` method diagram, which renders a *pipeline* (surface → attack → probe).

If a figure shows numbers from a results JSON, it is a plot and lives here. If it shows the
trust boundaries and the algorithmic sequence, it is a method diagram and lives in
`DIAGRAM-STYLE.md`. A page can carry both.

## One idiom, copied — never re-derived

The five Group-A plot types (below) share **one** SVG primitive: a margined plot box, linear
or log axes drawn as `.axis` lines + `.gridline`s + `.tick-label`s, data as `<circle>` /
`<rect>` / `<polyline>`, and an annotation badge. Build it once (the A1 scatter on
`vec2text.html` is the reference implementation); every other plot **copies that block** and
swaps the data and scales. Do not invent a second axis style, a second color for the primary
series, or a second caption layout. Consistency across pages is the cardinal rule (per
`STYLE.md`).

## Container — `.plot-frame` (mirrors `.diagram-frame`)

```html
<div class="plot-frame">
  <div class="plot-cap"><span class="l">FIG · A1 — leakage bits vs recovery</span>
    <span class="r">source: results/&lt;file&gt;.json · N=…</span></div>
  <svg class="plot" viewBox="0 0 680 440" role="img" aria-label="…one-sentence reading…">
    <title>…short title with the headline number…</title>
    … gridlines · axes · ticks · data · annotation …
  </svg>
  <div class="plot-cap" style="margin:14px 0 0;"><span class="l">Read</span>
    <span class="r" style="text-transform:none;letter-spacing:0;font-size:11px;">…one line…</span></div>
</div>
```

- The plot goes in the **Results** (or **Leakage Measures**) section, never inside a
  `.diagram-frame` and never inside a `.spec` table cell.
- The top `.plot-cap` carries the **FIG label** (`FIG · <plot-type> — <subject>`) and the
  **data provenance** (`source: results/<file>.json`), exactly as the `.spec` tables cite
  their source. Every plot states where its numbers came from.
- The bottom `.plot-cap` carries a one-line **Read** (what the reader should take away). Keep
  it factual: the correlation and its direction, or the operating window, not interpretation.

## CSS vocabulary (all in `css/site.css`, do not add per-page `<style>`)

| class | role |
|---|---|
| `svg.plot` | the plot root; sets the mono font + responsive width |
| `.axis` | the x/y axis lines (ink, 1.25px) |
| `.gridline` | dashed light reference grid (`--paper-3`) |
| `.tick` / `.tick-label` | axis ticks and their numeric labels |
| `.axis-label` | the uppercase axis titles |
| `.plot-line` / `.plot-line.b` | the sweep path / a second (contrast) series |
| `.plot-point` / `.plot-point.b` | data markers; `.b` = contrast series |
| `.point-label` | per-point labels (e.g. the ε value at each point) |
| `.ref-line` / `.ref-label` | a reference line (accessible cap, noise floor σ², plaintext marker) |
| `.stat-badge` / `.stat-sub` | the rank-correlation annotation (e.g. `ρ = +1.00`) |
| `.cell` / `.cell-label` / `.heatmap-axis` | heatmap rects, in-cell numbers, row/col labels (A2/A3/B2) |
| `.colorbar` + `.bar.seq` / `.bar.div` | the legend strip; `seq` = sequential ramp (recovery/bits), `div` = diverging (Gram interference, signed ρ) |

Color vars (in `:root`): `--plot-series` (navy) for the primary data series — it carries **no
trust-zone meaning**, so do not reuse `--tee`/`--gpu`/`--crypto` for data. `--warn` (amber) for
a second/contrast series. `--ramp-0…5` sequential parchment→terracotta for `seq` heatmaps;
`--div-neg`/`--div-pos` for diverging matrices.

## The five Group-A plot types

(Source: `refine-logs/visual-reporting-research/RESEARCH.md`, "Group-A rendering recipes".)

| # | plot | encoding | data source |
|---|---|---|---|
| A1 | bits-vs-recovery scatter + sweep path + ρ | x = leakage bits (canonical probe), y = recovery readout, one point per sweep param, `.plot-line` in sweep order, ρ in `.stat-badge` | `*_dp_sweep.json` / `spectral_mi_probe_eval.json` `(bits, recovery)` |
| A2 | layer × defense-parameter heatmap | rows = layer, cols = ε (or β), `.cell` fill = recovery via `seq` ramp, plaintext column | `localdp_depth_*`, the depth grid |
| A3 | Gram / interference heatmap | feature × feature, `.cell` fill via `div` ramp, clean-vs-defended pair | feature-Gram / feature-mix matrices |
| A4 | privacy–utility Pareto | x = defense param, leakage series + utility series, operating window marked with `.ref-line` | `refine-logs/utility-tradeoff/leakage_utility.json` |
| A5 | eigenspectrum + noise floor | sorted λ_i (log-y bars) + σ² `.ref-line` + per-direction ½log₂(1+λ_i/σ²) bits | `anisotropic_geometry_diagnostic.json` |

## Axes — rules

- **x = the attack-independent quantity** (bits / defense parameter / eigen-index); **y = the
  consequence** (recovery / utility / per-direction bits). Read left→right as cause→effect.
- Use the **canonical probe name + symbol** from `probes-registry.html` on the bits axis
  (`I_G`, `CLUB`, `V_cap`, …) — never an ad-hoc label.
- Use the **legible per-secret readout** (token-F1 / nDCG / recovery-rate / cosine / AUC), not
  a bare ratio, on the recovery axis (the `STYLE.md` metric convention applies to plots too).
- Linear scale by default; log-y only where the data spans >2 orders of magnitude (A5
  eigenspectra), and say so in the axis label.
- A plotted ρ (or any number) **must equal the number already stated in the page prose / table**
  — the plot visualizes the table, it does not introduce a new figure. Cross-check before
  shipping.

## Complete without JS; D3 only where it earns its place

Static plots are **pure inline SVG** and render with JavaScript disabled (same robustness bar
as the `.spec` tables). Reach for `/d3-viz` (vanilla-JS direct-DOM, D3 v7 vendored at
`js/d3.v7.min.js`) **only** when an R2 interactivity trigger from `DIAGRAM-STYLE.md` fires —
e.g. a sweep with too many points to label statically, or a hover-to-read heatmap with >25
cells. The static SVG is always the fallback baseline; interactivity is additive.

## Accessibility

- Every `svg.plot` carries `role="img"`, an `aria-label` with the one-sentence reading, and a
  `<title>` with the headline number.
- Encode by **position + label**, not color alone — a second series gets a distinct
  `.plot-line.b` dash pattern *and* `.plot-point.b` markers, not just a different hue.
- Animations (D3 only) respect `prefers-reduced-motion` via the existing site CSS guard.

## Cleanup pass

Plot captions and Read lines are prose: after adding a plot, the page still goes through the
mandatory `/humanize` → `/proofread` → `/term-audit` pass (`STYLE.md`) before the report gate.
Numbers, tags, and SVG geometry are never touched by the cleanup skills.
