---
type: research
status: current
created: 2026-06-24
updated: 2026-06-24
tags: [visualization, reporting, html, data-plots, diagram-style, campaign]
companion: docs/html/STYLE.md
---

# Visual reporting research — result/data illustrations for the surface reports

**Question.** Our `docs/html/*.html` reports state every result in `.spec` tables and carry
method/pipeline diagrams (per `DIAGRAM-STYLE.md`). What **result/data visualizations** (plots,
not architecture diagrams) would best help a reader see *information vs representation structure*,
i.e. how an attack exploits representation structure and how a defense parameter attenuates it?

**Method.** Read the four nominated source papers via the edgequake corpus + their author
blogposts/repos for the actual figures they ship; inventoried our existing pages and the numeric
result files to ground each technique in data we already have or will have.

**Headline finding.** Our pages are *table-only on the results axis*: of 16 pages, the SVGs present
are all method `diagram-frame`s; **none renders the bits-vs-recovery relationship as a plot**, which
is the single most load-bearing object in the whole thesis (the measurement loop's verdict). The
highest-leverage move is one small inline-SVG plotting harness plus a per-page bits-vs-recovery
scatter. Everything else is incremental on top of that.

---

## What the source papers actually plot (figure inventory)

Confirmed from the PDFs in edgequake and the authors' blogposts/repos.

### Voita & Titov 2020 — MDL probing (arXiv:2003.12298; repo: lena-voita/description-length-probing)
- **Per-category-by-layer regularity heatmap** — the canonical "CCG tags by layer" figure
  (`regularity-min.png`): rows = linguistic categories (CCG supertags), columns = layers, cell color
  = a per-category leakage/codelength value. Reads at a glance: *which structure lives at which depth.*
- **Codelength split into data-code + model-code** (Fig 2a/2b): stacked bars per layer; the *model*
  component is the discriminating signal accuracy misses (control tasks need a bigger model code).
- **Online-code learning curves** (Fig 2c): quality vs #training-examples; solid = real task, dashed
  = control. Area under the curve = the codelength.
- **Stability-across-settings strip** (Fig 2d, Fig 3): accuracy vs MDL across 10 probe settings / 5
  seeds, showing accuracy reorders layers while MDL does not.

### Voita et al. 2019 — Bottom-up evolution (arXiv:1909.01380; blog emnlp19_evolution)
- **Layerwise MI line plots**: x = layer, y = MI; one curve for MI(rep; input token) (decreasing) and
  one for MI(rep; output) (increasing); multiple curves overlaid for MT/LM/MLM objectives.
- **Change-between-consecutive-layers curve** (PWCCA distance vs layer).
- **Frequency-stratified curves**: the same per-layer quantity split into token-frequency bins
  (rare vs frequent), one curve per bin — "frequent tokens change more."
- **Nearest-neighbour identity/position-preservation curves**: fraction of same-token (or
  same-position) neighbours vs layer.

### Xu et al. 2020 — V-information / PVI (arXiv:2002.10689)
- **Predictive V-info vs distance** line plot (frame-distance experiment): predictability falls with
  separation — the readout of "usable information decreases as the channel degrades."
- **Cross-family attack/leak matrix** (Fig 3b): heatmap (i,j) = info family-i extracts from features
  hardened against family-j; the *diagonal is smallest in its row*, i.e. a defense tuned to one
  attacker family fails against the others. This is directly our "matched vs vacuous probe" story.
- **t-SNE of representations colored by the sensitive attribute**, with the decode AUC annotated.
- (PVI is defined per-example; the paper's framing of pointwise V-info as a *distribution over
  examples* is the seed for the difficulty-histogram technique below, even where this paper reports
  it as scalars/AUC.)

### Toy Models of Superposition 2022 (transformer-circuits.pub/2022/toy_model)
- **Gram / interference heatmap (WᵀW)**: square matrix, feature index × feature index, diverging
  color (red +, blue −, white 0); off-diagonal mass = interference between features sharing
  directions. This is *exactly* a feature-Gram leak / channel-mix readout.
- **Feature-direction geometry plots**: 2D projection of learned directions; antipodal pairs and
  polytope arrangements drawn as points/edges.
- **Dimensionality / feature-per-dimension scatter**: x = sparsity, y = D* = (features that fit) /
  (dimensions used); shows capacity as a function of input sparsity, with phase-change plateaus.
- **Phase-change curves vs sparsity**: a metric vs sparsity with sharp transitions.

---

## Mapping to our inventory

Our results files already contain the numeric shapes these techniques need:
- `results/anisotropic_geometry_diagnostic.json` — `per_epsilon`, eigenspectrum, `span_rank`,
  `nullspace_*` (spectral plots, capacity).
- `results/b3_decoupling_matrix.json` — `matrix_spearman` cross-surface 3×3 + diagonal-dominance,
  `grid` over epsilons × alphas × layers × seeds (interference/decoupling heatmap, channel matrix).
- `results/localdp_depth_L0_5_12_20.json` — depth × epsilon records (layer heatmaps, depth curves).
- `results/v2t_dp_sweep.json`, `results/spectral_mi_probe_eval.json`,
  `results/unified_dp_sweep.json` — `(bits, recovery)` pairs across a sweep (the core scatter).

---

## Prioritized techniques

Ranked by impact × fit × low implementation cost. Cost is for a self-contained inline-SVG plot
(no Plotly — the site defers it; D3 only where interaction earns its place).

### Group A — adopt now

#### A1. Bits-vs-recovery scatter with sweep path + rank-correlation annotation
- **Shows:** the central thesis object — does the attack-independent probe (bits) track attack
  recovery across the defense sweep? One point per sweep point; connect them in sweep order to show
  the trajectory from plaintext to maximal defense; annotate Spearman ρ. Co-monotone → probe
  predicts; scatter → the gap *is* the finding.
- **Source:** our own method (CLAUDE.md measurement loop); kin to Xu's predictive-V-info-vs-distance
  and the MDL accuracy-vs-MDL strips.
- **Reference impl:** ~40 lines of inline SVG (axes + polyline + circles + text), or `paper-figure`.
  No library.
- **Illuminates:** every (surface × attack × probe). I_G/CLUB/V_cap (bits) vs ridge/Vec2Text/Bayes-NN
  (recovery) across ε / β / κ / budget-B.
- **Record where:** the `(bits, recovery)` columns already in `spectral_mi_probe_eval.json`,
  `v2t_dp_sweep.json`, `unified_dp_sweep.json`, the resid-split / depth-inversion sweeps. One scatter
  per page, x = bits (probe ceiling), y = recovery readout, points = sweep params, ρ in the corner.
- **Lands on:** `vec2text.html` (B9/B10), `resid-split.html`, `resid-depth-inversion.html`,
  `resid-dp-attacks.html`, `embed-sgt.html`, `kv-cloak.html`. This is the spine plot for the whole site.

#### A2. Per-layer × defense-parameter leakage heatmap
- **Shows:** "depth ≠ privacy" and "where leakage lives" in one glance — rows = layers (resid_post),
  columns = ε (or β), cell color = recovery (or bits). The Voita MDL "tags-by-layer" pattern, with
  *defense parameter* on one axis instead of category.
- **Source:** Voita & Titov 2020 regularity heatmap (`regularity-min.png`).
- **Reference impl:** inline-SVG grid of `<rect fill=...>` + a colorbar; ~50 lines. We already
  hand-build rect grids in `kv-accumulation.html` / `defenses-existing.html`, so the idiom exists.
- **Illuminates:** resid_post across depth; input-DP propagation; the decoupling story.
- **Record where:** `localdp_depth_L0_5_12_20.json` (depth × ε); extend to all 36 layers × ε grid at
  fixed attack for the full heatmap (currently layers {0,5,12,20}). Plaintext column included.
- **Lands on:** `resid-depth-inversion.html`, `resid-dp-attacks.html`.

#### A3. Gram / interference heatmap (feature-mix leak)
- **Shows:** the off-diagonal structure an attack exploits — for GELO/KV-Cloak, the feature-mix
  matrix or row-Gram; diagonal-dominant = separable (leaks), mixed = covered. Directly visualizes
  *which channel is load-bearing.*
- **Source:** Toy Models of Superposition WᵀW interference heatmap.
- **Reference impl:** square `<rect>` grid, diverging palette (reuse warn/amber for high |value|);
  ~40 lines. Add one shared `.heatmap`/colorbar CSS block to `site.css`.
- **Illuminates:** GELO row-mix (κ), KV-Cloak feature/token/perm channels, AloePri keymat — the
  surfaces where the claim is "one channel carries the leak."
- **Record where:** the feature-mix / row-Gram matrices behind `gelo-orthogonal-gram-leak`,
  `kv-cloak-channel-decoupling`; recompute the |Gram| matrix at plaintext and at one defended κ for
  the before/after pair.
- **Lands on:** `resid-gelo.html`, `kv-cloak.html`.

#### A4. Privacy–utility tradeoff (Pareto) curve
- **Shows:** the operating window — leakage (or bits) vs utility (retrieval nDCG, task acc) across
  the sweep, with the knee marked. Already *described in prose* in `vec2text.html` B10 but not drawn.
- **Source:** standard in the privacy-ML defenses literature (CAPRISE/RemoteRAG framing referenced in
  our own B10); conceptually the dual of A1.
- **Reference impl:** dual-curve line plot, inline SVG ~40 lines; or overlay leakage and utility on a
  shared x = ε with two y-axes.
- **Illuminates:** every measurable defense (input-DP ε, Shredder, PriPert β, SGT budget-B).
- **Record where:** join `utility_retrieval_eval` columns with the leakage sweep (B10 already has the
  numbers); same for any defense with a utility metric.
- **Lands on:** `vec2text.html`, `embed-sgt.html`, `resid-split.html`, and a cross-defense panel on
  `synthesis.html` / `defenses-existing.html`.

#### A5. Eigenvalue / spectral-capacity plot (channel waterfilling)
- **Shows:** why I_G is what it is — the eigenvalue spectrum λ_i of the surface covariance and the
  per-eigendirection contribution ½log₂(1+λ_i/σ²) to capacity; the noise floor σ² as a horizontal
  line shows which directions survive the defense (waterfilling picture). Makes the spectral probe
  legible and shows *localization* (which eigendirections carry the leak).
- **Source:** our spectral channel-MI probe (claim `spectral-channel-mi-embedding-inversion`); visual
  idiom from capacity/waterfilling and the superposition dimensionality plots.
- **Reference impl:** sorted bar/step plot of λ_i (log-y) + a σ² threshold line + a second series for
  the per-direction bits; inline SVG ~50 lines.
- **Illuminates:** I_G on pooled-embedding (GTR) and resid_post; the nullspace/span-rank story.
- **Record where:** `anisotropic_geometry_diagnostic.json` (`per_epsilon`, eigenspectrum,
  `nullspace_trace_frac`) — already computed; one spectrum per ε overlays the noise floor moving up.
- **Lands on:** `vec2text.html` (measures section), `metric-std.html`, `resid-split.html`.

### Group B — nice-to-have

#### B1. Pointwise difficulty distribution (per-example PVI histogram)
- **Shows:** that leakage is not uniform across inputs — histogram/violin of per-example V_cap (PVI)
  or per-token recovery; long tail = some secrets are far more exposed. Per-token-frequency split
  reproduces Voita's "rare vs frequent" stratification.
- **Source:** Xu et al. 2020 (pointwise V-info); Voita 2019 frequency-stratified curves.
- **Reference impl:** binned `<rect>` histogram, inline SVG; or per-frequency-bin overlay curves.
- **Illuminates:** V_cap (token-id reader) on resid_post; Bayes-NN per-example margin. Useful but
  secondary to the sweep-level story; needs per-example logging we do not all keep yet.
- **Record where:** per-example PVI / per-token recovery at one (layer, ε), bucketed by token
  frequency. Lands on `resid-capacity-pvi.html`, `resid-depth-inversion.html`.

#### B2. Cross-family probe×defense matrix ("matched vs vacuous")
- **Shows:** the probe-failure dichotomy as a grid — rows = probes, columns = defense channels, cell
  = Spearman(probe, recovery); off-diagonal weakness = a probe not channel-matched. This is our
  `b3_decoupling_matrix.json matrix_spearman` (already 3×3 cross-surface) plus diagonal-dominance.
- **Source:** Xu et al. 2020 Fig 3b cross-family leak matrix.
- **Reference impl:** small labeled heatmap (A3 idiom) with ρ printed in each cell.
- **Illuminates:** the matched-probe program; `probe-failure-dichotomy`, `cross-surface-matched-probe`
  claims. High *conceptual* value but narrow (one or two pages).
- **Record where:** `b3_decoupling_matrix.json` is ready. Lands on `synthesis.html`,
  `resid-split.html`.

#### B3. NN-decode / codebook geometry (Voronoi / margin) plot
- **Shows:** how Bayes-NN decodes — 2D projection of the embedding codebook with the decision
  cells / nearest-neighbour margins, and how DP noise pushes a point across a boundary
  (Bhattacharyya overlap visualized as two shifted clusters).
- **Source:** Toy-models feature-geometry plots; standard NN-classifier illustrations.
- **Reference impl:** 2D PCA scatter + segments; inline SVG, but needs care to be honest (projection
  artifacts). Medium cost.
- **Illuminates:** Bayes-NN attack, Bhattacharyya-Fano bounds. Lands on `bnn-attack.html`.

#### B4. Stacked codelength bars (data-code vs model-code) / cost bars
- **Shows:** probe cost decomposition — for the MDL/SDL probes and the I_G-vs-CLUB-vs-V_cap cost
  comparison already tabled in vec2text B9 (60ms vs 1.7s vs 0.5s).
- **Source:** Voita & Titov 2020 Fig 2a/2b.
- **Reference impl:** stacked/horizontal bar, inline SVG ~30 lines.
- **Illuminates:** probe-cost narrative across `metric-std.html`. Low value relative to the table that
  already exists; adopt only if the cost story gets its own page.

### Group C — skip / not applicable

- **t-SNE / UMAP scatter of representations** (Voita, Xu): pretty but projection-dependent and easy to
  over-read; our claims are quantitative (bits, recovery, ρ), so a t-SNE adds aesthetic risk without
  evidentiary weight. Skip unless a single figure must convey "structure exists at all."
- **Antipodal-pair / polytope geometry** (toy models): specific to <10-dimension toy regimes; our
  surfaces are d=768–2560, so the exact polytope picture does not transfer. The *Gram heatmap* (A3)
  is the transferable part — keep that, skip the polytope drawing.
- **PWCCA change-between-layers curve** (Voita 2019): measures representational drift, not leakage;
  off-thesis for an attack/defense report.
- **Plotly / heavy interactive dashboards:** explicitly deferred by `DIAGRAM-STYLE.md` (too heavy);
  inline SVG + optional D3 only.

---

## Group-A rendering recipes (concrete)

Shared prerequisite: add **one** results-plot CSS block to `css/site.css` — `.plot-frame`
(mirrors `.diagram-frame`), `.plot-cap`, `.axis`, `.tick`, `.gridline`, `.heatmap`/`.cell`, and a
`.colorbar`. Plots go in the **Results** (or Measures) section as `.plot-frame > svg`, the data-plot
analogue of `.diagram-frame`. Keep the existing palette vars (`--warn`, zone colors); add a
sequential ramp var for heatmaps. No external JS for static plots.

| # | Plot | Page(s) | Data source | Encoding |
|---|---|---|---|---|
| A1 | bits-vs-recovery scatter + sweep path + ρ | vec2text, resid-split, resid-depth-inversion, resid-dp-attacks, embed-sgt, kv-cloak | `*_dp_sweep.json` `(bits,recovery)` | x=bits, y=recovery, points=sweep params, polyline in sweep order, ρ annotated |
| A2 | layer × ε leakage heatmap | resid-depth-inversion, resid-dp-attacks | `localdp_depth_*` (extend to 36 layers) | rows=layer, cols=ε, color=recovery; plaintext column |
| A3 | Gram / interference heatmap | resid-gelo, kv-cloak | feature-mix / row-Gram matrices | feature×feature, diverging color, before/after defense pair |
| A4 | privacy–utility Pareto | vec2text, embed-sgt, resid-split, synthesis | B10 leakage + utility eval | x=ε, dual series leakage vs utility, knee marked |
| A5 | eigenspectrum + noise floor | vec2text, metric-std, resid-split | `anisotropic_geometry_diagnostic.json per_epsilon` | sorted λ_i (log-y) + σ² line + per-direction bits |

Each plot ships with a `.plot-cap` (FIG-style label + one-line read), is complete without JS, and
carries its data provenance (`source: results/<file>.json`) the same way the `.spec` tables do.

---

## Proposed campaign breakdown (ralphex phases)

**Chosen split axis: visualization-family, harness-first.** Rationale: the five Group-A techniques
share *one* SVG plotting idiom (axes/scales/rects/polyline) but land on *different* pages and
*different* result files; splitting by surface would re-derive the same plotting primitives six times
and produce inconsistent axes. Building the shared primitive once, then rolling each plot family
across its target pages, gives consistent visual language (the STYLE.md cardinal rule) at minimum
cost. Each phase is one coherent, independently-reviewable chunk ending in `/auto-review-loop`.

- **Phase 0 — plotting harness + CSS vocabulary.** Add the `.plot-frame` / axis / heatmap / colorbar
  classes to `css/site.css`; write one reference inline-SVG plot (the A1 scatter on `vec2text.html`,
  which already has the `(bits,recovery)` data) as the copyable baseline. Document the idiom as a
  short `docs/html/PLOT-STYLE.md` companion to `DIAGRAM-STYLE.md`. *Rationale: every later phase
  depends on this; ship and review it alone so the visual language is locked before fan-out.*
- **Phase 1 — A1 bits-vs-recovery scatter across all sweep pages.** Roll the baseline scatter onto
  resid-split, resid-depth-inversion, resid-dp-attacks, embed-sgt, kv-cloak. *Rationale: the spine
  plot of the thesis; highest reader impact; pure reuse of Phase 0.*
- **Phase 2 — A2 + A5 (the spectral/depth family).** Layer×ε heatmap (extend depth sweep to full
  layer grid first) and the eigenspectrum+noise-floor plot. *Rationale: both read the geometry/
  spectral result files and share the "defense moves the floor" reading; group so the depth and
  spectral stories are rendered with one mental model.*
- **Phase 3 — A3 + A4 (channel-leak + tradeoff family).** Gram/interference heatmaps on
  resid-gelo/kv-cloak and the privacy–utility Pareto on vec2text/embed-sgt/synthesis. *Rationale:
  these are the "which channel leaks / what does it cost" pair; both new-ish data joins, naturally
  later than the spine.*
- **Phase 4 (optional) — Group-B selectives + synthesis panel.** Cross-family probe×defense matrix
  (B2, data ready) on synthesis; pointwise difficulty histogram (B1) if per-example logging lands.
  *Rationale: depends on per-example data and on the synthesis page being assembled; lowest urgency.*

Each phase: one GPU-free rendering pass per CLAUDE.md (no new experiments except the A2 layer-grid
extension, which is a single capture sweep and must clear the perf gate first), then the mandatory
`/humanize` → `/proofread` → `/term-audit` cleanup and `/auto-review-loop`.
