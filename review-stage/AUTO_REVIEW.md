# Auto Review — Task 9 (c10-viz-scatter): A1 bits-vs-recovery scatter rollout

Reviewer: Codex / gpt-5.5 (xhigh), thread 019efbe7. Backend: codex (medium difficulty).

## Round 1 — Score 5/10 — not ready
Critical: (1) resid-dp-attacks badge +0.76 did not match the 3 plotted monotone points (ρ=+1.0);
(2) resid-split plotted ridge but labelled badge "best recovery"; (3) plots inserted into invalid
DOM positions (inside <tbody> / mid-<p>); (4) contrast classes emitted as `class="plot-point.b"`
(dotted) instead of space-separated, breaking styling; (5) synthesis omission note hid the V_cap
−0.21 depth tail; (6) sgt "within each shape monotone" overstated (tail-loaded ρ=−0.20).

## Round 2 — Score 7/10 — almost (gate threshold met: ≥6 AND almost/ready)
Fixed all six. dp-attacks now sources every coordinate AND ρ from one file (b2_l0_bayes.json, 8
club-bits rows; Spearman recomputes to 0.7638 = +0.76). resid-split primary = best recovery (+0.96).
DOM placement valid. Classes space-separated. Synthesis surfaces −0.21. sgt wording corrected.
Remaining: dp plot not matching adjacent R1 table; resid-split mlp2 badge +0.92 not reproducing
from plotted points; sgt aria still overstated; synthesis manual provenance.

## Round 3 — Score 8/10 — almost (FINAL)
dp-attacks plot RELOCATED to end of §5 Results (decoupled from the R1 l0_fast table), source note
present. resid-split mlp2 +0.92 badge removed (only the reproducing +0.96 best badge remains; mlp2
shown as amber series, described qualitatively). sgt aria-label corrected. synthesis caption marks
"source: §02 overview table". Reviewer confirmed every annotated ρ recomputes from the cited JSON
(split best 0.956→+0.96, depth +0.85, dp +0.7638→+0.76, sgt across-shape 0.4816→+0.48, kv 0.7058→
+0.71 / channel-mean 0.7714→+0.77). All 6 plot SVGs XML-valid, no-JS, role/aria/title present, no
dotted classes, clean DOM placement. Final residual: removed the +0.92 mention entirely after the
round-3 review (strictly the reviewer's requested fix — committed state ⊇ reviewed state).

Verdict: score 8/10, almost → PASSES the report-quality gate (≥6 AND verdict ∈ {ready, almost}).

## Method Description
Task 9 rolled the Task-8 "A1" inline-SVG bits-vs-recovery scatter primitive (viewBox 680×440, plot
box x∈[64,656] y∈[20,384], `.plot-frame`/`.plot-line[.b]`/`.plot-point[.b/.c]`/`.stat-badge`/
`.ref-line`) across the five surface pages with paired (bits,recovery) sweep data and added a
cross-surface matched-probe ρ-summary dot plot to synthesis.html. A single generator
(refine-logs/viz-scatter/gen_scatters.py) reads each surface's results JSON, computes the plotted
coordinates and the Spearman ρ, and emits one fragment per page; each annotated ρ was cross-checked
to equal the value already stated in that page's prose (PLOT-STYLE.md cardinal rule). The synthesis
dot plot is hand-authored from the §02 overview table (single-valued ρ per surface; range rows
excluded and noted, including the V_cap −0.21 depth tail).
