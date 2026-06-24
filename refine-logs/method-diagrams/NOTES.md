# Task 5 — method diagrams (campaign-C-report-hardening)

Built 7 R1-compliant `.diagram-frame` method diagrams (DIAGRAM-STYLE.md) on the 7 backlog pages,
each inserted at the top of the page's Method section. All static inline SVG (viewBox 0 0 1320 ~430),
house-palette classes only, both measurement-loop arms (attack→recovery in `.box-warn`, probe→bits
"no attack"), defense-parameter locus marked, numbered sequence, every arrow labeled. D3 v7 vendored
at `docs/html/js/d3.v7.min.js` (no page is interactive yet; vendored for later step-through).

| page | sequence | defense locus | attack arm | probe arm |
|---|---|---|---|---|
| resid-rep2text | ① adapter (train/infer lanes) → ② frozen decoder → ③ recovered text | — (clean surface) | token-F1 (real−shuffled) | `I_G` "geometry-only · vacuous capacity" |
| resid-gelo | ① form H̃ (+shields S) → ② row-mix U=A·H̃ → ③ release | κ(A) condition number | BSS Hungarian cosine, margin vs baseline | `J` negentropy + feature-Gram leak UᵀU=H̃ᵀH̃ |
| resid-split | ① sparsify top-ρ → ② perturb β·scale → ③ release at cut | ρ (sparsify) + β (perturb) | ridge / mlp2 selectivity | `I_G`+Fano (+ CLUB secondary) "no inverter" |
| resid-dp-attacks | ① input embed → ② DP (ε) → ③ propagate to L → ④ release | ε at L0 + propagation | ridge / Bayes-NN / decoder (3 series) | CLUB / V_cap / SDL "no attack" |
| kv-cloak | ① M feature-mix ② S token-mix ③ P̂ permute ④ A mask | M = sole load-bearing (KKᵀ exact-invariant) | BSS Hungarian cosine | `J` negentropy + shared spectral capacity |
| embed-sgt | ① mean-pool e₀ → ② shaped noise N(0,D) → ③ release → ④ Vec2Text | noise shape at matched budget I_G | Vec2Text token-F1 | `I_G(D)` "equal budget ≠ equal recovery" |
| bnn-attack | ① token v → ② public table lookup → ③ clip+DP → ④ NN/MAP decode | σ = C·z/ε_dp | nearest-neighbour accuracy | Bhattacharyya upper + Fano lower bounds |

bnn-attack ALSO restyled off its inline stylesheet onto `css/site.css?v=2` (canonical topnav already
present): shared chrome now inherited; a trimmed page-local `<style>` keeps only the interactive-figure
classes (`.fig`/`.controls`/`.readout`/`.twocanvas`/canvas), the BNN palette vars, and an override of
the shared grid `.legend` for the inline dot legend. Masthead wrapped to the canonical 2-column form.
Interactive canvas figures (A–D) preserved unchanged.

Cleanup pass: /humanize (2 fixes: one "crucially" tell + one caption em-dash on resid-dp-attacks),
/proofread (clean — tag balance OK, no prose doubled words), /term-audit (clean — canonical probe
names used verbatim; no "spectral channel-MI" umbrella; the only em-dashes are the sanctioned
`FIG. NN —` figure label matching the vec2text exemplar).

Verification: all 7 pages div/table/section/svg tag-balanced; only house-palette diagram classes used.
