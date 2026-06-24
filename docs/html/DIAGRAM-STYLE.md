# Diagram style reference — `docs/html/`

How to build a **method diagram** that actually explains the surface, the attack, the probe, and the
defense — not a three-box flow. This is the depth standard `STYLE.md` §`method` points to. Read it
before drawing, and **research the source paper's figure first** (rule R0 below). Companion:
`STYLE.md` (page house style), `figure-spec` skill (deterministic SVG), `paper-figure` (data plots).

## Two diagram classes — pick the right tool

| class | what it shows | tool | interactive? |
|---|---|---|---|
| **Schematic method diagram** (mandatory per page) | the surface→attack and surface→probe pipeline, trust boundaries, the algorithmic/training sequence | hand-authored SVG (trust-zone palette) as baseline; **D3.js via the `/d3-viz` skill** when a sequence or boundary benefits from interaction | yes, where it aids understanding |
| **Data plot** | bits-vs-recovery curve, sweep, tradeoff frontier | `paper-figure` / hand SVG | static for now (**Plotly deferred** — too heavy) |

This file governs the **schematic method diagram**. Data plots follow `paper-figure`.

---

## R0 — Research the paper's figure first (do NOT reinvent)

Shallow diagrams come from inventing a schematic from scratch. Before drawing, **pull the source
paper's own attack/defense figure and adapt it** — the authors already solved the "what are the
boxes and boundaries" problem. Reference figures (talens doc server, open in browser):

- Vec2Text — `…/documents/ba61acf1-bb04-41bb-9eac-f413a2d7a09c/figures/fig_2_0`
- AloePri — `…/documents/812146cf-5c60-449b-855b-60e10ea8e003/figures/fig_7_8`
- KV-CLOAK — `…/documents/9ed851a6-655b-462a-87ee-8e3a915dbf2d/figures/fig_1_6`

For any new surface, find the paper's method figure (the doc server `figures/` endpoint, or the
arXiv PDF), screenshot it into the working dir, and redraw it in the house palette with our loop
annotations (attack path + probe path + defense locus). Cite the figure source in the `.diagram-cap`.
If no paper figure exists, sketch the data-flow on paper first; never go straight to boxes.

---

## R1 — Mandatory content (the depth rules)

A method diagram is **incomplete** unless it shows all of these. Use the worked example (§Anatomy)
as the template.

1. **Trust-zone bands.** Partition the canvas into labeled zones by who controls what, using
   `zone-band-*`: `client` (data owner, trusted) · `tee` (trusted-execution) · `crypto` (a defense /
   encrypted computation) · `gpu` (adversary / untrusted compute) · `storage` (KV-cache / embedding
   store). Every box sits inside the zone that owns it. Label each zone (`zone-label`) with the actor
   and a one-line capability (`zone-sublabel`), e.g. "Adversary — untrusted (WEIGHTS-PUB) · has φ,
   published (C,σ)".
2. **Trust boundaries as lines.** Draw the `boundary` line between zones and label what crosses it
   (`boundary-label`, e.g. "release"). The released surface is whatever crosses the boundary — name it
   on the crossing arrow.
3. **The algorithmic / training sequence, numbered.** Number the steps ①②③ in order
   (`box-label` "① base inversion", "② correct (T rounds)", …). For a learned attack/defense,
   **distinguish the train phase from the inference phase** (e.g. a dashed "train: fit on synthesized
   pairs" lane vs the solid "infer: invert e′" lane), and show what data each phase sees. An attack
   that iterates (Vec2Text correct loop, JD accumulation over T) must show the loop arc and its
   accept/stop condition.
4. **Both loop paths.** The measurement loop has two arms from the same surface: the **attack path**
   (→ recovery readout) and the **probe path** (→ bits). Draw both, from the released surface, so the
   reader sees the probe is computed independently of the attack (geometry-only / channel-matched).
   Annotate the probe box with its formula and "no attack" (see vec2text `I_G` box).
5. **Defense-parameter locus.** Mark where the swept parameter acts (the Gaussian-mechanism box with
   σ=C·z/ε; the GELO row-mix with κ(A); the PriPert perturbation β) and which arrow it attenuates.
6. **Typed, labeled arrows.** Every arrow carries the object that flows, via `arrow-label`. Use the
   arrow type for its semantics: `arrow-secret` (plaintext secret), `arrow-masked` (defended/released
   or query), `arrow-cipher`/`arrow-attest` (crypto/TEE), `arrow-warn` (leak/recovery), plain `arrow`
   (compute). Leak/recovery arrows end in a `.box-warn` target box (amber stroke); give its label the
   `.warn` modifier (e.g. `class="box-label warn"`). Inline `style="…"` is acceptable only for
   non-semantic geometry (per-instance `opacity`, `stroke-dasharray`); never for palette colors — those
   have classes (`.box-warn`, `.arrow-warn`, `.warn`, the `box-*`/`arrow-*`/`zone-band-*` set).

If a step has no arrow label or a zone has no actor label, the diagram is not done.

### Available CSS vocabulary (in `css/site.css`, do not invent classes)

- Frame: `.diagram-frame` > `.diagram-cap` (`.l` for the FIG label) + `<svg class="flow" viewBox=…>`.
- Zones: `.zone-band` + `.zone-band-{client,tee,crypto,gpu,storage}`, `.zone-label`, `.zone-sublabel`.
- Boundary: `.boundary`, `.boundary-label`.
- Boxes: `.box` + `.box-{client,tee,crypto,gpu,storage}`, `.box-warn` (amber stroke, leak/recovery),
  `.box-strong` (emphasis), `.box-label`, `.box-sub`, `.box-mono` (formulae). Amber text via the
  `.warn` modifier on `.box-label`/`.box-sub`.
- Arrows: `.arrow` + `.arrow-{secret,masked,cipher,attest,rrag,warn}`, `.arrow-label` (+ `.warn` for
  amber), marker `url(#ah)`.
- Interactive (added to `site.css`): `.diagram-controls`, `[data-step].is-active`, `[data-step].is-dim`,
  `@media (prefers-reduced-motion: reduce)`.
- Palette vars: `--client --tee --crypto --gpu --storage` (+ `-soft`), `--warn` (leak/amber).

Palette semantics (keep consistent across all pages): client = trusted owner · tee = terracotta
trusted-execution · crypto = defense · gpu = slate adversary · storage = KV/embedding store · warn =
amber leak.

---

## Anatomy of a good static diagram (the baseline)

`docs/html/vec2text.html` FIG.01 is the reference baseline — copy its structure. It shows: three
zones (owner / DP defense / adversary) with a labeled `release` boundary; the secret text X →
encoder φ → Gaussian mechanism (σ=C·z/ε, the defense locus) → the numbered Vec2Text loop ①base
inversion ②correct (T rounds, with the re-embed arc and the cosine accept condition) → recovered
text x̂ in a `.box-warn` box; and a separate **probe path** to the `I_G(σ)` box annotated
"geometry-only · no attack". FIG.01 has been brought to full compliance with R1 (every arrow labeled,
DP zone sublabel, `release` on the boundary `e′` actually crosses, `.box-warn` recovered-text box), so
it is the copyable baseline — reuse its skeleton and change the zones/boxes/sequence for the surface.

A static SVG like this is sufficient when the sequence is short and legible; otherwise add
interactivity per the R2 trigger rules below.

---

## R2 — Interactivity (D3.js), when it earns its place

Be **eager** to add interactivity where it aids understanding — do not animate for decoration, but do
not default to a flat static figure either.

**Trigger rules (add D3 if ANY holds):** the sequence has **4+ numbered steps**; there are distinct
**train vs inference phases**; the attack/defense is **iterative** (Vec2Text correct loop, JD
accumulation over T, depth propagation); a box/boundary hides a **sub-pipeline** worth expanding; or
the per-step **labels/formulae would crowd** the static SVG. If none hold, a static SVG is fine. When
in doubt, add the step-through — it is cheap and almost always clarifies a measurement loop.

The three sanctioned patterns:

1. **Step-through a sequence.** Prev/Next buttons (and ←/→ keys) advance the numbered steps ①②③,
   highlighting the active box + its arrows and dimming the rest, with a caption line per step. Best
   for multi-round attacks (Vec2Text correct loop, JD accumulation over T) and train→infer defenses.
2. **Hover-reveal detail.** Hovering a box reveals its formula / shapes / hyperparameters in a
   tooltip, keeping the base diagram uncluttered. Best for boxes whose `box-mono` detail would
   overflow.
3. **Zoom into a boundary.** Click a zone/boundary to expand the operation that crosses it (e.g.
   expand the Gaussian mechanism into clip → add-noise → release). Best when one box hides a
   sub-pipeline.

Engineering rules for D3 diagrams:

- **Author with the `/d3-viz` skill.** It is installed; use its vanilla-JS *direct-DOM* pattern
  (`d3.select('#fig')` + data-driven attribute/class updates) — that is the right integration for
  these static pages (no React/build step).
- **Loading D3.** The skeleton below loads D3 v7 from the CDN (`https://d3js.org/d3.v7.min.js`) and
  runs as-is — the site is tailscale-served static HTML with no CSP. For offline robustness, when you
  ship the first interactive diagram, vendor it once at `docs/html/js/d3.v7.min.js` and switch the
  `<script src>` to `js/d3.v7.min.js`. This is NOT an Artifact.
- **Shared CSS already present.** `.diagram-controls`, `[data-step].is-active`, `[data-step].is-dim`,
  and the `prefers-reduced-motion` block are already in `css/site.css` — do not redefine them per page.
- **Static-first, JS-enhances.** Render the full diagram as inline SVG (the baseline above) so it is
  complete with JS disabled; D3 only *adds* highlighting/stepping on top. Never require JS to see the
  pipeline.
- **Reuse the palette.** D3 toggles the same classes (`.is-active` / `.is-dim` on `[data-step]`
  elements); do not hard-code colors in JS.
- **Accessible.** `<svg role="img" aria-label="…">`; `<button type="button">` controls with
  `aria-controls` + `disabled` at the ends; an `aria-live="polite"` step caption; keydown handling
  scoped to the figure (not the whole window); the `prefers-reduced-motion` CSS already ships. The
  `.diagram-cap` states the same content statically.

### Worked D3 skeleton — step-through (copy and adapt)

```html
<div class="diagram-frame">
  <div class="diagram-cap"><span class="l">FIG. 0X — &lt;surface&gt; attack sequence</span>
    <span>step through ①→④ · ← / → keys</span></div>
  <svg id="fig0x" class="flow" viewBox="0 0 1200 420" role="img"
       aria-label="Step sequence: surface released, attack inverts in T rounds, probe bounds recovery">
    <!-- full static diagram (complete without JS): zones, boundary,
         boxes/arrows tagged data-step="1".."4" -->
  </svg>
  <div class="diagram-controls">
    <button type="button" id="fig0x-prev" aria-controls="fig0x">‹ prev</button>
    <span id="fig0x-cap" aria-live="polite"></span>
    <button type="button" id="fig0x-next" aria-controls="fig0x">next ›</button>
  </div>
</div>
<script src="https://d3js.org/d3.v7.min.js"></script>  <!-- or vendored js/d3.v7.min.js -->
<script>
(() => {
  const frame = document.querySelector("#fig0x").closest(".diagram-frame");
  const steps = ["① release e′ across the boundary","② base inversion → h₀",
                 "③ correct T rounds (re-embed, accept on cosine)","④ recovered x̂ vs probe ceiling I_G"];
  const svg = d3.select("#fig0x"), prev = d3.select("#fig0x-prev"), next = d3.select("#fig0x-next");
  let i = 0;
  const render = () => {
    svg.selectAll("[data-step]")
      .classed("is-active", function(){ return +this.getAttribute("data-step") === i + 1; })
      .classed("is-dim",    function(){ return +this.getAttribute("data-step") >  i + 1; });
    d3.select("#fig0x-cap").text(steps[i]);
    prev.attr("disabled", i === 0 ? "" : null);
    next.attr("disabled", i === steps.length - 1 ? "" : null);
  };
  const go = d => { i = Math.max(0, Math.min(steps.length - 1, i + d)); render(); };
  next.on("click", () => go(1));
  prev.on("click", () => go(-1));
  // scope keys to this figure (focus within it), so multiple diagrams on a page don't collide
  frame.setAttribute("tabindex", "0");
  frame.addEventListener("keydown", e => {
    if (e.key === "ArrowRight") { e.preventDefault(); go(1); }
    if (e.key === "ArrowLeft")  { e.preventDefault(); go(-1); }
  });
  render();
})();
</script>
```

The `.is-active` / `.is-dim` / `.diagram-controls` / reduced-motion CSS already ships in
`css/site.css`; do not redefine it per page.

---

## Anti-patterns (the "lazy diagram" checklist — reject in review)

- A three-box "input → model → output" flow with no trust zones and no boundary.
- Arrows with no labels; you cannot tell what flows.
- No numbered sequence; a learned attack/defense with no train-vs-infer distinction.
- Only the attack path drawn, probe path missing (or vice-versa) — the loop is invisible.
- Defense parameter not located on any arrow/box.
- A diagram invented from scratch when the source paper has a figure (R0).
- A `<table>` used where a diagram is required (STYLE: `.diagram-frame` is SVG only).
- Animation/interaction that conveys nothing (decorative motion).

---

## Per-surface backlog (pages currently without a method diagram)

These pages have **no** `.diagram-frame` SVG and need one built to this standard (campaign work):

| page | must show |
|---|---|
| `resid-rep2text.html` | last-token residual @L10 → adapter → frozen decoder; length axis; probe I_G path (vacuous-capacity annotation) |
| `resid-gelo.html` | residual rows → fresh row-mix A (κ locus, shield rows) → exposed U; BSS attack path + feature-Gram leak; row-negentropy probe path |
| `resid-split.html` | split layer boundary → PriPert sparsify+perturb (β locus) → released activation; ridge/mlp2 inverter; I_G + Fano probe path |
| `resid-dp-attacks.html` | input-embedding DP (ε locus) → propagation to depth L; ridge vs Bayes/decoder paths; CLUB/V_cap probe path |
| `kv-cloak.html` | KV rows → channels (feature-mix M / token-mix S·P̂ / mask A) as separate transforms → BSS path; negentropy probe; show M as the load-bearing channel |
| `embed-sgt.html` | pooled embedding → SGT heteroscedastic noise (budget B × shape) → Vec2Text path; I_G(D) probe; the shape axis |
| `bnn-attack.html` | (also restyle onto site.css) embedding table → DP release → nearest-neighbour decode; Bhattacharyya upper + Fano lower bound annotations |

Each redraw starts at R0 (find the paper figure) and satisfies R1 (all mandatory content).

---

## Checklist before a method diagram passes

- [ ] Source paper figure researched and adapted (R0), cited in `.diagram-cap`.
- [ ] Trust-zone bands + labeled actors; boundary line(s) with the crossing object named.
- [ ] Numbered algorithmic/training sequence; train vs infer distinguished where learned.
- [ ] Both loop arms drawn (attack → recovery, probe → bits); probe annotated "no attack".
- [ ] Defense parameter located; leak/recovery arrow uses `.arrow-warn`, target box uses `.box-warn`, amber labels use `.warn`.
- [ ] Every arrow labeled with the object that flows; arrow types match semantics.
- [ ] Static SVG complete without JS; D3 only enhances; accessible + reduced-motion honored.
- [ ] Only house palette classes used for color; inline `style` only for non-semantic geometry (opacity, dash). Any new shared class added to `css/site.css`, not per page.
- [ ] If an interactive diagram ships, D3 is vendored at `docs/html/js/d3.v7.min.js` and pages reference it (not the CDN).
