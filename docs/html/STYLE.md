# HTML report style guide — `docs/html/`

How to build a surface report page so the whole site reads as one document. **Match the existing
pages** (`vec2text.html`, `bnn-attack.html`) — do not invent a second visual language. One page
per surface: `docs/html/<surface>.html`. Always link the shared stylesheet; never inline a second
one.

## Page skeleton (copy from `vec2text.html`)

```html
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>transformer-attacks-lens — <Surface / attack></title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT,WONK@9..144,300..900,0..100,0..1&family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="css/site.css">
</head><body>
<div class="sheet">
  <span class="crosshair tl"></span><span class="crosshair tr"></span>
  <span class="crosshair bl"></span><span class="crosshair br"></span>
  <nav class="topnav">…</nav>
  <header class="masthead">…</header>
  <!-- numbered sections -->
</div></body></html>
```

External fonts via CDN are fine here (this is a tailscale-served static site, **not** an
Artifact — no CSP). Keep all styling in `css/site.css`; if a genuinely new component is needed,
add a class to `site.css` (shared) rather than a per-page `<style>`.

## Cardinal rule — every page is a short academic paper

Write each page as a research paper on one surface, not a blog post or a lab log: descriptive but
**terse and fact-packed** (match the density of a paper abstract). Three non-negotiables:

1. **Standard academic register only** (on par with the no-em-dash rule). Plain, precise, literal
   vocabulary; NO colloquial / conversational / figurative wording where a standard term exists.
   Avoid: "answer it", "boils/comes down to", "at once", "the whole point", "story", "buys" (=gains),
   "a lot", "kind/sort of", "sweet spot", "knob", "reads differently", "cracked", "the lever is",
   "the live frontier", figurative "geometry"/"concede". Prefer the plainest term ("simultaneously",
   "is essential", "relax a constraint", "applies differently"). When unsure, pick the plainer literal
   word. **No em-dashes** (`—`/`---`) — use commas, colons, or parentheses.
2. **No project/process jargon, ever** in visible text. Forbidden: internal experiment/run ids
   (`R1`, `R5`, `T1`, run_ids, surface slugs); review-process artifacts (`Codex`, `gpt-5.x`, `xhigh`,
   `result-to-claim`, `audit WARN`, `PARTIAL`, "medium confidence", thread ids); harness/skill names;
   spike file paths in prose. The reader is a researcher, not an operator. Refer to results
   descriptively ("under input-layer DP", "at depth 20"); spell things out ("Theorem 1", not "T1");
   state confidence scientifically ("supported at the input layer; the depth result is single-seed
   and not yet robust"), never as a review verdict.
3. **Substance, not hedging.** Every section states findings with numbers and mechanism. No
   throat-clearing, no significance inflation, no AI-voice transitions ("Moreover", "It is important
   to note", "Notably"). The cleanup pass enforces this, but write clean first.

## Masthead

`<header class="masthead">` holds `<h1 class="title">` and a `.subtitle` one-liner. Keep the title
short (≤ ~4 words) so it stays on one line.

**Do NOT add a long `.colophon` provenance block** (surface/attack/probe/verdict lines): it is
redundant with the body and its long lines starve the title column. Provenance belongs in the body
(Definitions table, Claims/Proof sections) and the footer. At most, an optional one-line
`<span class="rev">research report · YYYY-MM-DD</span>` directly under the subtitle.

## Section sequence — STANDARD ACADEMIC TITLES (anchor id → title)

Use these exact section **titles** (academic noun phrases), in order. NEVER colloquial titles like
"What this is", "How it works", or "Proof backbone".

| anchor id | section title | required content |
|----|------------------|----------|
| `introduction` | **Introduction** | the problem, the threat model in one line, and the headline finding with its key number; expands the subtitle. Active voice. |
| `preliminaries` | **Preliminaries** | `.spec` glossary table: surface, secret, threat model, measures, attacks. Terms only. |
| `method` | **Method** | the measurement procedure (past tense, reproducible) AND a **MANDATORY** `.diagram-frame` SVG figure of the surface→attack→probe pipeline, built to the depth rules in **`DIAGRAM-STYLE.md`** (research the paper figure first; trust-zone bands + boundary; numbered algorithmic/training sequence; BOTH loop arms — attack→recovery and probe→bits; defense-parameter locus). Author via `/figure-spec` (static) or D3 (interactive, per `DIAGRAM-STYLE.md`). A Method section without its figure, or with a shallow three-box flow, is incomplete. |
| `measures` | **Leakage Measures** | each probe, why it is attack-independent, what it bounds (bits). |
| `results` | **Results** | bits + per-secret readout `.spec` tables; the bits-vs-recovery relationship across the sweep. Data only, no interpretation. |
| `findings` | **Findings** | the claims as paper statements with scientific confidence; cite the claim node by name in the footer, not by internal id/verdict in prose. |
| `analysis` | **Analysis** | the theorem/bound explaining the results; proof sketch (full proof in the claim file). |
| `discussion` | **Discussion** | what leaks, how much, why; correlation→thesis or the gap and its cause; limitations; recorded negative results. |

Each section: `<header class="section-head"><div class="section-num">NN</div><h2 class="section-title">TITLE</h2><div class="section-meta">short label</div></header>` then body. Keep the order; omit a section only if genuinely empty.

## Stating claims (Findings & Analysis)

State each claim in **Toulmin form** and tag its strength with one **epistemic-status label** — this
is how you "state confidence scientifically," and it *replaces* any review verdict (never write
`PARTIAL`, "medium confidence", or a jury/process artifact).

Epistemic-status labels (pick one per claim):

| label | meaning | maps from internal verdict |
|---|---|---|
| **Established** | replicated, externally corroborated, consensus | a verified theorem / cross-checked result |
| **Supported** | evidence exists, not yet replicated | accepted single-setting result |
| **Preliminary** | single run / small sample / one seed | single-seed sweep |
| **Speculative** | from reasoning, not direct evidence | conjecture / next-step |
| **Contested** | conflicting evidence | mixed results across conditions |

Write each finding as: **claim · grounds (the number/result) · qualifier (label + scope) · rebuttal
(limitation)**. Example: "Information-efficient attacks restore the bits–recovery correlation
(**Supported**): a trained decoder tracks the leakage measures at depth 20 where the linear baseline
does not; single-seed, so not yet robust to seed variation." Cite the claim node by name in the
footer, not its internal id or verdict in prose.

## Components

- **`.spec` table** — the workhorse. Put tables **directly in the section** (a bare `<table class="spec">`),
  NOT inside `.diagram-frame` (that boxes them). Results tables MUST carry **both axes** per the metric
  convention: a `bits` column (canonical) and the per-secret human readout (perplexity / token-F1 /
  recovery-rate / cosine / AUC). One row per sweep point (plaintext → defense parameter).
- **`.diagram-frame` + `.diagram-cap`** — **method/architecture diagrams only** (never tables, never
  result plots). A deterministic SVG from `/figure-spec` (architecture/flow) of the surface→attack→probe
  pipeline embedded inline; caption in `.diagram-cap`. A measured relationship (the bits-vs-recovery
  curve, a sweep, a tradeoff frontier) is a result plot and uses `.plot-frame` below, not this. Reuse the trust-zone palette classes (`.box-tee`
  terracotta, `.box-gpu` slate, `.arrow-*`) for split-TEE/GPU diagrams. **Method-diagram depth rules,
  the full palette vocabulary, and the interactive-D3 step-through pattern: `DIAGRAM-STYLE.md`.**
- **`.plot-frame` + `.plot-cap`** — **result/data plots** (the measured bits-vs-recovery relationship,
  leakage-vs-depth heatmaps, privacy–utility curves), distinct from the method diagram above. Inline
  SVG, complete without JS, with `source:` provenance like the tables. **The plot idiom, the five
  Group-A plot types, the axis/heatmap/colorbar CSS vocabulary, and the accessibility rules:
  `PLOT-STYLE.md`.** The Results section should carry the bits-vs-recovery plot, not only the table.
- **`.prose` / `.lede`** — narrative; `.lede` for the section's opening emphasis line.
- **`.cgrid`** — card grid for parallel items (e.g. condition C0/C1/C2 comparisons).

## Cross-page nav + index

Add the new page to the `.topnav .links` (and any `.navgroup` it belongs to) on **every** page,
and to `docs/html/index.html` (the site landing page: one card/row per surface, plus the
cryptographic **zero-leakage reference** box for Euston/Fision/TwinShield-full — documented, not
swept). Serve the site with `scripts/harness/serve_docs.sh`.

## Cleanup pass — mandatory, after writing/editing a page

Generated HTML reads as AI-drafted (boilerplate transitions, cliché lexicon, em-dash overuse,
metaphor register). After the page is written and **before the report gate**, run all three
non-interactive cleanup skills on it, in order (they edit prose only — never tags, code, numbers,
or table data):

```
/humanize docs/html/<surface>.html      # strip AI-voice tells
/proofread docs/html/<surface>.html     # grammar / typos / structure
/term-audit docs/html/<surface>.html    # word-choice / register / remove em-dashes
```

Then re-open the page to confirm it still renders (tags balanced, tables intact). Only after this
does the page go to `/auto-review-loop`.

## Subtitle (abstract)

The `.subtitle` is a 1–2 sentence abstract: the problem and the headline finding. NOT a paragraph.
Substance goes in the sections, not the subtitle.

## Voice

Active voice, terse, fact-packed; comparative tables over prose lists; past tense for Method/Results.
Cite arXiv/DOI inline by author + year, never by internal id. Numbers come from the experiment
results and the claim node, never invented. No em-dashes. The mandatory cleanup pass
(`/humanize` → `/proofread` → `/term-audit`) strips residual AI tells after writing; its AI-tell
ruleset follows Wikipedia's AI-cleanup guide + Strunk & White (significance inflation, "delve/
leverage", negative parallelisms, rule-of-three, filler, hedging, signposting). Write to this
standard first.
