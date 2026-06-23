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

## Masthead + colophon (provenance is mandatory)

`<header class="masthead">` holds `<h1 class="title">`, a `.subtitle` one-liner, and a
`.colophon` block that states the cell's provenance — every report declares these four:

```html
<div class="colophon">
  <div><b>surface</b>  <which representation></div>
  <div><b>attack</b>   <attack + family></div>
  <div><b>probe</b>    <measure> — verified | candidate</div>
  <div><b>proof</b>    Codex gpt-5.5 xhigh · N rounds · PASS</div>
  <span class="rev">research report · YYYY-MM-DD</span>
</div>
```

## Section sequence (use these ids — the topnav links to them)

| id | `.section-title` | contents |
|----|------------------|----------|
| `overview` | Overview | the cell's thesis in 2–3 sentences (`.lede` + `.prose`) |
| `defs` | Definitions | glossary `.spec` table (surface, secret, threat model, terms) |
| `how` | How it works | mechanism; `.diagram-frame` figure (see below) |
| `probes` | Probe | the measure, why it is attack-*independent*, what it bounds |
| `results` | Results | the **bits + readout** table + the bits-vs-recovery correlation |
| `claims` | Claim | the claim statement; link the `research-wiki/claims/<slug>.md` node |
| `proofs` | Proof | proof sketch + provenance (full proof lives in the claim file) |
| `conclusions` | Conclusions | what leaks, how much, why; correlate→thesis or not→why |

Each section: `<header class="section-head"><div class="section-num">NN</div><h2 class="section-title">…</h2><div class="section-meta">…</div></header>` then body. Omit a section only if genuinely empty; keep the order.

## Components

- **`.spec` table** — the workhorse. Results tables MUST carry **both axes** per the metric
  convention: a `bits` column (canonical) and the per-secret human readout (perplexity / token-F1 /
  recovery-rate / cosine / AUC). One row per sweep point (plaintext → defense parameter).
- **`.diagram-frame` + `.diagram-cap`** — figures. Prefer a deterministic SVG from `/figure-spec`
  (architecture/flow) or a `/paper-figure` plot (bits-vs-recovery curve) embedded inline; caption
  in `.diagram-cap`. Reuse the trust-zone palette classes (`.box-tee` terracotta, `.box-gpu`
  slate, `.arrow-*`) for split-TEE/GPU diagrams.
- **`.prose` / `.lede`** — narrative; `.lede` for the section's opening emphasis line.
- **`.cgrid`** — card grid for parallel items (e.g. condition C0/C1/C2 comparisons).

## Cross-page nav + index

Add the new page to the `.topnav .links` (and any `.navgroup` it belongs to) on **every** page,
and to `docs/html/index.html` (the site landing page: one card/row per surface, plus the
cryptographic **zero-leakage reference** box for Euston/Fision/TwinShield-full — documented, not
swept). Serve the site with `scripts/harness/serve_docs.sh`.

## Voice

Terse, formal, structured. Comparative tables over prose lists. Cite arXiv/DOI inline. State what
is measured/decided and why. Numbers come from `refine-logs/<surface>/` and the claim node — never
hand-typed estimates.
