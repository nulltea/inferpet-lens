---
type: handoff
status: current
created: 2026-06-24
updated: 2026-06-24
tags: [html, report, synthesis, consistency, harness, campaign-A, campaign-B]
companion: docs/html/synthesis.html
supersedes:
superseded_by:
---

# HTML synthesis + cross-page consistency review (campaign A complete, B partial)

Review of the campaign-A (`docs/plans/campaign-A-consolidate.md`, all 8 tasks `[x]`) and campaign-B
(`docs/plans/campaign-B-expand.md`, Tasks 1–7 `[x]`, Tasks 8–11 open) report pages. This document
records what was built this session, the campaign-B "missing probes" investigation, the structural
inconsistencies found across the HTML pages, the data axes the sweeps do not yet populate, and the
deferred harness work item.

## Done this session

- **`docs/html/synthesis.html` auto-reviewed** (Codex gpt-5.5 xhigh, 2 rounds, `review-stage/AUTO_REVIEW.md`): R1 7/10 → R2 **8/10 "almost"** (positive). R1 caught one transcription error (depth-inversion L32 cosine-NN floor 0.390 → corrected to 0.000), recovery-only tables now marked, the GELO verdict softened to match its source's hypothesis status, an at-layer capacity-PVI table added, and internal run-IDs stripped from prose. `synthesis.html` is an intentional cross-cutting-page exception to STYLE.md's surface-page section sequence (documented inline in the file and accepted by the reviewer); inline heading styles follow the `vec2text.html` precedent.
- **`docs/html/synthesis.html`** — new cross-cutting page: one standardized table per surface family
  (residual, embedding, KV/QKV, permutation+defenses) over (surface × attack × defense × swept
  parameter), with the probe-bits-to-recovery rank correlation as the comparable axis, plus four
  cross-cutting findings and an open-axes table. Thirteen executed loops; nine track, four do not.
- **Index wiring** — `index.html` gained §02 cards and footer links for the five surfaces that had a
  page but no card (Rep2Text, GELO, split inference, Stained Glass, KV-Cloak), plus a cross-surface
  synthesis card and link.
- **Topnav normalized** — one canonical topnav (now including `synthesis`, `resid-gelo`,
  `resid-split`, `embed-sgt`) written across all 14 pages that link the shared stylesheet, replacing
  the drifted per-page copies. `bnn-attack.html` was deliberately left untouched (see below).
- **Cross-cutting claim stubs** — four `research-wiki/claims/*.md` nodes (proofs marked TODO),
  registered in `research-wiki/index.md` and linked into `research-wiki/graph/edges.jsonl`.

## Issue 1c: "campaign-B sessions ignore probes" — FALSE ALARM

Probe bits are present in both the data and the HTML for every campaign-B surface. Verified by
extracting table headers from each page and the paired correlation rows:

| Page | probe-bits column | bits-vs-recovery correlation |
|---|---|---|
| `resid-rep2text.html` | capacity I_G (bits) | yes (vacuous — stated as the finding) |
| `resid-depth-inversion.html` | measures (bits), MI bound (bits) | yes (Spearman 0.85) |
| `resid-gelo.html` | negentropy (bits) | yes (0.29 / 0.51 — does not track, stated finding) |
| `resid-split.html` | matched measure (bits) | yes (0.958 / 0.915 / 0.977) |
| `kv-accumulation.html` | negentropy (bits) | yes (0.92 genuine margin) |
| `kv-cloak.html` | negentropy (bits) | yes (0.71 / 0.77) |
| `embed-sgt.html` | measure (bits) + budget (bits) sweep | yes (0.48 / 0.97 / 0.95 — shape-blind finding) |

The underlying `refine-logs/<surface>/runs/.../*.json` carry paired probe-bits + recovery + a
precomputed correlation; the HTML cells match the JSON. Root cause of the false impression: a bare
`grep -c probe` returns 0 on `resid-split.html` and `embed-sgt.html` because those pages name the
specific measure ("matched measure (bits)", "spectral channel mutual information", "budget (bits)")
rather than the literal word "probe". Both `full` (campaign B) and `consolidate` (campaign A) recipes
enforce the probe step identically; `.ralphex/progress/progress-campaign-B-expand.txt` shows the loop
running, including the two deliberate non-correlations (GELO, Stained Glass) self-appending the
spawn-depth-1 follow-up Tasks 10 and 11 exactly per the measurement-loop doctrine.

**Action:** no code or report fix. Any future automated measurement-loop-completeness check should
look for a `(bits)` table column plus a bits-vs-recovery correlation row, not the string "probe".

## HTML structural inconsistencies (recorded, mostly not yet fixed)

Topnav drift is fixed (this session). The remaining items below are recorded for a follow-up pass;
they were left unfixed so this session's deliverable stayed scoped to synthesis + wiring.

| Page | Issue | Severity |
|---|---|---|
| `bnn-attack.html` | Separate visual language: own inline `<style>`, does not link `css/site.css`, uses canvas/JS figures and a non-standard colophon. Its topnav was NOT normalized (would break without the shared stylesheet) and so is missing the synthesis / GELO / split / Stained Glass links. | high — restyle onto `site.css` and re-add to nav |
| `resid-rep2text.html` | Missing Preliminaries, Findings, Analysis sections; no Method diagram. | medium |
| `resid-gelo.html`, `kv-cloak.html` | Method section present but no `.diagram-frame` SVG (STYLE makes the diagram mandatory). | medium |
| `resid-split.html`, `embed-sgt.html`, `resid-dp-attacks.html` | No Method-section SVG diagram (embed-sgt and dp-attacks use a table/formulas instead). | medium |
| `resid-dp-attacks.html`, `vec2text.html`, `bnn-attack.html` | Non-standard section titles ("Overview", "What the attack is") and lettered sub-results (R1–R6, B7–B10) — STYLE wants standard academic titles and no internal run ids in prose. | low–medium |
| 7 pages | Custom `.colophon` grid footer vs the plain footer; cosmetic but inconsistent. | low |
| several | `<table>` definition blocks using `<tbody>` without `<thead>`; numeric-column class usage inconsistent. | low |

Note for future readers: `embed-sgt` is **executed and verified** (claim
`sgt-channel-mi-shape-blind-metric-bound-vec2text`, `refine-logs/embed-sgt/runs/sweep/sgt_eval.json`
with `c3_utility`). A stale `EXPERIMENT_PLAN.md` in that folder still reads "planned"; ignore it.

## Data axes the sweeps do not populate (TODO, also on synthesis.html §05)

- **Downstream-task utility is missing on every surface.** No defense sweep records the protected
  model's accuracy or perplexity vs the defense parameter. The only utility-adjacent data are
  release-fidelity proxies (release cosine, total distortion) on the two sentence-embedding defense
  sweeps (`embed-sgt` `c3_utility`, `vec2text` privacy–utility table). The residual and KV defense
  sweeps record recovery only. → add a utility column per defense sweep before any leakage–utility
  claim.
- **No common recovery scale.** Token recovery, token-F1, p95 cosine-vs-floor, and permutation
  recovery-rate are not cross-comparable; synthesis uses the rank correlation as the comparable axis.
  → record a normalized recovery (fraction of attack-achievable ceiling) per surface.
- **Single base model per surface** (gemma-2-2b / Qwen3-4B / GTR-T5-base) → replicate one residual
  surface on a second model to test calibration transfer.
- **Single seed** on most sweeps → multi-seed the surfaces whose verdict turns on a near-0.6
  correlation (GELO, KV-Cloak channel-mean).
- **Queued matched-probe / stronger-attack follow-ups** (campaign-B Tasks 8–11) have not run; running
  them converts the Claim-2 dichotomy from diagnosed to demonstrated.

## Deferred: harness session drop-off on long waits (issue 2)

Campaign-B Task 7 (`embed-sgt`) was interrupted at least twice and had to restart from the top; the
current wait/keepalive fix is not robust. Recorded here as a future harness work item (not addressed
this session): make a long-running phase resumable across a session drop rather than restarting,
e.g. checkpoint the phase state so a restarted session resumes from the last completed step instead
of re-running the sweep. See the harness-hardening handoff
(`docs/handoffs/2026-06-24-ralphex-aris-harness-hardening.md`) for the broader harness context.
