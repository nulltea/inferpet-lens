# Auto Review Loop — Block A Task 8: site assembly

Reviewer backend: Codex (gpt-5.5, xhigh). Artifact: docs/html/index.html + site-wide topnav.

## Round 1 (2026-06-24)

### Assessment (Summary)
- Score: 8.5/10
- Verdict: ready
- The new index.html satisfies the deliverable: short academic landing page, one card per
  research surface, separate reporting-foundation band, explicit cryptographic zero-leakage
  reference, traceable numbers, canonical nav, no visible process jargon in the changed work.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Based on the inline evidence: **8.5/10**.

This looks ready as an internal research documentation site assembly. The new `index.html` appears
to satisfy the main deliverable: short academic landing page, one card per research surface,
separate reporting foundation, explicit zero-leakage reference, traceable numbers, canonical nav,
and no visible process jargon in the changed work.

**Remaining Weaknesses**

1. Older pages still violate the no-em-dash prose rule. Severity: medium. Inherited debt from
   accepted surface pages. Minimum fix: targeted prose-only pass replacing genuine body-prose
   em-dashes, leaving allowed title conventions and en-dashes intact.
2. Current-page marking by navgroup may be less precise than ideal. Severity: low-medium. Minimum
   fix: also mark the exact child link current (`aria-current="page"` or `.current`).
3. The landing page is dense. Severity: low. Minimum fix: trim each `finding` to one claim plus
   one qualifier.
4. Metric convention present but could be more visibly standardized per card. Severity: low.

**Residual Em-Dash Ruling**: acceptable noted tech-debt for this site-assembly deliverable, not
blocking. The assembly cleaned the new page, process-jargon, and nav without reopening all prior
accepted prose.

**Verdict**: Ready, with inherited polish debt.

</details>

### Actions Taken
- Built docs/html/index.html: topnav (residual/embedding navgroups + flat links), masthead, §01
  Introduction (threat model / probe-not-attack / bits+readout / how-to-read), §02 Surfaces
  (6-card .cgrid), §03 Reporting foundation, §04 Zero-leakage reference (.spec table:
  Euston/Fision/TwinShield-full, "~0 by construction"), footer.
- Added one identical canonical topnav to all 7 surface pages; restored it on the two newest
  pages (vec2text, defenses-existing) that had dropped it.
- Applied reviewer weakness #2 immediately: marked the exact navgroup child link current on the
  four navgroup pages (resid-capacity-pvi, resid-dp-attacks, vec2text, bnn-attack), in addition to
  the group label.
- Brought resid-dp-attacks.html to STYLE (presentation only, numbers/scope preserved): section
  titles "What this is"->Introduction, "How it works"->Method, "Proof backbone"->Analysis;
  "jury-gated · PARTIAL"->"claims and their support"; verdict "PARTIAL (Codex xhigh, medium
  confidence)"->epistemic labels "Supported at the input layer, Preliminary at propagated depth";
  "T1 (verified, Codex xhigh proof-checker)"->"Theorem 1 (verified)"; removed the em-dashes in the
  edited claim/proof prose.
- Stripped two "Codex gpt-5.5 xhigh · 3 rounds · PASS" colophon tooling artifacts from
  vec2text.html.

### Verification
- All 8 pages tag-balanced (HTMLParser); all 8 + css/site.css serve HTTP 200 locally; all internal
  .html links resolve to existing files.
- Index card numbers trace verbatim to their surface pages (capacity 0.80-1.00; permutation +0.60
  / three quarters; defenses 1.00->0.04 and >=0.45). No invented numbers.
- Site-wide process-jargon grep (jury-gated|PARTIAL|PASS|Codex|gpt-5.|xhigh|medium confidence|
  N rounds|forbidden titles) returns zero matches.

### Status
- Stopping: positive assessment on round 1 (score 8.5 >= 6 AND verdict "ready"). Loop complete.
- Difficulty: medium.

### Noted tech-debt (non-blocking, deferred)
- Residual body-prose em-dashes on older gated pages (vec2text, bnn-attack, metric-std, and the
  unedited sections of resid-dp-attacks). A site-wide prose polish spanning already-accepted
  deliverables; out of scope for assembly. Reviewer ruled non-blocking.
- Two vec2text section titles ("What the attack is", "How the attack works") are descriptive
  variants rather than the exact academic set; inherited, not flagged blocking.

## Method Description
The site is a multi-page static research report under docs/html/, one page per leakage surface
plus a landing index. Each surface page runs the project's measurement loop (attack-independent
probe in bits vs a recovery attack, swept plaintext -> defense parameter) and renders it on the
shared bits-canonical + per-secret-readout convention defined by the reporting layer. index.html
indexes the six research surfaces as cards, links the reporting-foundation page, and documents a
cryptographic zero-leakage reference (Euston/Fision/TwinShield-full) as the converse floor of the
leakage scale. A single canonical topnav (flat links + residual/embedding navgroups) makes the set
navigable; the active page is marked current. Served via scripts/harness/serve_docs.sh over the
tailnet.
