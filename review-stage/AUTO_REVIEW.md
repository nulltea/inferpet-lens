# Auto Review — campaign-C Task 6 (readout-metrics)

Reviewer: Codex gpt-5.5 (xhigh), difficulty medium. Surface: readout-metrics. Thread 019efb94.
(Previous campaign-C round logs are archived per-surface; this file now tracks Task 6.)

| Round | Score | Verdict | Outcome |
|---|---|---|---|
| 1 | 6/10 | almost | bnn H(V|Y)=0.03 bits at ε=80 ∈ (0,0.1) must be `30 mbit` (audit's "no value in (0,0.1)" false); `0.00` cells need contract not `toFixed(2)`; probe-vcap exemption needs in-page label; n/a wording vs report.py |
| 2 | 8/10 | almost | queue the source-rounded BNN precision debt; weaken "mirrors format_bits"; document the header-units-except-millibit convention |
| 3 | 9/10 | **ready** | all resolved; only non-blocking note (regeneration risk) — resolved: these HTML pages are the maintained artifact, no generator |

Final verdict: **9/10, ready** (gate ACCEPTED).

## Summary
A no-GPU verification + legibility pass. Audited every bits-bearing `.spec` table across all 22
`docs/html/*.html`: the bits + per-secret recovery-readout pairing was already satisfied by prior
standardization passes (the task's "most surface tables still show bare bits" premise was stale).
Made one harmonizing edit (constant `I_G (bits)=2856` column on the rep2text length table); after
round-1 cross-model review, fixed a genuine legibility-contract miss in the BNN equivocation column
(`30 mbit` for the sub-0.1-bit value, `≈0` for the source-rounded zeros, via a `fmtBits` JS helper +
static-cell edits); labeled the one exempt estimator-calibration table (probe-vcap) in-page; and
queued the BNN full-precision display-precision debt to Task 7. No new recovery-readout gaps; no GPU.

Outputs: `refine-logs/readout-metrics/AUDIT.md`, `refine-logs/readout-metrics/queued-for-utility.md`.
Edits: `resid-rep2text.html`, `bnn-attack.html`, `synthesis.html`, `probe-vcap.html`.

## Method Description
Audited `<th>` cells containing "bit" across the report, checking each bits-bearing table for a
same-row per-secret recovery readout and for `src/talens/report.py` `format_bits` legibility
(sub-0.1-bit → millibits; `n/a` distinct from measured zero). Verified compliance, applied targeted
legibility fixes where the contract was not yet honoured (BNN equivocation), and routed the one
model-required precision gap to the single GPU phase.
