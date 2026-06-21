---
type: reference
status: current
created: 2026-06-20
updated: 2026-06-20
tags: [refinement-report, scope-control, rejected-complexity]
companion: [FINAL_PROPOSAL, REVIEW_SUMMARY]
---

# Refinement Report — what the method kept, cut, and why

## Problem anchor (preserved)
Per-channel account of confidential-inference leakage: *what* leaks, *through
which surface*, *which independent probe predicts it* — replacing one conflated
TTRSR scalar + one fitted measure.

## Dominant contribution (one)
The **matched-probe taxonomy + decoupling law**. Everything else supports it.

## Smallest adequate mechanism
- Reuse the **existing** capacity-PVI reader (ch1), CLUB (ch3), VMA family
  (ch2), and ISA attacks (ch4). Only **new builds**: AloePri obf-table generator
  (Alg1), 4 defence Transforms/trainers, a PID QK/OV probe, and one independent
  Π-probe candidate. No new model, no new capture format.
- The Π-channel's calibration scaffolding (`WeightPair`/VMA/measures) already
  exists; the new defence *feeds* it, rather than a parallel pipeline.

## Complexity intentionally rejected

| Rejected | Why |
|----------|-----|
| Membership/attribute channel (LUMIA) | Probe *is* the attack (a linear probe) → cannot demonstrate the independence that is the whole point. |
| Full Vec2Text generative decoder | CLUB is already the embedding-geometry probe; a decoder adds engineering, not a sharper claim. |
| MoE-routing leakage surface | No MoE in gemma-2-2b / Qwen3-4B test models. |
| PML / α-leakage on activations; SAE effective-DoF | Parked (prior plan) until the matched-probe principle resolves. |
| Eager formal capacity bound | Demoted to a *conditional* block (B6-formal), triggered only by a matching failure. |
| Pre-data external review rounds | Deferred to post-M3 (user defers framing to data; Codex sandbox can't read files). |

## Frontier-primitive necessity
**Absent and correctly so** — this is a measurement/IT method, not an
LLM/VLM/Diffusion/RL contribution. No frontier-necessity block.

## Key claims & must-run ablations
- C1 per-channel matching (B1/B2/B5) with the **collinearity independence gate**.
- C2 decoupling law (B3) with the **CLUB-gradient artifact control**.
- C4 Π-probe selection (B2) — resolves a deferred decision.
- C5 framing verdict (B6) — resolves the other deferred decision.

## Deviation from the standard pipeline (logged)
The skill prescribes running `research-refine`'s GPT-5.5 review loop before
experiment planning. Here it is **deferred to post-M3** by deliberate choice: the
user is deferring the headline framing to experimental data, so an external review
now would push to prematurely narrow F-A/B/C; and the Codex MCP sandbox cannot read
this repo's files. The internal V3 discipline (anchor, one contribution, reject
complexity, isolate novelty) was applied in full. Resume external review at M3.

## Residual risks (see FINAL_PROPOSAL R1–R5)
Forward-consistency under obfuscation (R1); AloePri-full logit preservation (R2);
learned-Shredder cost (R3); matrix breadth (R4); Π-probe independence may not exist
(R5 — itself a result).

## Verdict
**READY** for B0–B3. Two open questions are data-gated by design, not unresolved.
