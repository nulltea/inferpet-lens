---
type: reference
status: current
created: 2026-06-20
updated: 2026-06-20
tags: [review-summary, decision-log, matched-probe, deferred-decisions]
companion: [FINAL_PROPOSAL, REFINEMENT_REPORT]
---

# Review Summary — matched-probe program

## How this proposal was refined

This is a **research-refine-pipeline** run that reshaped the prior monolithic
calibration thesis (`archive-capacity-pvi/`) into a per-channel matched-probe
program. The reshaping was driven by:
- the established **L20×input-DP decoupling** (token-id vs embedding reconstruction),
- the request to **separate attacks by (target × surface)** and pair each with an
  **independent** MI-correlated probe,
- and a codebase audit showing **3 of 4 channels + the Π τ-recovery family are
  already built** (`measures/vinfo_capacity`, `measures/club`, `attacks/*`,
  `weights/{vma,features,measures,types}`).

## Decisions taken (user, 2026-06-20)

| Fork | Decision | Consequence |
|------|----------|-------------|
| **Headline framing** | **Deferred to data** | Three framings (F-A matrix / F-B token-id + robustness / F-C Π-vs-obfuscation) kept live; B3+B6 adjudicate |
| **Channels** | **All four** (token-id, Π, embedding, attention) | Full breadth; membership cut (probe=attack) |
| **Defences** | **All four new** (AloePri perm-core, AloePri full Alg1, Shredder static, Shredder learned) + existing DP, split-depth | 6-family cross-scheme axis (B4) |
| **Π-probe** | **Deferred to data** | 3-way bake-off in B2 (C4): CLUB-on-φ vs capacity-reader-on-φ vs retrieval-PVI[=attack] |

## Reviewer-style concerns folded into the plan

1. **"The probe is just the attack."** Every channel has a per-instance
   collinearity test vs its attack-in-bits reference (token-id: vs retrieval-PVI,
   ρ=0.76 ✅; Π: vs VMA-PVI in B2). Independence is a *gate*, not an assumption.
2. **"The decoupling is an estimator artifact."** CLUB (independent estimator)
   must reproduce the off-diagonal gradient — it did at L20 (0.96→0.29). Kept as
   the B3 control.
3. **"AloePri obfuscation broke the model."** Logit-fidelity check in B0 (obfuscated
   forward ≈ plaintext) gates the AloePri sweeps.
4. **Stats framing** (carried from prior R3): within-layer/macro ρ primary,
   partial-ρ secondary.
5. **Scope creep.** V3 discipline applied: membership, Vec2Text-decoder,
   MoE-routing, PML/α-leakage explicitly rejected (REFINEMENT_REPORT).

## What is NOT yet reviewed externally

External gpt-5.5 (Codex MCP) review is **deferred to post-M3** — deliberately, because
(a) the user is deferring the framing to data, so a pre-data review would mostly push
to narrow prematurely, and (b) the Codex sandbox here cannot read repo files (paste
matrix tables inline when it runs). Resume via `/auto-review-loop` once B3's matrix exists.

## Verdict

**READY to build (B0–B3).** The dominant contribution (matched-probe taxonomy +
decoupling law) is stable; the two open questions (framing, Π-probe) are
*intentionally* data-gated, not unresolved weaknesses.
