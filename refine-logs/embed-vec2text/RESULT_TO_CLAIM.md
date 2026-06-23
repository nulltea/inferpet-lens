# Result-to-Claim Verdict — embed-vec2text (Task 4)

**Date**: 2026-06-23 · **Judge**: Codex gpt-5.5 xhigh (thread 019ef5f3) · **integrity_status**: warn (from EXPERIMENT_AUDIT.json; probe≠attack PASS)
**Evidence pre-check**: 5/5 cited numbers VERIFIED present in result JSONs.

## Verdicts

- **C1 (matched-probe correlation)**: `partial` — supported as an *in-sample rank-prediction* on this
  ε-sweep (Spearman +1.00 token-F1/cos/pos-acc, +0.71 exact; ≫ capPVI; 28× cheaper; attack-independent).
  NOT a broad calibrated-prediction or out-of-domain law (5 ε points, N=96<d). Already scoped this way
  in the claim (converse ceiling; validation, not implied).
- **C2 (converse)**: `partial`/`yes`-if-narrowed — `yes` as "0 observed violations across the sweep";
  the *proof* (T3) is the guarantee, the data is empirical consistency, not a tightness proof. Wording
  softened in the log to "consistent with the converse".
- **C3 (Vec2Text feedback null on per-position resid)**: `partial` — the negative empirical observation
  (feedback ≤ no-feedback at every ε) is solid; the bottleneck *mechanism* is lit-supported but not
  ablation-proven on our data. Caveat added to the log.

## Confidence: medium overall (C1 medium, C2 medium-high for "0 observed violations", C3 medium for the
negative result / low-medium for the causal explanation).

## Actions taken (this consolidation; no new GPU run)
- C1: claim already carries converse-ceiling + N<d scoping; exact-match qualifier added (from audit).
- C2: experiment log reworded to "consistent with the converse — 0 observed violations".
- C3: experiment log adds explicit "mechanism lit-supported, not yet ablation-proven here" note.

## Next experiments (queued, NOT run now — consolidation phase)
- Estimate Σ from n≫d embedding corpus (shrinkage / randomized) → un-rank-deficient d_eff + T4 tail.
- T4 bottom-mode ablation (achievability of the localization converse).
- Bootstrap CIs / denser ε for the rank correlations; second corpus/model to separate GTR-specific
  behavior.
- C3 controlled bottleneck ablation (compress residual / vary dim / strip logit-lens components).

These are recorded as the matched-probe-program follow-ups; the consolidated claim stands as a verified
converse probe with empirically validated in-sweep rank-correlation.
