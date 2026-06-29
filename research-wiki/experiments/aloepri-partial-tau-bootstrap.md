---
type: experiment
node_id: exp:aloepri-partial-tau-bootstrap
title: "Partial-τ cascade: a realistic-SIZE clean harvest bootstraps the table (ridge) and residual (decoder); label noise + alg2 are the real protection"
idea_id: "idea:matched-probe-program"
verdict: partial
confidence: medium
date: "2026-06-29"
hardware: "AMD Strix Halo iGPU (gfx1151), ROCm container"
duration: "~6 min across cells"
provenance: "refine-logs/aloepri/aloepri_partial_tau*.json; scripts/evals/static_obf/aloepri_partial_tau_sweep.py; src/talens/attacks/dp_inversion.py (cascade_attack)"
added: 2026-06-29
tags: [aloepri, cascade, tau-leak, bootstrap, ima-embedrow, isa-hiddenstate, measurement-loop, kill-argument-followup]
---

# Partial-τ cascade — full (surface × attack × label-quality × depth) matrix

Two-stage cascade (`cascade_attack`): harvest τ for top-k frequency token TYPES → train a supervised
inverter on those (deployment-basis rep, token) pairs → score recovery on the HELD-OUT (never-harvested)
types. The bootstrap signal is held-out recovery vs k. pythia-160m, 160 prompts, oracle-τ unless noted.

## Matrix — held-out (generalization) recovery
| surface | attack | k=20 | k=100 | k=293 | k=1024 |
|---|---|---|---|---|---|
| embed_table (static W̃, αₑ=1.0) | ridge | 0.000 | 0.006 | **0.864** | 0.998 |
| embed_table, order=random | ridge | — | 0.005 | 0.884 | 0.999 |
| embed_table, label-noise 0.5 | ridge | — | — | 0.02 | 0.943 |
| residual L0 | ridge | 0.000 | 0.001 | 0.003 | 0.126 |
| residual L0 | **decoder** | — | 0.025 | **0.882** | 0.997 |
| residual L0, label-noise 0.5 | decoder | — | — | 0.003 | 0.881 |
| residual L6 | decoder | — | — | 0.266 | 0.857 |
| kqv_out L0 (alg2@1.0) | ridge | 0.000 | 0.000 | 0.000 | 0.010 |
| kqv_out L0 (alg2@1.0) | decoder | — | — | 0.000 | 0.010 |
| kq L0/L6 (alg2@1.0) | ridge | 0.000 | 0.000 | 0.000 | 0.000 |

## Findings (rigorous, non-side-taking)
1. **A realistic-SIZE clean harvest (~293 diverse token pairs) DOES bootstrap** — to ~0.86–0.88 held-out
   recovery — on BOTH the static embedding table (linear ridge) AND the residual hidden state (but the
   residual needs a NON-LINEAR decoder; ridge fails at 0.003). Threshold ≈ embedding effective rank
   (~250), order-independent (freq 0.864 ≈ random 0.884). This overturns the earlier ridge-only "residual
   resists" reading: that was **attack-too-weak** (the measurement-loop failure mode) — the decoder
   re-breaks it.
2. **The real protection is LABEL QUALITY, not harvest size.** Realistic TFMA is ~52% clean; at 50% label
   noise k=293 collapses on BOTH surfaces (table 0.02, residual-decoder 0.003) and you need ~1024
   clean-equivalent pairs (table 0.943, residual-decoder 0.881). The clean ~293 "identity-fixed specials"
   are clean but low-diversity (may not span — the private-rag claim; not separately measured here).
3. **alg2's value/score surface is robust by construction.** Under alg2@1.0, kqv_out and kq resist BOTH
   ridge and the decoder at every k (≤0.01) — the secret per-head rotation defeats the cascade where the
   residual and table do not.
4. **Depth helps** (residual decoder L0 0.882 → L6 0.266 at k=293) but does not eliminate (L6 k=1024 0.857).

## Synthesis / implication for the realistic attacker
The attacker's partial-τ leverage is real and bites at a realistic harvest SIZE (~293) on the static
table and the residual — vindicating the "memorization is effective" intuition — but only under two
conditions the realistic harvest does not jointly meet: CLEAN labels (TFMA is noisy) AND a NON-LINEAR
attack for the residual. The alg2 value surface is safe regardless. So AloePri's residual/table safety
rests on (i) TFMA label noise, (ii) the low diversity of the clean (specials) harvest, and (iii) for the
residual, the attacker needing the right non-linear inverter — NOT on the harvest being too small. This
is the correct, measured replacement for "≤293 is below threshold".

## Queued
- specials-only (low-diversity clean) harvest control — does the actual leaked set span? (the one gap in
  the "clean+diverse needed" argument).
- decoder on residual at the realistic harvest under the ACTUAL TFMA label distribution (not a flat 50%).
- a fresh /result-to-claim and /kill-argument re-judge — the P5 verdict ("partial-τ doesn't break the
  residual") is overturned (attack-too-weak); the corrected claim is the conditional break above.

## Connections
Resolves+overturns kill-argument P5 (`refine-logs/aloepri/KILL_ARGUMENT.md`). Primitive:
`talens.attacks.dp_inversion.cascade_attack` (tests/test_cascade_attack.py). Eval:
`scripts/evals/static_obf/aloepri_partial_tau_sweep.py`. Report: `docs/html/static-obf.html` §04. Relates to the
dropped IMA-EmbedRow-ridge (same bootstrap, now measured: threshold = effective rank ~250, not 1024) and
[[matched-probe-program]].
