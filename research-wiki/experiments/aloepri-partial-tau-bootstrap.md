---
type: experiment
node_id: exp:aloepri-partial-tau-bootstrap
title: "Partial-τ cascade: a realistic-size CLEAN harvest (~293) bootstraps residual + table (ridge≈decoder); attention-value surface resists; protection is label quality"
idea_id: "idea:matched-probe-program"
verdict: partial
confidence: medium
date: "2026-06-29"
hardware: "AMD Strix Halo iGPU (gfx1151), ROCm container"
duration: "~15 min (11-cell matrix)"
provenance: "refine-logs/aloepri/ptau_fig/*.json; scripts/evals/static_obf/aloepri_partial_tau_sweep.py; src/talens/attacks/dp_inversion.py (cascade_attack)"
added: 2026-06-29
tags: [aloepri, cascade, tau-leak, bootstrap, ima-embedrow, isa-hiddenstate, measurement-loop, correction]
---

# Partial-τ cascade — attack × defence matrix (corrected, consistent pure k-pair bootstrap)

Two-stage cascade (`cascade_attack`): harvest τ for top-k token TYPES → train a supervised inverter on
those (deployment-basis rep, token) pairs → score recovery on HELD-OUT types. **All cells pure k-pair
bootstrap (`--no-aug`)**, oracle τ, L0, pythia-160m. Defence: keymat (αₑ=0), alg1 (αₑ=1.0), alg2 (head
rotation, only on the attention-value surface).

## Matrix — held-out recovery (k=293 / k=1024)
| attack | keymat | alg1 (αₑ1.0) | alg2 |
|---|---|---|---|
| ISA-HiddenState · ridge  | 0.882 / 0.997 | 0.612 / 0.980 | —* |
| ISA-HiddenState · decoder| 0.882 / 0.997 | 0.614 / 0.980 | —* |
| IMA-EmbedRow · ridge     | 0.989 / 1.000 | 0.864 / 0.998 | —** |
| IMA-EmbedRow · decoder   | 0.989 / 1.000 | 0.864 / 0.998 | —** |
| ISA-AttnValue (kqv_out)  | 0.005 / 0.085 | 0.000 / 0.010 | 0.000 / 0.010 |

`—*` alg2 is covariant-inert on the residual (the residual stays x·P̂; head transforms cancel) → no
distinct value. `—**` alg2 does not touch the static embedding table → not applicable.

## Findings
1. **A realistic-size CLEAN harvest (~293 types ≈ embedding effective rank) bootstraps both the residual
   (ISA-HiddenState, keymat 0.882) and the static table (IMA-EmbedRow, keymat 0.989), and ridge ≈ decoder
   on both** — the map is affine-saturated, no non-linear advantage. Order-independent (random ≈ frequency
   at k=293, measured separately on the table: 0.884 vs 0.864).
2. **CORRECTION (measurement-loop).** An earlier reading claimed the residual broke only under a
   non-linear decoder while ridge "resisted" at 0.003 (attack-too-weak). That 0.003 was an **artifact of
   blind synthetic-key augmentation**: the multi-key aug (designed for the k=0 fully-blind attack) was
   left on for k>0, mixing wrong-basis synthetic reps into the clean harvested deployment pairs and
   polluting the ridge fit. With a consistent pure k-pair bootstrap, ridge = decoder (0.882). The k=0
   fully-blind number (~0.06, the no-harvest ISA-HiddenState blind) is unaffected: that is the regime
   where the synthetic aug IS the attack.
3. **αₑ noise (alg1) is a SHARP lever; the paper αₑ=1.0 is under-parametrized for this attack.**
   residual ridge bootstrap (k=293, no-aug) vs αₑ: 0.882 (keymat) → **0.612 (αₑ=1.0, paper/VMA default)**
   → **0.017 (αₑ=2.0)** → 0.0 (αₑ≥4). αₑ=1.0 sits on the knee (barely helps); the bootstrap is defended
   only at αₑ≥2. So "alg1 barely helps" is an under-parametrized noise LEVEL (calibrated for the weaker
   VMA), not a bug. (`refine-logs/aloepri/ptau_fig/alpha_e_sweep_residual_ridge.json`.) Label quality is
   the other lever: realistic TFMA noise (~52% clean) collapses the k=293 bootstrap (table 0.864→0.02).
   Params: αₑ as stated, α_h was 0.0 in the runs (now set to the paper 0.2 in the eval) — **α_h noises only
   the output head, so it does not affect the residual / embedding-table / attention-value surfaces; the
   numbers stand**.
   - **ridge ≡ decoder is exact by design, not a bug.** `skip_decoder_attack` warm-starts the linear path
     to ridge, FREEZES it, ReZero-gates the GELU branch (init 0), and keeps the ridge-at-init `best_state`
     unless the branch beats ridge on a held-out split. Verified: on linear data decoder == ridge exactly
     (0.997); on the alg1 cell decoder 0.614 > ridge 0.612 (gate trained, +2 tokens) — so it is training,
     it just finds no non-linear gain (affine saturation; the table is literally the linear map (W+noise)P̂).
4. **The attention-value surface (kqv_out) resists at every setting (≤0.085), including keymat** where no
   head obfuscation is applied. The attention output mixes across positions, so it is not a
   per-token-invertible map and a type-harvest cannot bootstrap it. Algorithm 2 is redundant for this
   defense, not its cause.

## Synthesis
The attacker's partial-τ leverage is real and bites at a realistic harvest SIZE (~293) on the residual
and the static table, with a LINEAR attack (ridge), vindicating "memorization is effective" more strongly
than first reported. The protection is not harvest size and not attack non-linearity; it is (i) TFMA label
noise, (ii) the low diversity of the clean specials harvest, and (iii) for the attention-value surface,
structural contextuality (no per-token inverse). Replaces "≤293 below threshold" and the earlier
"non-linear decoder required" reading.

## Queued
- specials-only (low-diversity clean) harvest control.
- fresh /result-to-claim and /kill-argument re-judge — both P5 ("partial-τ doesn't break residual") AND
  the interim "non-linear-only" reading are overturned.

## Connections
Primitive `talens.attacks.dp_inversion.cascade_attack` (tests/test_cascade_attack.py). Eval
`scripts/evals/static_obf/aloepri_partial_tau_sweep.py`. Report `docs/html/static-obf.html` §04 FIG·01b
(5-subplot attack×defence). Relates to dropped IMA-EmbedRow-ridge and [[matched-probe-program]].
