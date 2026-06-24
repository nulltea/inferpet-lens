---
type: claim
node_id: claim:cross-surface-matched-probe-tracks-recovery
name: "Cross-surface: a channel-matched attack-independent probe predicts attack recovery on the majority of surfaces"
description: "Across 13 executed surfaces, a matched attack-independent probe tracks attack recovery (rho>=0.6) on 9; the correlation holds within a sweep and a defense family but does not give absolute cross-family/model calibration"
node_type: claim
status: drafted
provenance: "docs/html/synthesis.html (cross-surface synthesis); per-surface claim nodes listed in Connections"
tags: [cross-cutting, synthesis, matched-probe, measurement-loop, weights-pub, proof-todo]
date: 2026-06-24
added: 2026-06-24T12:40:00Z
companion: docs/html/synthesis.html
---

# A channel-matched, attack-independent probe predicts attack recovery across the privacy sweep on the majority of surfaces

**status:** `drafted` — empirically supported by the per-surface results aggregated in `docs/html/synthesis.html`; **proof: TODO** (a formal statement of the "matched-probe predicts recovery" relation and its scope conditions is not yet written).

## Claim

For a released surface with secret X, an attack-independent probe that is matched to the load-bearing
leakage channel produces a bits measure whose rank order across the defense sweep agrees with the
attack recovery order, at Spearman rho >= 0.6.

## Grounds

Of the thirteen executed surface loops, nine meet the bar:

- token-identity reading: depth inversion (rho +0.85 across L0–L32), split inference under PriPert
  (rho +0.958, +0.915 including the learned inverter), capacity-matched PVI under at-layer defenses
  (0.80–1.00);
- embedding inversion: Vec2Text under Gaussian DP (rho +1.00 token-F1), Bayes-NN where geometry-only
  union-Bhattacharyya and Fano bounds bracket the empirical error two-sidedly;
- source separation graded against a matched random-demixing floor: plaintext KV/QKV (rho +0.92 on
  the genuine margin), KV-Cloak (rho +0.77 channel-mean);
- permutation matching: full sorted-row matcher (rho +0.976).

## Qualifier (scope, epistemic status: Supported)

Single seed and single base model per surface; three different base models across the set
(gemma-2-2b, Qwen3-4B, GTR-T5-base).

## Rebuttal (limitation)

The correlation holds within a sweep and within a defense family. The absolute bits-to-recovery
calibration does not transfer across defense families or across base models, so the probe ranks
recovery, it does not yet predict an absolute recovery rate. Several surfaces sit near the 0.6 bar
on a single seed.

## Proof

TODO. Requires: (i) a precise definition of "channel-matched probe" per surface; (ii) the conditions
under which a matched converse is co-monotone with realized recovery (the PriPert and spectral-MI
work give one instance); (iii) a treatment of the calibration-non-transfer as a separate, weaker
statement.

## Connections

- Supporting per-surface claims: [[depth-inversion-certificate]], [[pripert-spectral-converse-slack-comonotone-tracking]],
  [[capacity-matched-pvi]], [[spectral-channel-mi-embedding-inversion]], [[bnn-error-bounds-bhattacharyya-fano]],
  [[kv-bss-subspace-floor-and-negentropy-probe]], [[kv-cloak-channel-decoupling-feature-mix-loadbearing]],
  [[perm-llr-threshold]], [[defense-channel-selectivity-mechanism-dependent]].
- Failure-mode companion: [[probe-failure-dichotomy-matched-or-vacuous]].
- Attack-strength companion: [[attack-strength-governs-realized-leakage]].
