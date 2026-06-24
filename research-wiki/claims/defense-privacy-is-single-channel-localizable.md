---
type: claim
node_id: claim:defense-privacy-is-single-channel-localizable
name: "Cross-surface: deployed defenses concentrate privacy in a single load-bearing parameter that a matched probe localizes"
description: "KV-Cloak feature mix, PriPert perturbation budget, AloePri dense key cover are each the sole load-bearing channel; no single scalar privacy level reduces leakage uniformly across secret kinds"
node_type: claim
status: drafted
provenance: "docs/html/synthesis.html; per-surface claim nodes in Connections"
tags: [cross-cutting, synthesis, defense, channel-decoupling, localization, proof-todo]
date: 2026-06-24
added: 2026-06-24T12:40:00Z
companion: docs/html/synthesis.html
---

# Deployed defenses concentrate their privacy in a single load-bearing parameter, and a channel-matched probe localizes it

**status:** `drafted` — supported across three defenses; **proof: TODO**.

## Claim

For the multi-parameter defenses studied, exactly one parameter (or channel) carries the privacy:
varying it moves attack recovery, while the others are recovery-irrelevant. A channel-matched probe,
together with a channel-ablation sweep, identifies which one.

## Grounds

- KV-Cloak: the feature mix is the only channel that reduces source-separation recovery; the token
  mix and the block size are inert.
- PriPert split inference: the perturbation budget collapses token recovery by a moderate value
  (beta ~ 0.5), while the sparsity ratio is secondary.
- AloePri permutation cover: the dense key cover is load-bearing, with the permutation core
  vulnerable up to noise alpha_e ~ 0.35.

## Qualifier (Supported)

Established by channel-ablation and parameter sweeps on each defense; single seed per defense.

## Rebuttal (limitation)

No single scalar privacy level reduces leakage uniformly across secret kinds: the additive-noise
defense drives permutation recovery from 1.00 to 0.04 while leaving token identity above 0.45.
Localization is therefore per-defense and per-secret, and the bits-to-recovery calibration does not
transfer across defense families (see [[defense-channel-selectivity-mechanism-dependent]]).

## Proof

TODO. Requires a per-defense statement of which channel is information-bearing (the KV-Cloak and
GELO Gram-identity arguments are one route) and a general criterion for "load-bearing channel".

## Connections

- Per-surface support: [[kv-cloak-channel-decoupling-feature-mix-loadbearing]],
  [[gelo-orthogonal-gram-leak-rowmix-defeats-bss]], [[pripert-spectral-converse-slack-comonotone-tracking]],
  [[perm-llr-threshold]], [[defense-channel-selectivity-mechanism-dependent]].
- Parent: [[cross-surface-matched-probe-tracks-recovery]].
