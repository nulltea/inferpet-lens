---
type: claim
node_id: claim:attack-strength-governs-realized-leakage
name: "Cross-surface: attack strength, not the measure, governs whether present information is realized as recovery"
description: "A weak linear attack can decorrelate from a probe that a stronger (Bayes-optimal / learned / maximal-invariant) attack re-correlates with, on the same surface"
node_type: claim
status: drafted
provenance: "docs/html/synthesis.html; per-surface claim nodes in Connections"
tags: [cross-cutting, synthesis, attack-efficiency, measurement-loop, weights-pub, proof-todo]
date: 2026-06-24
added: 2026-06-24T12:40:00Z
companion: docs/html/synthesis.html
---

# Attack strength, not the measure, governs whether present information is realized as recovery

**status:** `drafted` — supported across three surfaces; **proof: TODO** (the general "stronger
admissible attack re-correlates a matched probe" statement is the T1 information-efficiency line and
is not yet proved in this cross-surface form).

## Claim

On a fixed surface, a weak attack can show no correlation with a matched probe while a stronger
admissible attack (under public weights) re-correlates with it. The bits the probe certifies are
present are realized as recovery only by an attack efficient enough to extract them, so a
non-correlation under a weak attack is evidence about the attack, not about the measure.

## Grounds

- Propagated input-layer differential privacy at depth: a ridge attack gives rho −0.09 against the
  probe, a learned decoder gives +0.83.
- Input layer under differential privacy: a Bayes-optimal nearest-neighbour decode adds +0.98
  recovery over ridge.
- Permutation table: the full sorted-row matcher (the maximal invariant of the column permutation)
  recovers 0.43–0.60 more than a 64-bin quantile baseline and re-correlates the probe at +0.976.

## Qualifier (Supported)

Single seed. The stronger attacks are admissible under the public-weights threat model and are the
strongest tried, not provably optimal except at the input layer (Bayes-nearest-neighbour) and for
the permutation cover (maximal invariant).

## Rebuttal (limitation)

Because the attack class is not exhausted, a still-stronger attack could move a surface currently
labelled "tracks" or re-correlate one currently labelled "probe not matched". The claim is therefore
asymmetric: a re-correlation under a stronger attack is informative; a continued non-correlation is
not conclusive evidence of absent information.

## Proof

TODO. The input-layer Bayes-optimality and the permutation maximal-invariant arguments are proved on
their surfaces ([[thm-t1-info-efficient]], [[weak-domination]], [[perm-llr-threshold]]); the
cross-surface generalization is not yet stated.

## Connections

- Per-surface support: [[restore-correlation]], [[thm-t1-info-efficient]], [[weak-domination]],
  [[bayes-gap-diagnosis]], [[perm-llr-threshold]], [[depth-decoupling-input-dp]].
- Parent: [[cross-surface-matched-probe-tracks-recovery]]; sibling: [[probe-failure-dichotomy-matched-or-vacuous]].
