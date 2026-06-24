---
type: claim
node_id: claim:probe-failure-dichotomy-matched-or-vacuous
name: "Cross-surface: every probe-vs-recovery non-correlation resolves to probe-not-matched or vacuous-capacity, not a refutation"
description: "The four sub-threshold surfaces split cleanly into probe-not-matched (GELO, Stained Glass across shapes) and vacuous-capacity/extraction-limited (Rep2Text); each carries a queued follow-up"
node_type: claim
status: drafted
provenance: "docs/html/synthesis.html; per-surface claim nodes in Connections"
tags: [cross-cutting, synthesis, negative-result, matched-probe, measurement-loop, proof-todo]
date: 2026-06-24
added: 2026-06-24T12:40:00Z
companion: docs/html/synthesis.html
---

# Every failure of the probe to track recovery resolves to one of two diagnosable causes, never to a refutation of the thesis

**status:** `drafted` — supported by the four sub-threshold surfaces; **proof: TODO**. The dichotomy
is currently diagnosed, not demonstrated: the matched-probe and stronger-attack follow-ups that would
convert each diagnosis into a demonstration are queued, not run.

## Claim

When a probe fails to track recovery (Spearman rho < 0.6), the cause is exactly one of:
(a) **probe not matched** — the probe is a valid measure but reads a channel other than the
load-bearing leak; or (b) **vacuous capacity** — capacity exceeds the secret entropy by a wide margin,
so the probe is constant over the operating range and the attack is extraction-limited. Neither is a
counterexample to the thesis that a matched probe predicts recovery; each is a precondition being
unmet.

## Grounds

The four sub-threshold surfaces split cleanly:

- **probe not matched.** GELO: row negentropy rho 0.29–0.51 while the load-bearing leak is the
  feature Gram exposed by an orthogonal mixing. Stained Glass across noise shapes: scalar channel-MI
  rho +0.48 while read-subspace distortion (relCos / total distortion) predicts recovery at +0.97;
  the scalar capacity is provably allocation-blind at fixed budget.
- **vacuous capacity.** Rep2Text: capacity ceiling ~2856 bits far exceeds the sequence entropy, the
  capacity probe is constant across the plaintext operating range (rho +0.18 across length), and
  recovery is extraction-limited (it grows with length rather than decaying).

## Qualifier (Supported)

Each case carries a queued follow-up that would demonstrate the diagnosis: a feature-Gram-matched
probe and stronger source separation (GELO); a read-subspace probe and a shape-aware corrector
(Stained Glass); an extractable-information (V-information) probe and a stronger decoder (Rep2Text).

## Rebuttal (limitation)

Because the follow-ups have not run, the dichotomy is established by classification of the existing
results, not by re-correlating a matched probe (or re-saturating capacity with a stronger attack) on
these four surfaces. A third, currently-unobserved failure mode cannot be excluded a priori.

## Proof

TODO. Requires a formal statement that the failure set is partitioned by (a) and (b), ideally with
the matched-probe re-correlation (or stronger-attack re-saturation) actually exhibited on at least
one surface from each class.

## Connections

- probe-not-matched instances: [[gelo-orthogonal-gram-leak-rowmix-defeats-bss]], [[sgt-channel-mi-shape-blind-metric-bound-vec2text]].
- vacuous-capacity instance: [[rep2text-capacity-nonbinding-extraction-limited]].
- Parent: [[cross-surface-matched-probe-tracks-recovery]]; sibling: [[attack-strength-governs-realized-leakage]].
