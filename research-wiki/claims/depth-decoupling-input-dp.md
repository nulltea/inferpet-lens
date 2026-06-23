---
type: claim
node_id: claim:depth-decoupling-input-dp
name: "Measure–attack decoupling under propagated input-DP is depth-resolved (scoped)"
description: "Empirical, jury-PARTIAL/scoped single-seed: under propagated input-DP the attack-independent token-id probe's tracking of the embedding-reconstruction attack attenuates monotonically with depth (rho +0.96 L0 → −0.21 L20, sign-reversing at L20); CLUB shows a PARALLEL attenuation (stays +0.29 at L20) ⇒ depth-attenuation is a signal property, the L20 sign reversal is readout-specific."
node_type: claim
status: drafted
provenance: ".aris/traces/result-to-claim/2026-06-23_run01/ ; refine-logs/resid-capacity-pvi/"
tags: ["empirical", "partial", "resid-capacity-pvi", "input-DP", "depth", "decoupling", "negative-as-finding"]
date: 2026-06-23
updated: 2026-06-23
---

# Measure–attack decoupling under propagated input-DP is depth-resolved (scoped)

**status:** `drafted` (empirical; jury verdict `PARTIAL`/scoped, single-seed)

## Statement (scoped after jury, 2026-06-23)
Under **propagated input-DP** (Gaussian noise injected at the input embedding and propagated through
the network), the attack-independent token-id probe's tracking of the embedding-reconstruction attack
(TTRSR) **attenuates monotonically with the layer at which the residual stream is observed**:
Spearman(cap reader accuracy, TTRSR) = **+0.96 (L0) → +0.68 (L5) → +0.43 (L12) → −0.21 (L20)** in
gemma-2-2b. CLUB — an *independent* MI upper-bound estimator — exhibits a **parallel attenuation**
(+0.96 → +0.96 → +0.89 → **+0.29**): it weakens in the same direction but **stays positive at L20** and
does **not** reproduce the accuracy readout's sign reversal. The shared *attenuation* across two
independent measures argues the depth effect is a **property of the propagated signal, not a cap-PVI
estimator artifact**; the **L20 sign reversal is specific to the bounded token-id readout** and is the
sharper, less-replicated half. Interpretation: input-DP destroys **embedding geometry** (the attack's
target) *before* **token-identity decodability** (the measure's target); the divergence localizes,
by depth, what input-DP protects (geometry) vs not (id-decodability).

**Honest scope:** single model, single seed, n=7 ε-points per layer, no bootstrap CIs. This is the
measurement-loop "non-correlation IS the finding" branch — bounded and explained, not a defect.

## Empirical status (jury-gated, NOT self-certified)
`PARTIAL` / scoped (Codex xhigh result-to-claim, 2026-06-23, confidence medium; trace
`.aris/traces/result-to-claim/2026-06-23_run01/`). The jury explicitly noted CLUB shows *parallel
attenuation*, not the identical sign reversal — "establishing" was downgraded to "suggesting" pending
multi-seed firm-up. Integrity audit **WARN, no FAIL** (`EXPERIMENT_AUDIT.md`; Codex re-derived
cap-acc 0.991/0.679/0.429/−0.214 and CLUB 0.964/0.964/0.893/0.286 from raw records — they match).

## Evidence chain
- **R3 input-DP depth sweep** (`results/localdp_depth_L0_5_12_20.json`, ε∈{∞,4096,1024,768,512,384,256},
  n=7/layer, pca_softmax dim64). Per-layer Spearman(measure, TTRSR):

  | layer | clean TTRSR | ρ(cap acc, TTRSR) | ρ(CLUB, TTRSR) | ρ(cap-PVI bits) |
  |---|---|---|---|---|
  | L0  | 0.809 | +0.96 (+0.99 avg-ties) | +0.96 | +0.71 |
  | L5  | 0.559 | +0.68 | +0.96 | +0.68 |
  | L12 | 0.347 | +0.43 | +0.89 | +0.32 |
  | L20 | 0.462 | **−0.21** | **+0.29** | −0.21 |

- **Contrast (claim:capacity-matched-pvi, R2):** under **at-layer** representation-space defenses the
  same probe tracks at ρ 0.80–1.00 at *every* depth — so the decoupling is specific to **noise
  propagation**, not depth per se.
- **Class-PVI is uninterpretable here** (shuffle floor −45 to −51 b at every depth) — the catastrophe,
  not the divergence; cap-PVI floor stays ≈ −1.2.

## Why this is a result, not a probe failure
Two attack-*independent* measures (cap reader accuracy and the CLUB MI upper bound) both lose their
grip on the attack as noise propagates deeper. If the decoupling were a cap-PVI estimator artifact,
CLUB (a different estimator, also independent of the attack) would not attenuate. It does. The
divergence is therefore information-geometric: depthwise nonlinear processing reshapes
embedding-injected noise so that embedding-reconstruction collapses before token-id decodability.

## Robustness (leave-one-out, single seed)
Removing any single ε-point from the sweep, ρ(cap-acc, TTRSR) stays positive at L12 (range
+0.09..+0.60) and stays negative at L20 (range −0.94..−0.09). The L20 sign survives single-point
deletion; its magnitude is volatile. A floor below multi-seed CIs, not a substitute — the reversal is
reported as an observation in this run.

## Open (queued firm-ups, not this phase)
Multi-seed dense-ε sweep with bootstrap/permutation CIs on each Spearman; cross-model replication;
a matched embedding-reconstruction *cosine* readout to show geometry collapses while id-decodability
survives at L20; compare propagated input-DP vs at-layer noise at matched effective hidden-state
distortion.

## Connections
Companion repair claim [[capacity-matched-pvi]] (same surface, the positive regime + estimator theory).
Independence backbone [[threat-model-fairness]]. MI comparator [[mi-monotone-gaussian]].
_Edges recorded in `graph/edges.jsonl`._
