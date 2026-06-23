---
type: dev-log
node_id: exp:resid-capacity-pvi
status: current
created: 2026-06-23
updated: 2026-06-23
tags: [experiment, resid-capacity-pvi, PVI, capacity-matched, input-DP, depth-decoupling]
companion: [refine-logs/resid-capacity-pvi/RESULTS_STANDARDIZED.md, docs/html/resid-capacity-pvi.html]
---

# Experiment log — resid-capacity-pvi (capacity-matched class-PVI consolidation)

**Surface:** residual stream (`resid_post`) + L0 embedding. **Secret:** token_id. **Model:** gemma-2-2b.
**Probe:** `v_information_capacity` (attack-independent token-id reader). **Attack:** ridge
embedding-inversion (TTRSR). **Threat model:** WEIGHTS-PUB. Full standardized numbers (bits canonical
+ per-secret readout) in `refine-logs/resid-capacity-pvi/RESULTS_STANDARDIZED.md`.

## Verdicts (jury-gated, NOT self-certified)
- **result-to-claim** (Codex xhigh, 2026-06-23, thread 019ef5cb): both claims **PARTIAL / scoped** —
  strong within gemma-2-2b, not general (single model/seed). Confidence: C1 medium-high, C2 medium.
- **experiment-audit** (Codex xhigh, thread 019ef5cd): **WARN, no FAIL.** Ground-truth, normalization,
  result-existence, **probe≠attack circularity all PASS**; Codex independently re-derived the
  Spearmans (match). Two reporting-hygiene WARNs fixed (report.py wording; "robust"/"provably"
  softened).
- **proof-checker** (Codex xhigh, thread 019ef5d7): estimator-regime lemma **PASS** (2 rounds).

## Runs consolidated (provenance JSONs under results/)
| run | file | what |
|---|---|---|
| M1 floor screen | `capacity_screen.json`, `capacity_screen_dims.json` | class-PVI −49.7 → pca_softmax −1.9, 0.57× cost; dim-anchored |
| Representation-space defenses | `nondp_intervention.json`, `nondp_intervention_l2-0.1.json` | PCA-ablation + iso-noise; cap-acc ρ 0.80–1.00 vs TTRSR L5/12/20 |
| Input-DP depth sweep | `localdp_depth_L0_5_12_20.json` | ρ(cap-acc,TTRSR) +0.96→−0.21 L0→L20; CLUB +0.96→+0.29 (parallel attenuation) |
| Faithfulness M2 (early) | `localdp_m2_*.json` | l2 sweep; selectivity ρ 0.78, partial ρ\|r 0.67 |

## Findings
1. **Estimator-regime fix (claim:capacity-matched-pvi):** class-PVI's `d>n_val` shuffle-floor
   catastrophe (−44.9 to −51.4 bits across depth) → bounded cap-PVI floor (−1.23 to −1.27 at every
   depth), dim-anchored, 0.57× cost. Floor repair holds at all depths (not L12-only). Proven mechanism
   inline (Lemma 1 / Prop 2 / Prop 3, proof-checker PASS).
2. **Positive regime:** bounded reader accuracy tracks TTRSR at ρ 0.80–1.00 under at-layer
   representation-space defenses; CLUB tracks identically.
3. **Depth decoupling (claim:depth-decoupling-input-dp):** under propagated input-DP, ρ(cap-acc,TTRSR)
   attenuates monotonically with depth, sign-reversing at L20 (−0.21); CLUB attenuates in parallel
   (+0.29, no sign reversal) ⇒ depth-attenuation is a signal property, the L20 reversal is
   readout-specific. Single-seed, n=7/layer.
4. **Readout discipline:** accuracy primary (bounded), bits auxiliary (calibration-sensitive). Do not
   call accuracy "V-information".

## Named firm-ups (queued, not this phase)
Multi-seed + bootstrap CIs; cross-model replication; dim16 sensitivity; calibration diagnostic
(NLL/ECE) on the bits-fragile cells; matched embedding-cosine readout at L20 to show geometry
collapses while id-decodability survives.

## Claims
- [[capacity-matched-pvi]] — supports (scoped, accuracy readout; estimator lemma verified).
- [[depth-decoupling-input-dp]] — supports (scoped, single-seed; CLUB parallel attenuation).
