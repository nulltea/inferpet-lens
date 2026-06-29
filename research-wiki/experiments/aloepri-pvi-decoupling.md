---
type: experiment
node_id: exp:aloepri-pvi-decoupling
title: "AloePri PVI 'decoupling' resolved: a threat-model artifact; in-model PVI tracks the attack"
idea_id: "idea:matched-probe-program"
verdict: partial
confidence: medium
date: "2026-06-28"
hardware: "AMD Strix Halo iGPU (gfx1151), ROCm container"
duration: "~8 min (6 layers x 3 defence levels, K=64, in-model PVI)"
provenance: "refine-logs/aloepri/aloepri_isa_hidden_levels.json; scripts/evals/static_obf/aloepri_multikey_blind.py; src/talens/probes/vinfo_capacity.py (fit_X/fit_y)"
added: 2026-06-28
tags: [aloepri, pvi, v-information, threat-model, access-control, isa-hiddenstate]
---

# AloePri PVI "decoupling" resolved — it was a threat-model artifact

**verdict:** `partial` (corrected finding; the original negative claim is retracted, the corrected
positive claim is demonstrated single-seed and not yet jury-re-judged). Earlier Codex gate (2026-06-28
run01) judged the original claim `partial`; this entry supersedes its interpretation.

## Original claim (RETRACTED)
"PVI does not correlate with AloePri defence / ISA-HiddenState recovery; PVI measures information
presence, not accessibility, so it is not channel-matched to a key-based defence."

## Why it was wrong (user objection, correct)
PVI **is** accessibility by family V, not presence. The flat ~5.8-bit PVI across no-defence/keymat/alg1
came from computing PVI in the **matched regime**: the V_cap reader (`probe_vcap` →
`v_information_capacity`, family `pca_softmax` dim 64) was fit on (deployment-obf-rep, true-token) pairs
— `train_softmax_probe(Ztr, ytr=true labels)` on the deployment basis. A fixed (secret) keymat is a
lossless linear change of basis; a reader trained on labelled examples *in that basis* learns an
equivalent inverse from the labels without ever knowing the key. That paired-deployment data is exactly
what the in-model attack (no key, no τ, no paired data) cannot obtain. So the matched PVI measured the
matched-attacker ceiling, not the blind attack — out of threat model, same defect as IMA-ridge.

## The fix (library, generic)
`v_information_capacity` gained `fit_X`/`fit_y`: the reader trains on the supplied **attack-accessible**
(rep, label) pairs and is scored on the released `(X, y)`. Default `None` = row-split of the released
`X` (in-model when the release is attacker-reproducible, e.g. DP: the public mechanism lets the attacker
self-generate noised pairs — so the DP sweep is unchanged and stays correct). For a secret-key scheme,
pass synthetic own-key reps. Same probe, same function, used for DP and static obf; only the fit pairs
differ. (`probe_vcap` passthrough; `aloepri_multikey_blind.py`: plaintext uses default, keymat/alg1 fit
on `Xclean[tr]·P_aᵏ` for K synthetic keys, score on deployment reps. Tests: `test_vinfo_fit_pairs.py`.)

## Corrected metrics (in-model PVI, pythia-160m, 6 depths, single seed)
PVI now TRACKS recovery instead of decoupling:
- PVI bits, no-defence: 5.82 5.73 5.32 5.19 4.63 4.05  (positive, declines with depth)
- PVI bits, keymat:    −10.7 −11.4 −12.5 −11.2 −15.0 −37.6  (negative = transferred reader worse than prior = 0 accessible bits)
- PVI bits, alg1:      −10.7 −4.6 −4.3 −4.1 −4.7 −38.4
- recovery (disjoint), no-defence: 0.914 … 0.467 ; keymat/alg1: ≈0 at every depth.
So: PVI high ⇔ recovery high (no-defence); PVI ≤ 0 ⇔ recovery ≈ 0 (keymat/alg1). The "decoupling"
disappears once the probe is held to attack-accessible training data.

## Corrected claim (working; demonstrated, pending re-judge)
The same capacity-matched PVI, trained only on attack-accessible fit pairs, tracks ISA-HiddenState
recovery under AloePri: positive where the keyless attack recovers (no-defence) and ≤0 (no accessible
information) where it fails (keymat/alg1). The earlier flat PVI was a matched-regime artifact. AloePri's
keymat is information-preserving (a matched/paired-data attacker still recovers), but neither an in-model
attack nor an in-model probe can read it without the key. The generic lever is the `fit_X`/`fit_y`
override, which keeps one probe valid across DP (reproducible release) and secret-key schemes.

## Queued
multi-seed + CIs; an explicit DP positive control through the SAME override (default fit) confirming PVI
tracks recovery when the release is reproducible; sensitivity to `pvi_fit_keys` (K used for the reader);
a fresh `/result-to-claim` on the corrected positive claim.

## Connections
Report: `docs/html/static-obf.html` §04 (FIG.01, in-threat-model PVI). Library:
`src/talens/probes/vinfo_capacity.py` (`fit_X`/`fit_y`). Relates to [[matched-probe-program]]. Edges in
`graph/edges.jsonl`.
