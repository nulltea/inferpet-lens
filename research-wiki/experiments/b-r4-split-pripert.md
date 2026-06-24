---
type: experiment
node_id: exp:b-r4-split
title: "Experiment b-r4-split (PriPert split-inference defense sweep)"
idea_id: ""
verdict: yes
confidence: medium
date: "2026-06-24"
hardware: "AMD Strix Halo iGPU gfx1151 (ROCm)"
duration: "~9 min GPU (pilot 98s + sweep 434s, capture cached)"
provenance: "refine-logs/resid-split/"
added: 2026-06-24T00:00:00Z
tags: [residual, pripert, split-inference, spectral-channel-mi, fano-converse, capacity-slack, weights-pub]
---

# Experiment b-r4-split

**verdict:** `yes` (scoped)  ·  **confidence:** `medium`

PriPert (arXiv 2605.23158) as a `talens` Transform (`scripts/defenses/pripert.py`): per-row top-ρ
magnitude sparsification + additive perturbation δ (channel-matched Gaussian, σ=β·meanRMS plaintext,
fixed per layer). Swept (split layer × ρ × β) on Qwen3-4B `resid_post` (cached, release-gate-512)
vs ridge/nn/mlp2 token-embedding inverters (vocab-disjoint split + shuffle control + bootstrap CI),
with the matched attack-independent spectral channel-MI probe I_G + Fano converse, CLUB cross-check.
32 cells, L∈{0,8,16,24}.

## Metrics
Spearman(I_G,best-rec)=0.958, (I_G,mlp2)=0.915, CLUB 0.977; fixed-β layer×ρ slice 0.916; within-layer
β-sweep 1.0 @ L8/L16/L24, 0.5 (n=6, fragile) @ L0. C1: perturbation-dominant, depth-dependent (L8/16
collapse by β=0.5; L24 by β≈1; L0 resists β=2→0.363). C3: 0/32 converse violations; Fano binds only
1/32 (L8 β=2, I_G=10.0<H_X−1≈10.2; recovery 0.002) — slack.

## Reasoning
C1 supported (scope depth-dependent); C2 positive measurement loop (matched probe tracks recovery
incl. stronger mlp2 — no observed probe–attack gap → no follow-up); C3 valid-but-slack converse
(capacity-slack pattern). Theory L1–L3 proof-checked PASS (2 rounds): valid converse (L1),
β-axis co-monotonicity proved (L2, ρ-monotonicity refuted by counterexample → withdrawn), converse
vacuous when d_eff≥2H_X−2 (L3). Integrity audit WARN/no-fraud (probe≠attack confirmed).

## Connections
supports → claim:pripert-spectral-converse-slack-comonotone-tracking (Spearman 0.958, within-layer
1.0, 0 converse violations). Related: exp:b-r3-gelo (sibling residual defense), exp:b-r1-rep2text
(capacity-slack). Edges recorded in `graph/edges.jsonl`.
