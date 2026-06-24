---
type: experiment
node_id: exp:b-r1-rep2text-capacity-nonbinding
title: "Experiment b-r1-rep2text — Rep2Text on Qwen3 L10 residual is extraction-limited, not capacity-limited"
idea_id: ""
verdict: partial
confidence: medium
date: "2026-06-24"
hardware: "AMD Strix Halo iGPU (gfx1151, ROCm container)"
duration: "~16min (capture cached + 10-epoch adapter train + 6-σ × real/shuffled eval + 5-draw null)"
provenance: "refine-logs/resid-rep2text/"
added: 2026-06-24T00:00:00Z
tags: [rep2text, residual-stream, spectral-channel-mi, capacity, negative-result, qwen3, inversion, v-information]
companion: refine-logs/resid-rep2text/EXPERIMENT_RESULTS.md
---

# Experiment b-r1-rep2text — capacity probe vacuous; recovery extraction-limited

**verdict:** `partial` (C1 refuted; genuine-leakage + capacity-vacuity supported; across-σ ordinal only) ·
**confidence:** `medium` (single seed / one source-decoder pair / N=23 per bucket)

Tests whether a single last-token residual @ L10 is a binding information bottleneck (it is **not**) and
whether a matched geometry-only capacity probe predicts Rep2Text recovery (it does **not**, across
length). Supports [[claim:rep2text-capacity-nonbinding-extraction-limited]] (proof cross-model PASS).
Refines [[exp:vec2text-feedback-null]]; reuses the probe of
[[claim:spectral-channel-mi-embedding-inversion]]. Integrity audit WARN, no fraud, probe-≠-attack PASS.

## Setup
- Surface: Qwen3-4B last-token residual @ layer 10 (d=2560), captured via talens capture (cached).
- Attack (Rep2Text, arXiv 2511.06571): adapter MLP(2560→2048→2048→8·2048) → 8 soft-prompt embeddings →
  FROZEN Qwen3-1.7B decoder (shared Qwen3 tokenizer). Teacher-forced next-token CE, 10 epochs
  (CE 3.07→2.09), 762 train / 138 test, greedy generation. Code `scripts/spikes/rep2text_run.py`;
  length-stratified natural-text corpus `corpora/rep2text-stratified.txt` (1965 prompts, PIQA/MMLU/IFEval;
  `scripts/spikes/rep2text_build_corpus.py`).
- Privacy/defense sweep: isotropic Gaussian noise on the raw residual, σ = frac·rms, frac∈{0,.5,1,2,4,8},
  rms=0.527.
- Probe: geometry-only spectral channel-MI I_G of the FULL residual ensemble at matched σ (reference
  floor σ_ref=rms/√100). Covariance-spectrum-only; never sees text / decoder outputs.
- Controls: mean-residual (~0.002), 5-draw shuffled-residual null (~0.108, draw-std 0.003); genuine
  leakage = real − shuffled, paired bootstrap (5000) 95% CIs per bucket.

## Result
- **C1 REFUTED**: I_G(plaintext)=2856 b ≫ longest-prompt entropy (≤1026 b) ⇒ capacity NON-binding.
  Recovery does not decay with length; genuine leakage gap is largest for the longest prompts.
- **Genuine leakage significant**: real > shuffled at every bucket; all bootstrap 95% CIs exclude 0
  (p ≤ 0.009); gaps +0.015..+0.089 token-F1 (modest; decoder prior dominates raw F1).
- **C2 capacity probe vacuous across length**: rd_proxy ≈ 1 for all buckets, Spearman vs F1 = 0.18
  (n=36). Across-σ Spearman(I_G,F1)=1.0 / (I_G,gap)=0.94 but ordinal-only: >80% of capacity destroyable
  (2856→520 b) before the gap moves.
- **Theory (verified)**: capacity-slack vacuity lemma — when I_G ≥ H_X(L_max), the rate-distortion proxy
  is rank-constant (=1) across length, so its length-Spearman is undefined/0 independent of the attack;
  binding threshold L > I_G/h ≈ 167 tokens ≫ tested ≤59.

## Diagnosis / follow-up
Probe-not-channel-matched (capacity is the wrong probe when slack); recovery is extraction-limited
(𝒱-information), not capacity-limited. A stronger attack cannot rescue the capacity probe in the slack
regime. → follow-up `resid-rep2text-v2` (spawn-depth 1): matched V-information probe + stronger-attack
capacity ablation + derangement null + empirical per-token entropy.
