---
type: experiment
node_id: exp:dp-utility-vs-eps-160m
title: "DP input-noise utility collapse vs ε (perplexity-retention, Pythia-160M)"
idea_id: "idea:matched-probe-program"
verdict: yes
confidence: high
date: "2026-06-26"
hardware: "AMD Strix Halo iGPU (gfx1151), ROCm container, fp32 batched"
duration: "~few min (8 forwards × 1800 prompts, Pythia-160M)"
provenance: "refine-logs/pythia-depth/dp_utility.json; scripts/eval/dp_utility_sweep.py"
added: 2026-06-26T16:44:30Z
tags: ["dp", "utility", "perplexity-retention", "epsilon-sweep", "pythia-160m", "privacy-utility-tradeoff", "grid-calibration", "dp-forward", "scale-invariant"]
---

# DP input-noise utility collapse vs ε (perplexity-retention, Pythia-160M)

**verdict:** `yes`  ·  **confidence:** `high`  ·  tests `idea:matched-probe-program`

## Objective
Measure the **utility** side of the DP privacy–utility tradeoff for LocalDP (Gaussian noise on the
input embedding, propagated through the model), and check whether the leakage sweep's ε grid
(`∞,64,32,16,8`) sampled a regime where the model is still useful. Utility = the model's own output
fidelity vs its non-private (ε=∞) baseline — the DP-LLM convention (Yu et al. 2021 arXiv:2110.06500;
Li et al. 2021 arXiv:2110.05679; DP-Forward / Du et al. 2023 arXiv:2309.06746), scale-invariant
because every metric is referenced to the model's own ε=∞.

## Setup
Pythia-160M, fp32 batched capture, 1800 prompts (`corpora/rep2text-stratified.txt`). Teacher-forced
next-token: one forward per ε (no generation). σ = C·z/ε with C the 99.9-pct embedding norm (0.818),
z=4.845 (δ=1e-5) — same convention as the leakage sweep. Metrics per ε: perplexity, ground-truth
top-1 accuracy, accuracy-retention = acc(ε)/acc(∞), and top-1 agreement vs the clean run.
`scripts/eval/dp_utility_sweep.py`.

## Results
Baseline ε=∞: **ppl 42.6, next-token acc 0.341.**

| ε | σ/C | ppl | acc | acc-retention | agree-clean |
|---|---|---|---|---|---|
| ∞ | 0 | 42.6 | 0.341 | 1.000 | — |
| 512 | 0.009 | 44.9 | 0.335 | **0.984** | 0.87 |
| 256 | 0.019 | 55.7 | 0.312 | **0.917** | 0.73 |
| 128 | 0.038 | 222 | 0.203 | **0.596** | 0.43 |
| 64 | 0.076 | 2 929 | 0.049 | **0.145** | 0.10 |
| 32 | 0.151 | 9 432 | 0.013 | 0.038 | 0.04 |
| 16 | 0.303 | 16 971 | 0.010 | 0.029 | 0.03 |
| 8 | 0.606 | 29 948 | 0.011 | 0.033 | 0.02 |

**Utility budgets (linear interp on log ε): −10% / −20% / −50% accuracy-retention at ε ≈ 247 / 199 / 110.**

## Interpretation
- **The leakage ε grid was mis-calibrated.** Every noised point (`64,32,16,8`) sits **below ε≈110** —
  the model-destroyed regime. At ε=64 utility is already −85% (perplexity 69× worse, 10% next-token
  agreement with clean). The meaningful privacy–utility band (−10% to −50% utility) is **ε∈[110,247]**,
  which the leakage sweep never sampled.
- **Reframes the leakage results (worst-quadrant warning).** At ε≤64 the input token stays partly
  recoverable (ridge 0.28–0.84 across depth) **while the model is non-functional** — leakage from a
  broken model, not a useful defense operating point. The defensible operating regime is **ε∈[128,512]**.
- **Action:** the 1.4B (and any rerun) ε grid should target the utility band, e.g.
  `∞,512,256,192,128,110`, not `64,32,16,8`. Utility is one curve over ε (the noise enters once at
  input; independent of which layer is released), so it is cheap to sweep densely.

## Connections
- Utility counterpart to the leakage sweep `refine-logs/pythia-depth/dp_leakage_sweep.json`
  (recovery + PVI/I_G/sep), rendered in `docs/html/resid-dp-attacks.html`.
- Method grounded in [[paper:feyisetan2020_privacy_utility_textual]] and the DP-LLM utility convention
  (Yu 2021, Li 2021, DP-Forward 2023 — the forward-pass-noise analog).
- Bears on [[claim:depth-decoupling-input-dp]]: the depth decorrelation was measured in the
  utility-destroyed ε regime, so its operating-point relevance needs the [128,512] band.

