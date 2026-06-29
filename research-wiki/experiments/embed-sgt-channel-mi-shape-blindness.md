---
type: experiment
node_id: exp:embed-sgt-channel-mi-shape-blindness
title: "embed-sgt: SGT noise-shape sweep at matched channel-MI — Vec2Text recovery vs I_G"
idea_id: ""
verdict: partial
confidence: high
date: "2026-06-24"
hardware: "AMD Strix Halo iGPU gfx1151 (ROCm container)"
duration: "~14 min (13 Vec2Text inversions, N=96 each)"
provenance: "refine-logs/embed-sgt/runs/sweep/sgt_eval.json"
added: 2026-06-24T09:15:55Z
tags: [embedding, stained-glass, sgt, channel-mi, vec2text, shape-blind, negative-result, weights-pub]
---

# embed-sgt — SGT noise-shape sweep at matched channel-MI (Task 7, Block B)

**verdict:** `partial`  ·  **confidence:** `high`

## Question

Does the geometry-only spectral channel-MI converse `I_G` predict Vec2Text recovery on the pooled
GTR sentence embedding under the Stained Glass Transform (SGT) defense — the cleanest
MI-probe-predicts-attack test? SGT differs from the Block A isotropic DP study in **exactly one
thing**: the noise *shape* (anisotropic diagonal vs σ²I). Holding `I_G` fixed and varying the shape
isolates whether the scalar `I_G` is a complete predictor or only a within-shape summary.

## Setup

- **Surface:** pooled GTR sentence embedding (`gtr-t5-base`, d=768, mean-pooled).
- **Attack:** pretrained `gtr-base` Vec2Text iterative corrector (Morris 2023) — a FIXED decoder. `scripts/evals/vec2text/vec2text_attack.py`.
- **Defense:** SGT modelled as heteroscedastic Gaussian release `Y=e0+N`, `N~N(0,D)`, `D=diag(v)`. `scripts/defenses/sgt.py`.
- **Probe (attack-independent, geometry-only):** generalized spectral channel-MI `I_G=½Σlog2(1+μ_i)`, `μ=eig(D^{-1/2}ΣD^{-1/2})`. `src/talens/measures/spectral_channel_mi.py::spectral_channel_mi_diag` (+ 13 passing tests).
- **Sweep:** budgets `B∈{826.8,434.1,196.0,71.4}` bits × shapes {`iso`, `sgt_opt` (reverse-water-filling = distortion-minimizing SGT optimum), `tail_dump` (adversarial: noise on low-λ tail)}; all hit the same target `I_G=B` by bisection (auditor-verified matched to ≤2.3e-13 bits). Plaintext shared. N=96 held-out, max_tokens=32, num_steps=20, seed 20260624.

## Results (bits canonical + per-secret readout)

| B (bits I_G) | shape | I_G | token-F1 | exact | posAcc | relCos | D_tot |
|---|---|---|---|---|---|---|---|
| ∞ (plaintext) | clip-only | — | 0.842 | 0.323 | 0.567 | 1.000 | 0.00 |
| 826.8 | iso | 826.8 | 0.433 | 0.000 | 0.115 | 0.957 | 0.11 |
| 826.8 | sgt_opt | 826.8 | **0.566** | 0.031 | 0.162 | 0.964 | 0.09 |
| 826.8 | tail_dump | 826.8 | **0.048** | 0.000 | 0.014 | 0.023 | 2419 |
| 434.1 | iso | 434.1 | 0.262 | 0.000 | 0.042 | 0.848 | 0.48 |
| 434.1 | sgt_opt | 434.1 | 0.418 | 0.010 | 0.082 | 0.882 | 0.35 |
| 434.1 | tail_dump | 434.1 | 0.065 | 0.000 | 0.013 | 0.024 | 2419 |
| 196.0 | iso | 196.0 | 0.170 | 0.000 | 0.026 | 0.636 | 1.92 |
| 196.0 | sgt_opt | 196.0 | 0.271 | 0.000 | 0.058 | 0.721 | 1.19 |
| 196.0 | tail_dump | 196.0 | 0.063 | 0.000 | 0.017 | 0.022 | 2420 |
| 71.4 | iso | 71.4 | 0.100 | 0.000 | 0.022 | 0.383 | 7.68 |
| 71.4 | sgt_opt | 71.4 | 0.159 | 0.000 | 0.032 | 0.484 | 4.16 |
| 71.4 | tail_dump | 71.4 | 0.059 | 0.000 | 0.015 | 0.024 | 2423 |

## Metrics (headline)

- **C1 (within-shape):** Spearman(I_G, token-F1) = **1.0** for iso and sgt_opt (perfect monotone, reproduces Block A); tail_dump degenerate (−0.2, recovery floored).
- **C2 (HEADLINE):** at matched I_G=826.8, token-F1 spans **0.048 → 0.566** (~12×). Across the 12 noisy settings (plaintext anchor held out): Spearman(token-F1, I_G) = **0.482** (< 0.6 bar); Spearman(token-F1, relCos) = **0.972**; Spearman(token-F1, −D_tot) = **0.951**; head-localized I_G = 0.189. (Adding the plaintext anchor moves the I_G correlation only to ≈0.59, still < 0.6.)
- **C3 (utility):** sgt_opt has lower D_tot AND higher relCos than iso at every B (genuine privacy-utility winner) — yet yields HIGHER recovery. Utility-preservation buys attack success.

## Reasoning / verdict — does-NOT-correlate-across-shape (the finding)

The scalar geometry-only converse `I_G` is a **valid converse** and a **within-shape monotone
predictor**, but is **shape-blind** across noise shapes at matched budget. Recovery of the deployed
fixed Vec2Text corrector tracks read-subspace distortion (relCos / D_tot), not the MI budget. Per
the measurement-loop doctrine this *is* the result. Diagnosis (cross-model judged, confidence high):
primarily a **non-matched-probe** finding — proven structurally (claim L2: `I_G` is a scalar function
of the whitened spectrum, cannot distinguish matched-budget allocations differing only in which modes
carry noise). A **weak-attack** arm (a shape-aware decoder could exploit the tail-mode MI `I_G`
certifies present) is left open and queued.

## Integrity

`refine-logs/embed-sgt/EXPERIMENT_AUDIT.json` — PASS on all checks (cross-model gpt-5.5, read-only).
Auditor independently verified matched-budget `I_G` ranges ≤ 2.3e-13 bits (genuinely matched, not
unequal-budget fabrication) and that the probe never reads Vec2Text. Ground truth = true held-out
corpus text (real_gt).

## Theory

Gap bounded by lemmas L1–L3 (proof-checked PASS, 2 rounds, gpt-5.5 xhigh): `claim:sgt-channel-mi-shape-blind-metric-bound-vec2text`.

## Follow-up (spawn-depth-1)

`embed-sgt-followup-1` (surface `embed-sgt-v2`): matched read-subspace / 𝒱-information probe + a
shape-aware stronger attack. Test (a) a read-subspace-SNR / effective-distortion probe that should
re-correlate, and (b) a shape-aware decoder (whiten Y by D, or per-shape retrain) that would
re-saturate `I_G` on `tail_dump` — separating non-matched-probe from weak-attack.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
