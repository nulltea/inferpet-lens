---
type: experiment
node_id: exp:aloepri-ima-embedrow-transformer
title: "IMA-EmbedRow-transformer (paper §F.1) on pythia: τ-invariant inverter defended (≈0) at every αₑ; plain control passes (0.998)"
idea_id: "idea:matched-probe-program"
verdict: partial
confidence: medium
date: "2026-06-29"
hardware: "AMD Strix Halo iGPU (gfx1151), ROCm container"
duration: "~18 min (3 configs, n_eval=12k, 40 epochs)"
provenance: "refine-logs/aloepri/aloepri_ima_transformer.json; scripts/evals/static_obf/aloepri_ima_transformer.py; src/talens/attacks/dp_inversion.py (IMAInverter, ima_transformer_attack)"
added: 2026-06-29
tags: [aloepri, ima-embedrow, transformer, paper-F1, static-weights, defended]
---

# IMA-EmbedRow-transformer (paper §F.1) — ported and measured

Canonical IMA: a 2-layer / 8-head transformer inverter (`talens.attacks.dp_inversion.IMAInverter`,
ported from private-rag `run_ima_paper_like.IMAInverter`) trained on (obf-row → plain-embedding) MSE.
Two threat models, swept over αₑ:
- **plain control (matched, single key, row-split)** — validates the inverter inverts when a common
  inverse exists.
- **τ-invariant (in-model)** — train on K=4 synthetic OWN-key obfuscations of the public table
  (Kerckhoffs), decode the deployment table's held-out rows. The paper-faithful in-model attack.

## Results (pythia-160m, static embedding table)
| defence | αₑ | plain control (matched) | τ-invariant (in-model) |
|---|---|---|---|
| keymat | 0   | 0.998 | 0.000 |
| alg1   | 1.0 | 0.915 | 0.000 |
| alg1   | 2.0 | 0.121 | 0.000 |

## Findings
1. **τ-invariant ≈ 0 at every αₑ, including keymat (αₑ=0): AloePri defends against the canonical IMA, and
   the defense is the KEYMAT (basis hiding), not the noise.** No single inverter inverts a per-key random
   change of basis it never saw. Reproduces the paper's reported IMA ≈ 0 %.
2. **The plain control passes (0.998 at keymat)** — so the ≈0 is a real defense, not a broken attack.
   This is the control private-rag's driver FAILED ("trained inverter fails the identity-τ plain control");
   our port passes it. The blocker was undertraining: the d=768 inverter needs ample rows/epochs
   (≈12k tokens, 40 epochs); small sets underfit to ≈0 and are uninterpretable.
3. αₑ degrades even the matched control (0.998→0.915→0.121), the same sharp αₑ lever seen on the residual
   bootstrap — but it is irrelevant to the in-model τ-invariant number, which is ≈0 from the keymat alone.

## Scope
Single seed; static embedding-table surface; IMAInverter is a faithful-architecture port (vanilla
MHA/GELU, not Qwen3 RoPE/QK-norm — the privacy claim is architecture-agnostic). The L0 hidden-state
variant (IMA-L0-transformer) is redundant with this (same info up to RMSNorm rescaling, per the doc).

## Connections
Attack: `talens.attacks.dp_inversion.IMAInverter` / `ima_transformer_attack` (tests/test_ima_transformer.py,
single-key plain control). Eval `scripts/evals/static_obf/aloepri_ima_transformer.py`. Report
`docs/html/static-obf.html` §11. Relates to the dropped IMA-EmbedRow-ridge (out-of-model) and
[[aloepri-partial-tau-bootstrap]] (the in-model k-harvest ridge that DOES bootstrap the table).
