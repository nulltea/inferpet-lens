---
type: experiment
node_id: exp:b-kv1-accumulation-bss
title: "KV/QKV accumulation/BSS (gram_error · JADE · JD) on plaintext Qwen3 + matched negentropy probe"
idea_id: "idea:matched-probe-program"
verdict: partial
confidence: medium
date: "2026-06-24"
hardware: "CPU-only (cached operands; attacks/probe are numpy/BLAS) — GPU scale-up NO-GO (finding settled on CPU)"
duration: "~minutes (cached dev-24 capture; no GPU)"
provenance: "refine-logs/kv-accumulation/RESULTS.md; analysis_b3.json; c2_robustness.json; sanity_bss.json; pilot_dev24.json; scripts/spikes/kv_accumulation_bss.py; scripts/spikes/kv_bss_analysis.py; scripts/spikes/kv_c2_robustness.py; src/talens/attacks/bss.py; src/talens/measures/bss_separability.py"
added: 2026-06-24T00:00:00Z
tags: ["kv-cache", "qkv", "bss", "ica", "jade", "joint-diagonalization", "subspace-membership-floor", "negentropy-probe", "matched-probe", "accumulation", "weights-pub", "negative-result"]
---

# KV/QKV accumulation/BSS on plaintext Qwen3 + matched negentropy probe

**verdict:** `partial` · **confidence:** `medium` · tests
[[claim:kv-bss-subspace-floor-and-negentropy-probe]]

## Objective
Port the accumulation/BSS attack family (`gram_error` Gram-fingerprint, JADE single-observation ICA,
JD joint-diagonalization-across-`T`) onto the KV/QKV/activation surface and ask: (Q1) does recovery
scale with the number of observations `T`, and (Q2) does an attack-*independent* matched probe track
genuine recovery? Threat model WEIGHTS-PUB, Identity transform (plaintext) — the baseline before the
mixing defenses (Task 2 KV-CLOAK, Task 5 GELO).

## Setup
Qwen3-4B, dev-24 cached capture (24 prompts), layers `{0, 12, 20}`, kinds `{kq, kqv_out, resid_post}`,
`max_dim = 64`, `max_features = 256`. Attack: `talens.attacks.bss` (`jade`, `jd`, `jd_floor`,
`gram_error`). Matched probe: `talens.measures.bss_separability` (`negentropy_bits` — Hyvärinen
whitened-row negentropy; `shared_spectral_capacity_bits` — averaged-covariance water-filling). The
probe imports only the data-prep helpers `_operands, _subsample, _whiten` from the attack — never
`_joint_diag`/`jade`/`jd` — so it is computable with the attacks deleted (probe ≠ attack, audited).

## Key correction: the proper floor (the load-bearing methodological move)
The shipped `bss.jd_floor` compares a recovered source against **unrelated Gaussian** ground truth
(p95 ≈ 0.155). Under Identity (`U = H`, mixing `A = I`) that is the wrong control: any demixing `B` of
`U` yields rows in the row-span of `U`, so the Hungarian p95-cosine against `H`'s own rows is high
regardless of whether joint-diag found the right rotation. The matched null is a **random-orthogonal-
demixing floor**: same whitened data + same Hungarian pipeline, a Haar-random rotation in place of
joint-diag (`jade_proper_floor` / `jd_proper_floor`). Genuine separation = raw p95 **minus** this
floor. This is proved structurally in Lemma L1 of the claim (cross-model PASS).

## Results (bits canonical + per-secret readout)
- **gram_error** (appendix baseline): `cos_norm_distance = 0`, `row_gram_spectrum_error = 0` at every
  cell — trivially, `U = H` ⇒ the row-Gram is the fingerprint. Protocol-confirming, not a recovery claim.
- **Floor gap.** Median over 9 cells: jade raw p95 `0.776`; matched Haar-`B` floor `F_in ≈ 0.708`;
  Gaussian-GT floor `F_out ≈ 0.155`. Genuine margin = median per-cell `(raw − F_in)` = **0.027**. The
  floor-mismatch bias `F_in − F_out ≈ 0.553` ≈ 95–97% of apparent raw recovery.
- **(Q1) C1 — no accumulation.** Median JD p95 slope over 9 cells = **+0.009** per `log₂T` (flat); max
  genuine margin at any `(cell, T)` = `0.094` (`resid_post`, L20, `T = 8`); margins non-monotone in `T`.
  `T` axis `{1,2,4,8}` (`T = 16` had 0 disjoint stacks at dev-24).
- **(Q2) C2 — probe vs recovery (exploratory, n = 9).** negentropy → genuine margin: Spearman
  **0.92**, Pearson 0.95; negentropy → raw p95 (uncorrected): **−0.43**. Robustness
  (`c2_robustness.json`): permutation `p = 0.0013` (exact over 9!); leave-one-kind-out ρ ∈ {0.77,
  0.89, 0.77}; across-family-means ρ = 1.0 (monotone `kq ≪ kqv_out ≪ resid_post`); within-family
  ρ ∈ {0.5, −0.5, 0.5} (no within-family layer resolution). `shared_spectral_capacity` → JD p95:
  ρ = 0.56 (n = 36), below the 0.7 bar.

In bits: probe readout is `negentropy_bits` per cell (`kq ≈ 3.6`, `kqv_out ≈ 47`, `resid_post ≈ 280`);
recovery readout is the Hungarian p95-cosine margin (a cosine in [0,1], the natural per-secret readout
for a source-separation surface — there is no token/text decode here).

## Verdict (result-to-claim: partial; experiment-audit: WARN→resolved)
- **C1** is a sound **negative-control baseline**; it cannot assert "BSS meaningful only under a
  defense" without the mixing-defense sweeps (that is Tasks 2 & 5). Logged as the baseline that makes
  those sweeps interpretable.
- **C2** is real but statistically thin (n = 9, ordering dominated by 3 kind-families) → stated
  **exploratory** with the permutation test.
- Audit WARN: the genuine margin is `median of per-cell (raw − floor) = 0.027`, **not**
  `median(raw) − median(floor) = 0.067` — fixed in the RESULTS readout and the claim.

## GO/NO-GO for GPU scale-up: **NO-GO**
Genuine recovery is near-zero and flat in `T`; the qualitative claim (BSS ill-posed on plaintext; the
matched probe tracks the tiny genuine margin) is decided on CPU. A 512-corpus scale-up would only
tighten the slope CI on a flat line and add `T = 16/32` stacks — it cannot change the qualitative
claim, and C2's correlation is across `(kind×layer)` cells (more prompts tighten per-cell, do not
widen `n`). Conserve the single iGPU.

## Not-applicable family members (per WEIGHTS-PUB)
`sda`/`tfma` (aloepri) operate on a recovered token-id sequence, not on activations → `not_applicable`.
`ia` weight-axis needs an obfuscated weight pair `(W_plain, W_obf)`; WEIGHTS-PUB gives the true
weights → no weight secret → `not_applicable`.
