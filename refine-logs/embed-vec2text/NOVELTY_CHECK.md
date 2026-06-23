# Novelty Check — spectral channel-MI matched probe (embed-vec2text)

**Date**: 2026-06-23 · **Reviewer**: Codex gpt-5.5 xhigh (thread 019ef5f6) + WebSearch.

## Proposed contribution
Closed-form geometry-only Gaussian water-filling spectrum I_G(σ)=½Σlog₂(1+λᵢ/σ²), from the clean
embedding covariance + σ alone (no attack), as an attack-INDEPENDENT converse (Fano + RD + localization)
and empirical rank-predictor of Vec2Text inversion recovery for DP text embeddings; supersedes CLUB
(learned/loose) and capPVI (cluster-label) as the matched probe.

## Verdict: novelty 6.5/10 — PROCEED WITH CAUTION (narrow the claim)
Novel as a **framing/finding built from known pieces**, NOT a new information-theoretic method. No prior
work does exactly: empirical-covariance-spectrum AWGN MI, computed without running an inversion attack,
used as an attack-independent converse + rank-predictor for DP text-embedding inversion.

## Closest prior work
| Paper | Overlap | Delta |
|---|---|---|
| Eguard, arXiv:2411.05034 (AAAI) | MI for embedding-inversion privacy | learned autoencoder MI DEFENSE objective; not closed-form, not DP-channel-matched, not attack-independent |
| Zhuang et al. 2024, arXiv:2402.12784 | Vec2Text recoverability under Gaussian-noise sweeps | still RUNS Vec2Text; studies mitigation; no covariance-spectrum probe / no-attack converse |
| NVDP, arXiv:2601.02307 | DP + variational IB for transformer embeddings | RDP/BDP learned noisy embeddings, not spectral AWGN capacity or inversion converse |
| Concept-Aware/SPARSE, arXiv:2602.07090 | DP embeddings, anisotropic/Mahalanobis noise | not MI capacity, not attack-independent prediction |
| Constructing Privacy Channels, arXiv:1910.09235 | DP↔channel-capacity generically | not text embeddings, not inversion, not empirical rank prediction |
| Morris Vec2Text, arXiv:2310.06816; GEIA arXiv:2305.03010 | the attack | not the probe |

## Hostile-reviewer attacks + defensible positioning
1. Formula/Fano/RD are textbook → frame as "known AWGN MI **newly operationalized** for DP text-embedding
   inversion", geometry-only surrogate, not a new theorem.
2. **Monotonicity confound (most important):** +1.0 Spearman across a monotone σ-sweep may be "everything
   decreases with σ". Defense = anti-confound controls: compare I_G against raw σ, trace-SNR, effective
   rank, total variance, log-det-without-spectrum; vary model/dataset/clip/noise mechanism (not one
   monotone knob). This is exactly the queued R005 "break-monotone-knob" follow-up (Laplace/partial-dim).
3. MI-as-leakage is old (Eguard) → position as complement to learned MI estimators (CLUB/Eguard), the
   cheap attack-free converse, not a replacement for attacks "in general".

## Positioning line (adopted)
"A closed-form Gaussianized upper-bound leakage probe for DP text embeddings that converts the empirical
embedding spectrum into an attack-independent converse and empirically rank-predicts inversion difficulty
without training or running an attack — complementing learned MI estimators."

## Actions
- Claim updated with this positioning + Zhuang 2402.12784 / Eguard 2411.05034 as related-work refs.
- Anti-confound controls (vs raw σ / trace-SNR / effective-rank; Laplace/partial-dim DP) recorded as a
  matched-probe-program follow-up (already R005 in the tracker) — NOT run this consolidation phase.
