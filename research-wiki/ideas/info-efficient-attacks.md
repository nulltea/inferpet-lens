---
type: idea
node_id: idea:info-efficient-attacks
title: "Information-efficient attacks that saturate the MI ceiling to restore measure-attack calibration"
stage: active
outcome: pending
added: 2026-06-21T11:48:59Z
based_on: ["paper:kale2025_beamclean_language_aware", "paper:guo2004_mutual_information_minimum", "paper:dai2019_database_alignment_gaussian", "paper:blackwell1953_equivalent_comparisons_experiments", "paper:pananjady2017_denoising_linear_models", "paper:chung2022_diffusion_posterior_sampling"]
target_gaps: ["gap:G1"]
tags: ["headline", "attacks", "information-theory"]
---

# Information-efficient attacks that saturate the MI ceiling to restore measure-attack calibration

**stage:** `active`  ·  **outcome:** `pending`

## Thesis
The MI-recovery decorrelation (noise barely moves CLUB/PVI but collapses ridge TTRSR / RowSort VMA) is the Bayes-optimality gap. Replace weak attacks with information-efficient ones — denoise-then-invert MMSE/MAP for inversion (channel-aware, nonlinear), and full-row Gaussian-LLR Hungarian for permutation (sufficient statistic) — so recovery climbs to the Fano/I-MMSE/2-log-n ceiling and re-correlates with the IT probes. Each attack carries a theoretic guarantee (Blackwell weak-domination, strict MI-loss improvement, I-MMSE monotonicity / DCK 2-log-n threshold).

## Key risks
I-MMSE is Gaussian-only (Laplace via degradation-DPI); finite-variance needed for heavy-tailed embeddings; single-metric converse of sufficiency invalid; BeamClean already does the inversion attack empirically (novelty is the MI-tracking PROOF, not the attack).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

