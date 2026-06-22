---
type: paper
node_id: paper:feyisetan2020_privacy_utility_textual
title: "Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations"
authors: ["Oluwaseyi Feyisetan", "Tom Diethe", "Thomas Drake"]
year: 2020
venue: "WSDM 2020"
external_ids:
  arxiv: "1910.08902"
  doi: null
  s2: null
tags: ["metric-dp", "word-embedding-dp", "nns-attack", "canonical"]
added: 2026-06-22T00:00:00Z
---

# Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations

## One-line thesis
Apply calibrated multivariate Gaussian noise to word embeddings for metric (d_x-privacy) protection; the canonical reconstruction attack is nearest-neighbor search (NNS) against the known vocabulary embedding table.

## Problem / Gap
Word-level local DP (LDP) mechanisms for text typically add scalar Laplace noise in 1D mapped space, discarding the rich geometry of high-dimensional word embeddings. How to add noise that respects the embedding metric while providing formal DP guarantees?

## Method
- Parameterise a d-dimensional Gaussian noise calibrated to the L2 sensitivity of the embedding: σ = Δ_2 / ε (or similar for (ε,δ)-DP variant).
- Apply noise per token: Y = e_w + N(0, σ²I_d).
- The adversary's canonical attack: **NNS reconstruction** — given Y, output argmin_{w'} ‖Y − e_{w'}‖² over the vocabulary table. This is the MAP estimator under isotropic Gaussian noise and flat prior; it is the exact Bayes-optimal reconstruction attack.
- DP guarantee bounds the log-likelihood ratio P(M(w) ∈ S) / P(M(w') ∈ S) ≤ e^{ε·d(w,w')}, i.e. metric-DP / d_x-privacy, NOT reconstruction probability.

## Key Results
- Multivariate calibration preserves more semantic structure (cosine similarity, downstream task utility) than scalar Laplace in 1D.
- NNS reconstruction accuracy degrades gracefully with σ — but at ε values that preserve utility, NNS still substantially recovers tokens.
- The DP guarantee does NOT bound reconstruction probability; it bounds pairwise distinguishability.

## Assumptions
- Adversary knows the full vocabulary embedding table (WEIGHTS-PUB equivalent).
- Noise is isotropic Gaussian (not Laplace; metric-DP with L2 sensitivity).
- Token-level, not sequence-level; no model forward pass (embedding space only).

## Limitations / Failure Modes
- Only protects L0 (embedding space); NNS at L0 is the easiest possible attack.
- The pairwise DP bound is very loose as a reconstruction bound — high ε (weak noise) → NNS trivially reconstructs.
- Does not address propagated DP (noise injected at embedding, reshaped through transformer depth).

## Reusable Ingredients
- The NNS-against-table formulation of the reconstruction adversary is the canonical attack for any embedding-space DP mechanism.
- The distinction between DP-guarantee (log-likelihood ratio) and reconstruction probability is a recurring methodological point.

## Open Questions
- What ε is needed to actually defeat NNS? (Answer: very small ε destroys utility — the utility-privacy tradeoff is unfavorable at L0.)
- Does propagated DP (noise at L0, activation captured at L>0) provide better utility-privacy tradeoffs?

## Claims
Used by [[claim:bnn-nns-high-d-geometry]].

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Establishes NNS-against-table as the canonical L0 attack, which is exactly what BNN implements in closed form. Confirms that DP at L0 does not bound reconstruction probability, only pairwise distinguishability — the DP guarantee and BNN TTRSR=0.969 at r=3.63 are not a contradiction.
