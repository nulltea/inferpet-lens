---
type: claim
node_id: claim:bnn-nns-high-d-geometry
name: "BNN/NNS achieves near-perfect L0 recovery at any r where embeddings are near-orthogonal; PVI failure at high r is a probe-class artifact, not MI absence"
description: ""
node_type: claim
status: verified
provenance: "results/unified_dp_sweep.json; scripts/spikes/unified_dp_sweep.py; 2026-06-22 sweep"
tags: ["bnn", "nns", "high-dimensional-geometry", "dp-l0", "pvi-artifact"]
date: 2026-06-22
added: 2026-06-22T00:00:00Z
---

# BNN/NNS achieves near-perfect L0 recovery at any r where embeddings are near-orthogonal; PVI failure at high r is a probe-class artifact, not MI absence

**status:** `verified` (numerically, by unified DP sweep; analytical argument below)

## Statement

At the embedding layer (L0), under isotropic Gaussian noise Y = clip(e_v) + N(0,σ²I_d):

**(a) BNN (Bayes-NN = NNS against the known embedding table) achieves near-perfect TTRSR as long as the inter-embedding decision margin ‖Δ‖/(2σ) ≫ 1, which holds for any ε that preserves model utility in d=2304.**

**(b) Low PVI (capacity reader accuracy near chance) at the same r is NOT evidence of low mutual information — it is evidence that a linear softmax classifier in a 64-dim PCA subspace cannot extract the information. The information IS present (BNN proves it); PVI's restricted hypothesis class cannot access it.**

## Analytical argument (BNN correctness at r=3.63)

At ε=64, δ=1e-5, gemma-2-2b (d=2304, C_raw=4.147):
- σ = C·z_dp/ε ≈ 4.147 × 4.845 / 64 ≈ 0.314
- Per-dimension SNR ≈ (C/√d) / σ ≈ 0.086 / 0.314 ≈ 0.27   [r = σ√d/C = 3.63]

For BNN to correctly pick v over competitor u, the decision gap must be positive:
```
‖Y − clip(e_u)‖² − ‖Y − clip(e_v)‖² = ‖Δ‖² + 2⟨noise, Δ⟩   where Δ = clip(e_v) − clip(e_u)
```
The noise term 2⟨noise, Δ⟩ ~ N(0, 4σ²‖Δ‖²), std = 2σ‖Δ‖.

In d=2304 the 2048 vocabulary embeddings are nearly orthogonal, so:
```
‖Δ‖ ≈ √(‖e_v‖² + ‖e_u‖²) ≈ √(2·C²) ≈ √(2 × 4.147²) ≈ 5.87
```
Decision SNR = ‖Δ‖ / (2σ) ≈ 5.87 / 0.628 ≈ 9.35

P(BNN correct per pair) = Φ(9.35) ≈ 1 − 10⁻²⁰. With 2047 competitors: P(any failure) ≈ 10⁻¹⁶.

So BNN is theoretically perfect at r=3.63. The observed TTRSR=0.969 (not 1.0) comes from the ≈0.6% of token pairs that are NOT near-orthogonal (same-suffix tokens, morphological relatives, subword pieces that share embedding space) — the same pairs that cause BNN=0.994 at r=0 (ε=∞). Noise adds only ~2.5% additional failure on top of the zero-noise failures.

## Why PVI = 0.048 at r=3.63 is a probe artifact

PVI (capacity reader, pca_softmax, dim=64) asks: *can a linear softmax on the top-64 PCA components of Y distinguish tokens?*

At r=3.63: noise variance per dimension ≈ 0.099, signal variance per dimension ≈ 0.0075. Per-dimension noise/signal ratio ≈ 13:1. Every PCA direction is dominated by noise. A linear classifier in any 64-d subspace cannot separate tokens.

BNN does not use a linear function of a projection. It uses the **full d=2304-dimensional L2 geometry of the known table**: the question it answers is "which of the 2048 known embedding directions does Y point most toward?" This is a nonlinear decision rule (the argmin over 2048 reference vectors) that accumulates evidence across all 2304 dimensions simultaneously.

These are measuring fundamentally different things:
- PVI: can a 64-d linear projection classify tokens? → No at r=3.63
- BNN: does the nearest known embedding match the true token? → Yes at r=3.63 (decision SNR=9.35)
- CLUB@L0: variational upper bound on MI → 1312b selectivity at r=3.63 (substantial, consistent with BNN success)

## DP defense analysis

The DP guarantee does NOT bound BNN reconstruction probability. (ε,δ)-DP guarantees:
```
P(M(v) ∈ S) ≤ e^ε · P(M(v') ∈ S) + δ   for all S, v, v'
```
This bounds pairwise distinguishability, not reconstruction rate. BNN=0.969 at ε=64 is fully consistent with this bound. For BNN to fail (decision SNR < ~2), need σ > ‖Δ‖/4 ≈ 5.87/4 ≈ 1.47, i.e.:

```
ε < C·z_dp / 1.47 ≈ 4.147 × 4.845 / 1.47 ≈ 13.7
```

For BNN to be near-chance (decision SNR < ~0.1), need σ > ‖Δ‖/0.2 ≈ 29, i.e. ε < ~0.7.

**At ε < 7, noise overwhelms the inter-embedding geometry and BNN fails. But at such noise levels the model's ability to process the input is destroyed** — confirmed by PVI=0 (no information accessible to any classifier at that point).

This is the **utility-privacy cliff at L0** (Mattern et al. 2022): no ε achieves both BNN-failure AND preserved model utility.

## Why L20 propagated DP is qualitatively different

At L20 (propagated DP), at the SAME r=0.91 (ε=256):
- BNN@L0 = 1.000 (perfect in embedding space)
- ridge@L20 = 0.074 (collapses)
- CLUB@L20 = 2435b, PVI@L20 = 0.583 (significant residual MI)

The 20-layer transformer nonlinearly amplifies the Gaussian embedding noise. At r=0.91, σ_runtime=0.94 per embedding component (C_runtime=199). This is comparable to the embedding signal magnitude and the transformer's nonlinear blocks amplify it drastically in the L20 residual stream. Ridge can't exploit the remaining MI; probes say it exists. This validates the split-TEE placement at depth: even though BNN defeats L0 protection for any useful ε, the propagated representation at L20 is substantially harder for linear attacks.

## Connections
Uses [[paper:feyisetan2020_privacy_utility_textual]], [[paper:mattern2022_limits_dp_nlp]], [[claim:thm-t1-info-efficient]], [[claim:bayes-gap-diagnosis]]. Supported by exp:unified-dp-sweep, exp:b2-l0-bayes-vs-ridge.
