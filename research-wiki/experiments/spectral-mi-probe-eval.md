---
type: experiment
node_id: exp:spectral-mi-probe-eval
title: "B9 — spectral channel-MI I_G(σ) vs Vec2Text recovery on pooled GTR embedding under Gaussian DP"
idea_id: "idea:matched-probe-program"
verdict: supported
confidence: high
date: "2026-06-22"
hardware: "AMD Radeon 8060S iGPU (gfx1151, ROCm container)"
duration: "~5 min (probe) + recovery reuse"
provenance: "results/spectral_mi_probe_eval.json; src/talens/measures/spectral_channel_mi.py; tests/test_spectral_channel_mi.py"
added: 2026-06-23T00:00:00Z
tags: ["spectral-mi", "i_g", "vec2text", "embedding-inversion", "pooled-gtr", "dp", "matched-probe", "club", "cappvi", "fano", "geometry-only"]
---

# B9 — spectral channel-MI I_G(σ) vs Vec2Text recovery (pooled GTR embedding, Gaussian DP)

**verdict:** `supported`  ·  **confidence:** `high`  ·  validates probe in
[[claim:spectral-channel-mi-embedding-inversion]]

## Setup

Surface: pooled GTR-T5-base sentence embedding (mean-pool, d=768), the single-vector RAG/retrieval
surface where Vec2Text is the matched attack. Attack: pretrained `vec2text` `gtr-base` corrector
(Morris et al. 2023), num_steps=20, beam=1. DP: Gaussian mechanism Y=e₀+N(0,σ²I) on clip-C embeddings,
ε∈{∞,1024,512,256,128} (σ∈{0, 0.012, 0.024, 0.048, 0.095}). N=96 texts (N<d ⇒ rank-deficient Σ caveat
below). Probe: spectral channel-MI `I_G(σ)=½Σlog₂(1+λᵢ/σ²)` computed geometry-only from the clean
covariance spectrum + σ (no attack run), reported as `accessible_bits = min{H(e₀), I_G}`. Baselines:
CLUB (variational MI upper bound), capPVI (κ=40-cluster V-info accuracy).

## Metrics — bits canonical + per-secret readout

Readout (per-secret = recovered text): token-F1, cosine, exact-match, positional token-acc.
Bits: I_G / accessible_bits / CLUB-bits; Fano exact-match ceiling + RD per-token floor as converse.

```
   ε      σ │ token-F1   cos   exact │ I_G(bits) acc_bits CLUB(bits) │ Fano_exact RD_floor │ capPVI_acc
   ∞   0.000 │   0.790  0.954  0.198 │   1597     479      619       │   1.000    0.000    │   0.310
1024  0.012 │   0.446  0.827  0.000 │    312     312      541       │   0.653    0.291    │   0.241
 512  0.024 │   0.301  0.673  0.000 │    220     220      391       │   0.461    0.475    │   0.310
 256  0.048 │   0.163  0.432  0.000 │    135     135      201       │   0.284    0.656    │   0.276
 128  0.095 │   0.093  0.242  0.000 │     68      68       64       │   0.143    0.812    │   0.138
```

H(X)=H(e₀)-proxy = 479 bits (vocab 32100). At ε=∞, I_G=1597 > H(e₀) ⇒ accessible cap = min = 479 (T2:
low-noise binding ceiling is the discrete entropy, not I_G). In the privacy regime (ε≤1024) I_G<H(e₀),
so I_G is the informative ceiling and accessible_bits=I_G.

## Results

- **C1 (matched-probe correlation) — VALIDATED.** Spearman(I_G, recovery) = **+1.00** across the sweep
  for token-F1, cos, and positional token-acc — tied with CLUB (+1.00) and far above capPVI (+0.62).
  Exact-match: +0.71 (tie CLUB; floored at 0 for ε<∞ so rank-limited).
- **Cost.** I_G mean 0.060 s (cov-eigh) vs CLUB 1.67 s — **~28× cheaper** at equal correlation, and
  geometry-only (no attack, no critic training), so it satisfies the probe≠attack rule.
- **C2 (consistent with the converse) — 0 observed violations.** Every (Fano exact-match ceiling ≥
  observed exact, RD per-token floor ≤ observed token-error) pair respected across all 5 ε. RD floor
  rises 0.00→0.81 as ε falls, exact recovery 0.20→0.00 — converse never crossed. (Non-violation is the
  expected direction for a valid converse and does not by itself *prove* tightness; the proof T3 is the
  guarantee, this is empirical consistency.)
- **capPVI flat/orthogonal.** Cluster-label accuracy 0.31→0.14 only weakly tracks the order-of-magnitude
  token-recovery collapse — confirms the T2 contrast (saturates at ≤log₂κ≈5.3 bits, noise-robust on a
  scale orthogonal to fine token recovery).

## Reasoning

I_G is the certified channel-matched **converse ceiling** — its +1.0 rank-correlation with Vec2Text
recovery is empirical validation (not implied by a converse) that on this single-vector Gaussian-DP
channel the matched geometry-only probe predicts the matched attack, replacing the loose CLUB and the
orthogonal capPVI. The probe is attack-independent (computed from Σ and σ alone) so the correlation is
not circular.

## Honest scope / caveats

- **N=96 < d=768 ⇒ rank-deficient Σ.** d_eff (95→63) and the T4 localization tail are undersampled;
  the localization/bottom-mode-ablation achievability check (C3/M3) needs n≫d (estimate Σ from a large
  embedding corpus) and is **not yet run**.
- token-F1 is **not** bounded by the converse (T3b ceilings positional Hamming only); its co-monotonicity
  with the probe is consistent, not proven.
- Out-of-domain corpus ⇒ clean tF1 0.79 (below Morris in-domain 0.96); regime, not ceiling, is what
  matters for the correlation.

## Connections

- validates → [[claim:spectral-channel-mi-embedding-inversion]]
- companion → [[unified-dp-sweep]] (B8: CLUB↔recovery ρ=+1.0 on residual surfaces)
- sibling-attack → [[vec2text-feedback-null]] (the Vec2Text attack characterized on per-position resid)
