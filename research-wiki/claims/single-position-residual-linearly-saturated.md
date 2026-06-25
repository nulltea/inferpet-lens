---
type: research
status: current
created: 2026-06-25
updated: 2026-06-25
tags: [claim, residual, token-recovery, decoder, tuned-lens, affine, differential-privacy]
companion: docs/html/resid-dp-attacks.html
---

# Claim: single-position residual→token recovery is ~linearly saturated — a non-linear decoder losing to ridge at L0 is expected, not a bug

**Claim.** For recovering the token identity from a *single per-position* residual-stream vector
`r_L ∈ ℝ^d` (WEIGHTS-PUB; with or without input-DP), a **regularized affine map decoded against the
frozen embedding/unembedding table** — i.e. our ridge attack, which is exactly the **tuned-lens**
recipe — is the near-optimal attack. A non-linear decoder **at best matches** it at a near-linear
layer and is **expected to underperform a regularized linear baseline at L0** unless its linear part
is warm-started to the ridge solution. Recovery gains *beyond* the affine baseline require
**sequence/pooled context or an LM prior**, not a deeper per-vector regressor. Therefore the observed
"`decoder < ridge` at L0" is a **predicted property of the surface + data regime, not an
implementation defect**.

**Status: Supported** (literature + our L0 observation), single-seed empirics.

## Evidence

1. **Our measurement (gemma-2-2b, clean ε=∞, candidate pool 2048).** At L0 ridge top-1 = **0.966**
   vs the non-linear decoder **0.673**; ridge dominates at every ε at L0. A plain `Linear→ReLU→Linear`
   (h=384) decoder lost to ridge; widening to h=1024 + a linear skip **still** lost at L0. Only
   warm-starting/freezing the linear path to ridge makes the decoder ≥ ridge.
   (`refine-logs/dp-decoder-grid/dp_leakage_sweep.json`; pilots `pilot2.json`, `pilot3.json`.)
2. **Data regime is the binding constraint.** ~2,300 vocab-disjoint train rows vs a 5–10M-parameter
   decoder = **~2,000–4,000× overparameterized** (feature dim d=2304; ridge's α=1 penalty is
   load-bearing at n/d≈1). Capacity past the signal is pure variance on a vocab-disjoint test split.
3. **Why an MLP is not a free superset of linear here.** An MLP ⊇ linear only with (a) enough width
   — a hidden `h<d` routes the linear part through a rank-≤h bottleneck, not representable; and (b)
   an optimizer/data that *reaches* the linear least-squares solution — which ReLU-from-scratch on a
   few hundred rows does not. The embedding table `E` supplies the token geometry for free, so a
   linear map into `E` (ridge / tuned lens) exploits it data-efficiently where a free-target
   regressor must relearn it.

## Literature

- **Tuned Lens** (Belrose et al. 2023, arXiv:2303.08112): the validated residual→token decoder is
  **affine** (`A·h+b` composed with the frozen unembedding), **identity-initialised**; they report
  **no non-linear lens** and reject per-layer learned unembeddings as too data-hungry.
- **Embedding-inversion SOTA** uses shallow/near-linear front-ends + **generative LM-prior** decoders,
  not wide per-vector regressors: vec2text (Morris 2023, 2310.06816, iterative T5 corrector),
  GEIA (2305.03010, generative + linear projection), Rep2Text (2511.06571, narrow gated-skip MLP →
  soft tokens → frozen LLM), Song & Raghunathan (2004.00053).
- **Probing-power tradeoffs** (Hewitt & Liang 2019; Voita & Titov MDL 2020; Pimentel 2020): in the
  small, vocab-disjoint regime overfitting binds; low-capacity probes are preferred.
- **BeamClean** (2505.13758): under DP, recovery comes from a **frozen LM prior in a beam decode**
  fused with the noise model; per-vector decoders saturate, and the LM-prior advantage *grows* with
  noise.

## Implication (how it is used)

- Report **ridge as the principled single-position baseline** (= affine-into-`E` / tuned lens), not a
  strawman.
- Use a **ridge-warm-started, frozen-linear, gated GELU decoder** (narrow, early-stopped) as a
  `≥ ridge` probe of *residual non-linearity* — expect only modest, depth-localized gains.
- Pursue **LM-prior beam decoding (BeamClean-style)** for genuine gains, especially under DP — the
  real "stronger attack" axis. Recorded as a campaign-D task.
