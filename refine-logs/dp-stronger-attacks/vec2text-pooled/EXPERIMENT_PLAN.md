---
type: plan
status: current
created: 2026-06-22
updated: 2026-06-22
tags: [experiment-plan, vec2text, pooled-embedding, gtr, dp, matched-probe, embedding-geometry]
companion: [EXPERIMENT_TRACKER]
supersedes: []
---

# Experiment Plan — Faithful Vec2Text on a pooled embedding under DP (the surface where it belongs)

**Problem**: B7 + B7-analysis (`../EXPERIMENT_RESULTS.md`) showed Vec2Text is the WRONG
attack for the per-position `resid_post` surface — that surface gives each token its own
observation (per-position invertible, logit-lens readable; arXiv 2510.15511), so there is no
pooled bottleneck for Vec2Text's iterative re-embed + residual feedback to resolve, and our
"corrector" was a per-position MLP probe, not Vec2Text (trains in seconds vs the real
T5-seq2seq's days). Vec2Text's native surface is a **single pooled bottleneck vector** (a RAG /
retrieval sentence embedding). This plan moves the attack to that surface, using the **real
`vec2text` dependency with its PRETRAINED gtr-base corrector** (no multi-day training), and asks
the repo's actual research question there.

**Method Thesis**: On the embedding-geometry channel, the cheap matched IT probe (CLUB
`I(noised-emb; clean-emb)` / capacity-PVI) **tracks the recovery of the strongest available attack
(faithful, iterative+beam Vec2Text)** across a DP-noise sweep — the probe is a faithful predictor
of *achievable* leakage even against SOTA inversion, on the surface where that attack is sound.

**Threat model**: WEIGHTS-PUB-analog for embeddings — the adversary has the (public) encoder φ =
GTR-base and the published DP params (clip C, σ), observes the DP-protected pooled embedding, and
holds the public Vec2Text corrector matched to φ. Admissible = uses only this. Defense (clip+Gauss
on the released embedding) is an external `Transform`; the library asserts nothing (scheme-agnostic).

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| **C1** (sanity/anchor) | Proves the faithful dependency works & the surface leaks | Clean pooled GTR embeddings → Vec2Text (20 steps+beam) recovers text in the Morris ballpark (token-F1 ≳ 0.8, exact > 0 on 32-tok) | B1 |
| **C2** (calibration) | The leakage-vs-privacy-budget curve — the headline practical result | Recovery (token-F1/exact/cos) decays **monotonically** as ε falls over {∞,1024,512,256,128}; locate the collapse ε | B2 |
| **C3** (PRIMARY — matched probe) | Does the cheap probe predict what the SOTA attack achieves? | Spearman(probe, Vec2Text recovery) over the ε sweep ≥ ~0.8 for CLUB and/or capPVI; probe ranks ε-leakage like the strong attack | B2 |
| **C4** (info-efficiency / novelty isolation) | Separates *achievable* leakage (probe) from *search-limited* realized leakage (attack strength) | At FIXED ε, recovery rises 0-step < 1-step < 20-step+beam toward a ceiling the probe (constant in step/beam) predicts; Vec2Text ≫ NN-retrieval / BoW baseline | B3 |

**Anti-claims to rule out**:
- *"The probe tracks only because everything decays monotonically with ε (common-cause)"* — the
  exact confound B3/B4 of the main batch flagged. Mitigation: (a) the **search axis** (num_steps,
  beam at fixed ε) — recovery varies but the probe is **constant**, so C3 is read on the noise axis
  while C4 uses the orthogonal search axis; (b) B4 (nice-to-have) breaks the single monotone knob
  with partial-dim / Laplace DP and checks the probe still tracks.
- *"Vec2Text is decoration; a trivial attack does as well"* — ruled out by C4's NN-retrieval / BoW
  baseline (the information-efficiency gap).

## Paper Storyline
- **Main paper must prove**: C2 (the DP leakage-calibration curve under a SOTA attack) and C3 (the
  matched probe predicts it). Together: *a cheap IT probe forecasts the privacy budget at which even
  the strongest faithful inversion fails — on the embedding surface where that inversion is sound.*
- **Appendix can support**: C1 (clean replication), C4 strength-axis + baselines, B4 anti-confound,
  qualitative reconstructions.
- **Intentionally cut**: re-training a Vec2Text corrector (use pretrained); ada-002 (needs paid API
  — GTR is the offline, reproducible encoder); per-position resid (disproved in B7).

## Experiment Blocks

### Block 1 — Clean replication / pipeline sanity (C1)  — MUST-RUN
- **Claim tested**: C1. **Why**: prove the real dependency runs on the ROCm container and the pooled
  surface leaks before any DP is added (no false negatives downstream).
- **Dataset/split**: N≈500 texts truncated to 32 tokens (Morris regime). Source: a Wikipedia/NQ-style
  sample or the repo corpora truncated; held-out from any pool used in baselines.
- **Compared systems**: Vec2Text [0 steps] (base inverter) vs [20 steps, sbeam=4/8].
- **Metrics**: token-F1, BLEU, exact-match, cos(φ(recon), e). Decisive: token-F1 + exact.
- **Setup**: `vec2text.load_pretrained_corrector("gtr-base")`; encoder
  `AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder`; `mean_pool` +
  (GTR normalizes) → `invert_embeddings(emb, corrector, num_steps, sequence_beam_width)`. All via
  `scripts/run_in_rocm.sh`.
- **Success**: 20-step+beam token-F1 ≳ 0.8 (Morris reports ~0.97 tF1 / 92% exact at 32-tok+sbeam;
  ballpark, not exact, is enough). **Failure interpretation**: if clean recovery is poor →
  dependency/encoder/pooling/version mismatch, not a leakage finding — fix the pipeline first.
- **Table/figure**: Table "clean recovery, base vs Vec2Text".

### Block 2 — DP sweep: leakage calibration + matched probe (C2, C3) — MUST-RUN
- **Claim tested**: C2, C3 (primary). **Why**: the headline curve and the probe↔attack correlation.
- **Dataset**: same N≈500 (32-tok), fixed across ε.
- **Defense**: on the pooled embedding e: clip to norm C = percentile_{99.9}(‖e‖ over a calib set),
  then e' = e + N(0, σ²I), σ = C·z/ε, z = √(2 ln(1.25/δ)), δ=1e-5. ε ∈ {∞,1024,512,256,128}.
  (Mirror of `InputDPCover` but on the released embedding; Morris Limitations: corrector trained on
  un-noised embeddings — so this is the realistic non-adaptive defense.)
- **Compared systems**: Vec2Text [20 steps, sbeam=8] (the strong attack) at each ε.
- **Probes**: `talens.measures.club.club_mi_upper_bound(noised_emb, clean_emb)` → CLUB bits;
  `talens.measures.vinfo_capacity.v_information_capacity` adapted to the embedding (reader on noised
  emb predicting a text-derived label, e.g. token-set / cluster id) → capPVI. Compute per ε.
- **Metrics**: recovery (token-F1/exact/cos) per ε; CLUB bits, capPVI per ε; **Spearman(probe,
  recovery) over ε**.
- **Success**: recovery monotone ↓ in ε (C2); Spearman ≥ ~0.8 for ≥1 probe (C3). **Failure**: if
  probe is flat while recovery moves → probe is not matched to this channel (report honestly, like
  B7's negative).
- **Table/figure**: Figure "recovery & probe vs ε" (dual axis); the calibration curve.

### Block 3 — Attack-strength axis + weak baselines: information-efficiency (C4) — MUST-RUN
- **Claim tested**: C4. **Why**: separate *achievable* leakage (probe, constant in search) from
  *search-limited* realized leakage (attack strength); isolate that Vec2Text's machinery matters.
- **Design**: 2D cells = ε ∈ {∞,512,128} × strength ∈ {0-step, 1-step, 20-step+sbeam8}. Plus weak
  baselines at each ε: (i) NN-retrieval — cosine-match the noised emb to a held-out **text pool**'s
  clean embeddings, return the pool text (the trivial attack); (ii) BoW logistic (Song&Raghunathan
  2020 style) — optional 3rd family.
- **Metrics**: token-F1/exact per cell; gap(Vec2Text20 − NN), gap(Vec2Text20 − 0-step).
- **Success**: recovery strictly increases with search at fixed ε toward a ceiling ranked by the
  probe across ε; Vec2Text[20] ≫ NN-retrieval (info-efficiency gap > 0 at every ε with leakage).
  **Failure**: if NN-retrieval ≈ Vec2Text → the surface leaks trivially and Vec2Text is not the
  needed lever (still a valid finding about the channel).
- **Table/figure**: Table "info-efficiency: weak vs base vs Vec2Text × ε".

### Block 4 — Anti-confound: break the single monotone knob (C3 robustness) — NICE-TO-HAVE
- **Claim tested**: C3 is not a common-cause artifact. **Why**: B3/B4 showed a single monotone noise
  knob inflates probe↔attack correlation. **Design**: partial-dim DP (noise a random fraction p of
  the 768 dims, sweep p at fixed total σ) and/or Laplace vs Gaussian — a non-ε manipulation that
  changes geometry. Check Spearman(probe, recovery) holds across these too. **Success**: probe still
  tracks under a second defense-geometry axis.

### Block 5 — Qualitative privacy story (failure analysis) — NICE-TO-HAVE
- Example reconstructions at ε ∈ {∞,512,256,128} (the "what an attacker reads" panel); word-frequency
  vs recovery (Morris Fig 6 analog). **Table/figure**: qualitative box + frequency plot.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| **M0** | dependency smoke test | install `vec2text` in ROCm container; load gtr-base corrector; invert 4 clean texts | corrector loads + inverts on the iGPU (cuda-visible) | ~5 min | **vec2text torch/transformers pins vs ROCm torch** (highest risk) |
| **M1** | C1 clean replication (B1) | base vs 20-step+beam on N=500 clean | clean token-F1 ≳ 0.8 | ~10 min | pooling/normalization mismatch |
| **M2** | C2+C3 DP sweep (B2) | 5 ε × Vec2Text[20,beam8] + CLUB + capPVI | monotone decay + Spearman ≥ ~0.8 | ~15–20 min | probe adaptation to embedding label |
| **M3** | C4 strength + baselines (B3) | 3 ε × 3 strengths + NN-retrieval (+BoW) | Vec2Text ≫ NN; search→ceiling | ~15 min | pool leakage in NN baseline (use disjoint pool) |
| **M4** | anti-confound + qualitative (B4,B5) | partial-dim DP; example dumps | probe robust to 2nd axis | ~15 min | — |

Must-run: **M0 → M1 → M2 → M3**. Nice-to-have: M4.

## Compute and Data Budget
- **Total**: ~1 GPU-hour across must-run milestones (GTR-base encoder + T5-base corrector inference;
  N≈500; 20 steps + sbeam8). Each run ≤20 min on the single iGPU.
- **Data**: ~500 short (32-token) texts + a disjoint text pool for the NN baseline. No training data
  (pretrained corrector).
- **Human eval**: none (automatic metrics).
- **Biggest bottleneck**: M0 dependency compatibility on ROCm.

## Risks and Mitigations
- **Risk: `vec2text` pins (torch/transformers) conflict with the ROCm container.** Mitigation: install
  with `pip install vec2text` first; if it downgrades torch, retry `--no-deps` + pin a compatible
  `transformers`; ROCm presents as `cuda` so `.cuda()` calls work. Fallback: CPU inference for a small
  N=64 sanity (T5-base is slow but runnable) to validate the science, then optimize.
- **Risk: GTR sequence-length regime.** Use the 32-token corpus (Morris's trained regime); the
  gtr-base corrector is for ≤32-token recovery — do not feed long texts.
- **Risk: common-cause monotone confound (C3).** Mitigated by the search axis (probe constant) + B4.
- **Risk: NN-baseline pool leakage.** Use a held-out pool disjoint from the attacked texts.

## Final Checklist
- [x] Main paper tables covered (C2 curve, C3 probe↔attack)
- [x] Novelty isolated (C4 info-efficiency gap vs weak attacks)
- [x] Simplicity defended (pretrained dependency; no bespoke corrector; GTR not ada)
- [x] Frontier contribution justified (faithful SOTA inversion is the *point* — it is the strong
      attack the probe must predict; not decoration)
- [x] Nice-to-have (B4,B5) separated from must-run (B1–B3)
