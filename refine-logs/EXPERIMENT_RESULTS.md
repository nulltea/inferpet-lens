---
type: dev-log
status: current
created: 2026-06-20
updated: 2026-06-20
tags: [results, capacity-matched-pvi, randproj, faithfulness, DP-sweep, calibration]
companion: [EXPERIMENT_PLAN, idea-stage/IDEA_REPORT]
---

# Initial Experiment Results

**Date**: 2026-06-20  **Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Objective**: fix class-PVI (independent token-id V-family) so it tracks the attack, or formally retire it.

## Results by Milestone

### M0: Sanity — PASSED
`vinfo_capacity.py` (4 families: pca_softmax / randproj_softmax / gauss / knn), 9/9 new tests,
60/60 suite. Codex (gpt-5.5 xhigh) review: no critical bugs; **confirmed the measure never
touches the token-embedding table** (predicts token-id classes) → structurally independent of
the ridge→embedding attack. Finding: full-`d` gauss/knn are *also* overconfident, so every family
is capacity-matched by PCA/randproj reduction first.

### M1: Well-posedness + cost screen — PASSED (`results/capacity_screen.json`)
L12, every-n 2, 3 seeds. Healthy floor ≈ 0 (class-PVI = **−49.7**); cost must be ≤ class-PVI (2.85s).

| family | real PVI | shuffle floor | cost vs class-PVI | noise decay |
|---|---|---|---|---|
| class-PVI | 4.96 | **−49.7** ❌ | 1× | non-monotone (rises) ❌ |
| **randproj_softmax** | 5.17 | −3.9 | **0.42×** ✅ | graceful monotone ✅ |
| pca_softmax | 5.39 | −1.9 ✅ | 0.57× ✅ | monotone (steep) |
| gauss | −12.0 ❌ | −32.6 ❌ | 0.15× | non-monotone ❌ |
| knn | 1.35 | −1.1 ✅ | 0.13× ✅ | monotone, weak |

→ The independent family **can** be made well-posed *and* cheaper. Survivor: **randproj_softmax**.
gauss fails (miscalibrated reader — "the reader matters" ablation). Perf note: a redundant full-SVD
made an early run CPU-thrash 26 min at idle GPU; replaced with covariance-eigh on GPU
(11s→0.44s, GPU→100%).

### M2: Faithfulness — PASSED (`results/localdp_m2_randproj_L12.json`)
DP ε-sweep {∞,8192,4096,2048,1024,512,256}, L12, randproj dim 256, every-n 2. Ground truth = TTRSR
(falls 0.35→0.05). Spearman(measure, TTRSR):

| measure | Spearman | note |
|---|---|---|
| **cap-PVI selectivity (real − shuffle)** | **+0.929** | independent **and** faithful |
| retrieval-PVI | +0.929 | mechanical (= the attack in bits) |
| **class-PVI** | **−0.929** | independent but **anti**-correlated (fails) |
| CLUB | +0.536 | independent upper bound, weak on this sweep |
| cap-PVI raw bits | +0.643 | flat-then-cliff; selectivity is the faithful read |

## Summary
- **4/4 must-run blocks to date completed** (M0, M1, M2). Main result: **POSITIVE** — a
  capacity-matched member of the independent token-id family (randproj_softmax), read as
  shuffle-control selectivity, tracks the attack at Spearman **0.929** (= the mechanical attack
  measure) while costing **0.42×** class-PVI, where class-PVI is **anti-correlated (−0.929)**.
- **Independence argument (built-in)**: the *only* change class-PVI → cap-PVI is capacity-matching
  (dim reduction); no embedding/attack info is added. So tracking comes from fixing the estimator
  regime, not from becoming the attack. (Formal per-instance collinearity test vs retrieval-PVI =
  the next rigor step, B3.)
- **Open / next** (for the review loop to prioritise): (i) B3 independence isolation — per-instance
  cap-PVI vs retrieval-PVI collinearity + l2-only-vs-dim deletion study; (ii) extend M2 to L5/L20 +
  the 108-block control sweep (n=7 per layer is thin); (iii) raw-bits flat-then-cliff vs
  selectivity — characterise which to report; (iv) other defenses (split-depth, obfuscation) = B4.

## Next Step
→ `/auto-review-loop` to critique and prioritise (B3 independence + multi-layer robustness expected).

---
## Round 2 (review-driven) — multi-layer + control-anchored regularization

Reviewer R1 (5/10, "almost"): single-layer overstated it; need multi-layer, freeze family by floor health, control for noise knob, CIs.

- **Family frozen by M1 floor health only**: dim-sweep → `pca_softmax` (floor mild & dim-stable; randproj's floor blew up to −3.9 at high dim). Predeclared floor band [−1.5,1.5] b.
- **Multi-layer (L5/12/20) × denser ε (n=21)** breaks the single-noise-knob collinearity (DP noise is at the embedding so r depends only on ε; the layer axis gives TTRSR variation at fixed noise).
- **Control-anchored regularization** (`--capacity-l2`): l2=10 vs 0.1.

| pca64 cap-PVI selectivity | pooled ρ | partial ρ\|r | partial ρ\|retr | per-layer L5/L12/L20 | floor |
|---|---|---|---|---|---|
| l2=0.1 | 0.642 | 0.738 | −0.05 | 0.68/0.32/−0.21 | −1.2 stable |
| **l2=10** | **0.779** | **0.666** | **+0.275** | 0.96/0.89/0.32 | ~ stable |
| (ref) CLUB | 0.810 | 0.716 | — | 0.96/0.89/0.29 | — |
| (ref) class-PVI | −0.173 | — | — | −.86/−.64/−.79 anti | −49 |

**Verdict update**: capacity-matching + control-anchored reg **repairs the independent token-id family** — pooled ρ 0.78, tracks *beyond the noise knob* (partial ρ|r 0.67) **and beyond the attack-in-bits measure** (partial ρ|retr +0.275), at ~0.3× cost, where class-PVI is anti-correlated. Floor fixed −49→−1.2 (stable). 2/3 layers strongly faithful (0.89–0.96). **Residual**: L20 (late layer) still rises under moderate DP even at l2=10 → a genuine measure-vs-attack divergence (token-id decodability is more DP-robust than embedding reconstruction), not pure overfit. Independence empirically supported (ρ(cap,retr)=0.76 <0.9; adds beyond retr).

---
## Round 3 + FINAL (review-driven) — non-DP defense, the readout split, honest verdict

Reviewer R3: **7/10, scoped result ESTABLISHED if reframed.** The conceptual center shifted (correctly):
the fixed object is the **capacity-matched independent token-ID reader**, whose **bounded accuracy** is
the robust predictor; **PVI-in-bits is partially rescued, not solved**. Do NOT call accuracy "V-information".

**Faithfulness across THREE defenses** (Spearman vs TTRSR; default reg l2=0.1):
| defense | readout | L5 | L12 | L20 |
|---|---|---|---|---|
| PCA-subspace ablation | reader accuracy | 0.87 | 0.90 | 0.82 |
| isotropic hidden-state noise | reader accuracy | 0.90 | 0.90 | 1.00 |
| input-local-DP | reader accuracy | 0.68 | 0.43 | −0.21 (divergence) |
| input-local-DP | PVI bits (sel) | 0.68 | 0.32 | −0.21 |
| (PVI bits fragile) iso-noise | PVI bits (sel) | 0.40 | 0.40 | 0.70 |

**Verdict on the standing objective — ACHIEVED (scoped, honest):**
- **FIXED**: class-PVI's d>n_val catastrophe is removed by capacity-matching (floor −49→−1.5 b, *dim*-anchored
  not l2-anchored; ~0.3× cost). The **token-ID reader accuracy** faithfully tracks the attack (ρ 0.82–1.0)
  under representation-space defenses and at early/mid layers under DP, beyond the noise knob (partial ρ|r≈0.71),
  where the unfixed class-PVI is anti-correlated.
- **PARTIALLY RESCUED**: PVI-in-bits — the −48 floor & non-monotone rises are unbounded-log-loss artifacts;
  bits track only with regularization and stay fragile (iso-noise, late-DP). Report bits as auxiliary, accuracy as primary.
- **CHARACTERIZED LIMIT (a result)**: late-layer × input-DP divergence — token-id decodability outlives
  embedding-reconstruction; localizes what input-DP protects (embedding geometry) vs not (id-decodability).

**Remaining polish (non-blocking, for a paper)**: calibration diagnostic (NLL/ECE) on the artifact cells;
within-layer/macro-avg ρ as primary stat; cross-model replication; dim16 sensitivity. None change the verdict.
