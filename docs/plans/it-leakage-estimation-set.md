---
type: plan
status: current
created: 2026-06-15
updated: 2026-06-15
tags: [information-theory, V-information, MDL, PVI, PID, inversion-attacks, attention-leakage, leakage-measurement, experimental-design]
companion: [interpretability-leakage-bridge, mdl-vinfo-inversion-toolkit]
---

# IT-Leakage Estimation Set — Attacks × Measures

The decided experimental matrix for the research thread: **use
probe-complexity-aware information measures to predict how invertible a
GELO-exposed representation is** (the Cluster-A open contribution from
[`interpretability-leakage-bridge.md`](../research/interpretability-leakage-bridge.md)).
Measure definitions and tooling: [`mdl-vinfo-inversion-toolkit.md`](../research/mdl-vinfo-inversion-toolkit.md).

## Selection rule (why these and not others)

An attack joins the set iff it (1) operates on the **GELO secret** —
inference-time activations / hidden states / KV / attention / tokens that
cross the TEE→GPU boundary (training-set membership is excluded; the
secret is the *user's prompt*, not training data); (2) yields a **graded
success metric** usable as the regression target the IT measure predicts;
(3) **spans the threat surface** (secret type × depth × cover); (4) is
**runnable as ground-truth** (released code or already in
`evals/aloepri-attacks/`).

## The five attacks (calibration ground-truth)

| # | Attack | Paper(s) | Secret / where in GELO | Graded metric | Status |
|---|--------|----------|------------------------|---------------|--------|
| 1 | **Per-layer hidden-state inversion** | Depth-False-Sense, USENIX'25 ([2507.16372](https://arxiv.org/abs/2507.16372)); Shu IB-split ([2501.05965](https://arxiv.org/abs/2501.05965)) | residual stream at layer ℓ (the offloaded activation) | token-recovery F1 vs depth | artifacts exist |
| 2 | **Permutation / cover-break** | Hidden No More, ICML'25 ([2505.18332](https://arxiv.org/abs/2505.18332)) | covered activation under orthogonal/permutation cover | top-1 membership / token recovery | ✅ in `evals/aloepri-attacks/` |
| 3 | **Linear-probe membership/attribute** | LUMIA ([2411.19876](https://arxiv.org/abs/2411.19876)) | attribute/membership encoded in activations | probe AUC | trivial to build |
| 4 | **Embedding inversion** | Vec2Text, EMNLP'23 ([2310.06816](https://arxiv.org/abs/2310.06816)); LM-inversion, ICLR'24 ([2311.13647](https://arxiv.org/abs/2311.13647)) | pooled embedding (RAG path) | BLEU / token-F1 | released |
| 5 | **Attention-score inversion** | Unmasking Transformers ([2310.12462](https://arxiv.org/abs/2310.12462), recovers X from QK^T + output); Neural Breadcrumbs ([2509.05449](https://arxiv.org/abs/2509.05449), membership via attention distribution) | `QK^T` scores, if attention runs on the untrusted device | input-recovery error / membership AUC | repo has ISA-AttnScore |

## The measures (predictors)

| Measure | Role | Applied to | Source |
|---------|------|-----------|--------|
| **Conditional V-info / PVI** (above a public-weights baseline `B`) | **Primary predictor**, per-layer + per-instance. `B` = what public weights already reveal → matches `WEIGHTS-PUB`. For #3 the probe *is* the V-family, so the link is exact. | all 5 | Hewitt EMNLP'21; Ethayarajh ICML'22 |
| **MDL online-coding + Surplus Description Length** | Complexity cross-check — reports compressibility AND sample-complexity, not one scalar. | #1, #4 (continuous, depth-graded) | Voita 2003.12298; Whitney 2009.07368 |
| **PID-Flow + DAS/IIA** | Expressiveness upgrade. PID redundant-vs-synergistic isolates the invariant (= redundant) leak; DAS/IIA causally localises the secret subspace. | #2 (PID on the invariants), #1 (DAS) | PID-Flow 2602.15580; DAS 2303.02536 |
| **CLUB-upper + MINE/InfoNCE-lower** | **Optional** one-time bracket: confirm the V-info/PVI estimate respects the McAllester–Stratos `O(ln N)` ceiling. *Not* a per-layer headline number. | validation only | CLUB 2006.12013 |

## Central novel claim to test

A representation's **conditional-V-info / surplus-description-length for
input content is a calibrated predictor of an inversion attack's
token-recovery rate** — and under `WEIGHTS-PUB` the baseline `B` must be
the public-weights anchor. No prior work fits an IT measure to attack
success this way (closest: CaLE per-layer V-info for context-faithfulness;
Shu et al. per-layer MI but attack-construction-driven).

## Two whitespaces surfaced (fold into the contribution)

1. **Cover-invariance of attention scores as a *leakage* result is
   unpublished.** `softmax(QK^T)` is invariant under a shared rotation
   `(QO)(KO)^T = QK^T` and permutation-equivariant — folklore math, but no
   paper frames it as a confidentiality failure. This is the
   **attention-circuit analogue of the Gram-invariant membership result**
   in `gpu-offloaded-attention-with-value-cover.md` §2: the IT lens states
   it precisely as "an orthogonal/permutation cover leaves `MI(tokens; QK)`
   untouched." Novel, and a direct GELO security claim for attack #5.
2. **`MI(tokens; QK-circuit)` vs `MI(tokens; OV-circuit)` decomposition is
   unstudied.** The one MI-of-QKV paper (PMC12191707) treats Q/K/V
   symmetrically and finds attention scores *underestimate* token MI.
   PID-Flow is the natural tool to split QK vs OV leakage.

## Open design questions (next decisions)

- Target model/representations: Qwen3-1.7B vs 4B; which layers ℓ to sweep;
  capture via TransformerLens `run_with_cache` (or nnsight+NDIF if it
  outgrows the box, per [[feedback_no_cpu_for_gpu_workloads]]).
- Calibration protocol: fit V-info→recovery-rate regression across layers
  and across the 5 attacks; report `R²` / rank-correlation as the
  "predictor quality" headline.
- Whether to instrument #5's cover-invariance claim as a formal lemma +
  empirical check inside `evals/aloepri-attacks/`.

## Tooling / licensing

Per [`mdl-vinfo-inversion-toolkit.md`](../research/mdl-vinfo-inversion-toolkit.md):
substrate = TransformerLens (MIT); the official MDL and V-info repos are
**unlicensed** → reimplement the estimators (short: online-coding loop;
two-forward-pass PVI difference); `conditional-probing` (Apache-2.0) is the
clean reference.
