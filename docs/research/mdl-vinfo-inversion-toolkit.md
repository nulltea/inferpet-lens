---
type: research
status: current
created: 2026-06-15
updated: 2026-06-15
tags: [MDL, V-information, PVI, probing, information-theory, interpretability, inversion-attacks, leakage-measurement, tooling]
companion: [interpretability-leakage-bridge]
supersedes: []
---

# MDL / V-information as an Inversion-Vulnerability Predictor — Lineage & Toolkit

## Purpose

Cluster A of [`interpretability-leakage-bridge.md`](interpretability-leakage-bridge.md)
identified the cleanest open contribution: **use probe-complexity-aware
information measures (MDL, V-information) to predict how invertible a
representation is**, instead of the BLEU/F1 the attack papers report.
This doc supports building that, in two halves:

1. **The lineage** — techniques that extend / sharpen MDL probing and
   V-usable information into a *more precise and expressive* information-
   theoretic interpretability lens (what to measure).
2. **The toolkit** — confirmed, high-quality implementations of those
   measures plus the mechanistic-interpretability substrate they run on
   (what to build with), with license caveats.

The framing target: a **bounded-adversary leakage score per
layer/representation** that the GELO threat model can use as a
defendable, principled alternative to ad-hoc reconstruction metrics.

## Definitions

| Term | Meaning here |
|------|--------------|
| **MDL probing** | Voita–Titov: measure the *code-length* of labels given a representation, via **variational** (Bayesian, transmit probe params) or **online/prequential** (sum of cross-entropies over growing data prefixes) coding. Probe-complexity-aware MI surrogate. |
| **V-usable information** `I_V(X→Y)` | Xu et al.: MI restricted to a bounded predictive family `V`. *Not* bijection-invariant — computation can create usable info, encryption destroys it. The natural measure for a *compute-bounded* inversion adversary. |
| **Pointwise V-info (PVI)** | Per-*instance* decomposition of `I_V`: `pvi(x→y) = −log g[∅](y) + log g[x](y)`. Localises difficulty/leakage to individual examples and input slices. |
| **Loss-data curve** | Whitney et al.'s unifying object: loss vs. training-data-budget. MDL reads a slice along the *data* axis; **Surplus Description Length (SDL)** / ε-sample-complexity reads a slice along the *loss* axis. |
| **Conditional V-info** | `I_V(X→Y | B)`: usable information a representation adds *beyond a baseline* `B` — removes confounds (e.g. "what does this layer leak beyond public non-contextual features"). |
| **PID** | Partial Information Decomposition: split predictive info into **redundant / unique / synergistic** components — strictly more expressive than a scalar MI/V-info. |
| **DAS / IIA** | Distributed Alignment Search; **Interchange-Intervention Accuracy** — a *causal* (interventional) test of where a variable lives, vs. correlational probing. |

---

## Part 1 — The measurement lineage (ranked by how much it sharpens the lens)

Two parallel axes descend from the same root idea (information a *bounded*
learner can extract). They are now treated as siblings; V-information is
the more actively built-on branch post-2022.

### Axis 1 — Code-length / probe-complexity (the MDL branch)

| # | Paper | Year/venue | Sharpening gain over vanilla MDL | Code |
|---|-------|-----------|----------------------------------|------|
| A0 | **Voita & Titov, Information-Theoretic Probing with MDL** — [arXiv:2003.12298](https://arxiv.org/abs/2003.12298) | 2020 EMNLP | The anchor: codelength (variational + online coding) replaces probe accuracy. | [lena-voita/description-length-probing](https://github.com/lena-voita/description-length-probing) |
| A1 | **Blier & Ollivier, The Description Length of Deep Learning Models** — [arXiv:1802.07044](https://arxiv.org/abs/1802.07044) | 2018 NeurIPS | *Upstream source* of the online/prequential code that Voita–Titov's estimator uses. Read for the foundations. | — |
| A2 | **Whitney et al., Evaluating Representations by the Complexity of Learning Low-Loss Predictors** — [arXiv:2009.07368](https://arxiv.org/abs/2009.07368) | 2020 | **★ The key refinement.** Introduces **Surplus Description Length** + **ε-sample-complexity** and the unifying **loss-data curve**: MDL is one slice (fixed data → compressibility); SDL is the orthogonal slice (fixed loss → data needed). Makes the data-vs-loss-axis choice explicit and tolerance-parameterised. | `reprieve` library |
| A3 | **Pimentel et al., Pareto Probing** — [arXiv:2010.02180](https://arxiv.org/abs/2010.02180) | 2020 EMNLP | Replaces a single scalar with the **accuracy–complexity Pareto frontier** (Pareto-hypervolume). Keeps both axes MDL collapses into one. | [rycolab/pareto-probing](https://github.com/rycolab/pareto-probing) (GPL-3.0) |
| A4 | **Pimentel & Cotterell, A Bayesian Framework for IT Probing** — [arXiv:2109.03853](https://arxiv.org/abs/2109.03853) | 2021 EMNLP | Tells you *what estimand* MDL coding approximates and under which prior — recovers MDL/variational probing as a special case of Bayesian MI estimation. Conceptual precision. | — |
| A5 | **Shaw et al., Bridging Kolmogorov Complexity and Deep Learning** — [arXiv:2509.22445](https://arxiv.org/abs/2509.22445) | 2025, ICLR 2026 | Current frontier of the variational-code-length idea: proves asymptotically-optimal (Kolmogorov-grounded) description-length objectives exist for Transformers; gives the optimality limit the coder chases. | — |

### Axis 2 — Usable / bounded information (the V-information branch)

| # | Paper | Year/venue | Sharpening gain | Code |
|---|-------|-----------|-----------------|------|
| B0 | **Xu et al., A Theory of Usable Information Under Computational Constraints** — [arXiv:2002.10689](https://arxiv.org/abs/2002.10689) | 2020 ICLR | The anchor: `I_V` = MI under a bounded predictive family. The right primitive for a compute-bounded adversary. | — |
| B1 | **Ethayarajh, Choi, Swayamdipta, Understanding Dataset Difficulty with V-Usable Information** — [arXiv:2110.08420](https://arxiv.org/abs/2110.08420) | 2022 ICML (Outstanding Paper) | **★ The PVI paper.** Per-*instance* `pvi(x→y)` + feature/attribute slicing — turns a dataset-level scalar into a leakage score attributable to individual inputs and input transformations. | [kawine/dataset_difficulty](https://github.com/kawine/dataset_difficulty) |
| B2 | **Hewitt et al., Conditional Probing: Measuring Usable Information Beyond a Baseline** — [arXiv:2109.09234](https://arxiv.org/abs/2109.09234) | 2021 EMNLP | **★ Directly load-bearing for the threat model.** `I_V(X→Y|B)` isolates the *marginal* usable info a representation adds beyond a baseline — i.e. "what does a layer leak beyond what public features already give the adversary". | [john-hewitt/conditional-probing](https://github.com/john-hewitt/conditional-probing) (Apache-2.0) |
| B3 | **Choi, Jung, Watanabe, Understanding Probe Behaviors through Variational Bounds of MI** — [arXiv:2312.10019](https://arxiv.org/abs/2312.10019) | 2024 ICASSP | Unifies linear probing / fine-tuning / V-info under **variational MI bounds**; explains the convex per-layer probe curve (separability vs. content). The rigorous bridge from V-info to per-layer MI measurement. | [juice500ml/information_probing](https://github.com/juice500ml/information_probing) |
| B4 | **Yuan et al., CaLE: V-usable Information based Layer Enhancement** — [arXiv:2504.15630](https://arxiv.org/abs/2504.15630) | 2025 | Per-layer `I_V` to find where contextual info grows across depth, then intervene. Closest existing "where does usable info live across depth" tool — the layer-localisation pattern the inversion study wants. | — |

### More expressive than a scalar — Partial Information Decomposition

- **How Vision Becomes Language (PID-Flow)** — [arXiv:2602.15580](https://arxiv.org/abs/2602.15580),
  2026. Decomposes predictive info into **redundant / unique /
  synergistic** per layer; the **PID-Flow** estimator makes PID tractable
  in high-`d` reps via *dim-reduction → normalizing-flow Gaussianization
  → closed-form Gaussian PID*. **The transferable trick** for the GELO
  question: redundant info is defensible-by-compression; synergistic info
  cannot be removed without destroying utility.
- Companion at scale: **Information-Decomposition Analysis of LVLMs** —
  [arXiv:2603.29676](https://arxiv.org/abs/2603.29676), 2026.

### More expressive than correlation — causal localisation

Probing/V-info measure *correlational* extractability; causal abstraction
tests whether a subspace *causally carries* a variable — a stronger,
more expressive localisation of where a secret lives.
- **Geiger et al., Causal Abstractions of Neural Networks** — [arXiv:2106.02997](https://arxiv.org/abs/2106.02997), 2021 NeurIPS.
- **Geiger et al., Distributed Alignment Search (DAS)** — [arXiv:2303.02536](https://arxiv.org/abs/2303.02536), 2024 CLeaR. Learns non-basis-aligned causal subspaces; **Interchange-Intervention Accuracy (IIA)** is the graded, causal analogue of probe accuracy. Code: [stanfordnlp/pyvene](https://github.com/stanfordnlp/pyvene).
- **Boundless DAS** (scales to LLMs) — NeurIPS 2023; toolkit paper **pyvene** — [arXiv:2403.07809](https://arxiv.org/abs/2403.07809).

### Guardrails — the limits any leakage number must respect

- **McAllester & Stratos, Formal Limitations on the Measurement of MI** — [arXiv:1811.04251](https://arxiv.org/abs/1811.04251), AISTATS 2020. Any distribution-free high-confidence MI *lower* bound is **O(ln N)** in sample size. **This is the formal reason to report *usable/bounded* info, not "true MI".**
- **Song & Ermon, Understanding the Limitations of Variational MI Estimators** — [arXiv:1910.06222](https://arxiv.org/abs/1910.06222), ICLR 2020. Variance grows exponentially in true MI; proposes SMILE.
- **Cheng et al., CLUB: a Contrastive Log-ratio Upper Bound of MI** — [arXiv:2006.12013](https://arxiv.org/abs/2006.12013), ICML 2020. An MI **upper** bound — pair with the lower bounds to *bracket* leakage from both sides (most estimators only lower-bound). Code: [Linear95/CLUB](https://github.com/Linear95/CLUB).

### Adjacent privacy-side primitive worth tracking

- **Pointwise Maximal Leakage** (Saeidian et al., IEEE T-IT 2023, [doi](https://doi.org/10.1109/tit.2023.3304378)) — a *per-instance, operational, bounded-adversary* leakage measure conceptually parallel to PVI. Sits in the IT-privacy literature; a candidate formal target if the leakage score needs an operational-privacy interpretation rather than a probing one.

---

## Part 2 — The toolkit (confirmed implementations)

### (A) IT-probing method code — the measures themselves

| Repo | Implements | Framework | Maint. | License | Use in the study |
|------|-----------|-----------|--------|---------|------------------|
| [lena-voita/description-length-probing](https://github.com/lena-voita/description-length-probing) | **Official MDL probing** — variational + online coding | PyTorch + `jiant`, Jupyter | 75★, research artifact | **⚠ none shown** | Core codelength score "how much input info is extractable from an activation" |
| [kawine/dataset_difficulty](https://github.com/kawine/dataset_difficulty) | **Official V-info + PVI** | PyTorch + HF (BERT) | 91★ | **⚠ none shown** | Per-instance/per-slice usable-info leakage score |
| [john-hewitt/conditional-probing](https://github.com/john-hewitt/conditional-probing) | **Conditional V-info beyond a baseline** | PyTorch + HF, HDF5 cache, YAML | 21★ | **Apache-2.0** (cleanest) | Marginal leakage of a layer above public-feature baseline |
| [juice500ml/information_probing](https://github.com/juice500ml/information_probing) | Variational-bound unification of probing/FT/V-info (B3) | PyTorch | — | (verify) | Per-layer convex probe-curve theory + margin criterion |
| [john-hewitt/structural-probes](https://github.com/john-hewitt/structural-probes) | Structural (distance/depth) probes + control tasks | PyTorch | — | (verify) | Reference probe-architecture/training harness to adapt |
| [rycolab/pareto-probing](https://github.com/rycolab/pareto-probing) | Accuracy–complexity Pareto probing | Python | ~6★ | **GPL-3.0 (copyleft — flag)** | Express inversion risk as a Pareto curve |
| [ethanjperez/rda](https://github.com/ethanjperez/rda) | Rissanen Data Analysis (dataset MDL via prequential code) | PyTorch | ~37★ | (verify LICENSE file) | Dataset-level description-length curves |
| [Linear95/CLUB](https://github.com/Linear95/CLUB) | **CLUB + MINE + NWJ + InfoNCE + L1Out in one file** | PyTorch | ~361★ | (verify) | Single dependency for MI upper+lower bounds on activations |
| [gtegner/mine-pytorch](https://github.com/gtegner/mine-pytorch) | MINE (no first-party repo exists) | PyTorch | ~361★ | MIT | Neural lower-bound `I(activation; input)` |
| [RElbers/info-nce-pytorch](https://github.com/RElbers/info-nce-pytorch) | InfoNCE loss (on PyPI) | PyTorch | ~616★ | MIT | InfoNCE variational MI lower bound |

### (B) Mechanistic-interpretability substrate — actively maintained

| Repo | Role | Stars / version | License |
|------|------|-----------------|---------|
| [TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | **Activation capture** (`run_with_cache`) — feeds the probes | 3.6k★, v3.4.0 (2026-06) | MIT |
| [ndif-team/nnsight](https://github.com/ndif-team/nnsight) | Activation access/intervention/grads; **remote big-model exec via NDIF** | 958★, v0.7.0 (2026-05) | MIT |
| [jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) | SAE train/analysis (sparse codes → MDL/sparsity of activations) | 1.4k★, v6.44 (2026-05) | MIT |
| [pytorch/captum](https://github.com/pytorch/captum) | Attribution (IG, TCAV, TracIn) — input-recoverability baselines | 5.7k★, v0.9 (2026-04) | BSD-3 |
| [stanfordnlp/pyvene](https://github.com/stanfordnlp/pyvene) | Interventions / causal abstraction / **DAS** | 884★, v0.1.8 (2025-05) | Apache-2.0 |
| [inseq-team/inseq](https://github.com/inseq-team/inseq) | Seq-generation feature attribution | 467★, v0.7 (2026-02) | Apache-2.0 |
| [AlignmentResearch/tuned-lens](https://github.com/AlignmentResearch/tuned-lens) | **Tuned/logit lens** — per-layer learned lens ≈ residual-stream decodability proxy (**canonical repo**, not the thin `EleutherAI` fork) | ~594★, v0.2 | MIT |
| [google-deepmind/penzai](https://github.com/google-deepmind/penzai) | JAX NN-surgery + Treescope viz | 1.9k★, v0.2.5 (2025-04) | Apache-2.0 |
| [davidbau/baukit](https://github.com/davidbau/baukit) | Minimal hook-based activation tracing (`TraceDict`) | 253★ | MIT |
| [saprmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning) | SAE variants (gated/top-k/JumpReLU) via nnsight | 422★, v0.1 (2025-02) | MIT |

**Dormant / reference-only:** `openai/transformer-debugger` (abandoned —
Superalignment dissolved), `jalammar/ecco` (dormant ~4y), `jessevig/bertviz`
(8.1k★ but attention-viz only).

### License caveat (load-bearing)

The **two most central method repos — MDL (`description-length-probing`)
and V-info (`dataset_difficulty`) — display NO license** (no reuse grant;
treat as all-rights-reserved until cleared). **Pareto probing is GPL-3.0**
(copyleft). **`conditional-probing` (Apache-2.0) is the cleanest** to
vendor. Given the repo's memory note on licensing footguns
([[rust_ecosystem_audit]]), plan to **reimplement the MDL/PVI estimators
from the papers** rather than vendor the unlicensed repos — both methods
are short (online-coding loop; the two-forward-pass PVI difference) and
are partially re-derived in the Apache/MIT repos (`conditional-probing`,
`rda`).

---

## Recommended stack for the inversion-vulnerability study

1. **Substrate:** TransformerLens (`run_with_cache`) to harvest per-layer
   hidden states / KV tensors on the target Qwen3 model (or nnsight+NDIF
   if the model outgrows the box, per [[feedback_no_cpu_for_gpu_workloads]]).
2. **Primary score:** **conditional V-info / PVI** (B2 + B1) — per-layer,
   per-instance, *above a public-feature baseline*. This directly matches
   the `WEIGHTS-PUB` adversary: the baseline `B` = what public weights
   already reveal; the score = marginal leakage of the covered activation.
3. **Complexity-axis cross-check:** **MDL online-coding** (A0) and
   **SDL / loss-data curve** (A2) — to report code-length *and*
   sample-complexity, not a single scalar.
4. **Bracketing:** CLUB (upper) + MINE/InfoNCE (lower) to bound the score
   from both sides, respecting the McAllester–Stratos `O(ln N)` limit.
5. **Expressiveness upgrades when a scalar is too coarse:** PID-Flow
   (redundant vs synergistic → defensible vs irremovable leakage) and
   DAS/IIA (causal localisation of where the secret subspace lives).

## Open contribution restated

No paper found (per the bridge survey + this lineage pass) uses MDL,
SDL, or V-information/PVI to **rank representations/layers by
invertibility** or to **predict inversion-attack success** — the closest
is CaLE (B4, per-layer V-info but for context-faithfulness, not leakage)
and Shu et al. 2025 (per-layer MI but attack-construction-driven, not a
principled probe). The novel claim to test: *a representation's
conditional-V-info / surplus-description-length for input content is a
calibrated predictor of an inversion attack's token-recovery rate* —
and under `WEIGHTS-PUB` the baseline must be the public-weights anchor.

## Sources

All repo URLs verified via GitHub fetch; all paper URLs via arXiv/OpenAlex
during the 2026-06-15 scouting passes. Citation counts are OpenAlex/
Semantic Scholar snapshots (OpenAlex's `cites` graph is badly
under-populated for this subfield — counts are lower bounds). Companion:
[`interpretability-leakage-bridge.md`](interpretability-leakage-bridge.md).
