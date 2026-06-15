---
type: research
status: current
created: 2026-06-15
updated: 2026-06-15
tags: [interpretability, information-theory, MDL, information-bottleneck, SAE, inversion-attacks, probing, leakage-measurement]
companion: [mdl-vinfo-inversion-toolkit, it-leakage-estimation-set]
---

# Interpretability ↔ Confidential-Inference Attacks: the Information-Theoretic Bridge

## Purpose

This is a literature map for a planned research thread: using
**transformer-interpretability machinery** — probing classifiers,
**Minimum Description Length (MDL) probing** (Voita & Titov 2020),
**V-usable information** (Xu et al. 2020), the **Information Bottleneck
(IB)**, **sparse autoencoders (SAEs)** and the **logit/tuned lens** — to
give a *principled information-theoretic account* of why confidential-
inference attacks (Vec2Text, permutation/"Hidden No More", deep-layer
inversion, KV-cache leakage) succeed, and what a defence must remove.

The question that motivated the survey: **has anyone already connected
the interpretability-IT toolkit to the inference-attack literature?**
Four parallel scouting passes (OpenAlex citation-graph walks + web
search + arXiv/ACL/USENIX fetches, ~250 tool calls) answer: **partially,
and asymmetrically.** This doc records who, where the bridge is load-
bearing vs. incidental, and — most usefully — **where the bridge does
not yet exist** (the contribution surface).

## Definitions

| Term | Meaning in this doc |
|------|---------------------|
| **MDL probing** | Voita & Titov's online/variational *code-length* measure of how regularly a representation encodes a target — a probe-complexity-aware MI surrogate, strictly more principled than probe accuracy. |
| **V-usable information** | Xu et al.'s "Theory of Usable Information": MI *restricted to a predictive family* `V` — formalises "information a *bounded* adversary can actually use". The natural lens for a compute-bounded inversion attacker. |
| **Probe-as-MI-lower-bound** | Pimentel/Hewitt framing: a probe's accuracy lower-bounds `I(representation; target)`. An *attack* is exactly a probe whose target is the secret. |
| **IB / Privacy Funnel** | IB maximises `I(Z;Y)` (utility) under a `I(Z;X)` (compression) budget; the **Privacy Funnel** is its dual — *minimise* `I(Z;S)` (leakage of secret `S`) under a utility floor. Same Lagrangian, opposite sign on the secret. |
| **Inversion attack** | Recover input text / tokens / attributes from a representation (embedding, hidden state, KV tensor, logit vector, MoE routing). |
| **Leakage signature** | A per-layer scalar (entropy, MI, code-length) profile used to localise *where* in depth a secret is recoverable. |

## Headline findings

1. **The MDL / V-information ↔ privacy bridge is thin and one-
   directional.** MDL probing is used as a *defence-evaluation metric*
   in fair-representation work — never (found) to *explain or predict
   inversion-attack success*. The framing "a representation with low
   MDL-for-content is easier to invert" appears **unwritten**. This is
   the cleanest open contribution for the planned thread.

2. **The IB ↔ privacy bridge is mature and the closest prior art.**
   Privacy-Funnel theory (2014) → VIB defences (Shredder, NoPeek, DISCO,
   ReFIL) → and crucially **Shu et al. 2025 apply IB *per transformer
   layer* to a split-LLM inversion attack** — the single nearest
   neighbour to the planned work. The gap: nobody pairs this with the
   `WEIGHTS-PUB` adversary or a TEE-offload threat model.

3. **The most *direct* "interpretability-tool-as-attack" realisations
   are SAEs, the logit lens and activation steering.** PrivacyScalpel
   (defence) / UniLeak / PII-jailbreaking / LatentQA / Patil-et-al-logit-
   lens form a tight cluster where mechanistic-interpretability tooling
   is repurposed verbatim as the extraction mechanism. This is the
   clearest existing proof that the two fields are the same toolkit.

4. **The injectivity result is the algebraic anchor.** "Language Models
   are Injective and Hence Invertible" (Nikolaou et al. 2025) is the
   canonical statement of the hidden-state non-collision property that
   "Hidden No More" exploits — and is exactly the structural fact a
   `WEIGHTS-PUB` membership argument rests on.

5. **Real open gaps** (no paper found): a channel-capacity *bound* on
   text-recovery from a `d`-dim embedding; MDL/V-info used to *rank
   layers by invertibility*; IB analysis under `WEIGHTS-PUB` vs
   `WEIGHTS-BLIND`; superposition deliberately used as an obfuscation
   mechanism; and any TEE-resident execution paired with an IB leakage
   *proof* (current TEE-split work is purely cryptographic).

---

## Cluster A — MDL / probing-classifier as a *privacy metric* (the thin bridge)

The recurring pattern: privacy/fairness papers **adopt MDL probing from
interpretability as a leakage gauge**, but do not invert the direction
(no inference-attack paper invokes MDL to explain itself).

| Paper | Year / venue | Bridge | Centrality |
|-------|-------------|--------|-----------|
| **Fair NLP Models with Differentially Private Text Encoders** (Maheshwari, Denis, Keller, Bellet) — [arXiv:2205.06135](https://arxiv.org/abs/2205.06135) | 2022, Findings EMNLP | Uses Voita–Titov **online-code MDL** as the attribute-leakage metric: higher MDL ⇒ sensitive attribute (race/gender) harder for a probing adversary to extract. The strongest explicit MDL-as-privacy-bound found. | Central |
| **Learning Fair Representations via Rate-Distortion Maximization (FaRM)** (Basu Roy Chowdhury, Chaturvedi) — [arXiv:2202.00035](https://arxiv.org/abs/2202.00035) | 2022, TACL | MDL as primary metric for residual demographic info; framed as upper bound on adversary extraction ease. | Central |
| **Shielded Representations** (Iskander, Radinsky, Belinkov) — [arXiv:2305.10204](https://arxiv.org/abs/2305.10204) | 2023, Findings ACL | MDL (online-code) measures probe *effort* to extract attributes from INLP-protected reps; handles non-linear leakage. | Central |
| **Information Leakage in Embedding Models** (Song, Raghunathan) — [arXiv:2004.00053](https://arxiv.org/abs/2004.00053) | 2020, CCS | The foundational "**probing-classifier accuracy = attribute-inference attack**" framework (inversion / attribute / membership). Pre-MDL; establishes the accuracy side of the bridge. | Central |
| **Privacy-preserving Neural Representations of Text** (Coavoux, Narayan, Cohen) — [arXiv:1808.09408](https://arxiv.org/abs/1808.09408) | 2018, EMNLP | Earliest framing of an adversarial probing classifier *as* an attribute-inference attack; defence minimises what the probe recovers. | Central |
| **DP Representation for NLP (DPNR)** (Lyu, He, Li) — [arXiv:2010.01285](https://arxiv.org/abs/2010.01285) | 2020, Findings EMNLP | DP noise + adversarial-probing accuracy as empirical privacy gauge; links formal DP to probe success. | Moderate |
| **Disentangling Linguistic Competence of Privacy-Preserving BERT** (Arnold, Kemmerzell, Schreiner) — [arXiv:2310.11363](https://arxiv.org/abs/2310.11363) | 2023, BlackboxNLP | Edge-probing to measure linguistic structure *lost* under DP. A genuine reverse-use of interpretability probes for privacy eval (but accuracy, not MDL). | Incidental |

**V-usable information for privacy:** searches returned **nothing**.
Xu et al.'s V-information and the Pimentel/Hewitt probe-as-estimation
framing live entirely in linguistic interpretability; no privacy/attack
paper was found applying them. **Open.**

---

## Cluster B — Information Bottleneck & MI-estimation ↔ split / collaborative inference (the mature bridge, closest prior art)

### Theory spine
- **From the Information Bottleneck to the Privacy Funnel** (Makhdoumi, Salamatian, Fawaz, Médard) — [arXiv:1402.1774](https://arxiv.org/abs/1402.1774), 2014. The canonical IB↔privacy duality.
- **Deep Variational Information Bottleneck (VIB)** (Alemi et al.) — [arXiv:1612.00410](https://arxiv.org/abs/1612.00410), ICLR 2017. The tractable estimator every downstream defence uses.
- **Bottlenecks CLUB** (Razeghi, Calmon, Gündüz, Voloshynovskiy) — [arXiv:2207.04895](https://arxiv.org/abs/2207.04895), IEEE TIFS 2023. Unifies IB / Privacy-Funnel / CEB under one variational surface — a ready-made lens for cut-layer design.
- **MINE** (Belghazi et al.) — [arXiv:1801.04062](https://arxiv.org/abs/1801.04062), ICML 2018. Neural MI estimator; the workhorse for activation-level MI measurement.

### IB/MI defences for split inference
| Paper | Year | Mechanism |
|-------|------|-----------|
| **Shredder** (Mireshghallah et al.) — [arXiv:1905.11814](https://arxiv.org/abs/1905.11814) | ASPLOS 2020 | Learns additive noise minimising `I(input; smashed)`; −70.2% MI at −1.46% acc. |
| **NoPeek** (Vepakomma et al.) — [arXiv:2008.09161](https://arxiv.org/abs/2008.09161) | ICDMW 2020 | Distance-correlation (MI-proxy) regulariser on activations; NoPeek-Infer extends to inference-only. |
| **DISCO** (Singh et al.) — [arXiv:2012.11025](https://arxiv.org/abs/2012.11025) | CVPR 2021 | Learned MI-attributed channel pruning/obfuscation of sensitive channels. |
| **ReFIL / Fisher-MI** (Maeng, Guo, Kariyappa, Suh) — [arXiv:2209.10119](https://arxiv.org/abs/2209.10119) | 2022 | Diagonal **Fisher Information** → Cramér–Rao reconstruction-error bound; enforces leakage budget. |
| **Unsupervised Information Obfuscation** (Samragh et al.) — [arXiv:2104.11413](https://arxiv.org/abs/2104.11413) | 2021 | Removes null-space / low-energy components w.r.t. server's first layer; handles *unforeseen* attributes. |
| **Task-Oriented Semantic Comms vs MIA** (Wang et al.) — [arXiv:2312.03252](https://arxiv.org/abs/2312.03252) | IEEE TWC 2024 | VIB compression + adversarial anti-inversion loss; variational MI upper bound. |
| **NVIB-DP transformer embeddings** (El Zein, Henderson) — [arXiv:2601.02307](https://arxiv.org/abs/2601.02307) | 2026 | Nonparametric variational IB layer → Rényi/Bayesian-DP guarantees on embedding invertibility. |
| **Prompt-IB defence** (Noorbakhsh, Khalili, Sehatbakhsh) — [arXiv:2606.11592](https://arxiv.org/abs/2606.11592) | 2026 | "Privacy adapters" minimise `I(activations; prompt)`; −35% token recovery; derives reconstruction-error bounds. |
| **InfoDecom** (Deng, Lu, Duan) — [arXiv:2511.13365](https://arxiv.org/abs/2511.13365) | AAAI 2026 | IB-style split of smashed data into task-relevant/irrelevant; noise only on the residual. |
| **FIBNet** (Chen et al.) — IEEE TIFS 2024, [doi](https://doi.org/10.1109/tifs.2024.3424303) | 2024 | IB retains identity, discards soft-biometric attributes. |

### IB/MI **measurement & attack** (the load-bearing neighbours)
- **Model Inversion in Split Learning for Personalized LLMs: New
  Insights from Information Bottleneck Theory** (Shu, Li, Dong, Meng,
  Zhu) — [arXiv:2501.05965](https://arxiv.org/abs/2501.05965), 2025.
  **★ Closest prior art.** First to apply IB / per-layer **MI-entropy**
  to *Transformer-LLM split learning*, ranks layers by attack surface,
  then mounts a two-stage generative inversion (38–75% token recovery,
  >60% over SOTA). The planned thread is essentially "do this under
  `WEIGHTS-PUB` + with MDL/V-info rigour + against the GELO cover."
- **How Breakable Is Privacy: Probing & Resisting MIA in Collaborative
  Inference** (Liu, Zhu, Wang, Pan, He, Meng) —
  [arXiv:2501.00824](https://arxiv.org/abs/2501.00824), 2025. First
  **formal criterion** for MIA difficulty using MI + entropy + "effective
  information volume" (validated via MINE); defence SiftFunnel cuts them ≥50%.
- **Quantifying Privacy Leakage via Fisher-Approximated Shannon
  Information (FSInfo / FSInfoGuard)** (Deng, Lu, Duan, Hu) —
  [arXiv:2504.10016](https://arxiv.org/abs/2504.10016), 2025. Makes MI
  tractable per split layer; worst/avg-case reconstruction-error bounds.
- **Not All Features Are Equal** (Mireshghallah et al.) — WWW 2021,
  [doi](https://doi.org/10.1145/3442381.3449965). MI attribution
  separating "essential" vs "leaky" feature dims — an interpretability-
  flavoured leakage map.
- **Leakage Assessment via Neural MI Estimation** (Cristiani, Lecomte,
  Maurine) — COSADE 2020. MINE for side-channel leakage; technique ports
  directly to activation-level leakage auditing.
- **Variational Leakage** (Atashin, Razeghi, Gündüz, Voloshynovskiy) —
  [arXiv:2106.02818](https://arxiv.org/abs/2106.02818), 2021. Closest to
  *probe-as-MI-lower-bound* for privacy: adversary's inference net lower-
  bounds `I(Z;S)`.
- **No Free Lunch Theorem for Privacy-Preserving LLM Inference** (Zhang
  et al.) — [arXiv:2405.20681](https://arxiv.org/abs/2405.20681), Artif.
  Intell. 2025. Proves a fundamental privacy-utility tradeoff for
  perturbed-embedding inference.

---

## Cluster C — Mechanistic-interpretability tooling repurposed as attack/defence (the *direct* bridge)

This is the cluster where the two fields are visibly **the same toolkit**.

### SAE / dictionary learning
- **PrivacyScalpel** (Frikha, Razi, Nakka, Mendes, Jiang, Zhou) —
  [arXiv:2503.11232](https://arxiv.org/abs/2503.11232), 2025. *Defence.*
  k-SAE on the residual stream → identify PII-encoding monosemantic
  features → ablate/steer. Email leakage 5.15%→0.0%, >99.4% utility.
  Explicit argument: acting on *sparse monosemantic* features beats
  manipulating *polysemantic* neurons (directly the superposition lesson).
- **CRISP: Persistent Concept Unlearning via SAEs** (Ashuach, Arad,
  Mueller, Tutek, Belinkov) — [arXiv:2508.13650](https://arxiv.org/abs/2508.13650),
  ACL 2026. *Defence.* SAE-feature-targeted weight fine-tune for durable
  (not inference-time) concept removal.
- **PII Jailbreaking via Activation Steering** (Nakka, Jiang, Usynin,
  Zhou) — [arXiv:2507.02332](https://arxiv.org/abs/2507.02332), 2025.
  *Attack.* Linear probes on attention-head activations locate refusal
  behaviour → steer to suppress it → ≥95% disclosure. The dual of
  PrivacyScalpel (same toolkit, opposite sign).
- **Superposition as Lossy Compression** (Bereska, Tzifa-Kratira,
  Samavi, Gavves) — [arXiv:2512.13568](https://arxiv.org/abs/2512.13568),
  TMLR 2025. *Theory/measurement.* Shannon-entropy metric over SAE
  activations for "effective degrees of freedom". Privacy reading: high
  capacity ⇒ features recoverable ⇒ easier extraction; heavy
  superposition ⇒ irreversibly mixed ⇒ harder. Ties Elhage superposition
  to a measurable quantity.

### Logit lens / activation decoders
- **Can Sensitive Information Be Deleted From LLMs?** (Patil, Hase,
  Bansal) — [arXiv:2309.17410](https://arxiv.org/abs/2309.17410), 2023.
  *Attack+defence.* Uses the **logit lens** as a white-box extractor:
  "deleted" facts recovered 38% of the time by projecting intermediate
  hidden states to vocabulary. Shows editing the output layer leaves the
  secret in the middle layers.
- **LatentQA** (Pan, Chen, Steinhardt) —
  [arXiv:2412.08686](https://arxiv.org/abs/2412.08686), ICLR 2026.
  *Attack capability.* Fine-tunes a decoder LLM to answer open-ended
  questions about *what an activation encodes* — the most general
  "interpretability-tool-as-attack": query "what private info is here?".
- **Entropy-Lens** (Ali, Caso, Irwin, Liò) —
  [arXiv:2502.16570](https://arxiv.org/abs/2502.16570), 2025. *Measurement.*
  Shannon entropy of logit-lens predictions per layer. No privacy claim,
  but the entropy profile is directly usable as a **leakage signature**:
  the sharp-entropy-drop ("commitment") layers coincide with where
  inversion attacks find maximal recoverable content.

### Linear probes & internal-state classifiers on the residual stream
- **UniLeak — Universal Activation Directions for PII Leakage** (Marchyok,
  Coalson, Keum, Son, Hong) — [arXiv:2602.16980](https://arxiv.org/abs/2602.16980),
  2026. *Attack.* PII leakage as a residual-stream linear signal; a fixed
  steering direction amplifies PII across prompts, no labels needed.
- **Do LLMs Know What Is Private Internally?** (Wang, Xiong, Shu) —
  [arXiv:2604.00209](https://arxiv.org/abs/2604.00209), 2026.
  *Measurement+defence.* Contextual-integrity parameters are linearly
  separable directions in activation space; per-dimension steering.
- **LUMIA** (Ibanez-Lissen et al.) —
  [arXiv:2411.19876](https://arxiv.org/abs/2411.19876), 2024. *Attack.*
  Layer-by-layer linear probes for membership inference (+15.71% AUC).
- **Neural Breadcrumbs / memTrace** (Makhija et al.) —
  [arXiv:2509.05449](https://arxiv.org/abs/2509.05449), EACL 2026.
  *Attack.* Membership from layer-wise representation dynamics + attention
  patterns (AUC 0.85).
- **ISACL** (Zhang et al.) —
  [arXiv:2508.17767](https://arxiv.org/abs/2508.17767), EMNLP 2025
  Findings. *Defence/measurement.* Prefill-phase hidden-state classifier
  predicts copyrighted-content leakage *before* generation.

---

## Cluster D — Inversion-attack lineage & the injectivity anchor

### Embedding / hidden-state / logit inversion
- **Song & Raghunathan** (CCS 2020) — [arXiv:2004.00053](https://arxiv.org/abs/2004.00053). Embedding-as-lossy-channel framing (also in Cluster A).
- **Vec2Text — "Text Embeddings Reveal (Almost) As Much As Text"** (Morris, Chiu, Zhao, Shmatikov, Rush) — [arXiv:2310.06816](https://arxiv.org/abs/2310.06816), EMNLP 2023. IT framing is *titular* only; analysis is empirical BLEU.
- **Language Model Inversion** (Morris et al.) — [arXiv:2311.13647](https://arxiv.org/abs/2311.13647), ICLR 2024. Recover prompt from next-token logits (exploits low-dim geometry).
- **PILS** (Nazir, Finlayson, Morris, Ren, Swayamdipta) — [arXiv:2506.17090](https://arxiv.org/abs/2506.17090), NeurIPS 2025. Next-token distributions live in a low-dim subspace; 2–3.5× exact-recovery gain.
- **Depth Gives a False Sense of Privacy** (Dong, Meng, Li, Chen, Liu, Zhu) — [arXiv:2507.16372](https://arxiv.org/abs/2507.16372), USENIX Sec 2025. Inverts shallow→deep internal states; 86.88 F1 from a Llama-3 *middle* layer on a 4,112-token medical prompt. Empirical, no MI math — but the empirical complement to Shu et al.'s IB analysis.
- **Prompt Inversion Attack vs Collaborative Inference** (Qu et al.) — [arXiv:2503.09022](https://arxiv.org/abs/2503.09022), IEEE S&P 2025. 88.4% token acc on Llama-65B inter-layer activations.
- **What Does the Server See?** (Fan, Liu, Wang, Chen) — [arXiv:2605.23158](https://arxiv.org/abs/2605.23158), CCS 2026. ActInv attack + **Perturbation Amplification Factor**: per-layer non-uniform vulnerability (an empirical leakage signature). Defence PriPert.
- **Activation Inversion in Decentralized Training** (Dai, Lu, Zhou) — [arXiv:2502.16086](https://arxiv.org/abs/2502.16086), ACL 2025.
- **Expert Selections Reveal (Almost) As Much As Text** (Nuriyev, Kulp) — [arXiv:2602.04105](https://arxiv.org/abs/2602.04105), 2026. 91.2% token recovery from **MoE routing patterns** alone — a new leakage surface (relevant if GELO ever offloads MoE routing).

### Permutation attack ("Hidden No More") + injectivity
- **An Attack to Break Permutation-Based Private Third-Party Inference**
  (Thomas, Zahran, Choi, Potti, Goldblum, Pal) —
  [arXiv:2505.18332](https://arxiv.org/abs/2505.18332), ICML 2025.
  Reconstructs prompts from permuted hidden states by anchoring on
  permutation-*invariant* quantities (norms, Gram) recoverable from
  **public weights** — exactly the repo's `WEIGHTS-PUB` argument. (Already
  tracked in `aloepri-attacks.md`; logged here for the IT lineage.)
- **Language Models are Injective and Hence Invertible** (Nikolaou,
  Mencattini, Crisostomi, Santilli, Panagakis, Rodolà) —
  [arXiv:2510.15511](https://arxiv.org/abs/2510.15511), 2025. **★ Canonical
  injectivity / non-collision result** — decoder-only Transformers are
  generically injective on finite prompt sets (collisions are measure-
  zero); billions of empirical collision tests, zero found. Ships
  **SipIt**, a linear-time exact inverter. This is the algebraic fact
  underneath every "hidden states are lossless encodings" claim and the
  formal backbone of the GELO threat model's membership argument.
- **Transformer Injectivity & Geometric Robustness** (von Strauss) —
  [arXiv:2511.14808](https://arxiv.org/abs/2511.14808), 2025. Per-layer
  collision discriminants, bi-Lipschitz separation margins on LLaMA-3 /
  Qwen — a geometric refinement of the above.

### KV-cache leakage surface
- **Shadow in the Cache** (Luo, Shao, Zhang et al.) — [arXiv:2508.09442](https://arxiv.org/abs/2508.09442), NDSS 2026. Inversion / collision / semantic-injection on stored K/V tensors; defence KV-Cloak. The KV-cache holds *primitive* K/V (the attention "memory"), a richer surface than fused hidden states.
- **I Know What You Asked** — NDSS 2025, [doi:10.14722/ndss.2025.241772](https://doi.org/10.14722/ndss.2025.241772). Cross-tenant prompt leakage via prefix KV-cache sharing.

### Surveys
- **Model Inversion Attacks: A Survey** (Zhou et al.) — [arXiv:2411.10023](https://arxiv.org/abs/2411.10023), 2024. Broad MIA survey; no dedicated IT section. No survey found that *integrates* inversion + interpretability + IT — itself a gap.

---

## The contribution surface (gaps with no paper found)

These are the white spaces where the planned thread can be genuinely novel:

1. **MDL / V-information as an inversion-vulnerability predictor.** Nobody
   uses online-code MDL or V-usable-information to *rank layers/representa-
   tions by how invertible they are*, nor to explain Vec2Text/deep-layer
   success. MDL's probe-complexity awareness is the principled upgrade
   over the BLEU/F1 the attack papers report. **Most direct contribution.**
2. **A channel-capacity bound on text recovery from a `d`-dim
   representation** ("how many bits of the prompt survive into the
   embedding/hidden state"). No paper derives this; all report empirical
   recovery rates.
3. **Layer-wise MI curve (Tishby IB) ↔ inversion vulnerability** as a
   *measured* relationship across a full model. Shu et al. 2025 is the
   only neighbour and is attack-construction-driven, not a clean IB
   audit; Entropy-Lens supplies the per-layer scalar but never links it
   to attack success.
4. **IB / leakage analysis under `WEIGHTS-PUB` vs `WEIGHTS-BLIND`.** The
   entire IB-privacy literature implicitly assumes weight-blindness; none
   treats the public-weights adversary that makes invariants (norms,
   Gram) into a leakage anchor — the repo's central assumption.
5. **Superposition as a *deliberate* obfuscation mechanism.** Despite
   PrivacyScalpel's lesson, no paper proposes increasing polysemantic
   mixing to *resist* extraction — and superposition-as-lossy-compression
   (Bereska) gives the measuring stick.
6. **TEE-resident execution + an IB leakage *proof*.** TEE-split work is
   cryptographic; IB-split work ignores TEEs. Pairing the GELO cover
   invariant with an IB/MI bound on the covered activation is unclaimed.

## How this connects to the repo

- The injectivity papers (Nikolaou, von Strauss) **formalise** the
  `WEIGHTS-PUB` membership leak documented in `CLAUDE.md` and
  `gpu-offloaded-attention-with-value-cover.md` — worth citing wherever
  the cover invariant is justified.
- Shu et al. 2025 + "Depth Gives a False Sense of Privacy" are the direct
  threat-model references for GELO's TEE↔GPU activation offload; the
  per-layer leakage-signature idea maps onto the sensitive-layer-exclusion
  question (`tee_direct` / layer-skip) in the perf chronicles.
- PrivacyScalpel/UniLeak show the cover must defeat *linear residual-
  stream directions*, not just norms/Gram — a stronger bar than the
  orthogonal-cover membership gate already in `CLAUDE.md`.
- Cross-refs: `aloepri-attacks.md` (Hidden No More already tracked),
  `private-llm-inference.md`, `aloepri-vs-gelo.md`.

## Sources

All URLs verified via OpenAlex / arXiv / publisher fetch during the
2026-06-15 scouting passes. Citation counts where shown are OpenAlex/
Semantic Scholar snapshots and will drift. Primary anchors: see the
inline links above; foundational IT-probing references are Voita & Titov
*Information-Theoretic Probing with MDL* ([arXiv:2003.12298](https://arxiv.org/abs/2003.12298))
and Tishby–Shwartz-Ziv *Opening the Black Box* ([arXiv:1703.00810](https://arxiv.org/abs/1703.00810)).
