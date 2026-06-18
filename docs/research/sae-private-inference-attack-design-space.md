---
type: research
status: current
created: 2026-06-17
updated: 2026-06-17
tags: [SAE, sparse-autoencoder, inversion-attacks, confidential-inference, attribute-inference, triage, cover-breaking, leakage-measurement, threat-model]
companion: [sae-as-confidential-inference-attack, interpretability-leakage-bridge, it-leakage-estimation-set]
---

# How SAEs could power private-inference attacks: an offensive design space

## Purpose

The companion [sae-as-confidential-inference-attack](./sae-as-confidential-inference-attack.md)
established the **negative literature finding**: nobody has yet used a sparse
autoencoder (SAE) *as* the recovery mechanism in a confidential-inference
attack. This doc is the **constructive** counterpart: given what the SAE
literature actually proves about SAE capabilities, *how* could an adversary
weaponise them, against which attack goals, and where does an SAE genuinely
add value versus where a plain probe or a direct inverter dominates.

Synthesis, 2026-06-17, grounded in the repo threat model and cited evidence.

## Threat model (the repo's motivating example)

Split-TEE ⟷ untrusted-GPU inference (private-rag / GELO). Adversary =
**honest-but-curious GPU operator** under `WEIGHTS-PUB` (knows weights +
embeddings). It observes representations that traverse the untrusted device —
hidden states, Q·K·V, attention scores, the KV-cache — possibly wrapped by an
external `Transform` (cover/noise/permutation). The **secret** is the prompt
tokens and the sensitive content they encode.

Two consequences that make SAEs attractive here:

1. **`WEIGHTS-PUB` ⇒ the SAE is free.** The attacker can train an SAE on the
   *exact* target model's activations offline, on arbitrary corpora, with
   unlimited compute. No need to rely on transfer.
2. **Online compute is the scarce resource.** The attacker wants to be
   *selective* — spend heavy inversion only on requests worth it. This is the
   economic premise behind goal #3 below, and the SAE's biggest structural fit.

## The SAE decomposed into attacker primitives

An SAE is encoder `f: a∈R^d ↦ z∈R^m` (m≫d, k-sparse code) + decoder
`g: z ↦ â≈a`; decoder columns are the dictionary "features"/atoms, ideally
**monosemantic and human-labelable** (auto-interp / Neuronpedia).

| # | Primitive | What it gives the attacker | Key evidence |
|---|-----------|----------------------------|--------------|
| P1 | **Localization** | *Which* atom carries concept C, at which layer/position it fires | PrivacyScalpel isolates monosemantic PII features ([2503.11232](https://arxiv.org/abs/2503.11232)) |
| P2 | **Label-free readout** | A *named* detector ("email", "person", "diagnosis", "API key") with **no labeled attribute dataset** — zero/few-shot | Auto-interp + SAE-feature safety classifiers (jailbreak/chemical/toxic features) |
| P3 | **Sparse code as feature vector** | Cheap, binarizable input to a downstream classifier/linker | Binarized SAE features beat hidden-state probes & BoW ([2502.11367](https://arxiv.org/abs/2502.11367)) |
| P4 | **Decoder reconstruction** | An activation reconstruction `â` — but **lossy** (shrinkage, k-sparsity discards inactive dims) | Reconstruction-fidelity / shrinkage analyses ([2404.16014](https://arxiv.org/abs/2404.16014), [2406.04093](https://arxiv.org/abs/2406.04093)) |
| P5 | **Universality / transfer** | Features stable across models ⇒ robustness to model mismatch or defense-induced drift | Cross-model universal feature spaces ([2410.06981](https://arxiv.org/abs/2410.06981), USAEs [2502.03714](https://arxiv.org/abs/2502.03714)) |
| P6 | **Capacity / entropy measure** | A per-layer/position *recoverability* score over active features | Superposition-as-lossy-compression entropy ([2512.13568](https://arxiv.org/abs/2512.13568)) |

## Mapping primitives → attack goals

### Goal 1 — Recover sensitive *information* from the prompt (attribute inference)

- **Mechanism:** run the SAE on intercepted activations; read which
  sensitive-concept atoms fire (P1+P2) ⇒ infer presence/category attributes
  ("contains an SSN", "is about person Y", "mentions condition X").
- **For:** PII features are monosemantic and isolable (PrivacyScalpel);
  binarized SAE features beat hidden-state probes on text classification
  (2502.11367).
- **Against (important caveat):** *Are SAEs Useful? A Case Study in Sparse
  Probing* ([2502.16681](https://arxiv.org/abs/2502.16681)) finds SAE probes
  beat baselines on only **~2.2%** of datasets — a **linear probe on the dense
  activation usually matches or beats the SAE for raw extraction accuracy**.
- **Verdict:** realistic and near off-the-shelf, but the SAE's value here is
  **not** higher accuracy — it is (a) *label-free open-set targeting* (P2),
  (b) *interpretability* of exactly what leaked, (c) cheap monitoring (P3).
  For raw power, a probe is the baseline to beat. Strong for
  presence/categorical attributes; weak vs a trained inverter for fine detail.

### Goal 2 — Reconstruct prompt *data* (verbatim) — the strong case

- **SAE alone is the wrong tool.** P4 is a lossy bottleneck; verbatim recovery
  is what injectivity-inversion ([2510.15511](https://arxiv.org/abs/2510.15511)),
  Vec2Text-style inverters, and KV-cache reconstruction
  ([2508.09442](https://arxiv.org/abs/2508.09442)) do — none use SAEs.
- **Where the SAE *amplifies* verbatim recovery (the novel move):**
  1. **Targeting (P1/P6):** localize which positions/layers carry high-value
     tokens ⇒ point an expensive inverter only there (compute-selective).
  2. **Feature-conditioned inversion:** use the SAE feature set as a *semantic
     prior / conditioning* for a generative inverter — the activation-space
     analogue of "pretrained autoencoder as deep image prior" used in image
     reconstruction attacks. Readout (which concepts/entities present) →
     template or LM completion → fluent text. This **bridges Goal 1 → Goal 2**
     and is the cleanest unexplored contribution.
  3. **Seeding:** even a lossy `â` can initialise an optimisation-based
     inversion closer to the manifold.
- **Verdict:** SAE is an **enabler/amplifier, not the recoverer**. Pairing
  SAE readout with a back-end inverter is novel and untaken.

### Goal 3 — Flag sensitive prompts for triage (honest-but-curious compute budgeting) — STRONGEST FIT

- **Mechanism:** the *same* SAE-feature safety monitors that production systems
  use to detect jailbreak/toxic/chemical content (documented Neuronpedia
  features for "toxic roleplay", "chemical/explosive instructions", "ignore
  safety guidelines") — **sign-flipped**. Instead of *blocking* flagged
  requests, the curious operator *ranks* them: which requests are worth a full,
  expensive inversion.
- **Why the SAE is uniquely suited here (and not elsewhere):**
  * **Cheap + online:** a sparse code is a handful of feature lookups; runs
    continuously at ~free marginal cost over the forward pass already being
    computed on the untrusted GPU.
  * **Label-free, broad coverage (P2):** triage on "looks medical / legal /
    credentials / named-person" with no bespoke classifier per category.
  * **Interpretable triage reason:** the attacker learns *why* a request is
    juicy ⇒ picks the right targeted follow-up attack.
- **Verdict:** the **most natural, lowest-cost, most-already-realised**
  offensive use — an existing capability (SAE monitoring) with the intent
  reversed. High feasibility, essentially undiscussed *as an attack*.

### Goals you missed (additional offensive uses)

| # | Use | Sketch | SAE primitive | Maturity |
|---|-----|--------|---------------|----------|
| 4 | **Cross-request linkage / fingerprinting** | Stable sparse codes cluster/link requests from the same user/topic/session ⇒ re-identification, dossier-building, de-anonymising a request stream — even without full recovery | P3, P5 | Plausible, untaken |
| 5 | **Membership / seen-before & dedup** | Feature signatures test whether a target doc/prompt is present, or detect repeated confidential docs across requests | P3 | Plausible |
| 6 | **Cover/defense detection & breaking** | SAE reconstruction error / feature-distribution shift detects whether a representation is *natural* vs perturbed by a defense ⇒ (a) know a cover is active, (b) estimate its strength, (c) **denoise by projecting onto the SAE dictionary** (the natural-activation manifold). Directly relevant to GELO covers | P4, P6 | Novel, high-value |
| 7 | **Leakage prediction / layer ranking** | SAE-code entropy (P6) predicts which layer/token is most invertible ⇒ attack-planning *and* a defense-eval metric; bridges the repo's MDL/V-info "rank layers by invertibility" gap | P6 | Theory exists (2512.13568), unapplied |
| 8 | **Sensitive-span pinpointing** | Per-position feature firings highlight *exactly* which tokens are the secret (the SSN span) ⇒ minimal targeted recovery instead of whole-prompt inversion | P1 | Direct from P1 |

## Honest synthesis — where the SAE truly adds value

- **It is not a stronger extractor than a linear probe** for raw attribute
  accuracy (the 2.2% sparse-probing result). Do not pitch SAE-as-attack on
  power.
- **Its real, orthogonal advantages:**
  1. **Label-free, open-set, interpretable targeting** — know *what* and
     *where* with no attribute-labelled data.
  2. **Cheapness/sparsity ⇒ an always-on triage monitor** (Goal 3) — the
     killer app for a compute-bounded honest-but-curious adversary.
  3. **A manifold model of "natural activations"** ⇒ cover/defense detection
     and denoising (Goal 6).
  4. **A per-position/per-layer leakage map** ⇒ compute-selective targeting
     (Goals 2, 7, 8).
- **The strongest coherent attack narrative is two-stage:** SAE as the cheap,
  interpretable **triage + targeting front-end** (Goals 1, 3, 6, 7, 8) feeding
  an expensive **verbatim inverter back-end** (Goal 2) on only the few flagged,
  localized targets. That is precisely what makes the honest-but-curious
  attacker *compute-efficient* — the economic premise of goal #3.

## Connections back to the repo

- **Sharpens** [interpretability-leakage-bridge](./interpretability-leakage-bridge.md):
  its SAE branch is defence-only; triage (Goal 3), cover-breaking (Goal 6) and
  feature-conditioned inversion (Goal 2) are the open attack directions.
- **Feeds** the attack×measure matrix in `it-leakage-estimation-set`: add an
  **SAE row** — SAE-code entropy as a leakage *measure* (P6), SAE triage/readout
  as an *attack*, SAE reconstruction-error as a *cover-detection* signal.
- **Candidate experiments** (cheap, model-free-ish to start):
  1. SAE-readout attribute inference vs a linear-probe baseline on the same
     activations — quantify the *accuracy gap* (expected ≈0) and the *labelling
     cost gap* (expected large in SAE's favour).
  2. SAE-feature triage AUC for "request contains PII" vs forward-pass cost —
     establish the compute-saving curve for the two-stage attacker.
  3. SAE reconstruction-error as a detector of an applied cover/`Transform`,
     and dictionary-projection denoising vs the cover strength.
  4. SAE-code entropy (P6) vs measured per-layer inversion success — test the
     "entropy predicts invertibility" claim under `WEIGHTS-PUB`.

## Sources

1. Are Sparse Autoencoders Useful? A Case Study in Sparse Probing — https://arxiv.org/abs/2502.16681
2. SAE Features for Classifications and Transferability — https://arxiv.org/abs/2502.11367
3. PrivacyScalpel — https://arxiv.org/abs/2503.11232
4. Superposition as Lossy Compression — https://arxiv.org/abs/2512.13568
5. Quantifying Feature Space Universality Across LLMs via SAEs — https://arxiv.org/abs/2410.06981
6. Universal Sparse Autoencoders — https://arxiv.org/abs/2502.03714
7. Improving Dictionary Learning with Gated SAEs (shrinkage) — https://arxiv.org/abs/2404.16014
8. Scaling and evaluating sparse autoencoders — https://arxiv.org/abs/2406.04093
9. Language Models are Injective and Hence Invertible — https://arxiv.org/abs/2510.15511
10. Shadow in the Cache (KV-cache prompt recovery) — https://arxiv.org/abs/2508.09442
