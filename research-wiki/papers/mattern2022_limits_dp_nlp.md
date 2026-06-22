---
type: paper
node_id: paper:mattern2022_limits_dp_nlp
title: "Limits of Differential Privacy in NLP"
authors: ["Justus Mattern", "Zhijing Jin", "Benjamin Weggenmann", "Bernhard Schoelkopf", "Mrinmaya Sachan"]
year: 2022
venue: "arXiv"
external_ids:
  arxiv: "2301.09270"
  doi: null
  s2: null
tags: ["metric-dp", "word-embedding-dp", "nns-attack", "utility-privacy-tradeoff", "impossibility"]
added: 2026-06-22T00:00:00Z
---

# Limits of Differential Privacy in NLP

## One-line thesis
Word-level DP mechanisms face a fundamental utility-privacy tradeoff: the ε values that provide meaningful reconstruction privacy destroy downstream task utility, and vice versa.

## Problem / Gap
Word-level DP mechanisms (e.g. Feyisetan 2020, Carvalho 2023-TEM) claim to provide both privacy and utility. Is there a meaningful ε range that achieves both? What is the reconstruction adversary model?

## Method
- Systematically vary ε across a wide range for several word-level DP mechanisms.
- Use NNS (nearest-neighbor search against the known embedding vocabulary) as the reconstruction adversary — the exact MAP reconstruction attack under isotropic Gaussian noise.
- Measure both NNS reconstruction accuracy (privacy metric) and downstream task utility simultaneously.
- Fit the utility-privacy Pareto frontier.

## Key Results
- **At ε values preserving utility, NNS reconstruction accuracy remains high** — no mechanism achieves a "sweet spot" of both meaningful utility and meaningful privacy.
- **At ε values defeating NNS, downstream task performance collapses** to near-random.
- The utility-privacy tradeoff at the word/embedding level is essentially binary: you either have signal (NNS works) or you don't (utility gone).
- NNS is the dominant attack; no stronger adversary is needed to demonstrate the impossibility.

## Assumptions
- WEIGHTS-PUB threat model (adversary knows the vocabulary embedding table).
- Embedding-level noise only; propagated DP not considered.
- Word-level granularity (per-token); not sequence-level.

## Limitations / Failure Modes
- Only evaluates embedding-space (L0) mechanisms — propagated DP (noise at L0, activation at L>0) is not addressed.
- NNS is the only adversary considered; does not test stronger attacks.
- Results may differ for contextual embeddings (BERT-style) vs static (word2vec-style).

## Reusable Ingredients
- The utility-privacy "cliff" framing: at L0 embedding DP, no practical ε achieves both.
- NNS as the reconstruction adversary is sufficient to demonstrate impossibility — no stronger attack needed.
- The impossibility argument: ε needed to defeat NNS requires σ >> Δ/2 (inter-embedding distance / 2), which destroys signal entirely.

## Open Questions
- Does propagated DP (capture at L>0) change the tradeoff? (Our experiments suggest: yes, substantially.)
- Is there an ε range where deep-layer attacks fail while utility is preserved?

## Claims
Used by [[claim:bnn-nns-high-d-geometry]].

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
The canonical reference for the "DP at L0 cannot simultaneously protect and preserve utility" argument. Our unified DP sweep confirms and sharpens this at L0 (BNN=1.0 at r=0.91, BNN=0.969 at r=3.63), while showing propagated DP (L20) is qualitatively different — ridge collapses at r=0.91 while probes show residual MI.
