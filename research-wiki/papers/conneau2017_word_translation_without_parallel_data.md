---
type: paper
node_id: paper:conneau2017_word_translation_without_parallel_data
title: "Word Translation Without Parallel Data"
authors: ["Alexis Conneau", "Guillaume Lample", "Marc'Aurelio Ranzato", "Ludovic Denoyer", "Hervé Jégou"]
year: 2017
venue: "ICLR 2018 / arXiv"
external_ids:
  arxiv: "1710.04087"
  doi: null
  s2: null
tags: [cross-lingual-alignment, orthogonal-procrustes, seed-dictionary, technique-origin]
added: 2026-07-01T00:00:00Z
---

# Word Translation Without Parallel Data (MUSE)

## One-line thesis
Align two embedding spaces with an **orthogonal linear map** learned by iterative **orthogonal Procrustes**
on a (small or bootstrapped) seed dictionary of anchor pairs — with the closed form R = UVᵀ from svd(XᵀY).

## Relevance to us
The **technique origin** of our attack's core step: recover an orthogonal map from a handful of anchor
(plaintext, target) pairs via orthogonal Procrustes. VecMap (Artetxe et al.) shows seed dictionaries as
small as ~25 pairs suffice. Combined with the classical **known-plaintext attack** on a linear/orthogonal
cipher (Hill cipher: n pairs solve the n×n key), this is why AloePri Alg2's rotation is not an
information-theoretic defense once anchors are available — see [[claim:aloepri-kqvout-basis-alignment]].
This grounds our novelty assessment: the attack *technique* is known; only the AloePri-Alg2 instantiation
+ per-head block Procrustes + honest fully-known-prefix analysis are new (and evaluation-grade, not a new primitive).
