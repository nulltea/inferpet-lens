---
type: paper
node_id: paper:yukhimchuk2026_privacy_from_symmetry
title: "Privacy from Symmetry: Orthogonally Equivariant Transformers for LLM Inference"
authors: ["Alexander Yukhimchuk", "Andrey Shulga", "Mladen Kolar", "Martin Takáč"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2606.16461"
  doi: null
  s2: null
tags: [orthogonal-obfuscation, equivariant-transformer, privacy-preserving-inference, known-plaintext, related-defense]
added: 2026-07-01T00:00:00Z
---

# Privacy from Symmetry: Orthogonally Equivariant Transformers for LLM Inference

## One-line thesis (from abstract)
The client multiplies embeddings by a secret orthogonal matrix before transmission; an orthogonally
equivariant transformer computes on the rotated space, eliminating direct cosine-NN inversion (top-10
token recovery reduced from >35% to ≤1.3%).

## Relevance to us
Same **orthogonal-obfuscation family** as AloePri's Alg2 rotation — the defense our
[[claim:aloepri-kqvout-basis-alignment]] targets. The abstract establishes the *defense*; secondary
sources indicate this line discusses that orthogonal conjugation forces the adversary to first *recover
the hidden rotation via alignment* and that anchor/known-plaintext pairs enable it — **not verified from
the abstract; confirm on a body read** before citing for the attack. Our attack is the concrete
known-plaintext / orthogonal-Procrustes recovery instantiated on AloePri Alg2 with the per-head block
structure.
