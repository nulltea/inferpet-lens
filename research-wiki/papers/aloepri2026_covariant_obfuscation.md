---
type: paper
node_id: paper:aloepri2026_covariant_obfuscation
title: "Towards Privacy-Preserving LLM Inference via Covariant Obfuscation (Technical Report)"
authors: []
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2603.01499"
  doi: null
  s2: null
tags: [aloepri, covariant-obfuscation, orthogonal-obfuscation, privacy-preserving-inference, target-scheme]
added: 2026-07-01T00:00:00Z
companion: docs/html/static-obf.html
---

# Towards Privacy-Preserving LLM Inference via Covariant Obfuscation

## One-line thesis
"AloePri": a static, key-based covariant obfuscation that jointly re-parameterizes weights and data so the
untrusted server runs bit-equivalent inference on obfuscated tensors — a lossless change of basis (Alg1
keymat P̂/Q̂ + token permutation Π) plus per-head value rotation Û_vo + head-perm (Alg2) and αₑ embedding noise.

## Relevance to us
The **target scheme** of this repo's static-obf study. Our claim
[[claim:aloepri-kqvout-basis-alignment]] attacks its **Alg2 per-head value rotation** on `kqv_out`. Authors
field left empty (not confirmed from the abstract — fill on a body read). Reported inversion recovery <5%
of tokens at the default config (the number our threat-model-respecting attacks re-examine).

## Notes
Because the key material is a lossless re-parameterization it preserves mutual information — so it is
access control, not information hiding, and is absorbed by any attacker who recovers the basis. The αₑ
noise is the only information-destroying lever. See [[paper:yukhimchuk2026_privacy_from_symmetry]] (same
orthogonal-obfuscation family) and [[paper:conneau2017_word_translation_without_parallel_data]] (the
Procrustes-from-anchors technique our attack reuses).
