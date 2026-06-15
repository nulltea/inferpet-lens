# transformer-attacks-lens

An information-theoretic, interpretability-grounded account of
**confidential-inference attacks** on transformer representations.

## Premise

Transformer-interpretability research measures how much a representation
encodes about a target using **information-theoretic probes** — Minimum
Description Length probing (Voita & Titov 2020), V-usable information
(Xu et al. 2020), the Information Bottleneck. Confidential-inference
*attacks* (embedding inversion / Vec2Text, permutation "Hidden No More",
deep-layer inversion, KV-cache and attention-score leakage) measure the
same thing operationally, but report only BLEU / F1 / recovery rate.

This repo develops the bridge: **a probe-complexity-aware information
measure should be a calibrated predictor of an inversion attack's
success**, giving a principled leakage metric rather than an ad-hoc
reconstruction score. The central novel claim under test —

> a representation's conditional-V-information / surplus-description-length
> for input content predicts an inversion attack's token-recovery rate,
> with the baseline anchored on **public weights** (the conservative
> adversary).

## Document map

| Doc | What it is |
|-----|-----------|
| [docs/research/interpretability-leakage-bridge.md](docs/research/interpretability-leakage-bridge.md) | Literature survey: who connects interpretability (MDL/IB/SAE/probing) to inference attacks, and the gaps. |
| [docs/research/mdl-vinfo-inversion-toolkit.md](docs/research/mdl-vinfo-inversion-toolkit.md) | The MDL / V-information measurement lineage + confirmed implementations and licensing. |
| [docs/plans/it-leakage-estimation-set.md](docs/plans/it-leakage-estimation-set.md) | The decided experimental matrix: 5 attacks × the IT measures that predict them. |

## Origin & external references

This direction spun out of the **private-rag** repo (the GELO split
TEE↔untrusted-GPU offload system). Some documents cite private-rag
artefacts that remain there:

- `gpu-offloaded-attention-with-value-cover.md` — the Gram-invariant
  membership-leak result (the attention-score whitespace in this repo is
  its QK-circuit analogue).
- `evals/aloepri-attacks/` — the existing attack harness (Hidden No More,
  ISA-AttnScore) reused as ground-truth for attacks #2 and #5.
- The `WEIGHTS-PUB` / `WEIGHTS-BLIND` threat-model axis (private-rag
  `CLAUDE.md`).

These are referenced as prose pointers; they are not vendored here.
