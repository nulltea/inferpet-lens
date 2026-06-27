---
type: paper
node_id: paper:mireshghallah2019_shredder_learning_noise
title: "Shredder: Learning Noise Distributions to Protect Inference Privacy"
authors: ["Fatemehsadat Mireshghallah", "Mohammadkazem Taram", "Prakash Ramrakhyani", "Dean Tullsen", "Hadi Esmaeilzadeh"]
year: 2019
venue: "arXiv"
external_ids:
  arxiv: "1905.11814"
  doi: null
  s2: null
tags: []
added: 2026-06-26T17:17:59Z
---

# Shredder: Learning Noise Distributions to Protect Inference Privacy

## One-line thesis
_TODO: fill in after reading._

## Problem / Gap
_TODO._

## Method
_TODO._

## Key Results
_TODO._

## Assumptions
_TODO._

## Limitations / Failure Modes
_TODO._

## Reusable Ingredients
_TODO._

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
_TODO._

## Abstract (original)

> A wide variety of deep neural applications increasingly rely on the cloud to perform their compute-heavy inference. This common practice requires sending private and privileged data over the network to remote servers, exposing it to the service provider and potentially compromising its privacy. Even if the provider is trusted, the data can still be vulnerable over communication channels or via side-channel attacks in the cloud. To that end, this paper aims to reduce the information content of the communicated data with as little as possible compromise on the inference accuracy by making the sent data noisy. An undisciplined addition of noise can significantly reduce the accuracy of inference, rendering the service unusable. To address this challenge, this paper devises Shredder, an end-to-end framework, that, without altering the topology or the weights of a pre-trained network, learns additive noise distributions that significantly reduce the information content of communicated data while maintaining the inference accuracy. The key idea is finding the additive noise distributions by casting it as a disjoint offline learning process with a loss function that strikes a balance between accuracy and information degradation. The loss function also exposes a knob for a disciplined and controlled asymmetric trade-off between privacy and accuracy. Experimentation with six real-world DNNs from text processing and image classification shows that Shredder reduces the mutual information between the input and the communicated data to the cloud by 74.70% compared to the original execution while only sacrificing 1.58% loss in accuracy. On average, Shredder also offers a speedup of 1.79x over Wi-Fi and 2.17x over LTE compared to cloud-only execution when using an off-the-shelf mobile GPU (Tegra X2) on the edge.

