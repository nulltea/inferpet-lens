---
type: paper
node_id: paper:du2023_dpforward_finetuning_inference
title: "DP-Forward: Fine-tuning and Inference on Language Models with Differential Privacy in Forward Pass"
authors: ["Minxin Du", "Xiang Yue", "Sherman S. M. Chow", "Tianhao Wang", "Chenyu Huang", "Huan Sun"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2309.06746"
  doi: null
  s2: null
tags: []
added: 2026-06-26T17:17:58Z
---

# DP-Forward: Fine-tuning and Inference on Language Models with Differential Privacy in Forward Pass

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

> Differentially private stochastic gradient descent (DP-SGD) adds noise to gradients in back-propagation, safeguarding training data from privacy leakage, particularly membership inference. It fails to cover (inference-time) threats like embedding inversion and sensitive attribute inference. It is also costly in storage and computation when used to fine-tune large pre-trained language models (LMs). We propose DP-Forward, which directly perturbs embedding matrices in the forward pass of LMs. It satisfies stringent local DP requirements for training and inference data. To instantiate it using the smallest matrix-valued noise, we devise an analytic matrix Gaussian~mechanism (aMGM) by drawing possibly non-i.i.d. noise from a matrix Gaussian distribution. We then investigate perturbing outputs from different hidden (sub-)layers of LMs with aMGM noises. Its utility on three typical tasks almost hits the non-private baseline and outperforms DP-SGD by up to 7.7pp at a moderate privacy level. It saves 3$\times$ time and memory costs compared to DP-SGD with the latest high-speed library. It also reduces the average success rates of embedding inversion and sensitive attribute inference by up to 88pp and 41pp, respectively, whereas DP-SGD fails.

