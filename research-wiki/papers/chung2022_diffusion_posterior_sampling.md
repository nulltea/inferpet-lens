---
type: paper
node_id: paper:chung2022_diffusion_posterior_sampling
title: "Diffusion Posterior Sampling for General Noisy Inverse Problems"
authors: ["Hyungjin Chung", "Jeongsol Kim", "Michael T. Mccann", "Marc L. Klasky", "Jong Chul Ye"]
year: 2022
venue: "The Eleventh International Conference on Learning Representations (ICLR) 2023"
external_ids:
  arxiv: "2209.14687"
  doi: null
  s2: null
tags: []
added: 2026-06-21T11:47:15Z
---

# Diffusion Posterior Sampling for General Noisy Inverse Problems

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

> Diffusion models have been recently studied as powerful generative inverse problem solvers, owing to their high quality reconstructions and the ease of combining existing iterative solvers. However, most works focus on solving simple linear inverse problems in noiseless settings, which significantly under-represents the complexity of real-world problems. In this work, we extend diffusion solvers to efficiently handle general noisy (non)linear inverse problems via approximation of the posterior sampling. Interestingly, the resulting posterior sampling scheme is a blended version of diffusion sampling with the manifold constrained gradient without a strict measurement consistency projection step, yielding a more desirable generative path in noisy settings compared to the previous studies. Our method demonstrates that diffusion models can incorporate various measurement noise statistics such as Gaussian and Poisson, and also efficiently handle noisy nonlinear inverse problems such as Fourier phase retrieval and non-uniform deblurring. Code available at https://github.com/DPS2022/diffusion-posterior-sampling

