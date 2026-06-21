---
type: paper
node_id: paper:balle2018_improving_gaussian_mechanism
title: "Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising"
authors: ["Borja Balle", "Yu-Xiang Wang"]
year: 2018
venue: "arXiv"
external_ids:
  arxiv: "1805.06530"
  doi: null
  s2: null
tags: []
added: 2026-06-21T11:47:15Z
---

# Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising

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

> The Gaussian mechanism is an essential building block used in multitude of differentially private data analysis algorithms. In this paper we revisit the Gaussian mechanism and show that the original analysis has several important limitations. Our analysis reveals that the variance formula for the original mechanism is far from tight in the high privacy regime ($\varepsilon \to 0$) and it cannot be extended to the low privacy regime ($\varepsilon \to \infty$). We address these limitations by developing an optimal Gaussian mechanism whose variance is calibrated directly using the Gaussian cumulative density function instead of a tail bound approximation. We also propose to equip the Gaussian mechanism with a post-processing step based on adaptive estimation techniques by leveraging that the distribution of the perturbation is known. Our experiments show that analytical calibration removes at least a third of the variance of the noise compared to the classical Gaussian mechanism, and that denoising dramatically improves the accuracy of the Gaussian mechanism in the high-dimensional regime.

