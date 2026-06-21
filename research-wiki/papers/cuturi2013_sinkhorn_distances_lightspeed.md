---
type: paper
node_id: paper:cuturi2013_sinkhorn_distances_lightspeed
title: "Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances"
authors: ["Marco Cuturi"]
year: 2013
venue: "Advances in Neural Information Processing Systems 26, pages 2292--2300, 2013"
external_ids:
  arxiv: "1306.0895"
  doi: null
  s2: null
tags: []
added: 2026-06-21T11:47:15Z
---

# Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances

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

> Optimal transportation distances are a fundamental family of parameterized distances for histograms. Despite their appealing theoretical properties, excellent performance in retrieval tasks and intuitive formulation, their computation involves the resolution of a linear program whose cost is prohibitive whenever the histograms' dimension exceeds a few hundreds. We propose in this work a new family of optimal transportation distances that look at transportation problems from a maximum-entropy perspective. We smooth the classical optimal transportation problem with an entropic regularization term, and show that the resulting optimum is also a distance which can be computed through Sinkhorn-Knopp's matrix scaling algorithm at a speed that is several orders of magnitude faster than that of transportation solvers. We also report improved performance over classical optimal transportation distances on the MNIST benchmark problem.

