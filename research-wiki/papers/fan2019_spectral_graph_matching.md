---
type: paper
node_id: paper:fan2019_spectral_graph_matching
title: "Spectral Graph Matching and Regularized Quadratic Relaxations I: The Gaussian Model"
authors: ["Zhou Fan", "Cheng Mao", "Yihong Wu", "Jiaming Xu"]
year: 2019
venue: "arXiv"
external_ids:
  arxiv: "1907.08880"
  doi: null
  s2: null
tags: []
added: 2026-06-21T11:47:15Z
---

# Spectral Graph Matching and Regularized Quadratic Relaxations I: The Gaussian Model

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

> Graph matching aims at finding the vertex correspondence between two unlabeled graphs that maximizes the total edge weight correlation. This amounts to solving a computationally intractable quadratic assignment problem. In this paper we propose a new spectral method, GRAph Matching by Pairwise eigen-Alignments (GRAMPA). Departing from prior spectral approaches that only compare top eigenvectors, or eigenvectors of the same order, GRAMPA first constructs a similarity matrix as a weighted sum of outer products between all pairs of eigenvectors of the two graphs, with weights given by a Cauchy kernel applied to the separation of the corresponding eigenvalues, then outputs a matching by a simple rounding procedure. The similarity matrix can also be interpreted as the solution to a regularized quadratic programming relaxation of the quadratic assignment problem. For the Gaussian Wigner model in which two complete graphs on $n$ vertices have Gaussian edge weights with correlation coefficient $1-σ^2$, we show that GRAMPA exactly recovers the correct vertex correspondence with high probability when $σ= O(\frac{1}{\log n})$. This matches the state of the art of polynomial-time algorithms, and significantly improves over existing spectral methods which require $σ$ to be polynomially small in $n$. The superiority of GRAMPA is also demonstrated on a variety of synthetic and real datasets, in terms of both statistical accuracy and computational efficiency. Universality results, including similar guarantees for dense and sparse Erdős-Rényi graphs, are deferred to the companion paper.

