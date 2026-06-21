---
type: paper
node_id: paper:ding2018_efficient_random_graph
title: "Efficient random graph matching via degree profiles"
authors: ["Jian Ding", "Zongming Ma", "Yihong Wu", "Jiaming Xu"]
year: 2018
venue: "arXiv"
external_ids:
  arxiv: "1811.07821"
  doi: null
  s2: null
tags: []
added: 2026-06-21T11:47:15Z
---

# Efficient random graph matching via degree profiles

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

> Random graph matching refers to recovering the underlying vertex correspondence between two random graphs with correlated edges; a prominent example is when the two random graphs are given by Erdős-Rényi graphs $G(n,\frac{d}{n})$. This can be viewed as an average-case and noisy version of the graph isomorphism problem. Under this model, the maximum likelihood estimator is equivalent to solving the intractable quadratic assignment problem. This work develops an $\tilde{O}(n d^2+n^2)$-time algorithm which perfectly recovers the true vertex correspondence with high probability, provided that the average degree is at least $d = Ω(\log^2 n)$ and the two graphs differ by at most $δ= O( \log^{-2}(n) )$ fraction of edges. For dense graphs and sparse graphs, this can be improved to $δ= O( \log^{-2/3}(n) )$ and $δ= O( \log^{-2}(d) )$ respectively, both in polynomial time. The methodology is based on appropriately chosen distance statistics of the degree profiles (empirical distribution of the degrees of neighbors). Before this work, the best known result achieves $δ=O(1)$ and $n^{o(1)} \leq d \leq n^c$ for some constant $c$ with an $n^{O(\log n)}$-time algorithm \cite{barak2018nearly} and $δ=\tilde O((d/n)^4)$ and $d = \tildeΩ(n^{4/5})$ with a polynomial-time algorithm \cite{dai2018performance}.

