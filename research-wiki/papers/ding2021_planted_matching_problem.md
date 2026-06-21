---
type: paper
node_id: paper:ding2021_planted_matching_problem
title: "The planted matching problem: Sharp threshold and infinite-order phase transition"
authors: ["Jian Ding", "Yihong Wu", "Jiaming Xu", "Dana Yang"]
year: 2021
venue: "arXiv"
external_ids:
  arxiv: "2103.09383"
  doi: null
  s2: null
tags: []
added: 2026-06-21T11:47:15Z
---

# The planted matching problem: Sharp threshold and infinite-order phase transition

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

> We study the problem of reconstructing a perfect matching $M^*$ hidden in a randomly weighted $n\times n$ bipartite graph. The edge set includes every node pair in $M^*$ and each of the $n(n-1)$ node pairs not in $M^*$ independently with probability $d/n$. The weight of each edge $e$ is independently drawn from the distribution $\mathcal{P}$ if $e \in M^*$ and from $\mathcal{Q}$ if $e \notin M^*$. We show that if $\sqrt{d} B(\mathcal{P},\mathcal{Q}) \le 1$, where $B(\mathcal{P},\mathcal{Q})$ stands for the Bhattacharyya coefficient, the reconstruction error (average fraction of misclassified edges) of the maximum likelihood estimator of $M^*$ converges to $0$ as $n\to \infty$. Conversely, if $\sqrt{d} B(\mathcal{P},\mathcal{Q}) \ge 1+ε$ for an arbitrarily small constant $ε>0$, the reconstruction error for any estimator is shown to be bounded away from $0$ under both the sparse and dense model, resolving the conjecture in [Moharrami et al. 2019, Semerjian et al. 2020]. Furthermore, in the special case of complete exponentially weighted graph with $d=n$, $\mathcal{P}=\exp(λ)$, and $\mathcal{Q}=\exp(1/n)$, for which the sharp threshold simplifies to $λ=4$, we prove that when $λ\le 4-ε$, the optimal reconstruction error is $\exp\left( - Θ(1/\sqrtε) \right)$, confirming the conjectured infinite-order phase transition in [Semerjian et al. 2020].

