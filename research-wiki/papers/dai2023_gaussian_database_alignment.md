---
type: paper
node_id: paper:dai2023_gaussian_database_alignment
title: "Gaussian Database Alignment and Gaussian Planted Matching"
authors: ["Osman Emre Dai", "Daniel Cullina", "Negar Kiyavash"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2307.02459"
  doi: null
  s2: null
tags: []
added: 2026-06-21T11:47:15Z
---

# Gaussian Database Alignment and Gaussian Planted Matching

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

> Database alignment is a variant of the graph alignment problem: Given a pair of anonymized databases containing separate yet correlated features for a set of users, the problem is to identify the correspondence between the features and align the anonymized user sets based on correlation alone. This closely relates to planted matching, where given a bigraph with random weights, the goal is to identify the underlying matching that generated the given weights. We study an instance of the database alignment problem with multivariate Gaussian features and derive results that apply both for database alignment and for planted matching, demonstrating the connection between them. The performance thresholds for database alignment converge to that for planted matching when the dimensionality of the database features is \(ω(\log n)\), where \(n\) is the size of the alignment, and no individual feature is too strong. The maximum likelihood algorithms for both planted matching and database alignment take the form of a linear program and we study relaxations to better understand the significance of various constraints under various conditions and present achievability and converse bounds. Our results show that the almost-exact alignment threshold for the relaxed algorithms coincide with that of maximum likelihood, while there is a gap between the exact alignment thresholds. Our analysis and results extend to the unbalanced case where one user set is not fully covered by the alignment.

