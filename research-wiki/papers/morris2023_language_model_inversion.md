---
type: paper
node_id: paper:morris2023_language_model_inversion
title: "Language Model Inversion"
authors: ["John X. Morris", "Wenting Zhao", "Justin T. Chiu", "Vitaly Shmatikov", "Alexander M. Rush"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2311.13647"
  doi: null
  s2: null
tags: []
added: 2026-06-21T11:47:15Z
---

# Language Model Inversion

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

> Language models produce a distribution over the next token; can we use this information to recover the prompt tokens? We consider the problem of language model inversion and show that next-token probabilities contain a surprising amount of information about the preceding text. Often we can recover the text in cases where it is hidden from the user, motivating a method for recovering unknown prompts given only the model's current distribution output. We consider a variety of model access scenarios, and show how even without predictions for every token in the vocabulary we can recover the probability vector through search. On Llama-2 7b, our inversion method reconstructs prompts with a BLEU of $59$ and token-level F1 of $78$ and recovers $27\%$ of prompts exactly. Code for reproducing all experiments is available at http://github.com/jxmorris12/vec2text.

