---
type: paper
node_id: paper:morris2023_text_embeddings_reveal
title: "Text Embeddings Reveal (Almost) As Much As Text"
authors: ["John X. Morris", "Volodymyr Kuleshov", "Vitaly Shmatikov", "Alexander M. Rush"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2310.06816"
  doi: null
  s2: null
tags: []
added: 2026-06-21T11:47:15Z
---

# Text Embeddings Reveal (Almost) As Much As Text

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

> How much private information do text embeddings reveal about the original text? We investigate the problem of embedding \textit{inversion}, reconstructing the full text represented in dense text embeddings. We frame the problem as controlled generation: generating text that, when reembedded, is close to a fixed point in latent space. We find that although a naïve model conditioned on the embedding performs poorly, a multi-step method that iteratively corrects and re-embeds text is able to recover $92\%$ of $32\text{-token}$ text inputs exactly. We train our model to decode text embeddings from two state-of-the-art embedding models, and also show that our model can recover important personal information (full names) from a dataset of clinical notes. Our code is available on Github: \href{https://github.com/jxmorris12/vec2text}{github.com/jxmorris12/vec2text}.

