---
type: paper
node_id: paper:shi2021_selective_differential_privacy
title: "Selective Differential Privacy for Language Modeling"
authors: ["Weiyan Shi", "Aiqi Cui", "Evan Li", "Ruoxi Jia", "Zhou Yu"]
year: 2021
venue: "arXiv"
external_ids:
  arxiv: "2108.12944"
  doi: null
  s2: null
tags: []
added: 2026-06-26T17:17:59Z
---

# Selective Differential Privacy for Language Modeling

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

> With the increasing applications of language models, it has become crucial to protect these models from leaking private information. Previous work has attempted to tackle this challenge by training RNN-based language models with differential privacy guarantees. However, applying classical differential privacy to language models leads to poor model performance as the underlying privacy notion is over-pessimistic and provides undifferentiated protection for all tokens in the data. Given that the private information in natural language is sparse (for example, the bulk of an email might not carry personally identifiable information), we propose a new privacy notion, selective differential privacy, to provide rigorous privacy guarantees on the sensitive portion of the data to improve model utility. To realize such a new notion, we develop a corresponding privacy mechanism, Selective-DPSGD, for RNN-based language models. Besides language modeling, we also apply the method to a more concrete application--dialog systems. Experiments on both language modeling and dialog system building show that the proposed privacy-preserving mechanism achieves better utilities while remaining safe under various privacy attacks compared to the baselines. The data and code are released at https://github.com/wyshi/lm_privacy to facilitate future research .

