---
type: claim
node_id: claim:privacy-funnel-llm-regime-split
name: "Privacy-funnel regime split: I(rep;input)↓ at fixed I(rep;task) is achievable for downstream-label utility, not for faithful generation"
description: "Minimizing input leakage at fixed task utility is achievable for LLM/MLM iff the task is a coarse downstream label; not in any strong sense for next-token/generation (Privacy Funnel)."
node_type: claim
status: drafted
provenance: "refine-logs/pythia-depth/dp_utility.json; research literature (arxiv 1402.1774, 2310.09130)"
tags: ["privacy-funnel", "information-bottleneck", "privacy-utility-tradeoff", "mutual-information", "generation-vs-downstream", "llm", "mlm", "learned-noise", "adversarial-representation", "dp"]
date: 2026-06-26
added: 2026-06-26T17:18:58Z
---

# Privacy-funnel regime split: I(rep;input)↓ at fixed I(rep;task) is achievable for downstream-label utility, not for faithful generation

**status:** `drafted`

## Statement
For a released representation rep of input X under a privacy mapping, the achievability of minimizing I(rep;X) (input leakage) at a fixed lower bound on I(rep;T) (task utility) is governed by the Privacy Funnel (Makhdoumi et al. 2014 — the log-loss dual of the Information Bottleneck; tradeoff curve generally non-convex). (1) When T is FAITHFUL GENERATION / next-token prediction, T is near-deterministic in X, so the funnel floor is high: I(rep;X) cannot be reduced materially without collapsing I(rep;T). The tradeoff is intrinsically bleak — consistent with exp:dp-utility-vs-eps-160m, where next-token accuracy retention drops below 50% by ε≈110 while the input token stays recoverable. (2) When T is a COARSE DOWNSTREAM LABEL (sentiment, NLI, retrieval), I(rep;T) ≪ I(rep;X), and the funnel admits a large region where I(rep;X) is slashed at near-constant I(rep;T); learned/adversarial schemes realize it — Split-and-Denoise (+10% utility at fixed ε), MetaMorphosis, Shredder, adversarial representation learning.

## Honest scope
Does NOT give a numeric ε or closed-form floor for any specific model. Does NOT claim we have measured downstream-task utility on OUR models — that is the queued confirming experiment; the generation half is empirically supported by our own cliff, the downstream half by external SOTA only. The 'yes' requires the label to be a genuinely coarse function of input (low I(T;X)); tasks needing verbatim content (detailed summarization, code, extraction) move back toward the generation regime. 'No in any strong sense' = no DP/obfuscation mechanism (isotropic, anisotropic, OR learned) escapes the funnel floor for generation — not that achievable privacy is zero, only that meaningful input-leakage reduction costs proportional generation utility.

## Evidence chain
THEORY: Privacy Funnel [[paper:makhdoumi2014_from_information_bottleneck]] (log-loss dual of IB; perfect privacy with positive utility requires private-only info separable from the utility-relevant part); maximal-correlation/MI privacy bounds [[paper:asoodeh2015_maximal_correlation_mutual]]. GENERATION-REGIME (ours): exp:dp-utility-vs-eps-160m (next-token retention −50% at ε≈110, input still recoverable). DOWNSTREAM-REGIME (external SOTA): [[paper:mai2023_splitanddenoise_protect_large]] (+10% utility at fixed ε, downstream), [[paper:arefeen2023_metamorphosis_taskoriented_privacy]], [[paper:mireshghallah2019_shredder_learning_noise]], [[paper:shi2021_selective_differential_privacy]], [[paper:du2023_dpforward_finetuning_inference]].

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

