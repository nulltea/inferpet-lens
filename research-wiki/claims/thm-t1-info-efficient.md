---
type: claim
node_id: claim:thm-t1-info-efficient
name: "T1: info-efficient attacks dominate weak attacks; optimal recovery tracks MI"
description: ""
node_type: claim
status: verified
provenance: ".aris/traces/proof-checker/2026-06-21_run01/"
tags: ["backbone", "verified", "proof-gated"]
date: 2026-06-21
added: 2026-06-21T12:08:46Z
---

# T1: info-efficient attacks dominate weak attacks; optimal recovery tracks MI

**status:** `verified`

## Statement
Under S-X-Y Markov + finite 2nd moment / finite alphabet: (a) Bayes-optimal attack on Y weakly dominates any lossy/linear attack (every prior, stated losses, any noise); (b) strict for squared loss iff E[S|Y] non-affine (linear ridge), via MMSE orthogonality; (c) on the Gaussian/DP arm the optimal attack's recovery is monotone in MI (I-MMSE for S=X; degradation-order DPI+Blackwell for token target and for Laplace).

## Honest scope
Strictness asserted only via non-affinity (squared) or common-MAP-action (0-1), NEVER via MI-loss alone (single-metric converse invalid). I-MMSE 1/2-identity is channel-input(S=X)+Gaussian only; Laplace/token use degradation comonotonicity, no closed-form. Needs E||S||^2,E||Y||^2<inf (else L1/median). 'Weak attack need not track MI' is a non-implication remark, not proven positive.

## Evidence chain
gpt-5.5 xhigh proof-checker 2 rounds verdict PASS; round-1 I1-I7 (all LOCAL/MINOR, no FATAL/CRITICAL) + N1 resolved; PROOF_AUDIT.json; proof at docs/research/info-efficient-attack-guarantee.md

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

