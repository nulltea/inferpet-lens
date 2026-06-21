---
type: claim
node_id: claim:threat-model-fairness
name: "Attack-comparison fairness criterion (WEIGHTS-PUB honest-but-curious)"
description: ""
node_type: claim
status: drafted
provenance: ""
tags: ["threat-model", "methodology", "fairness"]
date: 2026-06-21
added: 2026-06-21T12:58:02Z
---

# Attack-comparison fairness criterion (WEIGHTS-PUB honest-but-curious)

**status:** `drafted`

## Statement
All attacks compared in this program operate under ONE fixed threat model: WEIGHTS-PUB honest-but-curious — the adversary knows the model weights + embedding table + the DP mechanism params (σ, clip C are the published privacy parameters), observes the DP-protected representation, and (since it has the weights) can run the public model on its OWN chosen inputs to generate unlimited (noised-representation, token) training pairs at any σ. An attack is ADMISSIBLE iff it uses only this information. A comparison is VALID only between admissible attacks.

## Honest scope
Does NOT hold under WEIGHTS-BLIND (no weight access): there neither ridge nor any trained decoder is admissible (cannot generate training pairs), so those comparisons would be invalid and out of scope. The σ-awareness is admissible ONLY because DP params are public; if σ were secret, channel-aware attacks would be inadmissible.

## Evidence chain
ridge: uses table + self-generated noised training pairs (admissible). L0 Bayes-NN: uses table + public σ, NO training set (admissible, uses strictly LESS). L>0 channel-aware decoder: table + self-generated noised pairs at σ (SAME admissible info as ridge). All fair under WEIGHTS-PUB. The capacity-PVI reader likewise trains on admissible noised data.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

