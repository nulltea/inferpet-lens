---
type: claim
node_id: claim:bayes-gap-diagnosis
name: "MI-recovery gap = Bayes-optimality gap"
description: "Empirical, jury-PARTIAL: the MI-recovery gap diagnoses attack inefficiency (directly at L0, directionally at depth), not poor MI prediction."
node_type: claim
status: drafted
provenance: ".aris/traces/result-to-claim/2026-06-23_run01/"
tags: ["backbone", "diagnosis", "empirical", "partial", "resid-dp-attacks"]
date: 2026-06-21
updated: 2026-06-23
added: 2026-06-21T11:48:34Z
---

# MI-recovery gap = Bayes-optimality gap

**status:** `drafted` (empirical; jury verdict `PARTIAL`)

## Statement
Recovery = f(MI, estimator quality); MI upper-bounds best-possible recovery (Fano/de Cherisey). The
observed effect (noise barely moves CLUB/PVI but collapses ridge recovery) is diagnostic of an attack
far below the information-theoretic ceiling — a **Bayes-optimality gap** — NOT of MI being a poor predictor.

## Empirical status (jury-gated, NOT self-certified)
`PARTIAL` (Codex xhigh, 2026-06-23, medium; trace `.aris/traces/result-to-claim/2026-06-23_run01/`):
- **Directly shown at L0:** the MI/probe signal stays high (CLUB 1912 bits, capPVI 0.74) while ridge
  collapses (0.02) and an exact Bayes-NN recovers ~1.0 — the gap is provably attack inefficiency there.
- **MI is not a generally poor predictor:** under at-layer noise ridge, decoder, CLUB, capPVI, and MDL/SDL
  all track together (R2 ρ=1.0, R3 +0.80).
- **At depth the gap is INFERRED, not measured to the ceiling:** R5 shows a *stronger-attack* gap closing
  (deep +0.83 vs ridge −0.09), not a demonstrated near-Bayes optimum. R6 shows attack optimality is
  regime-dependent. Closing it to the ceiling at depth is open.

## Honest scope
Does not claim a given weak attack tracks MI; only that the gap diagnoses estimator quality, not MI
failure. The optimal-attack-is-MI-governed direction is the **verified** theorem [[thm-t1-info-efficient]];
this is its diagnostic corollary.

## Evidence chain
- R1 L0 (`results/l0_fast.txt` + `results/b2_l0_bayes.json` bits) — exp `exp:b2-l0-bayes-vs-ridge`.
- R2/R3 at-layer (`results/b2_lpos_decoder.json`, `results/mdl_probe_check.json`) — all probes track recovery.
- R5 propagated (`results/b6_strong_decoder.json`) — exp `exp:b6-strong-decoder`.
- Theory: [[thm-t1-info-efficient]] (verified, Fano/de Cherisey + I-MMSE), [[restore-correlation]].

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

