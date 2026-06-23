---
type: claim
node_id: claim:restore-correlation
name: "Info-efficient attacks restore MI-recovery correlation (residual stream, partial)"
description: "Empirical, jury-PARTIAL on resid-dp-attacks: L0 shows uplift / ceiling-realization; propagated-DP depth shows ridge decorrelation AND decoder re-correlation; absent under at-layer noise."
node_type: claim
status: drafted
provenance: ".aris/traces/result-to-claim/2026-06-23_run01/"
tags: ["headline", "empirical", "partial", "resid-dp-attacks"]
date: 2026-06-21
updated: 2026-06-23
added: 2026-06-21T11:48:34Z
---

# Info-efficient attacks restore MI-recovery correlation (residual stream, partial)

**status:** `drafted` (empirical; jury verdict `PARTIAL`)

## Statement (scoped after jury, 2026-06-23)
Replacing the information-inefficient weak attack (ridge regression) with an information-efficient
attack restores the MI↔recovery correlation **in the regime where the weak attack decorrelates**:
- **L0 input-DP (uplift / ceiling-realization evidence):** an exact Bayes-NN realizes information ridge
  cannot (recovery ~1.0 where ridge collapses to 0.02), confirming recovery is limited by attack efficiency,
  not by lost information. This evidences the **attack-efficiency gap**, not literal re-correlation — at L0
  Bayes recovery is saturated, so its own Spearman-vs-probe is degenerate (=0).
- **Propagated input-DP at depth (L20) — the RE-CORRELATION evidence:** a trained channel-aware decoder
  partially re-correlates with the MI probes (Spearman vs capPVI: deep +0.83, iterative +0.71) precisely
  where ridge **anti**-correlates (−0.09).
This **does not hold under at-layer noise** (there even ridge already tracks MI, ρ=1.0 — no gap to close),
and a full-noise-range Bayes/Vec2Text ceiling attack at depth is **not yet built** (open frontier).

## Empirical status (jury-gated, NOT self-certified)
`PARTIAL` (Codex xhigh result-to-claim, 2026-06-23, medium confidence;
trace `.aris/traces/result-to-claim/2026-06-23_run01/`). Integrity audit: WARN, no FAIL
(`refine-logs/resid-dp-attacks/EXPERIMENT_AUDIT.md`). The underlying theory — Bayes-optimal weakly
dominates lossy/linear, recovery monotone in MI on the Gaussian arm — is the **verified** theorem
[[thm-t1-info-efficient]]; this claim is its empirical instantiation on the residual stream.

## Evidence chain
- **R1 (L0, `results/l0_fast.txt`, synthetic_proxy) — UPLIFT/ceiling-realization, not re-correlation:**
  Bayes-NN uplift +0.980 over ridge @r=1.82 (ridge 0.020 → Bayes 1.000); CLUB decays only 3084→1912 bits →
  information preserved, ridge inefficient. Exp node `exp:b2-l0-bayes-vs-ridge` (verdict yes).
- **R5 (propagated DP @L20, `results/b6_strong_decoder.json`, single seed):** deep decoder re-correlates
  +0.8286, iterative +0.7143 vs ridge −0.0857. Exp node `exp:b6-strong-decoder` (verdict partial).
- **R4 (`results/b2_propagated_dp.json`):** decoder uplift grows with propagated noise −0.07→+0.14.
  Exp node `exp:b2-propagated-dp` (verdict partial).
- **Counter-regime R2 (`results/b2_lpos_decoder.json`):** under AT-LAYER noise the MLP loses to ridge and
  ridge already tracks MI ρ=1.0 → bounds the claim to noise-propagation geometry. Exp `exp:b2-lpos-decoder-vs-ridge`.
- **R6 (`results/b6c_forward_model.json`):** forward-model Vec2Text closes the low-noise gap (+0.53) but is
  noise-fragile → no single attack dominates the range; embedding-space iteration is null (R5 C7).
- **Theory:** [[thm-t1-info-efficient]] (verified), [[bayes-gap-diagnosis]], [[mi-monotone-gaussian]],
  [[weak-domination]], [[strict-improvement]].

## Open (queued firm-ups, not this phase)
Multi-seed dense-ε R5 with CIs; **noise-aware FMV** (match E[Y|candidate] not a clean reference);
FMV-⊕-denoiser hybrid to dominate the full noise range; second model family.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

