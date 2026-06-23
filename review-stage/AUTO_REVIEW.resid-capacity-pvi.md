# Auto Review Loop — resid-capacity-pvi consolidation

## Round 1 (2026-06-23) — Codex gpt-5.5 xhigh (thread 019ef5e4)
### Assessment
- Score: 8/10 — Verdict: almost (POSITIVE_THRESHOLD met: score>=6 AND verdict in {ready,almost})
- Judged as a scoped consolidation report. Core surface coherent; estimator failure diagnosed,
  repaired probe separated from attack, standardized results consistent with the audit, proof/audit
  story clean, scope caveats honest not buried.
### Ranked weaknesses + MINIMUM fixes (all wording/robustness, no new experiments)
1. Single-seed n=7 correlations brittle (esp. L20) → add leave-one-out sensitivity. **DONE**: LOO range
   added to report + standardized results + claim (L12 stays + [+0.09,+0.60]; L20 stays − [−0.94,−0.09]).
2. L20 sign reversal must stay provisional → **DONE**: "observation in this run, not established".
3. Correlation = intervention-ordered co-tracking, not causal equivalence → **DONE**: caveat added to Discussion.
4. Accuracy first, bits as estimator-diagnostic only → already the framing; reaffirmed.
5. CLUB "tracks identically" too strong → **DONE**: split wording (co-tracks in rep-space; parallel
   attenuation under input-DP, no sign reversal).
6. Independence necessary not sufficient (shared geometry/frequency confounds) → **DONE**: caveat added.
### Status
- STOP — positive assessment at round 1 (8/10, almost). Cheap wording/robustness fixes applied inline.

## Method Description
An attack-independent token-identity reader (capacity-matched PVI) standardizes a hidden state,
reduces it below the validation count by train-only PCA, and fits a calibrated linear-softmax reader
over token-id classes (never reading the embedding table). Its bounded accuracy is correlated, across
representation-space and input-DP defense sweeps and per layer, against a ridge embedding-inversion
attack's top-1 recovery; CLUB (a variational MI upper bound) is an independent comparator.
