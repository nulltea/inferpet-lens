# Auto Review — perm-cover (consolidation)

## Round 1 (2026-06-23)

### Assessment
- Score: 7/10
- Verdict: almost
- Reviewer: Codex gpt-5.5 xhigh, thread 019ef65e-3884-7d51-ab8b-0c304ccb2dff (read artifacts directly)

### Key criticisms (all wording/style/honesty; no science FAIL)
1. Headline called full sorted row "the maximal invariant" without the per-row caveat.
2. CLUB-on-φ described as "upper bound on the information any matcher can use" (overscope).
3. "predicts the attack in both directions" stronger than evidence.
4. Title em-dashes + visible project jargon ("campaign thesis", "firm-up", inline claim id).
5. Findings not in epistemic-status form (Supported / Preliminary).
6. keymat row showed "≈0" instead of the -2.4 estimator floor; "-4% at α_e 0.2" should be ~2%.
7. EXPERIMENT_AUDIT Claim Impact stale ("needs proof") vs PROOF_AUDIT PASS.

Numbers verified by reviewer to match JSONs (+0.434, +0.600 uplift; Spearman +0.976).

### Actions taken (all 7 applied before finalizing)
- Headline + intro + analysis: "per-row maximal invariant" + shared-permutation relaxation note.
- Leakage Measures: scoped CLUB to the paired quantile-signature channel; DPI supplies dominance.
- Findings: relabeled "tracks recovery + key-matrix control"; added Supported / Preliminary labels.
- Removed title em-dashes and "campaign thesis"/"firm-up"/inline claim id from visible prose.
- Results table: keymat shows "-2.4 (estimator floor)"; claim % corrected to ~2% / ~4%.
- EXPERIMENT_AUDIT Claim Impact: supersession note, theory marked verified.

### Status
- Gate met at round 1 (score 7 >= 6 AND verdict "almost"). Fixes applied strengthen the artifact.
- Loop complete.

## Method Description
Consolidation of the permutation-cover surface. The AloePri cover hides a token-embedding table by a
shared column (feature) permutation plus per-row Gaussian noise and a secret row permutation; the
adversary recovers the row correspondence by matching column-permutation-invariant per-row signatures
with Hungarian assignment. The measurement loop runs, per noise level, the matching attack (graded
against the known permutation) and an independent variational mutual-information probe on the paired
signatures, then correlates probe bits against recovery across the sweep. The full sorted row (the
per-row maximal invariant) dominates the 64-bin quantile baseline both empirically (+0.43 to +0.60)
and by the data-processing inequality; a key-matrix cover collapses recovery and the independent probe
together.
