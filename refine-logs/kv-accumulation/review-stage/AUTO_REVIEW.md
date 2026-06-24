# Auto review — kv-accumulation (KV/QKV source separation)

Reviewer: Codex gpt-5.5 (xhigh), thread 019ef6f2-9b2a-7e21-8e36-5009a392cd85.

## Round 1 — Score 8/10, verdict "almost"
- Probe independence: PASS (negentropy imports only _operands/_subsample/_whiten; never joint_diag/jade/jd).
- Claim scope: PASS (proved L1 separated from empirical magnitude / C1 / exploratory C2).
- Number consistency: one integrity bug — "≈96%" wrongly attached to the 0.553 floor-mismatch gap (that is ≈71% of raw, not 96%).
- Negative result: honest, but defense extrapolation ("BSS meaningful only under defense") overreached the proof.
- Fixes: correct the percentage; reword "the quantity the separator maximizes"; make 0.155 traceable; weaken defense extrapolation; optional metadata split.

## Round 2 — Score 9/10, verdict "ready"
- Percentage: FIXED (margin 0.027 ≈ 3.4% of raw; ≈96% artifact; 0.553 reported separately as ≈71%).
- Probe language: FIXED ("fourth-moment separability surrogate"; explicit no-joint-diag).
- 0.155 traceability: FIXED (cited to bss.jd_floor / pilot_dev24.json; reviewer verified 0.154–0.159 range).
- Defense extrapolation: FIXED (conditional wording).
- Remaining: tiny polish only (HTML could name pilot_dev24.json inline; provenance already in claim/RESULTS). C2 remains scoped exploratory.

## STOP — positive assessment (score 9 >= 6 AND verdict "ready").

## Method Description
Plaintext KV/QKV (Qwen3-4B, kinds kq/kqv_out/resid_post at L0/12/20) under WEIGHTS-PUB, Identity
transform. Attack: BSS family (gram_error, JADE single-observation ICA, JD across stacked observations)
graded by Hungarian-aligned p95 cosine vs true rows. Matched probe: whitened-row negentropy (bits),
attack-independent. Key methodological move: a matched random-orthogonal-demixing floor replaces the
shipped unrelated-Gaussian chance floor; genuine separation = raw − matched floor. Result: BSS attack
framing is ill-posed on plaintext (identity mixing; metric subspace-membership-confounded, proved as
Lemma L1); genuine margin ≈0.027, flat in observation count; negentropy predicts the genuine margin
(ρ=0.92, exploratory) and anticorrelates with the raw score. Plaintext baseline for the mixing-defense
sweeps (KV-CLOAK, GELO).
