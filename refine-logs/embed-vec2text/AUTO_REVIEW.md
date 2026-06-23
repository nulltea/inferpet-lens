# Auto-Review Loop — embed-vec2text (consolidation)

## Round 1 (2026-06-23)
### Assessment
- Score: 8/10 · Verdict: almost
- Key criticisms (all writeup-honesty, all addressable now): (1) HTML "validated/verified+validated" overstates the partial/in-sweep cross-model verdict; (2) localization stated too strongly (T4 is theory; empirical eigen-ablation deferred, rank-deficient Σ); (3) B8/B9/B10 tables mix provenance (different N/source) without saying so; (4) B7 "faithful iterative corrector / feedback moot" too categorical (it's a per-position proxy; bottleneck mechanism not ablation-proven); (5) probe lede "estimates how much" should be "ceiling + rank-predicts".
- Correctly-deferred follow-ups (not blockers): n≫d Σ, T4 eigen-ablation, anti-monotonicity controls, bootstrap CIs/denser ε, second corpus/model.
### Actions taken (round 1 → all 5 writeup fixes)
- §05 claims: "validated"→"supported, in-sweep"; "verified+validated"→"proof verified; empirical support partial/in-sweep".
- §07 B9 + §08: localization → "theory only*"; added rank-deficient-Σ deferral.
- Added provenance src-notes to B8/B9/B10 tables (source file + N; B10 split-N juxtaposition).
- B7 prose → "Vec2Text-style per-position proxy"; mechanism "not ablation-proven here".
- §04 lede → "computes a closed-form Gaussianized leakage ceiling and empirically rank-predicts".
### Status: STOP CONDITION met round 1 (8/10, almost); fixes implemented; round 2 confirms.

## Round 2 (2026-06-23)
- Score 8.5/10 · almost. Two residual fixes: B8 src-note bits inconsistency (raw I_G vs accessible cap), §04 card localization still un-qualified. Applied both.

## Round 3 (2026-06-23)
- Score 9.0/10 · VERDICT: ready. No remaining must-fix defects. Loop terminates positive.
- (Optional non-blocking polish noted: §01 surface table "feedback is moot" slightly categorical; §07 supplies the caveat.)

## Method Description
Surface: pooled GTR-T5-base sentence embedding under a Gaussian-DP release Y=clip(e₀,C)+N(0,σ²I). Attack: pretrained Vec2Text gtr-base iterative corrector (Morris 2023). Matched attack-independent probe: spectral channel-MI I_G(σ)=½Σlog₂(1+λᵢ/σ²) from the clean-embedding covariance spectrum + σ alone (geometry-only). Verified converse (Fano exact-match + rate-distortion per-token ceilings, T4 top-eigendirection localization) folded inline into the claim. Empirically I_G rank-predicts Vec2Text recovery across the ε-sweep (+1.0 token-F1/cos, +0.71 exact), tying CLUB at ~28× lower cost, ≫ capPVI. Negative companion: Vec2Text feedback is null on the per-position residual stream (no bottleneck) — establishing the single-vector pooled surface as Vec2Text's domain.
