# Auto Review — resid-gelo (Task 5: GELO row-mixing defense)

## Round 1 — Score 8/10, verdict Almost
Reviewer (gpt-5.5 xhigh, thread 019ef85c) read the report page, claim+proof, results, audit, code.
No fraud, no page-vs-analysis.json number mismatch, register clean. Six presentation-precision fixes:
1. masthead/intro overbroad on exact Gram leak (shield-0 scope); 2. Findings C1 over-phrased
("defeats per-row separation" + margin<0.05 without scope); 3. page theory collapsed H̃→H (drop shield
caveat in §2 table + L2/L3); 4. C1 table: margin is median-of-per-layer-margins, not diff-of-displayed-
medians; 5. C2 "matched to the wrong channel" stated as settled vs hypothesis; 6. "cross-model
verification" process-flavored.

### Actions taken
Applied all six fixes: scoped the Gram leak to the exposed operand H̃ (exact HᵀH shield-0 / subtractable);
re-scoped C1 to tested single-seed JADE, shield-0 median, 17/45 cells>0.05; introduced H̃ notation across
§2 table and L1–L3; added the median-of-margins table note; softened C2 to "consistent with mismatched
probe OR too-weak attack"; removed process phrasing.

## Round 2 — Score 9/10, verdict Ready
The six concerns addressed; scope-honest, numbers aligned, reads as a rigorous internal research artifact.
Two trivial wording nits named and applied (Discussion "residual feature Gram exactly" → exposed-operand
with shield caveat; L3 "plaintext row space" → "exposed-operand row space").

STOP: score 9 ≥ 6 AND verdict ready. Gate met.

## Method Description
GELO is applied as a representation transform to Qwen3-4B resid_post (layers 0,12,20; cached capture),
exposing U=A·H̃ with A a fresh secret per-prompt n×n row-mixing (orthogonal at κ=1), optionally with
appended Gaussian shield rows. The sweep crosses κ(A)∈{1,3,10,30,100} × shield-frac∈{0,0.5,1.0}. Attack:
whiten U then joint-diagonalize (JADE), graded by p95 Hungarian |cosine| to plaintext rows vs a matched
random-orthogonal-demixing floor (genuine margin = recovery − floor). Anchor: amortized ridge U→H,
held-out. Probe: geometry-only whitened-row negentropy (bits), joint-diag deletable. Finding: orthogonal A
leaks the feature Gram exactly (C0); fresh row-mix holds per-row BSS near floor and defeats ridge (C1);
the row-negentropy probe does not track the BSS margin to the 0.6 bar (C2, partial, follow-up queued).
