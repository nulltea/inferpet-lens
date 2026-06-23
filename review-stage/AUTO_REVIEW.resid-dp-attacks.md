# Auto-review log — resid-dp-attacks (consolidation)

Reviewer: Codex gpt-5.5 xhigh (medium difficulty), thread 019ef525-4568-7421-a56e-6dc0409286c9.
Deliverable: standardized results + 2 claims (PARTIAL) + negative-result log + integrity audit (WARN) + HTML.

## Round 1 — Score 8/10 — almost
Critical: W1 (high) R1 used too loosely as "re-correlation" — it is uplift/ceiling-realization (Bayes
recovery saturated → its own Spearman-vs-probe is degenerate; ridge still +0.76..+0.88 at L0). W2 (med)
HTML headline "only where noise propagates through depth" overstates (L0 also in input-DP). W3 (med)
single-seed caveat not beside the +0.83. W4 (low) R2 per-row ρ=1.00 is a sweep metric. No numeric
mismatch; PARTIAL honest; probe≠attack defensible.

## Round 2 — Score 8.5/10 — almost
W3, W4 resolved. Residual stale wording: synthesis still said "R1 … ridge decorrelates"; claim frontmatter
still said re-correlate "L0 + propagated-DP depth"; §06 still "Supported at L0 … and at propagated depth";
§08 still "decorrelation … property of noise propagation through depth". All copy-consistency, not integrity.

## Round 3 — Score 9/10 — READY
All residuals fixed: R1 scoped to uplift/ceiling-realization throughout; re-correlation reserved to R5;
headline/§06/§08 narrowed to "at depth specifically requires propagation; absent at-layer". No
publication-blocking weaknesses for a scoped internal/site research report; PARTIAL / WARN / single-seed /
synthetic_proxy caveats remain visible. **STOP (positive).**

## Method Description
Consolidation of six prior residual-stream-under-DP attack runs (secret = token id, gemma-2-2b, WEIGHTS-PUB)
onto the bits-canonical + per-secret-readout metric. Attacks: L0 exact Bayes-NN, L>0 channel-aware MLP,
deep/iterative decoders, forward-model Vec2Text. Probes (attack-independent): CLUB MI upper bound,
capacity-PVI, MDL/SDL. Finding: the MI↔recovery decorrelation is propagation-specific — absent under
at-layer noise (ρ=1.0), an uplift gap at L0 (Bayes-NN +0.98), and a true anti-correlation at propagated
depth (ridge −0.09) that a stronger decoder re-correlates (+0.83). Open: no single attack dominates the
full noise range (forward-model best low-noise, decoder best high-noise) → noise-aware FMV is the frontier.
