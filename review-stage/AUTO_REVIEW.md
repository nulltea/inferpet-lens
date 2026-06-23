# Auto Review — defenses-existing (2026-06-24)

## Round 1 — Score: 8/10 — Verdict: ready
Reviewer: Codex gpt-5.5 xhigh (artifacts pasted inline; sandbox unavailable). Cross-checked all reported
numbers against an independent raw-JSON digest: no mismatch. probe!=attack integrity PASS, scope honest,
metric convention (bits + readout) satisfied, inline proof aligns with the verified package.

Minimum fixes applied (presentation-level, no evidence change):
1. token-recovery table: note all four measured depths + a seed/CI caveat.
2. cross-family calibration: "preliminary" qualifier carried into the introduction.
3. keymat single-config wording (already present, confirmed).
4. removed process residue from HTML prose ("re-rendered through shared reporting layer", "queued strengthening").
5. uncertainty note near the table.

STOP condition met (score >= 6 AND verdict in {ready, almost}). Loop complete.

## Method Description
Two implemented defenses (additive Laplace at a split activation; static invertible embedding-table
obfuscation) were swept across their privacy parameter on a single 2.6B model. Per sweep point, two
attacks (permutation matching, ridge token inversion) and two attack-independent probes (variational MI
upper bound, capacity reader) were run; bits and recovery were correlated within each defense and pooled
across the two families. Finding: additive noise is channel-selective (kills permutation, spares token-id)
and the bits-to-recovery calibration is mechanism-dependent. A Gaussian fixed-codebook critical-scale
separation proposition explains the channel selectivity (proved this phase).
