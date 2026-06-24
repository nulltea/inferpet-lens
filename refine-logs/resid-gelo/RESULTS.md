# Results — resid-gelo (Task 5, campaign-B-expand)

**Surface:** Qwen3-4B `resid_post` rows (the QKVO-feeding residual), cached capture
`capture-28a0ee6c41330ee3.pt` (512 prompts, layers {0..32}, d=2560). CPU-only; numpy GELO
transform + BSS attacks (`talens.attacks.bss`) + geometry-only negentropy probe
(`talens.measures.bss_separability`). No GPU capture.

**Defense.** GELO (arXiv 2603.05035, github.com/noskill/gelo): exposes `U = A·H`, `A ∈ R^{n×n}`
a **fresh secret per-prompt** row-mix of the `n` token rows (`A^{-1}=A^T` when orthogonal),
optionally padded with appended Gaussian **shield rows** energy-matched to the median real-row
norm. Correctness: left row-mix commutes with right projection, `U W = A H W`, un-mixed in the TEE
by `A^{-1}`. Privacy sweep: condition number `κ(A) ∈ {1,3,10,30,100}` (κ=1 ⇒ orthogonal A),
shield-fraction `∈ {0, 0.5, 1.0}`. Layers {0,12,20}, 48 prompts, max_dim 32, max_features 256, seed 0.

## C0 — orthogonal-A Gram-invariance leak (theory + empirical): CONFIRMED, headline
For orthogonal `A`, the column (feature) Gram of the **exposed operand** is exactly invariant:
`UᵀU = HᵀAᵀAH = HᵀH`. The row Gram `UUᵀ = A G_H Aᵀ` is only orthogonally *conjugated* (Frobenius
norm preserved, entries not). **Scope (jury correction):** with shield rows appended, the exposed
operand is `[H;S]`, so orthogonal A preserves the *augmented* Gram `[H;S]ᵀ[H;S] = HᵀH + SᵀS`, not
exactly `HᵀH`; the exact real-token-Gram leak holds at **shield-fraction 0** (or when shields are
known / subtractable). The empirical exact-invariance numbers below are shield-0.

Empirical feature-Gram rel-err `‖UᵀU−HᵀH‖_F/‖HᵀH‖_F` (shield 0, median over layers):

| κ(A) | 1 (orth) | 3 | 10 | 30 | 100 |
|---|---|---|---|---|---|
| feat-Gram rel-err | 2.5e-16 | 2.80 | 25.3 | 189 | 1780 |

Exact at κ=1, monotone↑ with κ. Sanity (B1) also verifies: cond(A)=1 at κ=1, row-Gram entries
change while Frobenius norm preserved, defender un-mix exact (`‖A⁻¹U−H‖/‖H‖ < 5e-15`) at all κ.
**A defense advertised as "secret mixing" publishes an exact d×d functional of the secret at its
recommended orthogonal setting — an attack-independent structural leak.**

## C1 — BSS recovery vs κ × shield + ridge anchor: CONFIRMED
Per-cell p95 Hungarian-matched |cosine| of JADE-recovered sources against plaintext rows, graded
against a **matched random-orthogonal-demixing floor** (same whitening, random rotation). Shield-0,
median over layers:

| κ(A) | 1 | 3 | 10 | 30 | 100 |
|---|---|---|---|---|---|
| JADE p95 | 0.721 | 0.705 | 0.705 | 0.701 | 0.692 |
| random-demix floor p95 | 0.674 | 0.668 | 0.663 | 0.660 | 0.666 |
| **genuine margin** | 0.047 | 0.037 | 0.022 | 0.025 | 0.026 |

The shield-0 median genuine BSS margin is small (≤ 0.05) and shows a **net decrease** as κ grows
(0.047→0.026; not strictly monotone — L20 κ=10 shield-0 = 0.069). Across the full sweep the margin
is not uniformly tiny: 17/45 cells exceed 0.05 (max 0.098, mostly shielded cells). The reading is
that the secret row-mix makes per-row ICA recovery only marginally above the subspace-membership
floor for the *tested* JADE/joint-diag attack, with no clear improvement as A becomes more
ill-conditioned. **Ridge anchor** (amortized linear `U→H` fit on train prompts, tested held-out):
median p95 = 0.288, **below** the random floor 0.667 → amortized inversion *fails* under
fresh-per-prompt A (the only thing that could survive fresh A is per-prompt BSS, and it barely does).

## C2 — matched probe tracks recovery: NON-CORRELATION (below 0.6 bar)
Geometry-only negentropy probe (bits, whitened-row negentropy — computable with the joint-diag
attack deleted). Spearman(probe bits, genuine margin):
- all 45 cells: ρ = 0.293 (p = 0.051)
- shield-0 only (cleanest κ axis, n=15): ρ = 0.507 (p = 0.054)

Both below the |ρ| ≥ 0.6 success bar. **Direction is right** (negentropy bits fall 66.6→48.1→30.5
→21.7→20.8 with κ, co-moving with the shrinking margin), but the correlation is weak/marginal.

### Why (hypothesis, NOT established): non-matched probe vs weak attack — both open
Working hypothesis: the load-bearing leak under GELO is the **feature Gram** (C0), not the row-BSS
channel — the tested JADE (joint-diag) margin is near-floor, so there is little row-recovery signal for the
row-negentropy probe to track, and that probe is matched to the *row-separation* (ICA) channel, not
to the *feature-Gram-recoverability* channel that actually leaks at κ=1. This mirrors the kv-cloak
channel-decoupling verdict (claim:kv-cloak-channel-decoupling-feature-mix-loadbearing).

**Jury caveat (do not overclaim):** the data cannot yet distinguish "non-matched probe" from
"weak attack". A stronger BSS attack (FastICA / FOBI / SOBI / multi-restart, larger max_dim,
multiple seeds) could raise the margin and re-correlate. The honest verdict is a *failed/weak
measurement loop* for this attack+probe pair; the feature-Gram-mismatch diagnosis is a hypothesis.
→ follow-up tests BOTH arms: stronger BSS suite AND a feature-Gram-matched probe + shielded-Gram
recovery analysis (spawn-depth 1).

## Verdict (result-to-claim jury, gpt-5 xhigh, 2026-06-24: overall partial)
- **Keeper claim (C0 yes/high + C1 partial→yes/medium-high), scoped:** orthogonal GELO exposes the
  feature Gram of the exposed operand exactly — at shield-0 this is exactly `HᵀH`, with shields the
  augmented Gram `[H;S]ᵀ[H;S]`. In this sweep the *tested* JADE (joint-diag) per-row recovery is only slightly
  above a matched random-demixing floor (net-decreasing with κ), while fresh-per-prompt mixing
  defeats amortized ridge inversion (p95 0.288 < floor 0.667). The demonstrated keeper result is a
  **structural feature-Gram leak + weak tested row-recovery channels** — the feature Gram, not the
  token rows, is the load-bearing observable for the attacks tried.
- **Measurement-loop C2 (no, as stated):** the row-negentropy probe does not meet the |ρ|≥0.6
  success bar (0.507 shield0 / 0.293 all). Reported as a failed/weak measurement loop. The
  feature-Gram-mismatch diagnosis is a hypothesis; follow-up (spawn-depth 1) tests stronger BSS +
  a feature-Gram-matched probe.

Artifacts: `sanity.json` (B1 identities), `sweep.json` (B2/B3 raw), `analysis.json` (C0/C1/C2).
