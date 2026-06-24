---
type: claim
node_id: claim:gelo-orthogonal-gram-leak-rowmix-defeats-bss
name: "GELO: orthogonal-A feature-Gram leak; fresh row-mix defeats per-row BSS + amortized inversion"
description: ""
node_type: claim
status: verified
provenance: "refine-logs/resid-gelo/proof/PROOF_AUDIT.json"
tags: [residual, gelo, bss, feature-gram, row-mixing, channel-decoupling, matched-floor, weights-pub]
date: 2026-06-24
added: 2026-06-24T06:34:44Z
companion: refine-logs/resid-gelo/RESULTS.md
---

# GELO: at its recommended orthogonal setting the feature Gram leaks exactly, while the fresh secret row-mix defeats per-row BSS and amortized inversion

**status:** `verified` (theory L1–L4 proof-checked PASS, 2 rounds; empirics integrity-audited WARN/no-fraud; measurement-loop verdict partial)

## Claim

GELO (arXiv 2603.05035, github.com/noskill/gelo) is a confidential-inference defense that, for remote
GPU compute, exposes only

    U = A · H̃ ,

where `H̃` is the residual operand fed to the QKVO projections — either `H ∈ R^{n×d}` (the `n` token
rows, feature dim `d`) or the row-augmented `H̃ = [H; S]` with `m` appended Gaussian **shield rows**
`S` — and `A` is a **fresh secret per-prompt** row-mixing matrix (`A^{-1}=Aᵀ` when orthogonal, the
recommended setting). Correctness holds because the left row-mix commutes with the right projection:
`U W = A H̃ W`, un-mixed inside the trusted enclave by `A^{-1}`. Against a `WEIGHTS-PUB` adversary
that observes only `U` and runs a key-free blind-source-separation (BSS) / left-demixing attack
(JADE / joint-diagonalization), graded by the p95 Hungarian-matched `|cosine|` of recovered sources
against the plaintext rows and referenced to a **matched random-orthogonal-demixing floor**:

1. **(C0, headline) Orthogonal A leaks the feature Gram exactly.** When `A` is orthogonal the
   feature (column) Gram of the exposed operand is exactly invariant: `Uᵀ U = H̃ᵀ Aᵀ A H̃ = H̃ᵀ H̃`.
   With no shields this is exactly the secret `Hᵀ H`; with shields it is the augmented Gram
   `Hᵀ H + Sᵀ S`, from which `Hᵀ H` is recoverable iff `Sᵀ S` is known/subtractable. So at its
   recommended orthogonal setting GELO publishes an exact `d×d` functional of the secret with no
   attack — an attack-independent structural leak. Empirically the feature-Gram rel-err
   `‖Uᵀ U − Hᵀ H‖_F/‖Hᵀ H‖_F` is `2.5e-16` at κ(A)=1 and grows monotonically to `1780` at κ=100
   (shield-0, median over layers {0,12,20}).
2. **(C1) The fresh secret row-mix defeats per-row BSS and amortized linear inversion.** The tested
   JADE/joint-diag per-row recovery sits only slightly above the matched random-demixing floor
   (shield-0 genuine margin `0.047 → 0.026` as κ:1→100, a net decrease; JADE p95 ≈ 0.69–0.72 vs
   floor ≈ 0.66–0.67), and an amortized ridge inverter `U→H` fits held-out p95 `0.288`, **below**
   the floor `0.667` — fresh-per-prompt A admits no fixed linear inverse (L4). So per-row recovery
   is near-floor and amortized inversion fails; the load-bearing observable for the attacks tried is
   the feature Gram (C0), not the token rows.
3. **(C2, measurement-loop, partial / NOT supported as stated) the geometry-only negentropy probe
   does not track the BSS margin to the preregistered bar.** Spearman(probe bits, genuine margin)
   = `0.507` (shield-0, n=15) / `0.293` (all 45 cells), below the `|ρ|≥0.6` success threshold (the
   direction is right: bits fall `66.6→20.8` with κ). Diagnosed as a failed/weak measurement loop;
   whether this is a non-matched probe (row-negentropy is matched to the row-BSS channel, not the
   feature-Gram channel that leaks at κ=1) or a too-weak BSS attack is **not** settled by the data —
   a follow-up tests both arms.

Consequence: GELO mirrors the KV-CLOAK channel-decoupling verdict
([[kv-cloak-channel-decoupling-feature-mix-loadbearing]]) on the residual surface — the secret
row-mix is near-cover for per-row BSS and kills amortized inversion, but a *static feature-Gram*
functional of the secret is exposed outright at the recommended orthogonal setting. Privacy is not
delivered by the mixing's apparent complexity; it would require breaking the feature-Gram channel.

## Theory (lemmas L1–L4, proof-checked PASS — full proof below)

Let `H̃` be the exposed operand (`H` or `[H;S]`), `G := H̃ H̃ᵀ` the row Gram, targets the
unit-normalized plaintext rows.

- **L1 (feature-Gram leak).** `A` orthogonal ⟹ `Uᵀ U = H̃ᵀ H̃` exactly; `= Hᵀ H` iff `S=0`, else the
  augmented Gram `Hᵀ H + Sᵀ S` (and `Hᵀ H` recoverable iff `Sᵀ S` known/subtractable).
- **L2 (row-Gram conjugation, not invariance).** `A` orthogonal ⟹ `U Uᵀ = A G Aᵀ` is an orthogonal
  similarity of `G`: spectrum and Frobenius norm exactly preserved, entries not (equality iff `A`
  commutes with `G`). The row Gram is hidden up to conjugation; the feature Gram is exposed.
- **L3 (rowspace preservation).** `A` invertible and `H̃≠0` ⟹ `rowspace(U)=rowspace(H̃)`, so every
  key-free left-demixing output lies in `rowspace(H̃)` (producible-direction set and oracle ceiling
  are `A`-independent). The *chance-floor* invariance holds for orthogonal `A` only; empirically the
  floor is recomputed on each observed `U`, so the within-cell genuine margin is well-posed at every
  κ without claiming cross-κ floor invariance.
- **L4 (fresh-per-prompt A defeats amortized linear inversion).** `A_t` i.i.d. with `E[A_t]=0`
  (Haar `O(n)`, or `Q₁ diag(s) Q₂` with `Q₁` Haar — both used in the sweep), independent of `H_t`,
  finite second moments ⟹ `W=0` is a global minimizer of the population least-squares/ridge objective
  over fixed no-intercept linear maps, and no such map improves population squared loss over the zero
  predictor. (Linear intercept-free class only; per-prompt BSS and nonlinear attacks out of scope.)

### Proof

#### L1 — feature-Gram leak.
By definition `U = A H̃`, so `Uᵀ U = (A H̃)ᵀ(A H̃) = H̃ᵀ Aᵀ A H̃`. If `A` is orthogonal, `Aᵀ A = I`,
hence `Uᵀ U = H̃ᵀ H̃` exactly (associativity, no approximation). With no shields (`H̃=H`),
`Uᵀ U = Hᵀ H`: the adversary computes `Uᵀ U` from the observation and obtains the exact `d×d` secret
feature Gram with no attack — an attack-independent functional of the secret. With shields
`H̃=[H;S]`, block multiplication gives `H̃ᵀ H̃ = Hᵀ H + Sᵀ S`, so `Uᵀ U = Hᵀ H + Sᵀ S`. The *identity*
`Uᵀ U = Hᵀ H` holds iff `Sᵀ S = 0`, i.e. `S=0` (real rows); when `S≠0` the exposed object is the
augmented Gram, from which `Hᵀ H` is exactly recoverable iff `Sᵀ S` is known/subtractable. ∎ (L1)

#### L2 — row-Gram conjugation, not invariance.
`U Uᵀ = (A H̃)(A H̃)ᵀ = A (H̃ H̃ᵀ) Aᵀ = A G Aᵀ`. For orthogonal `A`, `A^{-1}=Aᵀ`, so `U Uᵀ = A G A^{-1}`
is a similarity transform of `G`. Similar matrices share characteristic polynomials, so
`spec(U Uᵀ)=spec(G)`; in particular the eigenvalue multiset, the trace, and (both being symmetric
with equal eigenvalues) `‖U Uᵀ‖_F² = Σ λᵢ² = ‖G‖_F²` are preserved. Non-invariance: `U Uᵀ = G` for
all such `A` would require `A G = G A`. Counterexample: `G = diag(2,1)`, `A = [[0,1],[1,0]]`
(orthogonal); `A G Aᵀ = diag(1,2) ≠ G`, while spectrum `{1,2}` and `‖·‖_F = √5` are preserved. So
the row-Gram is hidden up to an orthogonal conjugation (spectrum exposed, entries not). ∎ (L2)

#### L3 — rowspace preservation ⟹ recoverable set / matched floor.
`rowspace(A H̃) = {xᵀ A H̃ : x∈R^{n+m}} = {(Aᵀ x)ᵀ H̃ : x∈R^{n+m}}`. As `x` ranges over `R^{n+m}` and
`A` is invertible, `Aᵀ x` ranges over all of `R^{n+m}` (bijection), so the set equals
`{yᵀ H̃ : y∈R^{n+m}} = rowspace(H̃)`; hence `rowspace(U)=rowspace(H̃)` (assuming `H̃≠0`, so
normalized alignments to nonzero targets are well-defined), and with `m=0`, `=rowspace(H)`. Every
key-free left-demixing output `BU` has rows in `rowspace(U)=rowspace(H̃)`, so the producible-direction
set and the per-target oracle ceiling depend only on `rowspace(H̃)` and the fixed targets — not on
which invertible `A` was used. **Floor-invariance requires orthogonality**: the value a random
demixing attains depends on the row-coordinate metric of `U`, which a non-orthogonal `A` changes
(e.g. `H=I₂`, `A=diag(k,1)`: a random unit row gives `(k cosθ, sinθ)`, whose alignment to `e₁`
depends on `k`); for orthogonal `A`, whitening removes the metric and the rowspace-uniform floor is
invariant. Empirically the matched floor is recomputed on each observed `U` (same whitening, random
rotation in place of joint-diag), so the within-cell genuine margin is well-posed at every κ even
for non-orthogonal `A`, without claiming cross-κ floor invariance. ∎ (L3)

#### L4 — fresh-per-prompt A defeats amortized linear inversion.
First, `E[A_t]=0`. Let `R = diag(-1,1,…,1) ∈ O(n)` (a reflection). Haar measure on `O(n)` is
left-invariant under `R`, so `A_t` and `R A_t` are identically distributed, giving
`E[A_t] = E[R A_t] = R E[A_t]`; the first row equals its own negation, hence is 0, and repeating
per coordinate gives `E[A_t]=0`. For the sweep's `A_t = Q₁ diag(s) Q₂` with `Q₁` Haar-orthogonal
independent of `(diag(s),Q₂)`, `E[A_t] = E[Q₁] E[diag(s) Q₂] = 0` since `E[Q₁]=0`. So `E[A_t]=0`
covers both κ=1 (orthogonal) and κ>1 (conditioned) mixings.

Consider the population ridge objective over fixed `W∈R^{d×d}`, `λ≥0`:
`J(W) = E‖U_t W − H_t‖_F² + λ‖W‖_F²`, with `U_t = A_t H_t`. Expand:
`E‖U_t W − H_t‖_F² = E⟨U_t W, U_t W⟩ − 2 E⟨U_t W, H_t⟩ + E‖H_t‖_F²`. The cross term is
`E⟨U_t W,H_t⟩ = tr(Wᵀ E[U_tᵀ H_t])`, and `U_tᵀ H_t = H_tᵀ A_tᵀ H_t`; by independence and the tower
rule, `E[H_tᵀ A_tᵀ H_t] = E[H_tᵀ (E[A_t])ᵀ H_t] = 0`. So the cross term vanishes for **every** `W`:
`J(W) = tr(Wᵀ(M+λI)W) + E‖H_t‖_F²` with `M := E[U_tᵀ U_t] = E[H_tᵀ A_tᵀ A_t H_t] ⪰ 0` (no
orthogonality assumed; when `A_t` orthogonal a.s., `M = E[Hᵀ H]`). Thus `J(W) ≥ E‖H_t‖_F² = J(0)`
for all `W`, so `W*=0` is a global minimizer — unique iff `M+λI ≻ 0`; when singular, every minimizer
`W'` satisfies `tr(W'ᵀ(M+λI)W')=0`, which at `λ=0` forces `U_t W' = 0` a.s. (an equally
uninformative predictor). At the optimum the prediction is `0` a.s. with residual `E‖H_t‖_F²`.
Therefore no fixed no-intercept linear map improves population squared loss over the zero predictor:
the amortized linear inverter carries no prompt-specific information. (Linear intercept-free class
only; per-prompt BSS, intercept, and nonlinear attacks are out of scope.) ∎ (L4)

Therefore L1–L4 hold under their stated assumptions. ∎

## Scope and caveats

- **C0 exactness is shield-0** (or known/subtractable `Sᵀ S`); with unknown shields the leak is of
  the augmented Gram `Hᵀ H + Sᵀ S`.
- **C1 is for the tested JADE/joint-diag attack at a single seed**; the margin's net decrease with κ
  is not strictly monotone (L20 κ=10 shield-0 margin 0.069), and 17/45 sweep cells exceed margin
  0.05 (mostly shielded). "Load-bearing channel is the feature Gram" is asserted for the attacks tried.
- **C2 is a partial / non-supported measurement-loop result** (ρ below 0.6); the feature-Gram-mismatch
  vs weak-attack diagnosis is a hypothesis — see follow-up. L4 covers only linear amortized inversion.
- Threat model `WEIGHTS-PUB`; CPU on cached `capture-28a0ee6c41330ee3.pt` (Qwen3-4B `resid_post`).

## Evidence

- Theory: `refine-logs/resid-gelo/proof/PROOF_PACKAGE.md` (full), audit `PROOF_AUDIT.json` (PASS,
  2 rounds, thread 019ef84c). Lemmas verified incl. L2 counterexample and L4 cross-term/`E[A]=0`.
- Empirics: `refine-logs/resid-gelo/{sanity,sweep,analysis}.json`; integrity audit
  `EXPERIMENT_AUDIT.md` (WARN, no fraud: real GT, matched floor, probe≠attack confirmed, held-out
  ridge). result-to-claim verdict: C0 yes/high, C1 partial→yes, C2 no, overall partial.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

- [[kv-cloak-channel-decoupling-feature-mix-loadbearing]] — same channel-decoupling verdict on the
  KV surface (secret feature mix load-bearing; row/permutation cover-invariant).
- [[kv-bss-subspace-floor-and-negentropy-probe]] — the matched-floor + negentropy-probe method GELO reuses.
