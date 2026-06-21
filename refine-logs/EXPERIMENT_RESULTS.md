---
type: dev-log
status: current
created: 2026-06-20
updated: 2026-06-20
tags: [results, matched-probe, aloepri, vma, permutation-pi, pid, shredder, code-review]
companion: [EXPERIMENT_PLAN, FINAL_PROPOSAL]
---

# Initial Experiment Results — matched-probe program (B0 + B2 + B3)

**Date**: 2026-06-20 / 2026-06-21  **Plan**: `refine-logs/EXPERIMENT_PLAN.md`

## M0 / B0 — Implement + unit-test — PASSED (76/76 suite, +16 new)

New code (defences in `scripts/defenses/`, agnostic measures in `src/talens/`):
- **AloePri Algorithm 1** (`defenses/aloepri.py::keymat_gen`) — invertible key pair
  `P̂ Q̂ = I_d` via null-space construction. Validated at **d=2304** (gemma-2-2b
  width): float32-stored `P̂@Q̂` max error **6e-9** (float64 build + `solve`, per review).
- **AloePri obf-table generator** + activation covers (`AloePriPermCover`,
  `AloePriKeyMatCover`); **Shredder** static-Laplace cover + learned-noise trainer
  (`defenses/shredder.py`); **MMI-PID** QK/OV probe (`measures/pid.py`).
- 16 new oracle tests (`test_aloepri/test_shredder/test_pid`), full suite green.

### Cross-model code review (gpt-5.5 xhigh) — 2 CRITICAL caught + fixed pre-run
1. **Shredder SNR sign was inverted** — `min task_loss + λ·(noise/signal)` shrinks
   noise. Fixed to `min task_loss + λ·SNR` (signal/noise) so the term *rewards*
   noise; test now pre-trains + freezes the head (faithful Shredder).
2. **V-info-in-MMI is not a sound Shannon PID** — reader bounds don't preserve
   lattice identities. Reframed to *operational reader atoms*; report raw unclamped
   `I_V`, a `lattice_ok` guard, and the **conditional increments** `I_joint−I_other`
   (the sound "additional usable leakage" read). Plus keymat numerics hardened.

## M2 / B2 — Permutation-Π channel: AloePri α_e sweep + Π-probe bake-off — PASSED

GPU-free (weight surface), gemma-2-2b embedding, N=1200 token rows,
`results/aloepri_vma_sweep.json`. Permutation-core regime; VMA τ-recovery = truth.

| α_e | VMA τ-recovery | CLUB-on-φ (indep) | retrieval-PVI-on-φ (dep ref) |
|-----|----------------|-------------------|------------------------------|
| 0.0 | 1.000 | 252.4 b | 3.34 b |
| 0.2 | 0.561 | 250.4 b | 3.31 b |
| 0.35| 0.212 | 246.4 b | 3.25 b |
| 0.5 | 0.088 | 240.4 b | 3.17 b |
| 1.0 | 0.022 | 214.1 b | 2.75 b |
| 1.5 | 0.007 | 180.7 b | 2.27 b |
| **keymat, α_e=0** | **0.000** (chance ≈8e-4) | **−2.4 b** | **0.00 b** |

**Spearman(measure, τ-recovery) over α_e: CLUB(indep) = +0.976, retr-PVI(dep) = +1.000.**

- **C1 (Π channel, weight surface): PASSED** — CLUB-on-φ is independent (an MI
  estimator on paired signatures, not the matching attack) **and** faithful (ρ 0.976).
- **C4 (Π-probe selection) RESOLVED on the weight surface → CLUB-on-φ.** retrieval-PVI's
  +1.000 is mechanical (it *is* the VMA in bits, per its docstring); the capacity-reader
  candidate is degenerate here (1 row = 1 class) → deferred to the activation surface (B4).
- **Keymat finding (replicated at d=2304):** the dense Algorithm-1 key matrix drives
  VMA *and* both φ-measures to floor — it erases the sorted-quantile channel entirely.
  So the **permutation-core** is the VMA-vulnerable regime; the **full keymat** defends it
  (attacking it needs the raw-row / trained EmbedRow inverter, not RowSort). A cross-scheme
  cell for B4 surfacing early.

## Matched-probe diagonal so far

| Channel | Matched independent probe | Attack | ρ(probe, attack) | Source |
|---------|---------------------------|--------|------------------|--------|
| Token-identity | capacity-PVI reader accuracy | ridge TTRSR | 0.82–1.0 | prior thread (depth sweep) |
| **Permutation-Π** | **CLUB-on-φ** | **VMA τ-recovery** | **+0.976** | **B2 (new)** |
| Embedding-geometry | CLUB I(rep;emb) | ridge cosine | ~0.81–0.96 | prior thread |
| Attention QK/OV | MMI-PID unique/cond-increment | ISA | — | B5 (pending) |

## Summary
- **Must-run done:** B0 (impl+review+test), B2 (Π channel + C4 resolved). Main result: **POSITIVE**.
- **Pending (need a unified GPU activation run):** B1 same-pipeline anchor for token-id/embedding
  rows under the *new* defences; **B3 decoupling matrix off-diagonal** (cross-apply each probe to
  each target — the diagonal + the L20×DP sign-flip seed are in hand); B4 cross-scheme; B5 attention.
- **Ready for `/auto-review-loop`:** YES (review to prioritise B3 off-diagonal + the GPU run plan).

## Next Step
→ `/auto-review-loop` (paste these tables inline — Codex sandbox can't read repo files).

---

## M3 / B3 — Decoupling matrix (headline) — RUN (gemma-2-2b, ε×depth×3 seeds, 72 settings)

`results/b3_decoupling_matrix.json`, GPU (~13 min). K×K Spearman M[probe][attack] over the shared grid:

| probe ↓ \ attack → | token_id (TTRSR) | embedding (cosine) | perm_Π (VMA) |
|---|---|---|---|
| **token_id** (cap-PVI acc) | **0.642** | 0.556 | 0.252 |
| **embedding** (CLUB) | 0.782 | **0.750** | 0.599 |
| **perm_Π** (CLUB-φ) | 0.633 | 0.641 | **0.812** |

**Diagonal-dominance Δ_i (bootstrap 95% CI):** token_id +0.087 [+0.018,+0.178] ✓ · embedding −0.033 [−0.086,+0.010] ✗(tie) · perm_Π +0.162 [+0.037,+0.312] ✓ → **2/3 channels diagonal-dominant** (CIs exclude 0); embedding's CLUB is a *generic* MI upper bound (ties with the token attack), not channel-specific.

**The decoupling lives on the DEPTH axis (per-layer diagonal ρ):**
- token_id: L0 **+0.888** → L5 +0.527 → **L12 −0.108 (sign-flip)** → L20 +0.082
- embedding: L0 +0.975 → L5 +0.959 → L12 +0.919 → L20 +0.360 (stays positive)

→ The token-id probe↔attack relationship **inverts at mid-depth** under input-DP while embedding's does not — the L20×DP seed reproduced and localized, and the matched/mismatched **decoupling demonstrated as a depth-resolved effect**.

**Controls (the methodological finding):**
- **Monotone-noise-index → every attack: −0.728 / −0.752 / −0.990.** A single monotone knob "predicts" all attacks (common-cause decay) → it **inflates the pooled off-diagonals and deflates Δ_i**. So the pooled matrix UNDERSTATES channel-specificity; read decoupling on the depth axis (off the noise axis), exactly why the 2D ε×depth grid was needed.
- Random probe ≈ 0 (−0.08/−0.07/−0.14) ✓; shuffled pairing ≈ 0 (−0.12) ✓; retrieval-PVI (dependent ref) +0.885 vs token attack ✓.

**B3 verdict (honest):** C2 **partially supported** — (a) 2/3 matched diagonals dominate (CIs exclude 0); (b) ≥1 sign-flip (token-id @ L12); (c) sanity controls clean. BUT the shared monotone-noise axis is a demonstrated confound that compresses the pooled margin → the **depth axis carries the channel-specific signal**, and embedding's generic CLUB is not channel-specific. Sharpest framing: *channel-specificity is a depth-resolved phenomenon; pooled scalar correlations are confounded by common-cause noise decay.* Firming up needs a second defence family (the noise axis alone can't separate channels) — the B4 direction.

---

## B4 — Cross-scheme calibration: Shredder vs input-DP — RUN (2026-06-21)

`results/b4_cross_scheme.json`, GPU. Second defence family = **Shredder static-Laplace
injected directly at the captured layer** (post-capture Transform; clean acts captured
once, noise swept in-memory). 6 levels × 4 layers × 3 seeds.

**Shredder matrix M[probe][attack]** (embedding row finite-corrected, 52/72 cells; see instability note):

| probe ↓ \ attack → | token_id | embedding | perm_Π |
|---|---|---|---|
| **token_id** | **0.389** | 0.275 | 0.108 |
| **embedding** (CLUB) | 0.769 | 0.702 | 0.383 |
| **perm_Π** | 0.147 | 0.229 | **0.425** |

→ Same pattern as B3: token-id and perm_Π diagonals dominate their rows (2/3); embedding's
CLUB is generic (ties/loses to the token attack). Magnitudes lower than DP (direct-inject Laplace
is less monotone-structured than propagated DP).

**Finding 1 — the decoupling is DEFENCE-INJECTION-SPECIFIC.** Per-layer token-id diagonal ρ:
- input-DP (embedding-injected, **propagated**): L0 +0.89 → L5 +0.53 → L12 −0.11 → L20 +0.08
- Shredder (layer-injected, **direct**):        L0 +0.16 → L5 −0.16 → L12 −0.18 → L20 +0.62

Completely different depth shapes → **the channel-specific decoupling is not a universal property;
it depends on where the defence injects noise.** (The naive "Shredder = flat like DP@L0" prediction
was wrong — but the stronger "defence-specific" claim is supported: the two schemes share no depth profile.)
Embedding diagonal stays high under both (Shredder: +0.98→+0.81; DP: +0.98→+0.36) — robust, scheme-agnostic.

**Finding 2 — cross-scheme transfer (C3) is CHANNEL-DEPENDENT** (per-channel ρ_DP / ρ_Shredder / ρ_pooled):
- **embedding: 0.750 / 0.702 / 0.722** → pooled ≈ within ⇒ **one calibration curve fits both schemes** (generic CLUB transfers).
- **token_id: 0.642 / 0.389 / 0.453** → pooled < within ⇒ **does NOT transfer** (the specific reader↔TTRSR shape is scheme-specific).
- **perm_Π: 0.812 (B3 draw) / 0.425 (B4 draw) / 0.569** → high **seed variance** (different random draws); flagged — needs the B2+ multi-seed firm-up to stabilise.

**Instability (flagged, non-fatal):** seed-1 CLUB diverged to `nan` (20/72 cells, all seed-1). Finite-cell
(52) estimates are stable. Fix: clamp/retry in `club_mi_upper_bound` (variational net init for that seed).

**B4 verdict:** C3 **partially supported + sharpened** — channel-specific calibration holds across two
defence families (2/3 diagonals dominate under both), but (a) the decoupling's **depth profile is
defence-injection-specific** (propagated DP ≠ direct Shredder), and (b) **transfer is channel-specific**:
the *generic* embedding CLUB transfers across schemes, the *specific* token-id reader does not. Sharpest
framing for the paper: *a matched probe calibrates its channel, but the calibration CURVE is jointly a
function of (channel, defence-injection-geometry) — there is no single scheme-agnostic leakage scalar.*

---

## B2+ — Π-channel firm-up (auto-review fix #4) — PASSED (CPU, 2026-06-21)

`results/b2plus_pi_firmup.json`. 5 seeds × 12 α_e (dense in 0.2–0.7) × 2 model widths.

| model (d) | ρ(CLUB-on-φ, VMA) per-seed | min | match-indep ρ(CLUB, VMA-nn) | pooled (raw) |
|---|---|---|---|---|
| gemma-2-2b (2304) | **+1.000 ± 0.000** | +1.000 | +0.998 | +0.438 [0.20,0.62] |
| Qwen3-4b (2560) | **+1.000 ± 0.000** | +1.000 | +0.998 | +0.895 [0.79,0.95] |

- **Π channel firmed: per-seed ρ = 1.000 across 5 seeds × 2 widths** — the within-sweep calibration
  is perfect and width-robust. retrieval-PVI (dependent ref) also +1.000 (mechanical).
- **Independent of the attack's assignment algorithm:** CLUB-on-φ tracks VMA equally under Hungarian
  AND nearest-neighbour matching (ρ +0.998) — it is not a reparameterisation of the assignment.
- **Resolves B4's apparent Π "seed variance":** it was a **pooling artifact**, not channel instability.
  Each seed's CLUB-vs-α curve is monotone (ρ=1), but pooling *raw* CLUB magnitudes across heterogeneous
  token draws adds baseline offsets that deflate the *pooled* rank-correlation (gemma 0.44, qwen 0.90).
  **Methodological refinement: the calibration unit is the within-condition sweep, not pooled raw
  magnitudes** — which also reframes B3/B4 (pool ranks within a defence axis, not raw values across draws).

## Updated matched-probe diagonal (post B2+/B3/B4)

| Channel | Matched independent probe | Attack | calibration (within-sweep) | cross-scheme | 
|---------|---------------------------|--------|----------------------------|--------------|
| Token-identity | capacity-PVI reader acc | ridge TTRSR | ρ 0.82–1.0; **depth sign-flip** | does NOT transfer (scheme-specific) |
| **Permutation-Π** | **CLUB-on-φ** | VMA τ-recovery | **ρ +1.000 ± 0.000** (5 seeds × 2 widths) | keymat closes channel |
| Embedding-geometry | CLUB I(rep;emb) | ridge cosine | ρ 0.70–0.98 (depth-robust) | **transfers** (generic) |
| Attention QK/OV | MMI-PID cond-increment | ISA | — (B5 pending) | — |

**CLUB nan bug FIXED** (`measures/club.py`): grad-clipping + non-finite skip + None-guard (never propagates
nan). Regression test `test_club_stability.py`; suite **78/78**.
