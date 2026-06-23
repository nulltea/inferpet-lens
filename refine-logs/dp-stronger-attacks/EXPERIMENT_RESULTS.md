---
type: dev-log
status: current
created: 2026-06-20
updated: 2026-06-22
tags: [results, matched-probe, aloepri, vma, permutation-pi, pid, shredder, code-review, vec2text]
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

---

## B2-L0 — Exact-Bayes attack vs ridge under input-DP (L0 embedding) — RUN (2026-06-21)

`results/l0_fast.txt` (attack) + capacity-PVI/CLUB from the same sweep. GPU-free, gemma-2-2b,
N=7000 (Zipf token-id sample, real embeddings), vocab-disjoint, pool=2048. **Proof-gated by T1.**
(Bug fixed: pool truncation had dropped large-valued true ids → clean recovery 0.616; now 1.000.)

| ε (r=σ√d/C) | ridge TTRSR | **Bayes-NN TTRSR** | uplift | capacity-PVI acc | CLUB (b) |
|---|---|---|---|---|---|
| ∞ (0.00) | 1.000 | 1.000 | +0.000 | 0.981 | 3084 |
| 512 (0.45) | 0.993 | 1.000 | +0.007 | 0.977 | 2942 |
| 256 (0.91) | 0.202 | 1.000 | **+0.798** | 0.935 | 2624 |
| 128 (1.82) | 0.020 | 1.000 | **+0.980** | 0.736 | 1912 |
| 96 (2.42) | 0.008 | 1.000 | +0.992 | — | — |
| 64 (3.63) | 0.002 | 0.993 | +0.992 | — | — |

**Findings.**
- **C1 (uplift) — CONFIRMED, large.** The channel-aware Bayes-NN attack recovers ~1.0 up to r≈2.4 while ridge collapses to ~0.02 by r=1.82 → uplift **+0.98**, growing with noise exactly as T1(a/b) predicts. At L0 the Bayes attack IS the exact optimum (no approximation slack).
- **Why:** in d=2304, isotropic DP noise is ~orthogonal to the (2048) inter-embedding directions, so NN-to-the-known-table is geometrically noise-robust → the information is *preserved*, and the optimal attack extracts it.
- **C2 (re-correlation) — supported.** The MI proxies decay slowly (CLUB −38%, capacity-PVI 0.98→0.74) and the *strong* recoverers (Bayes-NN ~1.0; the capacity-PVI reader, an approx-Bayes classifier, 0.74–0.98) **stay high — tracking the preserved information**; ridge **decorrelates** (crashes 50× while MI barely moves). The MI probes correctly predicted recoverability; ridge was the information-inefficient attack.
- **Honest limitation.** L0 is the *easiest* layer (observation ≈ noised embedding; attacker knows the exact table). It is a clean proof-of-principle, not the hard case. **The research question lives at L>0**, where DP noise has propagated through nonlinear blocks and NN-to-table no longer applies — a *learned channel-aware denoiser/decoder* is required (next batch).

**Next batch (next iteration):** L>0 channel-aware trained decoder vs ridge under input-DP (does the uplift + re-correlation survive noise propagation?), then other noise profiles (Laplace/Shredder) + MDL/SDL probe.

---

## Threat model & attack-comparison fairness (2026-06-21, per reviewer)

**Fixed threat model for ALL attack comparisons: WEIGHTS-PUB honest-but-curious** (the repo's
motivating model, CLAUDE.md). The adversary knows weights + embedding table + the DP mechanism
params (σ, clip C are *published* privacy parameters), observes the DP-protected representation,
and — having the weights — can run the public model on chosen inputs to synthesize unlimited
`(noised-representation, token)` training pairs at any σ. **An attack is admissible iff it uses
only this; a comparison is valid only between admissible attacks.**

| Attack | Information used | Admissible under WEIGHTS-PUB? | Fair vs ridge? |
|---|---|---|---|
| ridge (baseline) | table + self-generated noised training pairs | yes | — |
| **L0 Bayes-NN** | table + public σ, **no training set** | yes (uses strictly *less*) | ✓ |
| **L>0 channel-aware decoder** | table + self-generated noised pairs **at σ** | yes (same as ridge) | ✓ |
| capacity-PVI reader | trains on admissible noised data | yes | ✓ (probe, not attack) |

**Out of scope (would be invalid):** under WEIGHTS-BLIND (no weight access) neither ridge nor any
trained decoder is admissible (cannot synthesize training pairs); σ-awareness is admissible ONLY
because DP params are public — a secret σ would make channel-aware attacks inadmissible. We do NOT
claim cross-threat-model comparisons. (wiki: claim:threat-model-fairness)

---

## B2-L>0 — Channel-aware MLP decoder vs ridge under at-layer noise (L5/12/20) — RUN (2026-06-21)

`results/b2_lpos_decoder.json`, GPU. Cached clean resid_post (L5/12/20, gemma-2-2b), in-memory
Gaussian noise (level = σ/act-RMS), vocab-disjoint + **shuffle control** (selectivity = real − floor).
Threat model: WEIGHTS-PUB (all attacks admissible, [[threat-model-fairness]]).

| L | level | ridge sel | dec-CA sel | uplift-sel | shuffle floor (r/ca) | capPVI | CLUB |
|---|---|---|---|---|---|---|---|
| 5  | 0.0 | +0.779 | +0.556 | **−0.223** | 0.010/0.033 | 0.837 | 2959 |
| 5  | 1.5 | +0.167 | +0.091 | −0.076 | " | 0.420 | 617 |
| 12 | 0.0 | +0.704 | +0.406 | **−0.298** | 0.054/0.068 | 0.789 | 2902 |
| 12 | 3.0 | −0.002 | −0.014 | −0.011 | " | 0.160 | 164 |
| 20 | 0.0 | +0.720 | +0.495 | **−0.225** | 0.059/0.071 | 0.838 | 3043 |
| 20 | 3.0 | +0.077 | +0.030 | −0.047 | " | 0.237 | 375 |

**Findings (honest):**
- **NEGATIVE for the MLP decoder.** A 250-epoch MLP channel-aware decoder **loses to ridge at every
  depth and noise level** (uplift-selectivity always negative). The dramatic L0 uplift (+0.98) does
  **NOT** replicate at depth: there the clean embedding is *not* directly observable, ridge's
  closed-form linear map already captures the resid→embedding geometry well, and a vanilla MLP
  doesn't beat it. **Beating ridge at depth needs a genuinely stronger decoder** (iterative/Vec2Text
  refinement or noise-aware MAP+LM-prior à la BeamClean), not a plain MLP.
- **Shuffle control passes** (floor ≈ chance 0.01–0.07 → selectivity ≈ recovery): both attacks
  generalize, no memorization — consistent with vocab-disjoint.
- **KEY NUANCE — re-correlation is noise-geometry-specific.** Under **at-layer** additive Gaussian
  noise, **both ridge and decoder selectivity track the MI probes PERFECTLY** (Spearman(sel, capPVI)
  = Spearman(sel, CLUB) = **1.00** at L5/L12/L20). So ridge does NOT decorrelate here. The B3
  decorrelation (ridge↔MI breaks at L20) was specific to **input-DP noise *propagation*** (noise
  injected at the embedding, reshaped through depth) — NOT a generic property of ridge under noise.

**Synthesis (the sharpened thesis).** The MI↔recovery decorrelation is **not universal**: under
at-layer noise even the weak ridge tracks MI (ρ=1.0); it appears specifically under **noise
propagation** (input-DP through depth, B3 L20). Where decorrelation *does* occur, a stronger
information-efficient attack restores it — demonstrated decisively at **L0** (Bayes-NN +0.98). At
depth under propagation, the stronger attack is **not yet found** (the MLP isn't it) → the live
open problem. So: probes are faithful predictors except under noise-propagation geometry, where
attack strength is the limiting factor.

**Verdict:** C1 (uplift) holds at L0, FAILS for the MLP at depth. C2 (re-correlation) holds trivially
under at-layer noise (ridge already tracks); the interesting decorrelation is propagation-specific.
Next: a stronger depth decoder (iterative/MAP) targeting the input-DP-propagated regime where ridge breaks.

---

## MDL/SDL probe — completing the {PVI, CLUB, MDL/SDL} set (2026-06-21, model-free)

`results/mdl_probe_check.json`. Cached resid L12, at-layer Gaussian noise sweep, MDL
surplus-description-length (`measures/mdl.online_code_length`) with shuffle-control selectivity.

| level | MDL-SDL sel | ridge sel | capPVI | CLUB |
|---|---|---|---|---|
| 0.0 | +13898 | +0.704 | 0.789 | 2902 |
| 0.75 | +779 | +0.141 | 0.484 | 834 |
| 1.5 | +134 | +0.060 | 0.260 | 383 |
| 3.0 | +166 | −0.002 | 0.160 | 164 |

**MDL/SDL joins the faithful MI-probe set:** Spearman(MDL-SDL selectivity, ridge-selectivity) =
Spearman(·, capPVI) = Spearman(·, CLUB) = **+0.80** (monotone decay; the slight high-noise wobble
at c=3.0 is the known class-probe-family overfit floor). So **all three named probes — PVI, CLUB,
MDL/SDL — track attack recovery under at-layer noise** (ρ 0.80–1.0), closing the "MDL/SDL untested"
gap. CLUB/capPVI are smoother (ρ=1.0); MDL is noisiest (overfit + 6–7× cost), consistent with prior
findings — report CLUB/accuracy primary, MDL auxiliary.

---

## B2-propagated — stronger decoder vs ridge under PROPAGATED input-DP (L12/L20) — RUN (2026-06-21)

`results/b2_propagated_dp.json`, GPU. Propagated input-DP (embedding-DP hook → forward → capture
resid at L12/L20), gemma-2-2b, 160 prompts, ε∈{∞,1024,512,256}, vocab-disjoint + shuffle selectivity.
This is the **open regime** (B3 L20: ridge decorrelates from MI). Channel-aware decoder = MLP trained
on the propagated-noised resid.

| ε | L | ridge sel | decoder sel | uplift-sel | capPVI | CLUB |
|---|---|---|---|---|---|---|
| ∞ | 20 | +0.546 | +0.479 | −0.066 | 0.668 | 2525 |
| 1024 | 20 | +0.459 | +0.484 | **+0.025** | 0.693 | 2612 |
| 512 | 20 | +0.405 | +0.475 | **+0.070** | 0.825 | 2835 |
| 256 | 20 | +0.089 | +0.232 | **+0.143** | 0.527 | 1777 |

**Findings — POSITIVE (reverses the at-layer-noise negative):**
- **Under propagated DP the decoder increasingly BEATS ridge as noise grows** (uplift-sel
  −0.07→+0.03→+0.07→**+0.14** @L20). Ridge's linear obs→emb map breaks on the *structured propagated*
  perturbation (ridge sel collapses 0.546→0.089) while the channel-aware decoder holds (0.479→0.232).
  Contrast with at-layer noise (B2-L>0) where ridge was near-Bayes and the decoder *lost* — so the
  decoder's advantage is **specific to noise propagation**, exactly the regime where ridge decorrelated.
- **Re-correlation improves with the stronger attack:** Spearman(selectivity, capPVI) — decoder vs ridge
  = **0.80 vs 0.40 @L12** and **0.40 vs 0.20 @L20**. The decoder tracks the MI probe better than ridge
  where ridge breaks. (CLUB mixed/weaker; only 4 ε points → coarse.)

**Verdict:** the open regime is cracked in the *right direction* — a stronger (channel-aware) attack
recovers more AND re-correlates with MI under propagated DP, where the weak ridge both collapses and
decorrelates. **Suggestive, not conclusive** (4 ε, 1 seed, noisy Spearman); firming needs denser ε +
seeds + a genuinely stronger decoder (iterative/MAP). But the direction confirms the thesis:
*the MI probes are faithful; closing the recovery gap is an attack-strength problem, even at depth.*

---

## VMA permutation channel — RowSort-64 is a weak attack; full-sorted matcher is far stronger (2026-06-21, GPU-free)

`results/vma_stronger.json`. gemma-2-2b embed, N=1000, AloePri perm-core α_e sweep, 3 seeds,
Hungarian assignment. Directly answers the original VMA observation (α_e 0→0.2: τ 1.0→0.56, CLUB −1%).

| α_e | RowSort-64 (VMA baseline) | full-sorted-row | uplift | CLUB-on-φ |
|---|---|---|---|---|
| 0.1 | 0.977 | 1.000 | +0.023 | 243 |
| 0.2 | 0.565 | **0.999** | **+0.434** | 240 |
| 0.35 | 0.204 | **0.804** | **+0.600** | 235 |
| 0.5 | 0.099 | 0.442 | +0.343 | 230 |
| 1.0 | 0.023 | 0.054 | +0.031 | 206 |

**Finding — the permutation channel confirms the thesis (POSITIVE, tight).**
- **RowSort-64 is information-inefficient.** Its 64-quantile binning is a *lossy compression* of the
  sorted row; it collapses under small noise (α=0.2 → 0.56) while CLUB-on-φ barely moves (245→240b).
- **The full sorted row (all d values) — the sufficient statistic for the column-perm + Gaussian
  channel — extracts the preserved information**: τ-recovery 0.999 at α=0.2 (**uplift +0.43**), 0.804 at
  α=0.35 (**uplift +0.60**). (Euclidean vs cosine on the sorted vector identical here — the gain is the
  full sorted row vs the 64-bin compression, not the metric.)
- Both attacks track CLUB-on-φ over the full sweep (ρ≈1.0), but at *fixed small noise* RowSort
  **under-reports** leakage (says 56% recoverable) while the truth is ~100% — exactly the
  Bayes-optimality gap, on the permutation channel. The probe (CLUB-φ) was faithful all along.

**Verdict:** the information-efficiency thesis holds on BOTH channels — token/embedding (Bayes-NN @L0
+0.98; channel-aware decoder under propagated DP) AND permutation (full-sorted matcher +0.43–0.60 over
RowSort). In every case the weak deployed attack's collapse under small noise was attack weakness, not
information loss; the MI probes correctly indicated the leakage the stronger attack then realized.

---

## B6 — Stronger / Vec2Text-style decoder under PROPAGATED input-DP @L20 — RUN (2026-06-21)

`results/b6_strong_decoder.json`, GPU (≤20min budget). ridge / 1-shot MLP / deep MLP (capacity
control) / Vec2Text-style iterative corrector (T=1,2,3); propagated input-DP @L20, ε∈{∞,1024,768,512,
384,256}, vocab-disjoint + shuffle floors (ridge 0.006, mlp 0.035 ≈ chance). WEIGHTS-PUB.

| ε | ridge | mlp | deep | iter T1=T2=T3 | capPVI | CLUB |
|---|---|---|---|---|---|---|
| ∞ | 0.516 | 0.337 | 0.345 | 0.364 | 0.668 | 2523 |
| 1024 | 0.507 | 0.355 | 0.359 | 0.404 | 0.693 | 2610 |
| 768 | 0.483 | 0.379 | 0.402 | 0.422 | 0.741 | 2659 |
| 512 | 0.470 | 0.401 | 0.405 | 0.409 | 0.825 | 2834 |
| 384 | 0.349 | 0.372 | 0.356 | 0.393 | 0.787 | 2767 |
| 256 | 0.167 | 0.236 | 0.221 | 0.235 | 0.527 | 1780 |

**Findings:**
- **RE-CORRELATION (C6) — CONFIRMED, the headline.** Spearman(selectivity, capPVI) over ε:
  **deep +0.83, iterative +0.71, ridge −0.09**; vs CLUB: iter +0.71, ridge −0.09. The trained decoder's
  recovery **tracks the MI probes where ridge ANTI-correlates** (reproducing the B3 L20 decorrelation).
  A stronger attack restores the MI↔recovery correlation the weak ridge breaks — exactly the objective.
- **UPLIFT (C5) — crossover.** Decoder beats ridge at **high noise** (ε≤384: Δ +0.044/+0.068) but ridge
  wins clean/low-noise (Δ −0.15 at ∞). So the decoder is the stronger attack in the high-noise regime
  (where the defence "works") and tracks MI throughout; ridge is better only when barely defended.
- **ITERATION (C7) — honest null.** `iter_T3 − iter_T1 = +0.000`, `iter − deep = +0.023`. Pure
  embedding-space Vec2Text iteration adds **nothing** (it's a fixed function of Y), and capacity (deep)
  ≈ MLP. **Faithful Vec2Text needs the forward model in the loop** (re-embed hypothesis → compare to Y),
  which exceeds the 10–20min budget — the genuine open frontier (B6c, future work).

**Verdict:** the stronger attack is **implemented (Vec2Text-style corrector + deep decoder) and tested
optimally under budget**. The decisive objective-result holds: a trained nonlinear decoder **re-correlates
with the MI probes (+0.83) where ridge anti-correlates (−0.09)** under propagated DP — the MI probes are
faithful, ridge is information-inefficient. Honest limits: (a) the decoder beats ridge only at high noise
(crossover), (b) embedding-space iteration/capacity don't help — closing the low-noise gap and beating
ridge across all ε needs the forward-model-in-loop Vec2Text (the named next step).

---

## B6c — Forward-model-in-loop Vec2Text attack under PROPAGATED input-DP @L20 — RUN (2026-06-21)

`results/b6c_forward_model.json`, GPU (~1.5 min). FMV = re-embed each decoder-seeded candidate token
(top-k=16) through the actual model (clip-only reference) and match to the observed noised resid Y_obs;
teacher-forced prefix (oracle-prefix per-position upper bound). gemma-2-2b L20, ε∈{∞,512,256}, 400
scored test positions, vocab-disjoint, WEIGHTS-PUB.

| ε | ridge | decoder top1 | **FMV** | uplift FMV−ridge | uplift FMV−dec |
|---|---|---|---|---|---|
| ∞ | 0.212 | 0.380 | **0.738** | **+0.526** | **+0.357** |
| 512 | 0.489 | 0.431 | 0.495 | +0.006 | +0.064 |
| 256 | 0.227 | 0.245 | **0.025** | −0.202 | −0.220 |

**Findings — the forward model is the strongest LOW-noise attack but is noise-fragile:**
- **Closes the low-noise gap decisively.** At clean/low noise FMV recovers 0.738 vs ridge 0.212 / decoder
  0.380 (+0.53 / +0.36). Re-embedding candidates through the known model matches Y_obs almost exactly
  when noise is small — extracting what ridge/decoder miss. (Within-run comparison; absolute ridge varies
  across configs/pool, not cross-run-comparable — the relative FMV≫ridge/dec at clean is the valid claim.)
- **Collapses at high noise** (ε=256 → 0.025, below ridge). FMV matches the *clean* forward to a single
  heavily-noised observation, so at high σ the token signal is swamped → ~chance. **Mirror image of the
  decoder** (B6: wins high-noise, loses clean).
- **No single attack dominates the noise range** → the optimal attack is **regime-dependent**:
  forward-model match (low noise) ⊕ learned denoiser/decoder (high noise). FMV is non-monotone so it does
  not re-correlate over the full sweep (FMV↔capPVI +0.50 = ridge, coarse 3-ε); the decoder (B6, +0.83) is
  the re-correlating attack.

**Verdict (frontier built):** the faithful forward-model-in-loop Vec2Text attack is implemented + tested.
It confirms the thesis's strongest form at low noise (the model-in-loop recovers +0.53 over ridge where
the embedding-space corrector could not) and reveals the honest limit: it is noise-fragile because it
matches a clean reference to a single noisy draw. **The named optimum is a NOISE-AWARE FMV** —
denoise Y_obs (or match to E[Y|cand] under the noise model) before the forward-model match — which would
combine FMV's low-noise power with the decoder's high-noise robustness. Concrete next step.

---

## B7 — FAITHFUL Vec2Text: the iterative corrector loop B6c lacked — RUN (2026-06-22)

`results/b7_vec2text.json`, GPU (~4.5 min). The genuine Morris et al. (2023) loop B6c was
missing: each round RE-EMBEDS the current full hypothesis `ê^(t)=φ(h^(t))`, a learned corrector
`c(e, ê^(t), e−ê^(t), x^(t))` emits a NEW token sequence, accept iff closer to e (cosine, judged on
the test positions). Here φ = clip-only forward → resid_post @L20 (WEIGHTS-PUB); e = Y_obs (propagated
input-DP); corrector = per-position MLP on the paper's 4-block input (EmbToSeq collapsed). Known
(train-token) positions teacher-forced as fixed context (strictly less TF than B6c's full-prefix TF);
test positions recovered jointly + iteratively over a test-only pool; corrector trained symmetrically
on train positions over a train-only pool (vocab-disjoint). T=4 rounds, ε∈{∞,512,256}, gemma-2-2b.

| ε | ridge | base (t=0) | FMV-tf (B6c) | **V2T-feedback** (t0→t4) | **V2T-no-feedback** (t0→t4) | capPVI |
|---|---|---|---|---|---|---|
| ∞ | 0.407 | 0.270 | **0.527** | 0.270→0.273 | 0.270→**0.307** | 0.653 |
| 512 | 0.374 | 0.302 | 0.357 | 0.302→0.307 | 0.302→**0.347** | 0.798 |
| 256 | 0.126 | 0.121 | 0.020 | 0.121→0.111 | 0.121→**0.136** | 0.551 |

**Findings (honest — a surprise that contradicts Morris Fig 3):**
- **FEEDBACK DOES NOT TRANSFER (null/slightly negative).** V2T-*feedback* ≤ V2T-*no-feedback* at every
  ε (Δ −0.034/−0.040/−0.025). Morris Fig 3 shows feedback >> no-feedback. **Interpretation:** in the
  per-position-resid threat model `e = Y_obs` is ALREADY a per-token localized observation, so the
  hypothesis-embedding feedback `ê` — whose role in Morris is to localize a SINGLE GLOBAL sentence
  embedding across n tokens — adds little. (**Confound, flagged:** the feedback net has 2× nonzero input
  blocks → may overfit the fixed train set; the deciding control is below.)
- **Greedy iteration PLATEAUS at t=1.** A one-shot-trained corrector reaches a fixed point after one
  application → no multi-round gain. Morris's large exact-match jumps come from sequence-level BEAM
  (`[50 steps]` 40% → `[50 steps+sbeam]` 92%) + training on the iterative hypothesis distribution —
  neither implemented here (budget). So this reproduces the paper's *greedy* plateau, not its beam result.
- **POSITIVE — the trained corrector recovers the high-noise regime FMV lost.** ε=256: V2T-nf **0.136**
  vs FMV-tf **0.020** (collapsed, chasing a clean reference against corrupted Y_obs — the B6c failure)
  vs base 0.121 vs ridge 0.126. The high-noise lever is the **noise-aware refinement**, NOT the feedback.
  FMV still wins low-noise (0.527 @∞). → **reproduces B6c regime-dependence**: forward-match best at low
  σ, trained corrector best at high σ; the missing piece B6c flagged is supplied by training on noised Y,
  not by iterative re-embedding feedback.
- **Re-correlation:** V2T-fb↔capPVI ρ=**+1.00** (monotone with the MI probe across ε) vs FMV/ridge +0.50.
  Coarse (3 ε) — directional only.

**Verdict:** the faithful Vec2Text iterative corrector is **implemented + tested** (the T-step
re-embedding loop, residual feedback, seq-level acceptance gate — all the machinery B6c lacked). The
decisive result is a **negative for the feedback mechanism in this setting**: conditioning on the
re-embedded hypothesis does not help (and slightly hurts) because the per-position resid already gives a
localized target — Vec2Text's global-embedding-localization trick is moot here. The recovery B6c was
missing at high noise is delivered by the **noise-aware trained corrector**, not the iteration. **Named
firm-ups (in priority order):** (1) capacity-matched control — feed no-feedback `[e, e, 0, emb(x)]`
(same input width, no genuine feedback) to rule out the overfit confound and confirm the feedback-null
is real; (2) sequence-level beam search (the paper's actual lever); (3) train the corrector on its own
iterative hypothesis distribution (multi-round). Only (1) is needed to harden the headline claim.

### B7-analysis — why feedback failed + is Vec2Text the right attack for this surface? (2026-06-22)

Post-run analysis vs the reference impl (`github.com/vec2text/vec2text`) and the literature.
**Two independent causes; the second means the result above does NOT cleanly test Vec2Text.**

**Cause A — SURFACE/FRAMING MISMATCH (structural).** Vec2Text inverts a *single pooled bottleneck
vector* `e=φ(x)` that compresses an entire n-token sequence into one d-dim vector (mean-pool of GTR /
ada-002). That compression creates severe per-token under-determination, and the entire purpose of the
iterative re-embed + residual `e−ê` feedback is to resolve it by search. Our surface is the opposite:
per-position resid `Y_obs∈R^{n×d}` gives each token its OWN d-dim observation, dominated by that token.
*"Language Models are Injective and Hence Invertible"* (arXiv 2510.15511, 2025) proves the input is
almost-surely PER-POSITION recoverable from clean hidden states — the token is near-linearly readable
(logit-lens). So there is **no bottleneck under-determination per position for feedback to resolve** →
a one-shot per-position decoder is already near-sufficient and the feedback is structurally moot. This
is a category error in how we APPLIED Vec2Text (per-position), not a defect of Vec2Text.

**Cause B — OUR IMPL IS NOT FAITHFUL VEC2TEXT ("suspiciously fast" = correct).** Reference Vec2Text =
a **T5-base (~220M) autoregressive seq2seq** corrector trained on **MSMARCO 8.8M docs, 100 epochs,
days/GPU**, inference with **sequence-level beam search** (the lever for `[50 steps]`40%→`+sbeam`92%).
Ours = a 2-layer MLP regressing a per-position embedding (cosine loss), argmax to a 2048 pool, trained
in **seconds** on a few thousand rows — no autoregressive decoder, no generation, no beam. It cannot
explore sequence space; the loop is hollow (deterministic → fixed point at t=1). So B7 did not actually
run Vec2Text; it ran a per-position MLP probe inside Vec2Text's control flow.

**Literature on the residual-stream surface (claim verified):**
- `vec2text/vec2text` README: inverts **pooled sentence embeddings** (mean-pool, GTR/ada-002), MSMARCO
  8.8M, 100 epochs, "even a few days on a single GPU"; **no per-position activations**.
- **Rep2Text** (arXiv 2511.06571, 2025) — the genuine residual-stream analog: inverts a **single
  last-token representation `h^ℓ`** → full preceding text (~½ of 16-token seqs; ROUGE-1 0.6@8tok →
  0.3@64tok; best mid layers 10–15). The CORRECT single-vector framing for this surface. Notably it
  **dropped Vec2Text's iterative corrector** for an adapter+autoregressive-decoder — even single-token
  inversion does not clearly want the iterative-feedback machinery.
- Injectivity paper (2510.15511): per-position clean hidden states are exactly invertible via
  optimization search — per-position recovery is a (hard) exact-inversion/probing problem, not an
  embedding-inversion-search problem.

**CONCLUSION — Vec2Text is NOT the appropriate attack for the per-position `resid_post` surface.** Its
iterative-corrector + feedback machinery is built for a pooled single-vector bottleneck; per-position
activations are a probing/logit-lens problem (matched tools: linear/MLP probe — our "base decoder" — or
logit-lens/tuned-lens; under DP noise, a noise-aware learned decoder, which B7's no-feedback corrector
already is). Vec2Text WOULD be appropriate on a **single-vector framing** of this repo's space: (i) a
pooled RAG/retrieval sentence embedding (its native use, and the private-rag motivating surface), or
(ii) a single last-token resid à la Rep2Text. Per the standing instruction, **flagged and STOPPED** — no
experiment plan was built on the per-position surface. The decision of whether to pivot to the pooled /
last-token surface (using the real `vec2text` dependency, faithfully) is deferred to the user.

---

## B8 — Faithful Vec2Text on a pooled GTR sentence embedding under DP — RUN (2026-06-22)

The pivot from B7-analysis: Vec2Text on the surface where it is the CORRECT attack — a single
pooled bottleneck **sentence embedding**, not the per-position resid. Plan:
`vec2text-pooled/EXPERIMENT_PLAN.md`. Uses the **real `vec2text` dependency + PRETRAINED gtr-base
corrector** (no training; M0 dependency gate passed — transformers-4.44 shadow + apex JIT on the
ROCm iGPU; recipe in memory `vec2text-rocm-dependency-recipe`).

**Setup.** φ = `sentence-transformers/gtr-t5-base` encoder, mean-pooled → 768-d **sentence
embedding** `e0` (a real RAG/retrieval embedder; the encoder vec2text's corrector inverts → attacker
matched to target). Secret = the source text. Defense = DP on the *released embedding*:
`e' = clip(e0, C) + N(0,σ²)`, σ = C·z/ε, C = p99.9(‖e0‖) = 2.508. Attack = greedy 20-step Vec2Text
(`sequence_beam_width=1`; beam>1 is 4×+ cost, deferred). N=128 texts truncated to 32 GTR tokens
(corpus `release-gate-512.txt` — OUT-of-domain for GTR's NQ/MSMARCO-trained corrector; caveat below).
Recovery scored **against the ground-truth text** (BLEU / token-F1 / exact); cos(GTR(recon), e0) is a
secondary embedding-space check. Matched probe = CLUB `I(e'; e0)`; capPVI = kmeans-cluster reader.
`results/v2t_dp_sweep.json`, ~7 min total (≈0.66 s/text).

| ε | σ/C | BLEU | token-F1 | exact | cos | CLUB (b) | capPVI |
|---|---|---|---|---|---|---|---|
| **base [0-step], ε=∞** | — | 12.5 | 0.479 | 0.000 | 0.853 | — | — |
| **∞ (Vec2Text 20-step)** | 0 | 56.5 | **0.800** | **0.180** | 0.958 | 566.6 | 0.158 |
| 1024 | 0.005 | 10.8 | 0.433 | 0.000 | 0.821 | 496.1 | 0.158 |
| 512 | 0.010 | 5.9 | 0.299 | 0.000 | 0.693 | 401.4 | 0.132 |
| 256 | 0.019 | 2.8 | 0.162 | 0.000 | 0.442 | 248.6 | 0.132 |
| 128 | 0.038 | 2.0 | 0.098 | 0.000 | 0.244 | 71.1 | 0.105 |

**Findings — POSITIVE on the correct surface (contrast with B7's negative on the wrong one):**
- **C1 (clean leakage) — SUPPORTED.** Clean Vec2Text recovers **token-F1 0.80, exact 0.18, BLEU 56.5**
  vs the base 0-step model 0.48 / 0.0 / 12.5 — the iterative corrector roughly doubles token-F1 and
  produces exact reconstructions where the base model gets none (the information-efficiency gap, C4).
  Below Morris's in-domain greedy 20-step (tF1 0.96 / exact 0.40) — **consistent with our
  out-of-domain corpus** (GTR's corrector is NQ/MSMARCO-trained; `release-gate-512` is off-distribution).
- **C2 (DP leakage-calibration) — STRONGLY SUPPORTED.** Clean **monotone** decay across ε:
  tF1 0.80→0.43→0.30→0.16→0.10, cos 0.958→0.244, BLEU 56.5→2.0. Two-regime privacy story: even a
  *tiny* budget (ε=1024, σ/C≈0.5%) **kills exact reconstruction** (0.18→0) and halves token-F1, while
  **semantic/partial leakage persists** (cos 0.82, tF1 0.43) down to ε≈128. The pooled embedding is
  fragile to small absolute noise for *exact* recovery but degrades *gracefully* for partial recovery.
- **C3 (matched probe predicts the SOTA attack) — SUPPORTED for CLUB.** **Spearman(recovery, CLUB) =
  +1.00** for token-F1 / cos / BLEU (exact +0.71, degenerate — exact=0 off the clean point). CLUB
  `I(e';e0)` decays 566→496→401→249→71 bits, perfectly co-monotone with recovery → **the cheap probe
  forecasts what the strongest faithful inversion achieves at each privacy budget.** capPVI
  (kmeans-cluster reader) is weak/noisy (+0.67 tF1, barely moves 0.158→0.105) — **NOT the matched probe
  here**; CLUB is. (Consistent with the prior finding that embedding-geometry CLUB is the faithful,
  transferable probe for this channel.)

**Verdict:** On the pooled sentence-embedding surface — Vec2Text's *native* setting and the private-rag
motivating surface — the faithful pretrained attack works (clean tF1 0.80 / exact 0.18, ≫ base), DP on
the released embedding yields a clean monotone leakage-vs-ε calibration curve, and the **cheap CLUB
probe tracks the SOTA attack's recovery with ρ=+1.00**. This is the headline the matched-probe program
wanted, delivered with a real SOTA attack — and it sharpens B7/B7-analysis: Vec2Text is the right tool
**here** (single pooled bottleneck) and the wrong tool on per-position resid. **Honest caveats / firm-ups:**
(a) out-of-domain corpus depresses absolute C1 (swap to NQ/Wikipedia for domain-matched numbers, won't
change C2/C3); (b) greedy beam=1 (run beam≥4 for the full-strength exact-match anchor — affordable, ~4×);
(c) capPVI needs a genuinely matched reader for embeddings (cluster-id is not it); (d) C4 strength-axis
(0/1/20-step) + NN-retrieval baseline still to run (B3 in the plan). Threat model: attacker has φ (public
GTR) + DP params + matched corrector (WEIGHTS-PUB analog for embeddings).

---

## B9 — Spectral channel-MI probe: implementation + empirical validation (C1/C2) — RUN (2026-06-22)

Validates `claim:spectral-channel-mi-embedding-inversion`. Probe implemented as the geometry-only
`src/talens/measures/spectral_channel_mi.py` (Codex xhigh reviewed — no critical; hardened; 10/10
model-free unit tests `tests/test_spectral_channel_mi.py`). Eval `scripts/eval/spectral_mi_probe_eval.py`,
`results/spectral_mi_probe_eval.json`. Pooled-GTR DP sweep, N=96, 32-tok, greedy 20-step Vec2Text.
Plan: `vec2text-pooled/../spectral-mi-probe/EXPERIMENT_PLAN.md`.

| ε | σ | token-F1 | exact | pos-acc | cos | **I_G (b)** | d_eff | CLUB (b) | capPVI | RD-floor |
|---|---|---|---|---|---|---|---|---|---|---|
| ∞ | 0 | 0.790 | 0.198 | 0.491 | 0.954 | 1597 | 95 | 619 | 0.310 | 0.00 |
| 1024 | .012 | 0.446 | 0 | 0.087 | 0.827 | 312 | 95 | 542 | 0.241 | 0.291 |
| 512 | .024 | 0.301 | 0 | 0.060 | 0.673 | 220 | 95 | 391 | 0.310 | 0.475 |
| 256 | .048 | 0.163 | 0 | 0.022 | 0.432 | 135 | 94 | 201 | 0.276 | 0.656 |
| 128 | .095 | 0.093 | 0 | 0.018 | 0.242 | 68 | 63 | 64 | 0.138 | 0.812 |

**C1 (matched-probe, headline) — VALIDATED.** Spearman(probe, recovery) over ε:
- **I_G = +1.00** (token-F1 / cos / pos-token-acc), +0.71 (exact, degenerate since exact=0 off clean) —
  **identical to CLUB** (+1.00 / +0.71), and **≫ capPVI** (+0.62 / +0.54, weak/noisy).
- **Cost:** I_G **60 ms** (closed-form eigh, geometry-only, no attack/training) vs CLUB **1.7 s**
  (variational net) vs capPVI 0.5 s (reader). → I_G **dominates CLUB**: same predictiveness at ~28×
  lower cost, zero estimator variance, never runs the attack, AND decomposable (per-mode/d_eff) where
  CLUB is a single scalar.

**C2 (ceilings hold) — 0 violations.** RD per-token-error floor (T3b) rises monotonically
0→0.29→0.48→0.66→0.81 with noise; actual per-token error (1−pos-acc) stays above it at every ε.
Fano exact-match ceiling is trivial at low σ (I_G>H_X proxy → clamps to 1) but **non-trivial at high
noise** (ε=128: ceiling 0.144 ≥ actual exact 0). H_X≈479b = 32·log2(32100) and H_e0=H_X are flagged
upper proxies — C2 is "no violation + qualitative tightness", not a sharp number.

**Honest caveat (sample-size / spectrum estimation).** `d_eff≈95` is pinned at `n−1`: with **N=96 < d=768
the sample covariance is rank-deficient** (Marchenko–Pastur regime), so the eigen-spectrum, `d_eff`, and
tail profile are **undersampled** (only ~95 nonzero eigenvalues). C1/C2 are robust to this (they need only
that I_G is monotone in σ, which holds), but the **localization profile (M3/B4) requires n≫d** — estimate
Σ from a large embedding corpus (cheap: GTR-encode only, no inversion) decoupled from the small attacked
set. This is the standing "estimate Σ" open risk in the claim, now quantified.

**Verdict:** C1 (the headline) and C2 are **validated** — the geometry-only spectral channel-MI is a
matched, attack-independent leakage probe that predicts SOTA Vec2Text recovery as well as CLUB at ~28×
lower cost and ≫ capPVI, with ceilings that hold. Remaining must-run: **M3 eigen-ablation** (localization
"where" test) — re-scoped to estimate Σ from n≫d embeddings first.

---

## B10 — Utility side: retrieval ranking fidelity → the privacy–utility tradeoff — RUN (2026-06-22)

The other half of the tradeoff. Literature scan (EdgeQuake + web): privacy/RAG schemes measure utility
as **retrieval ranking fidelity** (CAPRISE/RemoteRAG — alignment of the perturbed-query ranking with the
plaintext ranking + top-k' expansion), **downstream-task accuracy + Retained-Performance%** (OSNIP), or
**end-to-end QA EM/F1** (DP-KSA — where text-overlap/BLEU lives). Our released object is a GTR mean-pooled
**sentence embedding = a retrieval encoder**, so the matched, standard, robust, cheap metric is retrieval
ranking fidelity — *not* BLEU (which measures generation and needs an LLM+QA set, off-surface here).

Probe: `scripts/eval/utility_retrieval_eval.py` (`results/utility_retrieval_eval.json`). DP on the QUERY
embedding (same clip C + Gaussian σ=C·z/ε as the leakage side; corpus clean), each of N=256 texts a
leave-one-out query, clean cosine ranking = ground truth. Metrics: nDCG@10 (graded rel=clean cosine),
Recall@{1,5,10}, Spearman rank-corr of the full ordering, CAPRISE top-k' expansion. Cheap (cosine only,
no LLM / no training / no inversion; ~1 min).

**Privacy–utility tradeoff** (C≈2.51, aligned ε; leakage from B8/B9, N≈96–128):

| ε | σ/C | nDCG@10 | R@1 | R@10 | rankρ | k'/k | ‖ Vec2Text tF1 | exact | I_G (b) |
|---|---|---|---|---|---|---|---|---|---|---|
| ∞ | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.0 | ‖ 0.790 | 0.198 | 1597 |
| 1024 | .005 | 0.998 | 0.875 | 0.914 | 0.992 | 1.2 | ‖ 0.446 | 0 | 312 |
| 512 | .010 | 0.992 | 0.816 | 0.851 | 0.969 | 1.5 | ‖ 0.301 | 0 | 220 |
| 256 | .019 | 0.970 | 0.629 | 0.729 | 0.895 | 2.5 | ‖ 0.163 | 0 | 135 |
| 128 | .038 | 0.897 | 0.391 | 0.518 | 0.721 | 6.4 | ‖ 0.093 | 0 | 68 |
| 64 | .076 | 0.749 | 0.188 | 0.287 | 0.485 | 13.3 | ‖ — | — | — |

**Finding — utility degrades far more gracefully than leakage (favorable tradeoff).** At **ε≈512**
retrieval is near-lossless (nDCG@10 **0.992**, rank-ρ 0.97, R@10 0.85) while Vec2Text **exact
reconstruction is already dead** (0.198→0) and partial leakage halved (tF1 0.79→0.30, I_G 1597→220).
Clean **operating window ε≈256–512**: nDCG ≥0.97 (retrieval essentially intact) while exact text recovery
is gone and I_G cut 5–7×. Utility only seriously bites at **ε≈128–64** (nDCG 0.90→0.75; top-k' expansion
blows up 6×→13× — must over-retrieve to keep the true top-10) — well below where reconstruction leakage
was already defeated. **nDCG (graded) ≫ more robust than Recall@k**: the true neighbors stay top-ranked
even as exact top-k set membership shuffles, so ranking quality survives noise that halves Recall@10.

**Verdict:** the privacy–utility curve is favorable for embedding-DP against Vec2Text — there is an ε band
where the defense kills exact inversion at near-zero retrieval cost. Utility probe = retrieval ranking
fidelity (nDCG/Recall/rank-ρ/top-k'), the standard distance-preserving-RAG metric; cheap and robust.
Caveat: ground truth = the clean retriever's own ranking (self-referential, à la CAPRISE's vs-plaintext
eval); the publication-grade upgrade swaps in real BEIR/MTEB qrels (same code path).
