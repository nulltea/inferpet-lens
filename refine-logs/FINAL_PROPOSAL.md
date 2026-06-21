---
type: plan
status: current
created: 2026-06-20
updated: 2026-06-20
tags: [matched-probe, leakage-channels, permutation-recovery, AloePri, Shredder, V-information, CLUB, PID, cross-scheme-calibration]
companion: [it-leakage-estimation-set, EXPERIMENT_PLAN]
supersedes: [archive-capacity-pvi/EXPERIMENT_PLAN]
---

# Final Proposal — Matched information-theoretic probes for confidential-inference leakage channels

## Problem anchor (do not widen)

Confidential-inference leakage is currently scored as **one scalar** — token
top-1 recovery success rate (TTRSR) of a ridge `X→embedding` attack — and a
single IT measure is fit to predict it. The established capacity-matched-PVI
result already contains the crack this proposal widens: under input-local-DP the
**token-id reader and the embedding-reconstruction attack decouple with depth**
(ρ +0.99 @L0 → −0.21 @L20; CLUB shows the same gradient, so it is a property of
the signal, not the estimator). That decoupling is not noise — it is a
**probe↔target mismatch**: the reader measures *token identity*, the attack
reconstructs *embedding geometry*, and input-DP destroys those two at different
depths.

**The problem this proposal solves:** give a *per-channel* account of leakage —
*what* leaks (target), *through which surface*, and *which independent probe
predicts that channel's attack* — instead of one conflated scalar.

## Stable thesis: the matched-probe principle

Leakage is a small set of **(target × surface) channels**. Each channel `c` has:

- an **attack** `A_c` with a graded success metric `m_c`,
- a **matched probe** `P_c` chosen so that `I(P_c-readout ; target_c)` is high
  **but** `P_c` is computed *without* the attack's fitted map/target (independent
  by construction, not a reparameterisation of `A_c`).

**Central claim — a channel-specific calibration principle (falsifiable protocol):**

> A leakage probe is meaningful only relative to a *target secret*, an *observation
> surface*, and an *attacker family*. Matched pairs `(P_c, A_c)` calibrate across a
> defence's knob (within-layer/macro ρ ≥ 0.9); **mismatched** pairs `(P_c, A_{c′})`
> decouple — they anti-calibrate or become statistically irrelevant. The decoupling
> is the *prediction*; the L20×DP divergence is the first instance, generalised to a
> full probe×target matrix.

(Per the round-1 reviewer: **do not call this an information-theoretic "law"** until
it replicates across ≥2 defence families with repeated sign-flips — it is a
**channel-specific calibration principle / empirical decoupling protocol**.) This is
the dominant contribution **regardless of which headline framing the data selects**
(below). It upgrades "an IT measure correlates with an attack (one scheme)" — the
level of PAF / Jacobian-Rank-Recovery / FSInfo — to **matched + cross-channel +
cross-scheme + with a falsifiable decoupling protocol that explains when a measure
mispredicts**.

### Channel-definition table (the protocol's contract — per reviewer fix #5)

| Channel | Secret (target) | Surface | Matched probe `P_c` | Attack `A_c` (metric) | Independence criterion | Expected failure mode |
|---------|-----------------|---------|---------------------|------------------------|------------------------|------------------------|
| Token-id | token id @ pos | resid/kqv/kq | capacity-PVI reader acc | ridge (TTRSR) | never reads embed table | predicts id where reconstruction fails (L20×DP) |
| Perm-Π | secret τ∈S_n | obf weight rows / permuted acts | CLUB-on-φ | VMA (τ-recovery) | MI estimator, not the matching/assignment | keymat erases φ → both →0 |
| Embedding | continuous hidden vec | resid/pooled | CLUB I(rep;emb) | ridge cosine | continuous MI, not the ridge map | tracks reconstruction, misses id @ depth |
| Attn QK/OV | QK invariants / OV | kq/kqv_out | MMI-PID cond-increment | ISA (recovery err) | V-info reader, not ISA | cover-invariant QK leaks where scores don't |

## Three live framings (the data decides — deferred per 2026-06-20 decision)

The user explicitly deferred the headline framing until matrix data arrives. The
program is structured so **B3 (the decoupling matrix) + B4 (cross-scheme)
adjudicate** between:

| | Framing | Selected if… |
|---|---------|--------------|
| **F-A** | Matching principle as headline: a matrix of matched channels + the decoupling law | Matched diagonal ≥0.9 holds on **≥3** channels AND off-diagonal is reliably lower |
| **F-B** | Token-id calibration headline; Π + obfuscation/Shredder as cross-scheme robustness | Only token-id (+ maybe embedding) calibrate cleanly; other channels are messy |
| **F-C** | Permutation-recovery vs static obfuscation (AloePri) as a standalone thread | The Π-channel is the sharpest / most novel and the others add little |

No framing is baked into the build — every block produces evidence usable under
all three. The **decision gate is B6**.

## The four leakage channels (3 of 4 already largely built)

| # | Channel (target × surface) | Attack `A_c` (status) | Matched probe `P_c` (status) | MI link |
|---|----------------------------|-----------------------|------------------------------|---------|
| **1** | **Token-identity** — token id @ position; resid_post / kqv_out / kq | ridge-inversion TTRSR (`attacks/hidden_state`✅), vocab-match (`attacks/vocab_match`✅) | **capacity-PVI reader accuracy** (`measures/vinfo_capacity`, pca_softmax dim-anchored ✅) | `I(rep; token-id)` |
| **2** | **Permutation Π** — secret τ∈S_n under a static cover; obfuscated embed/head rows | **VMA τ-recovery** (`weights/vma`✅ RowSort+Hungarian), vocab-match (cover-agnostic ✅) | **bake-off** (B2): CLUB-on-invariants (`weights/measures.club_mi_weights`✅) vs **capacity-reader-on-invariants** (new, independent) vs retrieval-PVI-on-invariants (`weights/measures.v_information_weights`✅ — *= the attack in bits, the dependent reference*) | `I(invariant φ; row-id)` |
| **3** | **Embedding geometry** — continuous hidden/pooled vec; resid / pooled | ridge `X→emb` cosine (`attacks/hidden_state`✅; Vec2Text conceptually) | **CLUB** `I(rep; emb)` (`measures/club`✅) | `I(rep; emb)` |
| **4** | **Attention QK/OV** — Q·K invariants / per-head OV; kq (~48%), kqv_out (~97%), attn_score (low) | ISA-AttnScore (`attacks/attn_score`✅) | **PID QK-vs-OV split** (redundant/synergistic info about tokens) — **new build**; folds in the `softmax(QK^T)`-cover-invariance whitespace | `I(tokens; QK)` vs `I(tokens; OV)` |

Membership/attribute (LUMIA) is **cut** — there the probe *is* the attack
(a linear probe), so it cannot demonstrate independence, the whole point.

### The Π-channel is more built than assumed (2026-06-20 user correction)

`src/talens/weights/` already implements the full τ-recovery family on a
`WeightPair(plain, obf, perm, token_ids)`: the VMA attack, the
obfuscation-invariant `sorted_quantile` signature φ, CLUB `I(obf-φ; plain-φ)`,
and retrieval-PVI on φ. **What is missing is the defence that produces the
obfuscated table** — explicitly flagged "Part 2 / tests." That generator (AloePri
Alg1 keymat + permutation + Gaussian) is precisely the new defence work, and it
**feeds the existing VMA/measure scaffolding directly**, with the obfuscation
noise α_e as the calibration knob the module was designed to sweep.

## Defences — the cross-scheme calibration axis (6 families)

| Defence | Axis | Status | Wiring |
|---------|------|--------|--------|
| **input-local-DP** | lossy noise (embedding) | ✅ `localdp_runner.InputDPCover` (embed forward-hook) | exists |
| **split-depth** | abstraction by depth | ✅ (choose capture layer) | exists (depth sweep) |
| **AloePri permutation-core** | static invertible permutation | **new** Transform | token + embed-row permutation Π; `per_prompt_matrices(transform=…)` seam + embed hook for forward-consistent vocab-match |
| **AloePri full key-matrix (Alg1)** | static invertible rotation/scale + perm | **new** | KEYMATGEN/INVKEYMATGEN (null-space `P̂Q̂=I`) + per-component covariant obfuscation → obfuscated `WeightPair` and/or activation Transform; **must verify logits preserved** |
| **Shredder static-Laplace proxy** | lossy noise (learned-shape skipped) | **new** Transform | additive Laplace at a split activation |
| **Shredder learned-noise** | lossy noise (faithful) | **new** | train noise tensor to min `1/SNR` (+task loss) at the cut, fit Laplace, sample per-input |

The 6 span the two-corner design space (AloePri = static-invertible /
search-hardness; Shredder = learned-lossy / MI-destruction; DP between them),
which is exactly what a *cross-scheme* calibration claim needs.

## Dominant contribution (singular)

**A matched-probe taxonomy of confidential-inference leakage + the decoupling
law**: each (target×surface) channel has an independent IT probe that
calibratedly predicts its attack across defences, and mismatched probe↔target
pairs provably decouple. The capacity-PVI/token-id result is *one row*; the L20
divergence is the *first datum* of the law.

## Intentionally rejected complexity (V3 discipline)

- **Membership/attribute channel** — probe = attack; cannot show independence. Cut.
- **Full Vec2Text decoder** — CLUB already is the embedding-geometry probe; a
  generative decoder adds engineering, not a sharper claim. Cut to "conceptual."
- **MoE-routing surface** (Expert-Selections-Reveal) — no MoE in gemma-2-2b /
  Qwen3-4B test models. Parked.
- **PML / α-leakage on activations; SAE effective-DoF** — parked (prior plan).
- **Formal capacity bound** — kept only as a *conditional* block (B6-formal),
  triggered if a channel's matching *fails* and needs explaining.

## Key risks (carried into the plan)

- **R1 — forward-consistency under obfuscation.** The Π attack `vocab_match`
  needs `forward_fn` in the *same* covered space as the targets; only valid for a
  *deterministic/static* cover (AloePri qualifies; per-forward-random covers
  break it). VMA on `WeightPair` sidesteps this (pure table algebra) → prefer VMA
  for the Π-calibration, use vocab-match as the forward cross-check.
- **R2 — AloePri full reparam must preserve logits.** Covariant obfuscation is
  exact only if every key matrix cancels (`P̂Q̂=I`) and RMSNorm rescale κ is fused.
  Verify accuracy ≈ plaintext on a held-out batch before trusting the sweep;
  λ regulates norm blow-up (collapses at λ=3 in bf16).
- **R3 — learned-Shredder training cost.** A GPU training loop per cut/SNR point;
  gate behind the static-Laplace proxy passing, and behind C1/C2 holding.
- **R4 — matrix breadth vs depth.** 4 channels × 6 defences × layers is large;
  heavy gating (cheap cached-capture phases decide framing before expensive
  new-defence sweeps).
- **R5 — Π-probe independence.** retrieval-PVI-on-invariants *is* the VMA; the
  bake-off must find a probe with per-instance collinearity <0.9 vs VMA yet
  ρ≥0.9 vs τ-recovery, or conclude the Π-channel admits no independent probe (a
  result, feeding F-C scepticism).

## Relation to the prior thread

Subsumes `archive-capacity-pvi/EXPERIMENT_PLAN.md`: that plan's C1 (capacity-PVI
fixed + faithful) is **channel 1's matched-probe row, already PASSED**; its C2
(cross-scheme) is this proposal's B4 generalised across channels. The depth
decoupling becomes the seed of the decoupling law (C2/B3).
