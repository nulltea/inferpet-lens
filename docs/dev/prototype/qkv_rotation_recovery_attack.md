---
type: prototype-note
status: current
created: 2026-07-01
updated: 2026-07-01
tags: [aloepri, isa-attnvalue, kqv_out, rotation-recovery, known-plaintext, orthogonal-procrustes]
companion: research-wiki/claims/aloepri-kqvout-basis-alignment.md
supersedes: none
---

# qkv rotation-recovery attack (`rotation_recovery_attack`)

`rotation_recovery_attack` = recover AloePri Alg2's secret value-rotation from a few known-plaintext
anchor pairs, un-rotate the leaked tensors, then read them with a self-trained inverter.

**Name.** Renamed from `basis_align_attack` (too abstract) to a method-descriptive name: it *recovers a
rotation*. **Surface: `kqv_out`** (the per-head attention value output / ISA-AttnValue) — *not* `kq`
(scores carry rotary position-coupling, so their distortion is not a single fixed rotation). Code:
`src/talens/attacks/dp_inversion.py` (`rotation_recovery_attack`, `orthogonal_procrustes_R`,
`blockwise_procrustes_R`).

**This is a KNOWN attack** — orthogonal-Procrustes known-plaintext recovery of a linear/orthogonal cipher
(cross-lingual embedding alignment MUSE/VecMap; Hill-cipher KPA; orthogonal-obfuscation LLM-inference line
arXiv:2606.16461 / 2603.01499). See [[claim:aloepri-kqvout-basis-alignment]] for the prior-art verdict.

## Phase 0 — precompute (once, keyless)
1. **Self-gen inverter.** Run the *public plaintext* model on the attacker's own prompts → plaintext
   `kqv_out` rows `Xp_tr` with known tokens `ytr`. Fit ridge `W = (Xpᵀ Xp + αI)⁻¹ Xpᵀ E[ytr]` (rep→embedding).
2. **Harvest.** TFMA on the id-wire reveals τ for K token types → those tokens are identifiable in the victim stream.

## Phase 1 — attack (per victim request observed)
3. **Gather anchors.** For each victim position `i` whose *entire causal prefix* `0..i` is harvested,
   reconstruct the plaintext prefix (de-permute via τ), run the public model → `Xp_align[i]`; pair with the
   observed deployment rep `Xd_align[i]`. (Guard: a position with any non-harvested prefix token is unusable.)
4. **Estimate the rotation** `R̂` s.t. `Xp·R̂ ≈ Xd`:
   - global `orthogonal_procrustes_R`: `R̂ = U Vᵀ` from `svd(Xp_alignᵀ Xd_align)` — needs `n_align ≥ d=768`.
   - block `blockwise_procrustes_R`: per (plaintext head h, deployment head h') fit `O(64)` Procrustes;
     assign the head-permutation by min residual (Hungarian); assemble `R̂ = perm ∘ blkdiag(Q_h)` — needs `n_align ≥ 64`.
5. **Un-rotate**: `x̂ = Xd_te · R̂ᵀ` (R̂ orthogonal ⇒ R̂ᵀ = R̂⁻¹).
6. **Decode**: `nearest_token(x̂ · W, E[pool], pool)`.

## The one subtlety
Step 3 is the whole threat-model story. `kqv_out_i = Σ_{j≤i} a_ij V_j` is contextual, so `Xp_align[i]`
needs the *full* prefix — only **fully-harvested-prefix** positions yield honest anchors (position 0 always;
deeper only if every earlier token is harvested). `R̂` is context-independent, so any valid anchors suffice;
scarcity of honest anchors — not correctness — is the binding constraint.

## Worked example
d=4 toy, 1 head, secret `R` = 90° rotation. Harvest reveals "the" at position 0 of many prompts →
`Xp = kqv_out("the")` (no context, computable), `Xd` = observed. 4+ such pairs → `svd(XpᵀXd)` → `R̂ = R`.
Held-out "cat" leaks `Xd_cat`; `Xd_cat·R̂ᵀ ≈ Xp_cat` → `W` decodes → "cat". Never used the key.

## Complexity
Symbols: `d = 768` (model dim), `H = 12` heads, `hd = 64` head dim, `n_a` anchors, `n_tr` self-gen rows,
`n_te` test rows, `V_p = |pool|`.
- **Anchors** (step 3): one O(n_positions) prefix scan per prompt.
- **R̂ global** (step 4): `XpᵀXd` is `O(n_a·d²)`; SVD of `d×d` is `O(d³)`.
- **R̂ block** (step 4): `H²` head-pairs, each `O(n_a·hd² + hd³)`, + assignment `O(H³)` → `O(H²(n_a·hd²+hd³))`.
  With `H=12, hd=64` this is ~2–3 orders cheaper than one `d³` global SVD.
- **Self-gen ridge** (step 1): `O(n_tr·d² + d³)`.
- **Un-rotate + decode**: `O(n_te·d²)` + nearest-token `O(n_te·V_p·d)`.
- **Sample complexity (the point)**: global needs `n_a ≥ d = 768`; block needs `n_a ≥ hd ≈ 64` (each
  anchor supplies all H heads). The block estimator cuts the anchor requirement **~H× (768 → 64)** — decisive,
  because honest anchors are scarce.

## Limitations
- **Honest anchors are scarce** — fully-known-prefix positions are ~10× fewer than harvested-token
  occurrences (measured L0: 43–99 vs 441–869 at K=50–700); global Procrustes is infeasible honestly, only
  the block estimator clears its ~64 threshold (at K≳100).
- **Depth** — deeper `kqv_out` mixes longer contexts, so fully-known-prefix positions get rarer; the attack
  is expected to weaken with depth (untested; queued).
- **Noise cap** — αₑ noise (alg1/alg2@1.0) caps recovery independent of `R̂` (alg2@1.0 ≤ 0.22); the attack
  removes the rotation, not the noise. So the residual AloePri defense is the noise, not the basis.
- **Needs victim traffic** — deployment reps `Xd` come only from observed victim inference (the attacker
  can't run the deployment model); not a fully offline attack.
- **Head-perm recovery** from few anchors (min-residual assignment) is the fragile step.
- **`kq` surface** — rotary position-coupling breaks the single-fixed-`R` assumption; this attack is for
  the value surface `kqv_out` only.
- **Not a new primitive** — a known attack (see the claim). The contribution is the AloePri-Alg2 evaluation
  + block variant + honest analysis.
