---
type: plan
status: current
created: 2026-06-30
updated: 2026-06-30
tags: [aloepri, matched-vs-selfgen, invariance, threat-model, isa-attnscore, per-head-fingerprint]
companion: docs/handoffs/2026-06-30-attnvalue-pool-generalization.md
---

# Experiment Plan — "Matched is realistic iff the surface is invariant"

**Problem**: The matched-ridge inverter (FIG·02) reports high recovery, but it trains on
deployment-basis (rep, token) pairs with true labels. A keyless attacker can only obtain those pairs
by *self-generation* (running the PUBLIC plaintext model on its own prompts) — which works only if the
obfuscation leaves the read surface **invariant**. On a rotated surface, self-generation produces reps
in the wrong basis and fails, so the matched number there is an out-of-threat-model **ceiling**.

**Method thesis (one sentence)**: For each (surface × config), a *self-generated* ridge equals the
*matched* ridge iff the obfuscation leaves that surface invariant; otherwise self-gen collapses to the
shuffled-label floor while matched stays high — and the matched−self_gen gap is exactly the
"unrealistic advantage" the threat model forbids.

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| **C1** Matched is realistic iff surface invariant | Tells which reported recoveries are real keyless attacks vs ceilings | (surface×config) matrix where `self_gen ≈ matched` on invariant cells AND `self_gen ≈ floor` (matched ≫ floor) on rotated cells; 3 seeds | B1 |
| **C2** Invariance is per-(surface×config), not per-algorithm | Same config rotates one surface, leaves another invariant; Alg2 *flips* kqv_out | keymat: kqv_out invariant (gap≈0) but residual rotated (gap large); Alg2: kqv_out flips to rotated | B1 |
| **C3** (open) Per-head SVD fingerprint recovers Π_head even under Alg2 | Tests the user's "defeated by Alg2" assumption; Uvo is orthogonal → SVs invariant | Π_head recovery accuracy ≈ 1.0 under keymat/alg1 AND under Alg2 (falsifies "defeated") | B2 |

**Anti-claims to rule out (built into B1):**
- *"self_gen just is a weaker attack"* → ruled out by the invariant cells where `self_gen ≈ matched` (identical training procedure; only the rep *source* differs).
- *"the gap is the candidate pool"* (the prior FIG·01b/02 finding) → ruled out by using the **same pool** for matched and self_gen within each cell.
- *"self_gen collapse is undertraining"* → ruled out by reporting self_gen **in-set** recovery (must be high on plaintext / invariant cells).

## Invariance ground truth (a-priori; B1/B2 test against this)

| surface (read point) | keymat | alg1 (αₑ noise) | Alg2 |
|---|---|---|---|
| **residual** (ISA-HiddenState) | rotated `·P̂` | rotated `·P̂` + noise | rotated `·P̂` (Alg2 residual-inert) |
| **kqv_out** (ISA-AttnValue) | **invariant** | **invariant** + noise | **rotated** `Uvo` + head-perm |
| **kq** (ISA-AttnScore) | invariant | invariant + noise | ~invariant (head-perm only, matched-absorbable) |

Confirmed from `scripts/defenses/aloepri.py::_apply_alg2`: kqv_out = `einsum(Uvo, value)` per head, then `[head_perm]` → rotated. Prediction: `self_gen` recovery follows the **bold/invariant** cells; collapses on **rotated** cells.

## Experiment Blocks

### Block 1 — Matched vs Self-generated (MAIN, MUST-RUN)
- **Claim**: C1, C2.
- **Why**: the direct, decisive discriminator.
- **Data/split**: pythia-160m, `corpora/release-gate-512.txt`, 160 prompts, layer 0. **Primary split = rowsplit** (shared vocab) so any self_gen collapse is *basis mismatch*, not the (orthogonal) kqv_out generalization issue. Secondary: vocab-disjoint (appendix).
- **Compared systems** (per surface×config cell, same pool, same rows):
  - `matched` — `W = ridge(X_dep[tr] → emb[y[tr]])`, decode `X_dep[te]` (ceiling / FIG·02).
  - `self_gen` — `W = ridge(X_plain[tr] → emb[y[tr]])` (reps from the PLAINTEXT model), decode `X_dep[te]` (realistic, keyless).
  - `floor` — shuffled-label ridge.
- **Surfaces**: residual, kqv_out, kq. **Configs**: plaintext (control), keymat, alg1@1.0, Alg2@1.0.
- **Metrics**: top-1 token recovery (decisive); report `matched`, `self_gen`, `floor`, **gap = matched − self_gen**, and self_gen **in-set** recovery. 3 seeds, mean ± sd.
- **Success criterion (proves C1)**: gap ≤ ~0.05 on every predicted-invariant cell AND `self_gen − floor ≈ 0` with `matched − floor ≫ 0` on every predicted-rotated cell.
- **Failure interpretation**: self_gen collapses on a predicted-*invariant* cell → harness bug or self_gen not faithful; self_gen ≈ matched on a predicted-*rotated* cell → that surface isn't really secret-rotated (the attacker can self-gen anyway).
- **Figure**: grouped bars, x = surface×config ordered invariant→rotated, series = {matched, self_gen, floor}; the self_gen line tracking-then-dropping is the headline.

### Block 2 — Per-head fingerprint Q/K/V/O (MUST-RUN, structural)
- **Claim**: C3 (resolve the user's assumption).
- **Why**: a *keyless* attack on the weight surface; tests whether Alg2's head layer is actually defended.
- **Method**: per head h and projection P∈{q,k,v,o}, signature = sorted singular values of the obfuscated head-block `Ŵ_P[h]`; match each obf head to the nearest public-head signature (Hungarian) → recover Π_head. Metric = head-permutation recovery accuracy.
- **Configs**: keymat, alg1, Alg2.
- **Prediction (to test)**: ≈1.0 under keymat/alg1 (heads untouched); **likely still ≈1.0 under Alg2** for V/O (Uvo orthogonal → SVs invariant) — would *falsify* "defeated by Alg2". Q/K depend on the `Mq` transform — report separately.
- **Failure interpretation**: if Alg2 *does* drop it, Uvo is not a clean within-head orthogonal on that projection (check `_apply_alg2`).
- **Note**: this recovers the head *permutation*, not tokens; it does not by itself make kqv_out self-gen work under Alg2 (the within-head Uvo rotation remains). Keep separate from B1.

### Block 3 — αₑ noise robustness of self_gen on the invariant surface (NICE-TO-HAVE)
- **Claim**: supports C1 — on an invariant+noised surface (alg1 kqv_out), does self_gen degrade *like* matched (public mechanism, self-replicable noise) rather than collapsing?
- **Method**: αₑ ∈ {0, 0.5, 1.0, 2.0} on kqv_out; self_gen samples its own noise. Report self_gen vs matched vs αₑ.
- **Success**: self_gen tracks matched across αₑ (gap stays small) → noise is not a basis secret.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision gate | Cost | Risk |
|---|---|---|---|---|---|
| **M0** sanity | harness + self_gen training valid | plaintext config, all surfaces: `self_gen == matched`? self_gen in-set high? | self_gen ≈ matched on plaintext (else bug) | ~1 min | low |
| **M1** main | B1 matrix | 3 surfaces × 4 configs × 3 seeds, rowsplit | C1/C2 verdict | ~3–4 min | capture point for kqv_out must be post-Uvo |
| **M2** fingerprint | B2 | Q/K/V/O × 3 configs (CPU, weights only) | C3 verdict | <1 min | Mq effect on Q/K |
| **M3** polish | B3 + disjoint split | αₑ sweep kqv_out; disjoint appendix | — | ~2 min | low |

## Compute and Data Budget
- pythia-160m, 160 prompts, L0. Capture = plaintext + 3 obf configs = 4 forward passes; ridge/SVD trivial. **Total < 10 min, single iGPU.** Perf-gate (`scripts/harness/perf_gate.md`) before M1.
- Reuses: `aloepri_score_surface_sweep` capture (already captures all configs with shared token ids → `X_plain` and `X_dep` come for free), `defenses.aloepri.reparam_pythia`, `talens.attacks.dp_inversion.{ridge_W, nearest_token}`. Residual capture via `dp_leakage_sweep._stack`/`resid_capture`.

## Risks and Mitigations
- **kqv_out capture point**: must capture the post-Uvo `attention.dense` *input* (it already does via the dense pre-hook) so Alg2 actually rotates the captured tensor. Verify in M0 (Alg2 kqv_out ≠ plaintext kqv_out elementwise).
- **self_gen "collapse" not reaching floor**: random basis overlap could give >floor. Mitigate with the floor baseline + 3 seeds; judge by `self_gen − floor`, not absolute.
- **αₑ noise realization**: self_gen uses its own noise draw; on alg1 cells self_gen should still ≈ matched (public mechanism). If not, that itself is a finding.

## Final Checklist
- [ ] Main figure (B1 grouped bars) covers C1+C2
- [ ] Anti-claims (weaker-attack / pool / undertraining) controlled in B1
- [ ] C3 fingerprint resolves "defeated by Alg2?" with a measured number
- [ ] Same pool for matched vs self_gen within each cell
- [ ] MUST-RUN (B1, B2) separated from NICE-TO-HAVE (B3)
