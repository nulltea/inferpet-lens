---
type: claim
node_id: claim:aloepri-kqvout-basis-alignment
name: "Harvest-aligned basis recovery makes the keyless kqv_out attack strong under Alg2, by spending the harvest on the low-dim secret rotation, not the token map"
description: "A modest TFMA harvest of K token types, used to estimate AloePri Alg2's context-independent per-head value rotation (not to learn tokens), lets an unlimited self-generated inverter recover held-out tokens on kqv_out under Alg2 — lifting recovery from the shuffled-label floor toward the invariant-surface ceiling as K grows"
node_type: claim
status: proposed
provenance: "devised from §05 matched-vs-selfgen (docs/html/static-obf.html) + FIG·01b partial-τ cascade; L0 pilot pending"
tags: [aloepri, isa-attnvalue, kqv_out, basis-alignment, orthogonal-procrustes, self-gen, harvest, threat-model, weights-pub, proof-todo]
date: 2026-07-01
added: 2026-07-01T00:00:00Z
companion: docs/html/static-obf.html
supersedes_reading: "FIG·01b ISA-AttnValue ≈ 0 (naive cascade) reads kqv_out as harvest-resistant"
---

# Harvest-aligned basis recovery for kqv_out under Alg2

**status:** `supported (L0)` — core mechanism confirmed by the L0 pilot
(`scripts/evals/static_obf/aloepri_basis_align_pilot.py` → `refine-logs/matched-invariance/basis_align_pilot.json`,
pythia-160m, 3 seeds). Still to firm up: honest fully-known-prefix alignment, depth sweep, and the
finer sub-threshold K-curve. Not self-certified beyond the pilot's scope.

## Pilot result (L0, kqv_out)

| config | K=0 (naive self-gen floor) | K≥50 harvested types | `R̂` rel-err |
|---|---|---|---|
| **alg2@0** (pure rotation, no noise) | 0.010 | **0.646** (= invariant ceiling) | **0.0** |
| **alg2@1.0** (rotation + αₑ=1.0 noise) | 0.003 | 0.15 → 0.22 (rising in K) | ~0.82 |
| keymat (control, R = I) | 0.646 | 0.646 | 0.0 |

**Confirmed.** On the pure Alg2 rotation, a harvest of just K=50 types (→ >768 = d aligned rows) recovers
`R̂` **exactly** (rel-err 0.0) and basis-alignment jumps to the **invariant-surface ceiling (0.646)** — the
secret rotation is fully defeated keylessly. The K-lift is a threshold in the number of aligned rows
(orthogonal `O(d)` Procrustes is exactly determined at n_align ≥ d; K controls how many aligned rows the
harvest yields). Under alg2@1.0 the rotation is still partially removed (0.003 → 0.22) but the **αₑ noise
caps recovery** (rel-err stays ~0.82) — the noise, not the rotation, is the residual defense, exactly as
the limits below predicted. Control clean (R = I ⇒ R̂ = I, K-independent, 0.646).

## Honest-alignment result (L0, alg2@0) — corrects the pilot's magnitude

The pilot aligned on ALL harvested-token rows (optimistic: assumes the attacker can compute the plaintext
rep at any harvested position). The faithful attacker forms a pair only at a position whose ENTIRE causal
prefix is harvested. Redone honestly (`aloepri_basis_align_honest.py` → `basis_align_honest.json`, 3 seeds,
alignment restricted to leading fully-harvested-prefix victim positions):

| K | honest n_align (optimistic) | global O(768) Procrustes | per-head block Procrustes |
|---|---|---|---|
| 50 | 43 (441) | NA (< 768) | NA (< 64) |
| 100 | 47 (514) | NA | **0.589** |
| 300 | 61 (670) | NA | **0.589** |
| 700 | 99 (869) | NA | **0.634** |

**Corrected findings.** (1) Honest aligned pairs are **~10× scarcer** than the pilot's count (43–99 vs
441–869) and never reach d=768 across ~80 victim prompts, so the **global O(768) Procrustes is infeasible
honestly** — the pilot's "K=50 → exact" was an artifact of the optimistic overcount. (2) But `R` is
block-structured (`perm · blkdiag(Û_vo)`), so a **per-head O(64) Procrustes + head-permutation assignment
needs only ~64 aligned rows**; with ~64–99 honest pairs it recovers the rotation and reaches **0.589–0.634,
≈ 97% of the 0.646 ceiling**. (3) So the attack **survives honestly** — Alg2's rotation is still defeated
keyless — but the realistic recipe is: exploit the block structure (global is out), and gather ≥ 64
fully-known-prefix victim positions (reached at K ≳ 100 here; more victim traffic lowers the K needed).
Honest n_align grows with K only slowly, bottlenecked by how often victim prompts begin with long runs of
harvested tokens.

**Takeaway.** Alg2's per-head value rotation provides ~no *information-theoretic* defense against a
harvest-equipped keyless attacker on kqv_out: with the block-structured estimator it is recovered from a
few dozen fully-known-prefix labels, lifting recovery from the floor (~0.01) to ≈ 0.6 (near the invariant
ceiling). Only the αₑ noise caps recovery (alg2@1.0 stays ≤ 0.22). This overturns the FIG·01b naive-cascade
reading (ISA-AttnValue ≈ 0) for the rotation and matches the report-wide thesis: the single
information-theoretic lever is noise, not the basis obfuscation. Queued: depth sweep (honest n_align
shrinks with depth as prefixes lengthen), and promoting `blockwise_procrustes_R` into
`talens.attacks.dp_inversion` if the attack graduates to a FIG·01b recovery-vs-K subplot.

## Setting

AloePri Algorithm 2 obfuscates the per-head attention value output `kqv_out` by a **secret, fixed,
context-independent block-orthogonal map** `R` (per-head `Û_vo ∈ O(64)` + a 12-way head permutation),
baked into the weights: `deployment_kqv_out = plaintext_kqv_out · R` for every position and context.
Threat model (WEIGHTS-PUB, Kerckhoffs): the attacker has the public plaintext model and the obfuscation
algorithm, observes deployment tensors, holds **no key** (`Û_vo`/head-perm seed) and **no token
permutation τ**. A TFMA wire attack legitimately reveals τ for **K token types** (the harvest).

Two prior readings, both spending the harvest wrong for this surface:
- **§05 (self-gen)**: the keyless self-generated inverter recovers kqv_out where the surface is invariant
  (keymat/alg1) but **collapses to the shuffled-label floor under Alg2** — it cannot reproduce the secret
  basis `R`.
- **FIG·01b (naive cascade)**: a ridge trained on the K harvested *rows* reads ≈ 0 — kqv_out is a
  contextual mixture, so the map memorizes the seen tokens and does not generalize to held-out types.

## The claim (insight: separate the two unknowns)

Recovering kqv_out under Alg2 has **two independent unknowns**, and the naive attacks conflate them:
1. **The secret rotation `R`** — low-dimensional (12 orthogonal 64×64 blocks + a permutation),
   **context-independent**, and a Kerckhoffs-known *structure* (only its value is secret).
2. **The token map** (rep → token) — needs all token types, but is **not secret**: the attacker can
   generate it for free in the plaintext basis (§05).

**Claim.** Spend the harvest on **(1) only**. K harvested pairs suffice to *estimate `R`* (few samples,
because `R` is orthogonal and block-structured — far fewer DOF than a token map). Then **(2)** is solved
by unlimited self-generation, exactly as in §05's invariant case. Concretely:

### The attack — harvest-aligned basis recovery (`basis_align_attack`)
1. **Aligned pairs.** For victim positions whose causal prefix is entirely harvested (trivially, a
   harvested token at a self-attending position — position 0, or self-dominated attention — needs *no*
   context), run the public plaintext model to get `plaintext_kqv_out_i`; pair with the observed
   `deployment_kqv_out_i`.
2. **Estimate `R`** by orthogonal Procrustes: `R̂ = argmin_{Q∈O(d)} ‖P Q − D‖`, closed-form
   `R̂ = U Vᵀ` from `svd(Pᵀ D)` (block-structured variant: per-head `O(64)` + head-perm from the residual
   match). `R` orthogonal ⇒ few aligned samples pin it.
3. **Un-rotate** all deployment reps: `D · R̂ᵀ → ` (estimated) plaintext basis.
4. **Decode** the un-rotated held-out positions with the self-generated inverter (ridge trained on the
   attacker's unlimited own `(plaintext_kqv_out, token)` pairs, §05).

## Why K makes it stronger (prediction)

K drives only the **number of aligned pairs** (fully-known-prefix / self-attending positions), hence
`R̂`'s accuracy. Prediction: as K grows, held-out recovery under Alg2 rises **monotonically from the
shuffled-label floor (~0.02) toward the invariant-surface ceiling (~0.55 at L0)**; once K supplies
≳ (per-head rank) clean aligned pairs, `R̂ ≈ R` and Alg2's rotation is effectively removed. This is the
FIG·01b ISA-AttnValue-vs-K curve the naive cascade flatlined at ~0. Control: at keymat `R = I`, so `R̂ ≈ I`
and recovery is K-independent at the invariant level (self-gen already suffices).

## Threat-model compliance

No key, no oracle, no paired data beyond the legitimate TFMA harvest. Uses (a) the public model, (b) the
harvest, (c) the Kerckhoffs-known *structure* of `R` (block-orthogonal, context-independent). The harvest
is spent on the secret's low-dimensional part; self-generation (keyless) supplies the token map. Strictly
stronger than the naive cascade and strictly keyless — it is the realistic attacker of §05 augmented with
exactly the one thing it lacked under Alg2 (the rotation), obtained legitimately.

## Limits (to test, not assume)

- **Depth**: deeper `kqv_out` mixes long contexts, so fully-known-prefix positions become rare and `R̂`
  degrades — the K-lift likely shrinks with depth (consistent with L0 being the strong surface).
- **Noise floor**: αₑ (alg1+alg2) caps recovery independent of `R̂`; alignment removes the rotation, not
  the noise. So the ceiling under alg2@1.0 is the αₑ-noised self-gen level, not plaintext.
- **Rotary/position coupling** makes the score surface `kq` rotation position-dependent (not a single
  `R`); this trick is cleanest on `kqv_out` (value axis, no rotary).
- **Head-permutation recovery** from few pairs is the fragile step; global `O(d)` Procrustes absorbs it
  but needs ~d aligned samples, while the block-structured variant needs fewer — validate which.
- **Honest alignment**: the faithful attacker aligns only on fully-known-prefix positions; a pilot that
  aligns on all harvested-token rows is an **optimistic ceiling** on `R̂` quality (flag in results).

## Relationships

- Extends [[cross-surface-matched-probe-tracks-recovery]] / the §05 matched-vs-selfgen finding
  (self-gen collapses under Alg2) — this claim supplies the missing rotation from the harvest.
- Refines the FIG·01b partial-τ reading (see [[aloepri-partial-tau-bootstrap]]): ISA-AttnValue ≈ 0 is the
  *naive* cascade; the harvest CAN bootstrap kqv_out once spent on `R`, not the token map.
- Subsumes the deferred per-head-fingerprint (Π_head) attack: the head permutation is part of `R` and is
  recovered jointly by the Procrustes step.
