---
type: experiment
node_id: exp:aloepri-kqvout-rotation-recovery
title: "Known-plaintext rotation-recovery defeats AloePri Alg2's kqv_out value rotation keyless (L0); honest anchors need the per-head block Procrustes"
idea_id: "idea:matched-probe-program"
verdict: supported
confidence: medium
date: "2026-07-01"
hardware: "AMD Strix Halo iGPU (gfx1151), host .venv"
duration: "~10 min (2 pilots, 3 seeds, L0)"
provenance: "scripts/evals/static_obf/aloepri_basis_align_pilot.py + aloepri_basis_align_honest.py; refine-logs/matched-invariance/{basis_align_pilot,basis_align_honest}.json; src/talens/attacks/dp_inversion.py (rotation_recovery_attack, orthogonal_procrustes_R, blockwise_procrustes_R)"
added: 2026-07-01
tags: [aloepri, isa-attnvalue, kqv_out, known-plaintext, orthogonal-procrustes, rotation-recovery, self-gen, threat-model]
companion: research-wiki/claims/aloepri-kqvout-basis-alignment.md
---

# Rotation-recovery (known-plaintext) attack on kqv_out under Alg2 — L0

Tests [[claim:aloepri-kqvout-basis-alignment]]: a keyless attacker recovers AloePri Alg2's secret per-head
value rotation from a harvest (orthogonal Procrustes on aligned anchor pairs), un-rotates, then decodes
with a self-generated inverter. Attack is KNOWN (see the claim's prior-art section); this is the AloePri-Alg2
evaluation. pythia-160m, release-gate-512 (160 prompts), L0, 3 seeds.

## Result — pilot (optimistic alignment: all harvested-token rows)
| config | K=0 (self-gen floor) | K≥50 | R̂ rel-err |
|---|---|---|---|
| alg2@0 (pure rotation) | 0.010 | **0.646** (= invariant ceiling) | 0.0 |
| alg2@1.0 (rotation + αₑ noise) | 0.003 | 0.15 → 0.22 | ~0.82 |
| keymat (control, R=I) | 0.646 | 0.646 | 0.0 |
Pure rotation is recovered exactly once aligned rows ≥ d=768 → recovery jumps to the invariant ceiling.
αₑ noise caps recovery independent of R̂ (the residual defense is the noise, not the rotation).

## Result — honest alignment (leading fully-known-prefix victim positions only)
| K | honest n_align (optimistic) | global O(768) | per-head block |
|---|---|---|---|
| 50 | 43 (441) | NA (<768) | NA (<64) |
| 100 | 47 (514) | NA | 0.589 |
| 300 | 61 (670) | NA | 0.589 |
| 700 | 99 (869) | NA | 0.634 |
Honest anchors are ~10× scarcer and never reach 768 → global Procrustes is infeasible honestly (the
pilot's global success was the optimistic overcount). But R is block-structured, so per-head O(64)
Procrustes + head-perm assignment needs only ~64 anchors and reaches 0.589–0.634 (~97% of the 0.646
ceiling) at K≳100.

## Verdict
Alg2's per-head value rotation provides ≈0 information-theoretic defense on kqv_out against a
threat-model-respecting keyless attacker with a harvest: recovered from a few dozen fully-known-prefix
anchors via the block estimator, lifting recovery from ~0.01 to ≈0.6. Only αₑ noise caps recovery
(alg2@1.0 ≤ 0.22). Overturns the naive-cascade FIG·01b reading (ISA-AttnValue ≈ 0). Queued: depth sweep
(honest anchors shrink as prefixes lengthen), a body-read of arXiv:2606.16461/2603.01499 for prior attack
discussion.
