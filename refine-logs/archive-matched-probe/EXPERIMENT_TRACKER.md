---
type: reference
status: current
created: 2026-06-20
updated: 2026-06-20
tags: [experiment-tracker, checklist]
companion: [EXPERIMENT_PLAN]
---

# Experiment Tracker

Status: ☐ todo · ◐ in-progress · ☑ done · ✗ failed/blocked

## B0 — Implement + unit-test (MUST-RUN, <20 min CPU) — ☑ DONE
- ☑ AloePri Alg1 obf-table generator (`keymat_gen`, `P̂Q̂=I` @ d=2304, err 6e-9) → obf `WeightPair`
- ☑ AloePri permutation-core + keymat activation covers (`AloePriPermCover`, `AloePriKeyMatCover`)
- ☑ Shredder static-Laplace cover + learned-noise trainer (SNR-sign fixed per review)
- ◐ Capacity-reader-on-invariants Π-probe — degenerate on weight surface (1 row=1 class); deferred to activation surface (B4)
- ☑ PID QK-vs-OV probe (`pid_mmi`; reframed to operational reader-atoms + cond-increments per review)
- ☑ Unit tests green: 16 new, **76/76 suite**
- ☑ Cross-model code review (gpt-5.5 xhigh): 2 CRITICAL caught + fixed (Shredder sign, PID soundness)
- **Gate:** PASSED → B1/B2

## B1 — Per-channel matched calibration, cheap defences (MUST-RUN, 20–40 min ROCm)
- ☐ ch1 capacity-PVI reader-acc vs ridge TTRSR (DP-ε + split-depth) — expect ρ≥0.9 (prior: 0.82–1.0)
- ☐ ch3 CLUB vs ridge cosine — expect ρ≥0.85
- ☐ partial-ρ|knob + per-instance collinearity logged
- **Gate (C1, ch1/3):** both meet threshold

## B2+ — Π firm-up (5 seeds × 12 α × 2 widths) — ☑ DONE (CPU, 2026-06-21)
- ☑ per-seed ρ(CLUB-on-φ, VMA) = **+1.000 ± 0.000** (gemma d2304 + qwen d2560)
- ☑ match-mode independence: ρ(CLUB, VMA-nn) = +0.998 (not tied to assignment algo)
- ☑ B4 "seed variance" = pooling artifact (raw-magnitude offsets); within-sweep ρ=1
- ☑ **CLUB nan FIXED** (`club.py` grad-clip + None-guard; `test_club_stability.py`; 78/78)

## B2 — Π bake-off + AloePri α_e sweep → select Π-probe (MUST-RUN) — ☑ DONE (CPU, ~9s)
- ☑ AloePri-obfuscated WeightPair, α_e ∈ {0,.1,.2,.35,.5,.75,1.0,1.5}; VMA τ-recovery truth (1.00→0.007)
- ☑ CLUB-on-φ (independent): ρ = **+0.976**
- ☑ retrieval-PVI-on-φ (dependent ref): ρ = +1.000 (mechanical)
- ◐ capacity-reader-on-φ: degenerate on weight surface → activation surface (B4)
- ☑ keymat point: VMA→0.000, CLUB/retr→floor (keymat defeats RowSort φ-channel)
- **Gate (C4): RESOLVED → CLUB-on-φ is the independent Π-probe** (weight surface)

## B3 — Decoupling matrix (MUST-RUN) — ☑ DONE (GPU, ~13 min, 72 settings, 3 seeds)
- ☑ K×K ρ(P_c, m_{c′}) over {token-id, Π, embedding} (`results/b3_decoupling_matrix.json`)
- ☑ diagonal-dominance Δ_i + bootstrap CIs: **2/3 dominant** (token +0.087✓, Π +0.162✓; embedding tie)
- ☑ controls: random≈0, shuffled≈0, retr-PVI dep +0.885; **monotone-index confound demonstrated (−0.73/−0.75/−0.99)**
- ☑ **sign-flip: token-id @ L12 (ρ −0.108)**; per-layer depth axis carries the decoupling
- **Gate (C2): PARTIALLY MET** — depth-resolved channel-specificity holds; pooled matrix confounded by common-cause noise → needs 2nd defence family (B4)

## B6 — Framing verdict (MUST-RUN, analysis)
- ☐ apply decision rule → F-A / F-B / F-C
- ☐ lock Π-probe
- ☐ (conditional) C1′ formal argument if any channel's matching failed
- **→ run `/auto-review-loop` here (paste matrix tables inline)**

## B4 — Cross-scheme calibration (Shredder vs input-DP) — ☑ DONE (GPU, 2026-06-21)
- ☑ Shredder static-Laplace as 2nd family (post-capture transform; `results/b4_cross_scheme.json`)
- ☑ Shredder matrix: token-id + perm_Π diagonals dominate (2/3); embedding CLUB generic (same as B3)
- ☑ **Finding 1**: decoupling is **defence-injection-specific** — token-id per-layer diagonal differs
  entirely (DP propagated +0.89→+0.08 vs Shredder direct +0.16→+0.62)
- ☑ **Finding 2 (C3)**: transfer is channel-specific — embedding transfers (0.75/0.70/0.72 pooled≈within),
  token-id does NOT (0.64/0.39/0.45 pooled<within); perm_Π high seed variance (flag → B2+)
- ⚠ instability: seed-1 CLUB → nan (20/72); finite-cell estimates stable. Fix: clamp/retry in `club`.
- **Gate (C3): partially met + sharpened** — no single scheme-agnostic leakage scalar; calibration curve = f(channel, injection-geometry)

## B5 — Attention QK/OV PID (NICE-TO-HAVE, 30–60 min)
- ☐ PID unique-QK vs kq-leakage; unique-OV vs kqv_out-leakage
- ☐ shared-rotation cover-invariance lemma check (I(tokens;QK) & ISA unchanged)

## Milestones
- ☐ M0 (B0) ☐ M1 (B1) ☐ M2 (B2) ☐ M3 (B3) ☐ M4 (B6 verdict) ☐ M5 (B4) ☐ M6 (B5)
