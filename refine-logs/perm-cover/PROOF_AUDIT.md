# Proof Audit — perm-llr-threshold

**Date**: 2026-06-23 · **Reviewer**: Codex gpt-5.5 xhigh (read-only) · thread `019ef653-ad3d-7940-bc96-bd7d3cb67e51`
**Verdict**: **PASS** (zero FATAL/CRITICAL; four MINOR wording conditions applied)

## What was verified
- **Lemma 1 (DPI domination, unconditional)** — `I(φ(X);φ(Y)) ≤ I(s(X);s(Y))` via two DPI
  applications (φ=Q∘s deterministic). DPI direction correct; holds at every σ. ✅
- **Lemma 2 (per-row channel)** — (i) sorted vector = maximal invariant of the Sᵈ action; (ii) Pe
  stays isotropic since P orthogonal; (iii) `min_P‖y−Px‖² = ‖s(y)−s(x)‖²` via the rearrangement
  inequality (direction and norm-preservation algebra confirmed); (iv) one-to-one assignment =
  Hungarian. Conclusion holds **as a profile-MLE / joint-MAP result for the per-row channel**. ✅
- **Remark (shared-P scope)** — confirmed correct: per-row sort is Sᵈ-invariant and DPI-dominates
  RowSort but is **not** the joint maximal invariant under a shared P; `YYᵀ = Π(X+E)(X+E)ᵀΠᵀ` is
  genuinely shared-P invariant because `PᵀP=I`, and per-row sorting discards it. ✅

## Issues (all MINOR, all fixed)
| id | category | fix |
|---|---|---|
| MAP-01 | probability mode | Labeled profile-MLE = **joint MAP over (Π,{Pᵢ})**, explicitly **not** marginal MAP over Π (which would log-sum-exp over nuisance permutations). |
| TIE-01 | edge case | "an optimizer"; coordinate ties make the optimizing permutation non-unique, value identity unaffected. |
| DPI-EQ-01 | DPI equality | Q-injectivity stated as **sufficient**, not necessary, for equality. |
| THR-01 | external threshold scope | 2 log n scoped to DCK arXiv:1903.01422 (achievability `I ≥ 2 log n + ω(1)`); **not** implied by L1/L2, not transferring to quantized φ or the shared-P channel. |
| COPY-01 | copy consistency | reviewer could not read files; `PROOF_PACKAGE.md` and the claim's inline Theory section synced by the executor after the fixes. |

## Scope honesty (what is and is not proven)
- **Proven:** L1 unconditionally; L2's profile-MLE/joint-MAP optimality for the **per-row**
  column-permutation isotropic-Gaussian channel.
- **Not proven (explicit):** exact joint-MAP optimality under the **shared** AloePri permutation
  (per-row sort is a sound, DPI-dominating relaxation there); the exact `2 log n` constant for the
  sorted-feature or shared-P channel (cited as an external benchmark only).

Acceptance gate: zero open FATAL/CRITICAL → **PASS**.
