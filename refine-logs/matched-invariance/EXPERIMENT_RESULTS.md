# Initial Experiment Results — matched vs self-generated invariance

**Date**: 2026-06-30
**Plan**: refine-logs/matched-invariance/EXPERIMENT_PLAN.md
**Data**: refine-logs/matched-invariance/matched_vs_selfgen.json (pythia-160m, release-gate-512, 160 prompts, L0, 3 seeds)

## Results by Milestone

### M0 — Sanity (plaintext control): PASSED
On plaintext, `matched == self_gen` exactly for every surface (residual 0.993, kqv_out 0.547, kq 0.269),
floor ≈ chance (0.02–0.04). Harness validated: when the surface is untouched the two attackers are
identical, and `self_inset` confirms the self_gen map trains correctly.

### M1 — Matched vs Self-generated (C1/C2): PASSED
matched (M) / self_gen (S), mean over 3 seeds; floor ≈ 0.015–0.030:

| surface | plaintext | keymat | alg1 (αₑ=1.0) | alg2 (αₑ=1.0) |
|---|---|---|---|---|
| residual (768→1024) | M .993 / S .993 | M .993 / **incompat** | M .985 / **incompat** | M .985 / **incompat** |
| kqv_out (768) | M .547 / S .547 | M .547 / **S .547** | M .478 / S .231 | M .478 / **S .004** |
| kq (192) | M .269 / S .269 | M .269 / **S .269** | M .283 / S .193 | M .283 / **S .022** |

## Verdict

- **C1 — matched is realistic iff the surface is invariant: PROVEN.**
  - Invariant cells (kqv_out/kq under keymat): self_gen = matched, **gap = 0.000**.
  - Rotated cells (kqv_out/kq under alg2): self_gen → floor (0.004 / 0.022) while matched stays high
    (0.478 / 0.283) → matched there is an out-of-threat-model **ceiling**.
- **C2 — invariance is per-(surface × config), not per-algorithm: PROVEN.**
  - keymat leaves kqv_out/kq invariant (self_gen=matched) but the **residual is dim-incompatible**
    (P̂ widens 768→1024) → a keyless self-gen attacker cannot even apply its map; matched is a ceiling.
  - alg2 **flips** kqv_out/kq from invariant to rotated (secret Uvo rotation + head perm).
- **Anti-claims ruled out**: (weaker-attack) refuted — self_gen = matched on invariant cells, same
  procedure; (pool effect) refuted — same pool for matched/self_gen within each cell; (undertraining)
  refuted — `self_inset` high.

## Bonus finding — αₑ noise is a third regime
alg1 (αₑ=1.0) is "invariant basis + secret noise": self_gen degrades (kqv_out 0.547→0.231, kq 0.269→0.193)
but stays **well above floor**. Only the secret noise *realization* breaks exact transfer, not the basis.
αₑ is a public mechanism, so a **noise-aware self_gen** (samples its own αₑ; queued as B3) should recover
most of the gap — to be tested. Contrast alg2 (true basis rotation) where self_gen → floor regardless.

## a-priori correction
kq was predicted "~invariant under alg2 (head-perm matched-absorbable)". The matched ridge *does* absorb
the head-perm (0.283), but the **keyless self_gen cannot** (it doesn't know the permutation) → collapses to
0.022. So kq is rotated-for-the-keyless-attacker under alg2, same as kqv_out. Matched-absorbable ≠
self-gen-recoverable — itself a clean illustration of the thesis.

## Deferred
- **B2 (per-head fingerprint Q/K/V/O)** — NOT run. Design flaw found: the shared residual rewrite Q̂ᵀ is
  non-orthogonal and applied to every head, so a naive SVD signature won't match the public heads even
  under keymat; and under keymat/alg1 there is no head-perm to recover (identity — Π_head exists only
  under alg2). Needs a genuinely Q̂ᵀ-invariant fingerprint before implementing. Flagged, not shipped.
- **B3 (noise-aware self_gen αₑ sweep)** — NICE-TO-HAVE; would test whether the alg1 intermediate regime
  closes when self_gen models the public noise.
- **B1 disjoint split** — appendix (orthogonal generalization axis).

## Ops note (blocker hit + fix)
Host `.venv` lacked `transformers` (it lived only in the retired container). Installed transformers 5.12.1
+ accelerate 1.14.0 via uv. **accelerate pulled torch 2.12.1+cu130, clobbering the `.pth` rocm torch**
(uv's resolver does not see the `.pth` torch as installed). Fixed by `pip uninstall torch` → the shared
rocm torch resurfaced (2.10.0+rocm, cuda True). **The `.pth` torch is fragile under any uv install that
pulls torch**; the robust fix is the wheel/uv-cache approach (`~/scripts/install_rocm_torch.sh`) so torch
is a tracked package. Recommend switching `.venv` to the wheel before more dep installs.

## Ready for next step
Main claim defended (C1+C2, 3 seeds). → figure for the report + `/result-to-claim` to promote to a wiki
claim. B2 needs a design pass; B3 is a cheap firm-up.
