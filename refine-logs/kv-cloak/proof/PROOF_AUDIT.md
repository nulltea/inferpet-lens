# Proof Audit — KV-CLOAK channel decoupling (L1-L4)

**Reviewer**: gpt-5.5 xhigh (cross-model), thread 019ef73c-ac1b-7391-846c-7771850ac970, 3 rounds.
**Verdict**: PASS (zero open FATAL/CRITICAL). Claim: research-wiki/claims/kv-cloak-channel-decoupling-feature-mix-loadbearing.md

## Round 1 — FAIL
Two GLOBAL defects: (I-06 INVALID) the per-target Jensen bound E‖P_W k̂_j‖≤√(s/d) was over-lifted to
the p95/Hungarian metric without an order-statistic argument; (I-07 OVERSTATED) √(s/d) wrongly
identified with the random-demixing floor (floor is ~1/√d, below the oracle ceiling). Plus LOCAL:
S not stated orthogonal (I-02), targets not unit-normalized (I-03), L3 b-independence overstated as
pure theory (I-04), probe invariance over-attributed to S not just P̂ (I-05), edge cases (I-10/12).

## Round 2 — WARN
Reframed L4: ρ ≤ max_j‖P_W k̂_j‖; Bernstein-type Beta tail + union bound. Remaining: tail not stated
explicitly (R-01), (1+o(1)) needed log n=o(s) not log b=o(d) (R-02), s-vs-b target count (R-03),
floor an order statistic not Θ (R-04), "squeezed into band" wrong — realized attack can score below
floor (R-05).

## Round 3 — PASS
Stated the Bernstein tail Pr(‖P_W k̂_j‖²−s/d ≥ t) ≤ exp(−c min(d²t²/s, dt)); union over n≤s targets →
E_M max ‖·‖² ≤ s/d + C(√(s log n)/d + log n/d); Jensen → E_M ρ = O(√(s/d)+√(log n/d)), O(√(s/d)) when
log n=O(s). Floor restated as order statistic O(√(log(ns)/d)) below the ceiling, not a lower bound.
Reviewer: "No remaining FATAL/CRITICAL; the lift is rigorous." Cosmetic cleanups (n≤s, C-S
zero-projection, claim mirrors proof asymptotics) applied after.

## Status of lemmas
- L1 (M Gram-invariant): exact. PASS.
- L2 (S·P̂ orthogonal similarity ⇒ spectrum invariant; permutation preserves symmetric functionals): exact. PASS.
- L3 (S·P̂ preserves row space ⇒ oracle recovery + chance floor invariant for fixed K; cross-b operator-level + empirical): PASS.
- L4 (secret Haar M caps E_M ρ at O(√(s/d)+√(log n/d)) → chance for d/s→∞): PASS as the security-relevant upper bound. Conditional on 0<s<d, s≪d.
