---
type: dev-log
status: current
created: 2026-06-22
updated: 2026-06-22
tags: [experiment-results, shaped-noise, geometry-diagnostic, gate, pivot]
companion: [EXPERIMENT_PLAN, EXPERIMENT_TRACKER]
---

# Experiment Results — shaped-noise study

**Date**: 2026-06-22 · **Plan**: `refine-logs/EXPERIMENT_PLAN.md`

## M-1 / B(-1) — geometry diagnostic (the go/no-go gate): GATE = GO-UNSTABLE → **PIVOT-to-narrow**

**Script** `scripts/spikes/anisotropic_geometry_diagnostic.py` · **result** `results/anisotropic_geometry_diagnostic.json`
· geometry-only (no labels/attack), gemma-2-2b pool K=2048, d=2304, C_raw=4.147. Codex sanity audit (inline-verbatim,
thread 019ef0ae): **VERDICT yes** — supports the gate; no fatal artifact; mark complete with verdict PIVOT-to-narrow.

```
                            eff_rank(S)  top10-mass  top40-pair-mass  edge_eff#   eigenspace-stab(cosθ)
 ε=128  r=1.82  σ=0.157        7.1        0.814        0.986            5.9        128↔64 : 0.240
 ε= 64  r=3.63  σ=0.314      341.1        0.093        0.003        669 786        64↔32  : 0.959
 ε= 32  r=7.27  σ=0.628      198.7        0.115        0.000      1 938 900        32↔16  : 0.992
 ε= 16  r=14.5  σ=1.256      147.7        0.129        0.000      2 081 628
 span_rank = 2047 / 2304   ·   nullspace_trace_frac = 0.1115 (isotropic's wasted ~11%)
```

### Finding (the headline did NOT survive the gate)
The error-weighted pairwise scatter `S` (which sets where shaped noise should go) is **strongly regime-dependent**:
- **Low noise (ε=128, r=1.8 — the regime where BNN is near-perfect and the attack actually matters):** `S` is
  **spiky** — effective rank ≈ **7**, ~6 morphological-twin pairs carry **98.6%** of the scatter, touching only **~3%**
  of pool tokens. The dominant pairs are exactly the case/space/script twins (`'Hardware'~' Hardware'`, `' pasta'~' Pasta'`,
  `' six'~' seven'`, plus rare-script glyphs).
- **High noise (ε≤64):** `S` broadens (eff-rank 148–341) — but that is the utility-destroyed regime (the cliff), so it's
  irrelevant to a useful defense.
- **The low- and high-noise optimal shapes are nearly orthogonal** (top-10 eigenspace cosθ = **0.24**): there is no single
  stable anisotropic structure to exploit.

### Consequence (gate decision)
The strong premise — *"anisotropic noise broadly beats isotropic against BNN@L0 at smaller budget"* — is **not supported**
for this codebook in the attack-relevant regime. Shaped noise has only two levers here:
1. **trivial**: remove the ~11% of isotropic variance wasted in the token-irrelevant nullspace (`iso_full → iso_inspan`);
2. **narrow**: target the handful of morphological-twin directions that dominate low-noise `S` — protecting ~3–6% of
   tokens (those with a close twin), regime-specifically.

Per the plan's pre-registered pivot clause (M-1 → if pivot), the broad **B0 theorem and M2 broad-dominance headline are
NOT pursued**. The honest study is the narrow one: *quantify* the nullspace-removal + morphological-targeting win and show
it is small and concentrated — which also explains, geometrically, why "smarter noise" cannot broadly rescue local-DP
against the embedding-distance attack (consistent with the metric-DP "local-deniability-only" framing).

### Scope / caveats (from the sanity audit)
- Numbers are pool-local (K=2048); full-vocab or multi-pool replication would strengthen a paper but is unnecessary to
  reject GO-BROAD. The σ-trend and the 0.24 eigenspace flip are **physical** (the exponential distance kernel; different
  graph Laplacians), not numerical (weights are log-rescaled, scale-invariant metrics). `tok_cov` is σ-invariant by
  construction (w monotone in ‖Δ‖). Geometry-only — no ground-truth/attack leakage.
- This gate rules out the broad premise; it does **not** itself measure downstream efficacy (that would need the
  attack+utility runs, only worth doing in the narrow framing).

## Status
M-1 complete (sanity PASS). Gate = **PIVOT-to-narrow** → study direction decision required before M0/M1/M2.
