---
type: plan
status: current
created: 2026-06-22
updated: 2026-06-22
tags: [experiment-plan, bhattacharyya-fano-bounds, union-bhattacharyya, fano, bnn, dp-probe, geometry-only]
companion: [PROOF_PACKAGE, EXPERIMENT_TRACKER]
supersedes: []  # NNS-PVI plan rejected; archive dir removed in cleanup
---

# Experiment Plan — Union-Bhattacharyya & Fano error bounds: a geometry-only probe matched to BNN

**Problem**: BNN/NNS is the MAP-optimal attack on the L0 embedding-DP channel; it recovers
0.969 at r=3.63 while CapPVI=0.048 and CLUB tracks it only at ρ=0.22. No existing probe is
matched to its decision geometry. The first candidate (NNS-PVI) was rejected: it evaluates the
MAP posterior on the attack's own observations, sharing the per-point distance matrix — it is
the attack re-scored (proof review + literature both confirmed).

**Method thesis**: The L0 channel is textbook M-ary Gaussian signaling. The matched, *genuinely
independent* probe brackets the MAP (=BNN) error from a route that never touches the attack's
test decode: **union-Bhattacharyya upper bound** + **Fano-equivocation lower bound** (fresh-noise
MC), both computed from the codebook `{e_v}` + σ alone. Validated proof package (PROOF_PACKAGE.md,
3 Codex rounds, PASS): T1 two-sided bound validity, T2 BNN achieves the bracketed error, T3 independence
by construction, T4 estimator consistency, T5 σ-monotonicity.

**Date**: 2026-06-22

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|-------|----------------|-----------------------------|--------|
| **C1 (primary)** — measured BNN error `1−TTRSR` falls inside `[P_e^lb, P_e^ub]` at every ε | The two-sided bound is a valid, attack-independent predictor of the MAP attack. | At each of 5 ε, `P_e^lb ≤ (1−TTRSR_n) ≤ P_e^ub` (with Hoeffding CI on TTRSR, CLT CI on P_e^lb). | B2 |
| **C2 (primary)** — the geometry-only upper bound predicts BNN-not-1.0 | The probe reproduces the ≈0.6% morphological-confusion floor from the *real* distance histogram, not an idealization. | `P_e^ub(ε=∞ clip-only)` ≈ measured `1−TTRSR` ≈ 0.03; the dominant union terms are small-‖Δ‖ (morphological) pairs. | B2, B3 |
| **C3 (independence)** — the probe value is unchanged when computed without any test observation | Demonstrates the property NNS-PVI lacked. | Probe computed from `{e_v}`+σ+fresh-noise RNG only; bit-identical regardless of which test split / which `Y_i` (by construction; B1 asserts via code path, B2 confirms numerically). | B1, B2 |
| **C4 (comparison)** — the error bounds track BNN across ε better than CLUB/CapPVI, at a fraction of the cost | Establishes it as the matched probe for this channel. | ρ(P_e^ub-implied-TTRSR, BNN-TTRSR) and bound-gap vs CLUB/CapPVI ρ; wall-clock ⊥ n. | B2, B4 |

**Anti-claims to rule out**
- **A1 "it's the attack again."** → T3 (proven) + B1 code path: probe never computes `‖Y_i−e_v‖²`. Only the codebook self-distance Gram and fresh synthetic noise.
- **A2 "bounds are vacuous so it predicts nothing."** → report the bound gap; at the SNRs where BNN is interesting (r ≤ 3.63, near-orthogonal) the upper bound is tight (B2). Flag the vacuous low-SNR regime honestly.
- **A3 "uniform-prior mismatch."** → compare BNN TTRSR macro-averaged uniformly over the pool (matched to the uniform-prior bounds); report empirical-prior variant via Remark-N formulas as a secondary.

## Paper storyline
- **Main**: C1 (bounds bracket BNN) + C3 (independence) — a proof-backed, attack-independent, geometry-only predictor of the optimal L0 attack.
- **Appendix**: C2 (morphological floor from geometry), C4 (cost + tracking vs CLUB/CapPVI), Remark-N empirical-prior variant.
- **Cut**: α-information (Arimoto) tighter Fano (note as future tightening); full-vocab (V=256K) bounds (the pool=2048 bounds already matches BNN's hypothesis space); L>0 (channel is not Gaussian-in-embedding-space at depth — BNN doesn't apply).

## Experiment Blocks

### B1 — Implement the error-bounds probe — MUST-RUN FIRST (CPU)
**Builds**: `src/talens/measures/channel_error_bounds.py` with:
- `union_bhattacharyya(table, pool, sigma, clip_norm, exact=True)` → `P_e^ub` (exact-Q union) and `P_e^ub_B` (Bhattacharyya). Computes the pool self-distance Gram once; re-exponentiates per σ.
- `fano_equivocation(table, pool, sigma, clip_norm, M=64, seed)` → `Ĥ_M` (bits), stratified `se`, and `P_e^lb` = `(Ĥ_M − z·se − 1)/log2(K−1)`. Draws fresh `ε_vj ~ N(0,σ²I)`; **never takes Y observations as input** (enforced by signature — no `X`/`Y` arg).
**Sanity / unit tests** (`tests/test_channel_error_bounds.py`, CPU, synthetic d=32, K=8):
- T-a: σ→0 ⟹ `P_e^ub → 0`, `Ĥ_M → 0`, `P_e^lb → ~0`.
- T-b: orthonormal codebook, known σ ⟹ `P_e^ub` matches the closed-form `(K−1)Q(1/(√2 σ))` within MC error.
- T-c: `union_bhattacharyya` exact-Q ≤ Bhattacharyya form (T1 inequality).
- T-d: `Ĥ_M` unbiased — mean over many seeds ≈ brute-force `H(V|Y)` on a fine grid (small K,d).
- T-e: **independence smoke test** — probe output identical for two disjoint synthetic "test sets" (it ignores them by construction; assert the function has no Y parameter and output depends only on (table,pool,σ,seed)).
- T-f: monotonicity — `P_e^ub(σ)`, `Ĥ_M(σ)` non-decreasing across a σ grid (T5).
**Success**: T-a..T-f pass. **Cost**: <30 min code + <5 min test. **Priority**: MUST-RUN.

### B2 — Error-bounds vs BNN on the ε-sweep — MUST-RUN (GPU)
**Claim**: C1, C2, C3, C4.
**Dataset/split**: gemma-2-2b, corpora/release-gate-512.txt, 256 prompts, vocab-disjoint, pool=2048, seed=20260622, ε∈{∞,1024,512,256,64}, C_raw clip. Reuse the unified_dp_sweep harness for BNN TTRSR.
**Compared quantities** at each ε: measured BNN TTRSR (Hoeffding CI) | `P_e^ub`, `P_e^ub_B` | `Ĥ_M`, `P_e^lb` (CLT CI) | existing CLUB-sel, CapPVI-sel for reference.
**Metrics**:
- Primary (C1): indicator `P_e^lb ≤ 1−TTRSR ≤ P_e^ub` at each ε; report the bound gap.
- C2: at ε=∞ (clip-only, σ=0⁺) does `P_e^ub` reproduce the ≈0.03 floor? Inspect which pool pairs dominate the union sum (expect morphological/subword neighbors).
- C3: confirm probe values are independent of the test split (recompute on a 2nd disjoint split → identical).
- C4: ρ(`1−P_e^ub`, BNN-TTRSR) over ε vs ρ(CLUB,BNN)=0.22, ρ(CapPVI,BNN)=0.45; wall-clock per ε.
**Setup**: add a `--probe error-bounds` path to `unified_dp_sweep.py` calling `channel_error_bounds`; M=64 fresh draws/codeword; z for 95% one-sided.
**Success (C1)**: bracketing holds at ≥4/5 ε (the low-SNR ε=64 may have vacuous upper bound — report honestly). **Failure**: if BNN error exits the bounds at a high-SNR ε, the bound or clip-norm handling is wrong → debug C_raw and the pairwise term.
**Table/fig**: main error-bounds plot (BNN error + bands vs ε); union-term histogram at ε=∞.
**Priority**: MUST-RUN. **Cost**: ~10 min GPU (probe is ~6 s; BNN/TTRSR reuse).

### B3 — Morphological-floor attribution — NICE-TO-HAVE
**Claim**: C2 sharper. List the top-50 pool pairs by `exp(−‖Δ‖²/8σ²)` at ε=∞; verify they are subword/morphological relatives (shared stems, casing, leading-space variants). Confirms the geometry-only upper bound "knows" where BNN fails. **Cost**: trivial (post-process the Gram). **Priority**: NICE-TO-HAVE.

### B4 — Cost & tracking comparison table — NICE-TO-HAVE (freeride in B2)
Wall-clock per ε for error-bounds vs CLUB vs CapPVI vs MDL; ρ-vs-BNN for each; n-independence demonstration (rerun probe at n=300 vs n=3000 → identical). **Priority**: NICE-TO-HAVE.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Implement + unit-test probe | B1 + pytest | T-a..T-f pass | <1h CPU | Low (textbook math) |
| M1 | Error-bounds vs BNN sweep | B2 (+B3,B4 freeride) | C1 bracketing ≥4/5 ε | ~15 min GPU | Med (clip-norm/SNR regime) |
| M2 | Verify claims + wiki | analysis | C1+C3 ⟹ headline confirmed | 30 min | Low |

**Must-run**: M0 → M1 → M2. **Nice-to-have**: B3, B4.

## Compute and Data Budget
- M0 <5 min CPU; M1 ~15 min GPU (probe ~6 s, rest is BNN/TTRSR reuse). Total GPU **<20 min**.
- Data: reuse release-gate-512.txt + pool=2048 from prior sweep. No new data, no human eval.
- Biggest bottleneck: none of note — this is the cheapest probe in the suite (⊥ n).

## Risks and Mitigations
| Risk | Mitigation |
|------|-----------|
| Upper bound vacuous at ε=64 (low SNR) | Report `min(1,·)`; the interesting regime r≤3.63 is tight (near-orthogonal); the bound gap is itself the diagnostic |
| Fano lower bound loose (≤0 when H(V|Y)≤1 bit) | Note as honest scope; α-information tightening deferred |
| Uniform-prior vs empirical-TTRSR mismatch | Compare BNN macro-uniform over pool; empirical via Remark-N formulas as secondary |
| clip-norm mismatch (C_raw vs C_runtime) | L0 uses C_raw (table space) — same as BNN; assert in B1 |

## Final Checklist
- [ ] B1: `channel_error_bounds.py` implemented; T-a..T-f pass
- [ ] B1: probe signature takes NO Y/X observation argument (enforces T3 independence)
- [ ] B2: `--probe error-bounds` added to unified_dp_sweep; 5-pt ε sweep run
- [ ] C1: bracketing `P_e^lb ≤ 1−TTRSR ≤ P_e^ub` confirmed (≥4/5 ε)
- [ ] C2: ε=∞ upper bound reproduces ≈0.03 floor; union dominated by morphological pairs
- [ ] C3: probe identical across disjoint test splits (independence confirmed numerically)
- [ ] C4: cost + ρ-vs-BNN table vs CLUB/CapPVI; n-independence shown
- [ ] Wiki: `claim:bnn-error-bounds-bhattacharyya-fano` (verified), edges, log; new paper nodes (Proakis ref, de Cherisey already present)
