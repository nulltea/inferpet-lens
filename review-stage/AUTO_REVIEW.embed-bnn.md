# Auto Review Loop — embed-bnn (BNN error bounds: union-Bhattacharyya ⊕ Fano)

Surface: embed-bnn · Reviewer: Codex `gpt-5.5` xhigh (thread `019ef63c`) · Backend: codex · Difficulty: medium
Gate file: `refine-logs/embed-bnn/REVIEW_STATE.json` · Started 2026-06-23

Consolidation of an already-audited result: the geometry-only two-sided error-bound probe
(union-Bhattacharyya upper + Fano-equivocation lower) for the BNN/NNS L0 embedding-DP attack.
Load-bearing property: attack-independence (probe ≠ attack).

## Round 1 (2026-06-23)

### Assessment (Summary)
- Score: **7/10** (scoped top-venue contribution)
- Verdict: **almost**
- STOP CONDITION met (score ≥ 6 AND verdict ∈ {ready, almost}) — loop terminates at round 1.
- Reviewer checks: (a) attack-independence **PASS** under A0 (bound fns consume only `pool_emb`,
  `sigma`, synthetic RNG; no `{Y_i}` path); (b) numbers **MATCH** raw JSON (n_inside=10/n_eps=10,
  rho_ub=0.887625→0.888, rho_lb=0.937436→0.937, ε=56 upper clamped, rare-glyph top pairs); (c) proof
  **mostly sound** — T1 upper + Fano lower standard & correctly attributed, T3 sound under declared
  inputs, T4 stratified-MC unbiased/consistent sound; (d) scope **mostly honest**.
- "No fatal flaw in independence, numbers, or core proof. Remaining issues are wording, uncertainty,
  and scope hardening."

### Reviewer Raw Response

<details><summary>Full reviewer response (Codex gpt-5.5 xhigh, thread 019ef63c)</summary>

Score: 7/10 for a scoped top-venue contribution. Clean consolidation of textbook bounds into a
defensible probe; not a broad breakthrough. Verdict: almost.

(a) Attack-independence: Pass under A0 — bound functions consume only `pool_emb`, `sigma`, synthetic
RNG; no `{Y_i}` path. Caveat real: if pool/prior/clip/σ estimated from the transcript, T3 fails.
(b) Numbers: Match — n_inside=10/n_eps=10; rho_ub 0.887625→0.888; rho_lb 0.937436→0.937; table,
clamped upper at ε=56, rare-glyph top pairs all match raw JSON.
(c) Proof: Mostly sound. T1 upper + Fano lower standard & correctly attributed. T3 sound conditional
on declared public inputs. T4 stratified-MC unbiased/consistent sound. Main issue: finite-M Fano
"certified" language too strong unless always phrased as asymptotic CLT coverage.
(d) Overclaim: Scope mostly honest. Risky phrase still the C2 "morphological floor" framing — closest
pairs are rare Unicode/PUA tail tokens; case/space/number twins only the recognizable subset.

Ranked weaknesses + minimum fixes:
1. Finite-M lower-bound wording too certificate-like → replace "certified finite-M" with "CLT lower
   confidence bound"; state only liminf_{M→∞} coverage everywhere.
2. C2 attribution still risks overclaiming morphology → rename to "small-distance / under-trained-tail
   floor" unless BNN confusion counts show morphological twins dominate realized errors.
3. Empirical validation one model/pool/seed → frame as single-instance validation, or add 3-5 pool
   seeds; one extra model/table would help top-venue strength.
4. BNN empirical error lacks uncertainty reporting → add binomial/stratified MC CIs for bnn_err.
5. Independence test weaker than proof → replace blacklist signature test with exact input whitelist
   + invariance test (bounds unchanged when attack observations regenerated).
6. Upper/lower bounds loose in transition regimes → frame as bracketing diagnostics, not estimators;
   defer tighter refinements explicitly.

Bottom line: no fatal flaw in independence, numbers, or core proof.

</details>

### Actions Taken (zero-compute integrity hardening; #1, #2 + #6 already framed)
- **#1** Claim: renamed "Fano lower bound, certified finite-$M$" → "Fano lower confidence bound,
  finite-$M$ (CLT, asymptotic coverage only — not a finite-$M$ certificate)"; proof step relabelled
  "Fano lower (CLT lower confidence bound, asymptotic coverage)". (Claim already carried the
  `liminf_{M→∞}` coverage caveat verbatim.)
- **#2** Experiment log: renamed C2 finding to "small-distance / under-trained-tail floor
  (morphological twins are the *recognizable* subset, not the dominant one)"; claim Viability already
  states the very closest pairs are rare Unicode/PUA tail tokens.
- **#6** Already framed: claim Viability + Limitations state bounds are loose at extremes and "the gap
  between the upper and lower bounds is itself a diagnostic"; α-information-Fano / min-distance union
  refinement explicitly deferred.

### Deferred (non-blocking, recorded for follow-up)
- **#3** Multi-seed / second embedding table — scope is explicitly single-instance (uniform-prior, L0,
  gemma-2-2b) in Limitations; an extra pool-seed sweep is a strengthening, not a correctness fix.
- **#4** Binomial CIs for `bnn_err` — Hoeffding half-width `bnn_hoeffding_hw=0.00375` is already stored
  per record (T2 gives the certificate); render as an explicit interval in a future revision.
- **#5** Exact-signature whitelist + regenerate-observations invariance test — audit already logged this
  as future hardening; current blacklist test + proof T3 + data-flow inspection establish independence.

### Status
- STOP at round 1 (gate met). Difficulty: medium. Fixes #1/#2 applied; #3–#5 deferred non-blocking.

## Method Description
The probe is a geometry-only two-sided bound on the Bayes (= nearest-neighbour, uniform-prior MAP)
error of the L0 embedding-DP channel `Y = clip(e_V) + N(0,σ²I)` with public codebook `{e_v}`. The
**union-Bhattacharyya upper bound** `P_e^ub = (1/K)Σ_vΣ_{u≠v}Q(‖Δ_vu‖/2σ)` is a function of the
codebook self-distance multiset and σ alone; the **Fano-equivocation lower bound**
`(Ĥ_M−1)/log₂(K−1)` uses a fresh-noise Monte-Carlo estimate `Ĥ_M` of the channel equivocation `H(V|Y)`
computed over synthetic noise drawn around each codeword. Neither path touches the attack's
observations `{Y_i}` — independence is by construction under the declared-public-inputs assumption A0
(proof T3), discharged additionally by a data-flow unit test. Validated on gemma-2-2b (pool=2048):
the measured BNN error is bracketed at all 10/10 ε points, with the upper bound tight at low noise and
the Fano lower bound tracking at high noise (crossover r≈3.6).
