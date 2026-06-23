---
type: plan
status: current
created: 2026-06-22
updated: 2026-06-22
tags: [experiment-tracker, spectral-mi, matched-probe, validation]
companion: [EXPERIMENT_PLAN]
---

# Experiment Tracker — Spectral channel-MI probe validation

Validates `claim:spectral-channel-mi-embedding-inversion`. Companion to
`../vec2text-pooled/` (reuses its Vec2Text attack pipeline + B8 sweep).

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | implement + unit-test I_G (geometry-only) | `spectral_channel_mi.py` | model-free | Gaussian-exact, monotone, Σt_i=I_G, d_eff | MUST | **DONE ✓** | 10/10 tests pass on host .venv (tests/test_spectral_channel_mi.py); Gaussian-exact ½logdet, monotone, d_eff, tail, ceilings, γ-inverse |
| R002 | M1 | probes on GTR sweep | I_G / CLUB / capPVI | N=96, ε{∞,1024,512,256,128} | bits, monotone vs ε | MUST | **DONE ✓** | `results/spectral_mi_probe_eval.json`; I_G monotone 1597→68b, sane vs CLUB |
| R003 | M2 | C1 matched-probe comparison | I_G vs CLUB vs capPVI ↔ Vec2Text recovery | same | Spearman + cost | MUST | **DONE ✓** | **I_G ρ=+1.00 (=CLUB) ≫ capPVI +0.62; I_G 60ms vs CLUB 1.7s (~28× cheaper)**. Headline POSITIVE |
| R004 | M2 | C2 ceiling holds + tightness | Fano/RD ceiling vs actual Vec2Text | same | violation count(=0) | MUST | **DONE ✓** | 0 violations; RD-floor 0→0.81 monotone, respected; Fano non-trivial at high σ. H_X/H_e0 proxies flagged |
| R005 | M3 | C3 eigen-ablation (where) | keep/drop top-k principal modes of Y | ε{∞,512,128}×k | recovery vs k; knee vs d_eff; CLUB(Y_k) control | MUST | **TODO (re-scoped)** | **N=96<d=768 → rank-deficient Σ (d_eff pinned at n−1).** Must estimate Σ from n≫d embeddings (cheap GTR-encode, no inversion) decoupled from the attacked set, THEN ablate. OOD caveat stands |
| R006 | M4 | d_eff vs knee + H(e0) collisions | — | — | d_eff overlay; collision count | NICE | TODO | sharpens C2 band |

**Decision gates**: M0 tests green before M1; M1 I_G sane vs CLUB before M2; M2 no ceiling violation + I_G tracks before claiming C1/C2; M3 keep-saturates-by-d_eff before claiming C3.
**Output**: results JSON under `results/`; analysis appended to `../EXPERIMENT_RESULTS.md` (dp-stronger-attacks) and the claim's evidence chain; NOT the refine-logs root files.
