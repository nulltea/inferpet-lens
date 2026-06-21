---
type: reference
status: current
created: 2026-06-21
updated: 2026-06-21
tags: [experiment-tracker, information-efficient-attacks]
companion: [EXPERIMENT_PLAN]
---

# Experiment Tracker — information-efficient inversion attacks

Status: ☐ todo · ◐ in-progress · ☑ done · ✗ blocked

| Run | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status |
|-----|-----------|---------|------------------|-------|---------|----------|--------|
| R001 | **M0 proof** | theoretic guarantee T1 (proof-writer↔checker) | weak-domination + strict-MI-loss + I-MMSE-monotone | — | **VERIFIED (PASS, 2 rounds)** — `docs/research/info-efficient-attack-guarantee.md`; wiki claim:thm-t1-info-efficient | MUST | ☑ |
| R002 | M1 sanity | impl denoise-then-invert + reader-as-attack | toy jointly-Gaussian MMSE | — | MMSE = closed form | MUST | ☐ |
| R003 | M1 sanity | clean-case parity | denoise+ridge @ σ=0 | vocab | TTRSR == ridge | MUST | ☐ |
| R004 | **M2 anchor (L0)** | exact-Bayes vs ridge under DP | ridge / Bayes-NN-to-table | vocab | **uplift +0.98 @ε128; clean 1.0; C1✓ C2✓** (`results/l0_fast.txt`; wiki exp:b2-l0-bayes-vs-ridge) | MUST | ☑ |
| R004b | **M2 anchor (L>0)** | channel-aware MLP decoder vs ridge | ridge / MLP-decoder (noised-trained) + shuffle ctrl | vocab | **MLP LOSES to ridge (uplift-sel −0.01..−0.30, L5/12/20); at-layer noise → ridge already tracks MI ρ=1.0; decorrelation is input-DP-propagation-specific** (`b2_lpos_decoder.json`) | MUST | ☑ (negative) |
| R004c | M2 (L>0, propagated DP) | stronger depth decoder (iterative/MAP) vs ridge under PROPAGATED input-DP | iterative-refine / MAP+LM-prior | vocab | uplift + re-correlation where ridge breaks (B3 L20 regime) | MUST | ☐ next |
| R005 | M2 width | width robustness | denoise+ridge (Qwen3-4b) | vocab | uplift | NICE | ☐ |
| R006 | **M3 re-corr** | re-correlation w/ probes | strong vs ridge | vocab | Spearman(rec, CLUB), Spearman(rec, capPVI) | MUST | ☐ |
| R007 | **M4 novelty** | channel-awareness vs capacity | σ-aware vs noise-naive (matched) vs linear+σ | vocab | uplift attribution | MUST | ☐ |
| R008 | M4 simplicity | LM-prior deletion | denoise+retrieval vs +LM-prior MAP | vocab | uplift delta | MUST | ☐ |
| R009 | M5 transfer | Laplace/Shredder arm | strong vs ridge | vocab | uplift, Spearman | NICE | ☐ |

**Gates**: R001 must pass before R004+ claims are asserted. R004 uplift>0 ∀ε. R006 Spearman(strong)≫Spearman(ridge).
**Discipline**: heavy runs via `scripts/run_in_rocm.sh`; validate GPU saturation; L0/5/12/20 only; inspect if >10 min.
