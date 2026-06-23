# Experiment Audit Report — defenses-existing

**Date**: 2026-06-23
**Auditor**: Codex gpt-5.5 xhigh (cross-model, read-only filesystem)
**Trace**: `.aris/traces/experiment-audit/2026-06-23_defenses-existing_run01/`

## Overall Verdict: WARN
## Integrity Status: warn

Load-bearing checks **A (no ground-truth leak)** and **D (no probe==attack circularity)** both **PASS**.
The WARNs are reporting-granularity, not evidence fraud, and have been addressed in the doc.

## Checks

### A. Ground Truth Provenance — PASS
Π GT is `rng.permutation` stored as `WeightPair.perm` (`aloepri.py:134-137`); VMA reconstructs via
signatures + Hungarian (`vma.py:44-54`) and reads `inverse_perm()` only to **score** (`vma.py:63`).
Token GT = tokenizer `input_ids` (`b4:51,55`); ridge fits on train labels, scores held-out
(`_inversion.py:58-96`). No secret leaks into reconstruction.

### B. Score Normalization — PASS
Recovery = raw equality/hit means (`vma.py:65`; `ridge.py:113`). CLUB bits = `mi_nats/ln2` verbatim
(`club.py:197`); keymat estimator floor `−2.3937` preserved in JSON (`aloepri_vma_sweep.json:91`).
No division by model-own max.

### C. Result File Existence — WARN
Core numbers trace. Shredder Π=0.037 is an honest 3-seed mean of raw 0.036/0.031/0.045
(`b4:385,721,1057`). cross_scheme Spearman exist (`b4:76`); aloepri spearman exist (`aloepri:18`).
**WARN**: "n_prompts=192" is the spike default, not stored in the JSON → now annotated in the doc.

### D. Probe==Attack Circularity — PASS
CLUB is computed independently of ridge/VMA (`weights/measures.py:41`; `b4:111`). retrieval-PVI **is**
the inversion attack rendered in bits (`vinfo.py:134`; `weights/measures.py:14`); the doc labels it
"VMA-in-bits / never as independent confirmation" (`RESULTS_STANDARDIZED.md:21`). No attack reports
its own recovery as an independent probe.

### E. Scope Honesty — WARN
Doc flags single model + probe instability (seed-1 token_probe collapse, embed NaN: `b4:492,494`).
**WARN (addressed)**: AloePri sweep is **single seed** (`aloepri_vma_sweep.py:49`) and keymat is a
**single α_e=0 config** (`aloepri:84`; `json:87`) — both now stated explicitly in the doc.

### F. Evaluation Type: synthetic_proxy
Defense applied to **real** gemma-2 activations / embed table; Π = synthetic AloePri perm over real
rows; token-id = real tokenizer GT under synthetic static-Laplace.

## Action Items (done)
- Annotated "n_prompts=192 = spike default, not in JSON".
- Stated AloePri = single seed, keymat = single config explicitly.

## Claim Impact
- C1 channel-selectivity: **supported**
- C2 cross-family transfer: **needs_qualifier** (soften "non-transfer" → mechanism-dependent; n small, embed NaN)
- C3 no depth sign-flip: **supported**
- C4 AloePri keymat: **needs_qualifier** (keymat single config → "this config", not general)
