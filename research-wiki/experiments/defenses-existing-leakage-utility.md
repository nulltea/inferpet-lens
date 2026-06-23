---
type: experiment
node_id: exp:defenses-existing-leakage-utility
title: "Implemented-defense leakage–utility consolidation (Shredder static-Laplace + AloePri)"
idea_id: ""
verdict: partial
confidence: medium
date: "2026-06-23"
hardware: "gemma-2-2b / gemma-2 embed table (cached); CPU re-render via report.py"
duration: ""
provenance: "results/b4_cross_scheme.json ; results/aloepri_vma_sweep.json ; refine-logs/defenses-existing/"
added: 2026-06-23T21:57:46Z
tags: ["defenses-existing", "shredder", "aloepri", "leakage-utility", "channel-selective", "calibration", "consolidation"]
---

# Implemented-defense leakage–utility consolidation

**verdict:** `partial`  ·  **confidence:** `medium`

Consolidation (Block A) of the two implemented defenses in `scripts/defenses/`, read as **defenses**
(parameter sweeps), standardized to bits-canonical + per-secret readout via `src/talens/report.py`.
Standardized tables: `refine-logs/defenses-existing/RESULTS_STANDARDIZED.md`.

## What was run (pre-existing data, re-standardized)

- **Shredder static-Laplace** (`results/b4_cross_scheme.json`): i.i.d. Laplace(0,b) at the observed
  activation; b∈{0,0.109,0.218,0.381,0.545,0.817} × layers{0,5,12,20} × 3 seeds (72 records); also the
  input-DP arm for the cross-scheme comparison.
- **AloePri** (`results/aloepri_vma_sweep.json`): embed-table obfuscation; perm-core α_e sweep (8 points,
  single seed) + keymat single α_e=0 config.

## Metrics / Results (bits canonical + readout)

1. **Channel-selectivity.** Shredder Π VMA-recovery 1.000→0.037 over b (27× collapse) while token-id
   ridge-recovery stays ≥0.45 at all depths even at b=0.817 (L0 0.747→0.670).
2. **Cross-family non-transfer (softened).** Spearman(probe-bits, recovery): within input-DP
   token 0.642 / Π 0.812 / embed 0.750 > within Shredder token 0.389 / Π 0.425; pooled 0.453 / 0.569.
3. **No depth sign-flip** under Shredder (token recovery positive at every depth) — corroborates
   `claim:depth-decoupling-input-dp` as injection-locus.
4. **AloePri regimes.** perm-core VMA 1.0→0.007 with CLUB & retr-PVI tracking (Spearman 0.976/1.0);
   keymat α_e=0 → VMA 0, CLUB −2.4 bits (estimator floor). Keymat is the load-bearing defense.

## Reasoning / Verdict & integrity

- result-to-claim (Codex gpt-5.2 xhigh): **KEEPER/scoped** — (1)/(3) yes; (2) partial→soften; (4) yes
  perm-core / keymat single-config.
- experiment-audit (Codex gpt-5.5 xhigh): **WARN, no FAIL** — **A (no GT leak) PASS**, **D (no
  probe==attack circularity) PASS**; WARNs fixed in the doc.
- Theory backbone proof-checker **PASS** (Round-1 FAIL → Round-2 PASS), folded into
  `claim:defense-channel-selectivity-mechanism-dependent`.

## Limits / queued firm-ups

Single model; Shredder 3 seeds, AloePri single seed + keymat single config; no bootstrap CIs;
token_probe/embed_probe numerically unstable (claim uses attack recovery). Queue: multi-seed + CIs,
held-out cross-family calibration test, repair embed_probe, keymat parameter sweep.

## Connections
_Edges are recorded in `graph/edges.jsonl`._
Supports → `claim:defense-channel-selectivity-mechanism-dependent` (partial).
