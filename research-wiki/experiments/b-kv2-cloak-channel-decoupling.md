---
type: experiment
node_id: exp:b-kv2-cloak-channel-decoupling
title: "Experiment b-kv2-cloak-channel-decoupling"
idea_id: ""
verdict: yes
confidence: medium
date: "2026-06-24"
hardware: "CPU host venv + numba (cached operands)"
duration: "~11min sweep"
provenance: "refine-logs/kv-cloak/"
added: 2026-06-24T01:45:23Z
tags: [kv-cloak, kv-cache, bss, defense-sweep, matched-probe]
companion: refine-logs/kv-cloak/RESULTS.md
---

# Experiment b-kv2-cloak-channel-decoupling

**verdict:** `yes` (C1 supported) · **confidence:** `medium` (C2 between-channel)

KV-CLOAK defense + block-size sweep (Task B-2). Supports
claim:kv-cloak-channel-decoupling-feature-mix-loadbearing. Integrity audit WARN (no fabrication /
phantom / probe-attack circularity; 3 reporting/faithfulness items fixed). Proof cross-model PASS.

## Setup
- Surface: raw per-head KV-cache (kind `k`), Qwen3-4B, layers {0,12,20}, 48 prompts (110-178 tokens),
  capture `capture-7de5ef8d6e14afe9.pt` (capture.py patched to emit `k`/`v` pre-`repeat_kv`).
- Defense: KV-CLOAK `K'=S·P̂·(K+A)·M` (arXiv 2508.09442 eq.9) as `scripts/defenses/kv_cloak.py`
  Transform; channels {m, sp, scx, naive, a, full}; b∈{16,32,64}; mask α∈{0,1,4}; seeds {0,1,2};
  273 cells.
- Attacks: gram_error, jade (single-obs ICA), jd/jd_floor (accumulation + chance floor). Probes:
  negentropy_bits, shared_spectral_capacity_bits (geometry-only, attack-independent).

## Key results (bits canonical + p95-cosine readout)
- Channel decoupling (B1 exact): M-only row-Gram invariant (2.1e-9); S·P̂ Gram-spectrum invariant
  (3.2e-9) but full Gram rotated (1.40); A-only moves spectrum (0.20). 8/8 unit checks.
- Recovery (L0, floor 0.157): identity 0.626 → M/naive/full 0.126 (at floor); sp/scx 0.612 (≈ plaintext);
  a 0.581. Negentropy 1044b → M 1.5b; sp ~1075b.
- JD accumulation (floor 0.157→0.168): identity 0.49→0.75, sp 0.49→0.73, scx 0.49→0.72; M/naive/full ~0.12 flat.
- b-flatness: jade exactly flat for M (0.145 at every b); spectral-cap spread ≤0.27 for token-axis channels.
- Matched probe (C2): negentropy↔jade Spearman 0.706 (p=5e-42, n=270) aggregate, channel-mean 0.77;
  WITHIN-channel weak/sign-flipped (anticorrelates under mask) — a between-channel diagnostic, not a
  within-channel oracle. Spectral-cap↔jade 0.33 (not matched).

## Reading
The secret right-orthogonal feature mix M is the only load-bearing channel (drives recovery to the
chance floor); block size b and the permutation/token-mix are cover-invariant (recovery-inert). The
additive mask only perturbs the Gram spectrum and has a small non-floor recovery effect. Proof L1-L4
(verified) gives M an attack-independent recovery ceiling O(√(s/d)+√(log n/d)).

## Reproduction
- Capture (GPU, once): `scripts/run_in_rocm.sh python3 scripts/spikes/kv_cloak_capture.py`
  (corpus `corpora/kv-cloak-long-48.txt`, kinds k/v, layers 0/12/20 → `capture-7de5ef8d6e14afe9.pt`).
- Sweep (CPU, host venv with numba+scipy): `python3 scripts/spikes/kv_cloak_sweep.py --kinds k
  --layers 0,12,20 --channels m,sp,a,scx,naive,full --b-values 16,32,64 --mask-energies 0,1,4
  --seeds 0,1,2 --max-dim 16 --max-features 256` (the reported run; the script's own arg defaults
  differ and are NOT the reported config). Sanity: append `--sanity`. Analysis: `python3
  scripts/spikes/kv_cloak_analysis.py` (writes analysis.json; jd_floor reported at T=1 and T=4).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

- Supports claim:kv-cloak-channel-decoupling-feature-mix-loadbearing.

## Artifacts
refine-logs/kv-cloak/{RESULTS.md, analysis.json, sweep.json, sanity.json, EXPERIMENT_AUDIT.md,
proof/PROOF_AUDIT.md}; scripts/defenses/kv_cloak.py; scripts/spikes/kv_cloak_{capture,sweep,analysis}.py.
