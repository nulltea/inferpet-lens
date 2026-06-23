---
type: plan
status: current
created: 2026-06-22
updated: 2026-06-22
tags: [experiment-tracker, vec2text, pooled-embedding, dp, matched-probe]
companion: [EXPERIMENT_PLAN]
---

# Experiment Tracker — Faithful Vec2Text on a pooled embedding under DP

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | dependency smoke test | gtr-base corrector, 4 clean texts | n=4 | inverts? loads? | MUST | **DONE ✓** | PASS — near-perfect recon on iGPU. Recipe: transformers==4.44.2 shadow in .deps + apex JIT to /tmp/torch_ext + --no-deps pure-python deps. See memory vec2text-rocm-dependency-recipe |
| R002 | M1 | C1 clean replication | Vec2Text 0-step vs 20-step | N=128, 32-tok clean | tF1, BLEU, exact, cos | MUST | **DONE ✓** | clean tF1 0.80 / exact 0.18 vs base 0.48 / 0.0 (out-of-domain corpus → below Morris 0.96/0.40). beam=1 greedy (beam≥4 anchor pending) |
| R003 | M2 | C2 leakage curve + C3 probe | Vec2Text[20,beam1] × ε{∞,1024,512,256,128} | N=128 | tF1/exact/cos, CLUB, capPVI, Spearman | MUST | **DONE ✓** | C2 clean monotone decay; **C3 CLUB ρ=+1.00** (tF1/cos/bleu); capPVI weak (+0.67, not matched). results/v2t_dp_sweep.json |
| R004 | M3 | C4 info-efficiency | {0/1/20-step} × ε{∞,512,128} + NN-retrieval (+BoW) | N=500 + disjoint pool | tF1/exact, gaps | MUST | TODO | NN pool must be disjoint |
| R005 | M4 | C3 anti-confound | partial-dim / Laplace DP × Vec2Text + probe | N=500 | Spearman(probe,recovery) | NICE | TODO | break monotone knob |
| R006 | M4 | qualitative privacy story | example recon dumps + word-freq | subset | qualitative | NICE | TODO | Morris Fig6 analog |
| R007 | UTIL | utility side → privacy–utility tradeoff | retrieval ranking fidelity (DP query vs clean GTR retriever) | N=256, ε{∞..64} | nDCG@10, Recall@k, rankρ, top-k' expansion | MUST | **DONE ✓** | `scripts/eval/utility_retrieval_eval.py` → B10. Favorable tradeoff: ε≈256–512 kills Vec2Text exact recon at near-lossless retrieval (nDCG≥0.97); utility bites only at ε≤128. Cheap (cosine, no LLM/inversion). Std metric (CAPRISE/RemoteRAG); BLEU rejected (off-surface). Caveat: clean-ranking GT; BEIR-qrels upgrade available |

**Matched-probe theory (2026-06-22)**: CLUB/capPVI shown inadequate (capPVI flat = coarse cluster label; CLUB loose/non-localizing). Adopted + VERIFIED probe = **spectral channel-MI** `I_G(σ)=½Σlog2(1+λi/σ²)` (geometry-only converse ceiling + Fano/RD recovery bound + top-eigendirection localization). Proof: `PROOF_PACKAGE.md` (Codex PASS, 3 rounds); claim `research-wiki/claims/spectral-channel-mi-embedding-inversion.md`. TODO: (R007) implement as a geometry-only `talens.measures` fn (cov-eigh+σ, like channel_error_bounds.py) + recompute over the B8 sweep; (R008) T4 achievability = bottom-mode ablation (ablate λi≪σ² dirs of Y → Vec2Text recovery ~unchanged).

**Decision gates**: M0 must pass (dependency runs) before M1; M1 tF1≳0.8 before M2; M2 monotone+Spearman before claiming C3.
**Output**: results JSON under `results/`; analysis appended to `../EXPERIMENT_RESULTS.md` (the dp-stronger-attacks results doc), NOT the refine-logs root files.
