---
type: plan
status: current
created: 2026-06-21
updated: 2026-06-21
tags: [experiment-plan, information-efficient-attacks, MMSE, denoise-then-invert, I-MMSE, Fano, proof-gated, calibration]
companion: [FINAL_PROPOSAL, EXPERIMENT_RESULTS]
supersedes: [archive-matched-probe/EXPERIMENT_PLAN]
---

# Experiment Plan — information-efficient inversion attacks (proof-gated)

**Problem**: IT leakage probes (CLUB, capacity-PVI) **decorrelate from attack recovery
under small noise** (B3/B4: input-DP barely moves CLUB/PVI but ridge TTRSR collapses
0.81→0.60; VMA 1.0→0.56 at α_e=0.2). This is the **Bayes-optimality gap** — the deployed
attacks (linear ridge `X→emb`+cosine; lossy RowSort) sit far below the Fano/I-MMSE recovery
ceiling, so recovery falls for reasons MI cannot see. Either MI is a poor predictor (rejected)
**or** the attacks are information-inefficient (the hypothesis we test).

**Method thesis**: A **noise-aware, nonlinear denoise-then-invert MMSE/MAP** token-recovery
attack **provably (weakly) dominates** linear ridge (Blackwell/Rao-Blackwell/DPI), strictly
when `E[token|X]` is non-affine, and on the Gaussian-DP arm its recovery is **monotone in MI**
(I-MMSE). So it recovers more **and** re-correlates with the IT probes — the ridge↔strong-attack
gap *is* the measurable Bayes-optimality gap. (Frontier-LLM-component block: **N/A — skipped**.)

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|-------|----------------|-----------------------------|--------|
| **T1 (theory, proof-gated FIRST)** — Bayes-optimal denoise-then-invert weakly dominates ridge; strict under non-affine `E[token｜X]`; recovery monotone in MI on the Gaussian arm. | This is the *theoretic guarantee* the user requires before any empirical claim. | A `/proof-writer`→`/proof-checker` PASS (status ≥ sound-modulo-imports) on the theorem, with the verified caveats (I-MMSE Gaussian-only; finite variance; strict-MI-loss framing, not single-metric converse). | B0 |
| **C1 (primary)** — the denoise-MMSE attack recovers strictly more than ridge under input-DP, uplift **growing** with noise. | The headline: a stronger attack that the defence does not stop. | TTRSR(strong) − TTRSR(ridge) > 0 at every ε, with the gap increasing as ε falls (more noise), across L0/5/12/20, 3 seeds. | B2 |
| **C2 (primary)** — the strong attack **re-correlates with the IT probes** where ridge decorrelates. | Resolves the user's critique: makes the probes predictive by fixing the attack, not the probe. | Spearman(recovery, CLUB) and Spearman(recovery, capacity-PVI) over the ε-sweep are substantially higher for the strong attack than for ridge (target: ridge≈flat/неmonotone → strong ≥ 0.8). | B3 |
| **C3 (supporting)** — the gain is **channel-awareness**, not capacity. | Anti-claim defence: rules out "just a bigger model." | A *noise-naive* nonlinear decoder (same capacity, no σ) does NOT achieve the channel-aware uplift; a *linear* denoiser with σ does NOT match the nonlinear one (unless ~Gaussian). | B4 |
| **C4 (transfer)** — uplift + re-correlation hold under **Laplace/Shredder** noise. | Generality beyond the Gaussian arm where I-MMSE applies. | Repeat B2/B3 under Shredder static-Laplace; degradation-DPI guarantee (not I-MMSE); uplift + Spearman improvement persist. | B5 |

**Anti-claims to rule out**
- **A1 "just more capacity/params."** → B4 capacity-matched noise-naive decoder gets no channel-aware gain.
- **A2 "row-split memorisation."** → strong attack evaluated **vocab-disjoint** (same honest regime as ridge); report row-split only as an upper anchor.
- **A3 "it secretly is the probe."** → the attack outputs a *token*, scored by TTRSR vs ground truth; the probe (CLUB/PVI) is a separate bits/accuracy quantity. Independence by construction.
- **A4 "the reader↔ridge comparison is unfair"** (different task/pool). → fix the comparison protocol: every attack predicts token-id on the **same split + same candidate pool**.

## Paper storyline
- **Main paper proves**: T1 (B0) + C1 (B2) + C2 (B3) — a proof-backed stronger attack that restores MI↔recovery calibration. + C3 (B4) novelty isolation.
- **Appendix**: C4 (B5) Laplace transfer; the attack ladder (reader-as-attack → denoise+ridge → denoise+decoder → +LM-prior MAP).
- **Cut**: full BeamClean reproduction (cite as external empirical PoC; we add the *theory* + the MI-tracking demonstration). MoE/attention surfaces.

## Experiment blocks

### B0 — Proof gate (theoretic guarantee) — MUST-RUN, FIRST
- **Claim**: T1. **Gate**: no empirical claim (C1–C4) may be *asserted as supported* until this passes.
- **Deliverable**: a theorem doc + `/proof-writer`→`/proof-checker` loop until status ≥ `sound-modulo-imports`. Theorem (from the verified backbone, wiki claims `weak-domination`/`strict-improvement`/`mi-monotone-gaussian`):
  > Let token S and observation X have fixed joint law; Y = X through the DP channel. Let A* = Bayes-optimal estimator on Y (MAP for top-1), A_ridge = best affine `X→emb`+cosine. (a) **risk(A*) ≤ risk(A_ridge)** for every noise level (Blackwell post-processing / Rao-Blackwell). (b) If `I(S; φ_ridge(Y)) < I(S;Y)` then **strict** on a positive-measure set (orthogonality; linear-MMSE suboptimal unless jointly Gaussian). (c) **Gaussian arm**: along the DP SNR path, I(S;Y) and −MMSE are both monotone (I-MMSE `dI/dsnr=½·mmse`), so A*'s recovery is monotone in MI.
- **Caveats to encode (proof-checker will flag)**: I-MMSE is Gaussian-only (Laplace→degradation-DPI, part (c) restated as monotone-under-degradation); finite second moment required (heavy-tailed embedding → conditional-median variant); do NOT claim the single-metric converse of sufficiency (use strict-MI-loss).
- **Success**: proof-checker PASS. **Failure**: weaken C1–C4 to empirical-only + document the unprovable step.
- **Priority**: MUST-RUN, FIRST.

### B1 — Implement + sanity (model-free + clean) — MUST-RUN
- **Builds**: the denoise-then-invert attack `attacks/denoise_invert.py` (defense-eval-side): (i) **channel-aware denoiser** `g_σ(ỹ)≈clean` — closed-form posterior-mean for the L0/Gaussian case (Balle-Wang shrinkage), a small trained MLP for L>0; (ii) feed to the existing ridge/retrieval inverter. Plus the **reader-as-attack** wrapper (capacity-PVI reader scored as token recovery).
- **Sanity**: on a toy jointly-Gaussian (S,X) the MMSE denoiser matches the closed-form posterior mean; on clean (σ=0) the attack reduces to ridge (no regression); finite-variance guard.
- **Success**: toy MMSE matches closed form; clean-case parity with ridge. **Cost**: <15 min CPU.

### B2 — Main anchor: strong attack vs ridge under input-DP — MUST-RUN
- **Claim**: C1. **Data**: `localdp_runner` captures, gemma-2-2b, L0/5/12/20, ε∈{∞,4096,2048,1024,512,256}, 3 seeds, **vocab-disjoint** (A2/A4). Qwen3-4b for width robustness (appendix).
- **Compared systems**: ridge (baseline) · denoise+ridge (channel-aware) · denoise+decoder · reader-as-attack (row-split anchor) — **all on the same split + candidate pool**.
- **Metrics**: TTRSR per ε; **uplift = TTRSR(strong)−TTRSR(ridge)**; recovery-vs-ε monotonicity.
- **Success**: uplift > 0 at every ε, increasing as ε↓ (where ridge collapses); ≥ one ε where ridge is "defended" (<0.1) but strong attack is not.
- **Failure**: no uplift → either denoiser mis-specified (debug σ/manifold) or MI genuinely destroyed (then defence is sound — a result). **Priority**: MUST-RUN.

### B3 — Re-correlation with the IT probes — MUST-RUN
- **Claim**: C2. Reuse B2 runs; for each attack compute Spearman(recovery, CLUB) and Spearman(recovery, capacity-PVI) across the ε-sweep (within-layer, per B2+ "within-condition" rule — pool ranks, not raw magnitudes).
- **Success**: Spearman(strong-recovery, probe) ≫ Spearman(ridge-recovery, probe); ideally strong ≥ 0.8 while ridge is flat/non-monotone. This is the figure that *answers the critique*.
- **Failure**: strong attack recovers more but still doesn't track MI → MI is genuinely a loose predictor in this regime (supports critique (1); pivot to a tighter measure). **Priority**: MUST-RUN.

### B4 — Novelty isolation: channel-awareness vs capacity — MUST-RUN
- **Claim**: C3 (rules out A1). **Systems**: channel-aware denoiser (known σ) vs **capacity-matched noise-naive** decoder (no σ) vs **linear** denoiser+σ. **Metric**: uplift attributable to σ-awareness and to nonlinearity.
- **Success**: the channel-aware nonlinear denoiser carries the gain; noise-naive (same capacity) and linear (with σ) do not match it. **Doubles as the simplicity/deletion check**: does the LM-prior MAP add anything over denoise+retrieval? If not, prefer the simpler denoiser. **Priority**: MUST-RUN.

### B5 — Laplace/Shredder transfer — NICE-TO-HAVE
- **Claim**: C4. Repeat B2/B3 under Shredder static-Laplace. Guarantee via degradation-DPI (NOT I-MMSE). **Success**: uplift + re-correlation persist under non-Gaussian noise. **Priority**: NICE-TO-HAVE.

## Run order & milestones

| M | Goal | Blocks | Decision gate | Cost |
|---|------|--------|---------------|------|
| **M0** | **proof gate** (proof-writer↔checker on T1) | B0 | proof PASS (≥ sound-modulo-imports) → unlock empirical claims | analysis (Codex xhigh) |
| **M1** | implement + sanity | B1 | toy MMSE = closed form; clean = ridge | <15 min CPU |
| **M2** | main anchor (DP) | B2 | uplift>0 ∀ε, growing with noise | ~30–60 min ROCm |
| **M3** | re-correlation | B3 | Spearman(strong,probe) ≫ ridge | reuse M2 (~10 min) |
| **M4** | novelty isolation | B4 | gain = channel-awareness, not capacity | ~30 min |
| **M5** | Laplace transfer | B5 | uplift + re-corr persist | ~30 min |

**Auto-review**: run `/auto-review-loop` after M3 (the main result) and again after M4 (paste tables inline; Codex can't read files). **Proof loop** (`/proof-writer`+`/proof-checker`) runs at M0 and is re-invoked if a reviewer challenges the guarantee.

## Compute & data budget
- **~2–4 GPU-hr total.** Denoiser is cheap (posterior-mean closed form at L0; small MLP for L>0). Inverter reuses ridge/retrieval. Captures reuse `localdp_runner`. The **LM-prior MAP (BeamClean-style)** variant is the only heavy piece — gated behind B4 showing denoise+retrieval is insufficient.
- **Data**: cached captures + existing corpora; paired (clean, noised) captures for training the L>0 denoiser (one extra capture pass per ε).

## Risks & mitigations
- **R1 — L>0 denoiser hard to train** (noise propagated nonlinearly): start at L0 (closed-form MMSE, cleanest proof), then L5/12/20 with a trained denoiser; if it fails, scope the headline to L0/early layers honestly.
- **R2 — proof stalls at M0** (a step unprovable): fall back to the weak-domination + strict-MI-loss core (very robust) and present I-MMSE-monotonicity as Gaussian-arm-only; never block the empirical work beyond the robust core.
- **R3 — fair comparison** (reader vs ridge task mismatch, A4): fix split+pool across all attacks; report row-split reader only as an anchor, vocab-disjoint as the honest bar.
- **R4 — heavy LM-prior MAP**: gate behind B4; reuse BeamClean's released code rather than reimplementing.

## Final checklist
- [x] Main tables covered (uplift-vs-ε, re-correlation, channel-awareness isolation)
- [x] Novelty isolated (B4 channel-awareness vs capacity)
- [x] Simplicity defended (B4 denoiser-only vs +LM-prior deletion)
- [x] Frontier contribution **explicitly not claimed** (non-frontier method)
- [x] Theoretic-guarantee gate FIRST (B0 proof, with verified caveats)
- [x] Must-run (B0–B4) vs nice-to-have (B5) separated

---

# Block B6 — Stronger decoder attacks (Vec2Text-style / forward-model) — appended 2026-06-21

**Problem**: R004c showed a plain MLP decoder beats ridge under propagated input-DP @L20 but only
*suggestively* (4 ε, 1 seed, uplift ≤+0.14). Is a properly-strong attack (more capacity, iterative
refinement, or forward-model optimization) **conclusively** stronger AND clearly re-correlating?

**Method thesis**: An information-efficient decoder that (i) has adequate capacity, (ii) iteratively
refines, and/or (iii) uses the known forward model closes the gap ridge leaves in the propagated-DP
regime — making uplift-vs-ridge conclusive and re-correlation clearly > ridge.

## Claim map
| Claim | Min convincing evidence | Block |
|---|---|---|
| **C5 (primary)** — a strong decoder beats ridge conclusively under propagated-DP @L20 | uplift-selectivity > 0 across ≥5 ε × ≥2 seeds, CI excludes 0; and > the one-shot MLP | B6a |
| **C6 (re-correlation)** — its selectivity tracks MI clearly better than ridge | Spearman(strong-sel, capPVI) ≥ ridge+0.2, ≥0.6 absolute | B6a |
| **C7 (novelty isolation)** — the win is iteration/forward-model, not just capacity | iterative/forward-model > capacity-matched deep one-shot decoder; OR honest null (pure embedding iteration = capacity) | B6b |

**Anti-claims**: "just more parameters" (B6b capacity control); "memorization" (vocab-disjoint + shuffle
selectivity, established); "unfair info" (WEIGHTS-PUB — corrector trained on attacker-generated triples,
forward-model uses public weights; all admissible, [[threat-model-fairness]]).

## Blocks
### B6a — strong decoder vs ridge/MLP under propagated-DP @L20 — MUST-RUN
- Systems: ridge · one-shot MLP (R004c) · **deep/long-trained decoder** · **iterative corrector**
  c(Y,ê_t)→ê_{t+1}, T∈{1,2,3} rounds (Vec2Text-style, embedding space, trained on WEIGHTS-PUB triples).
- Data: propagated input-DP capture @L20 (reuse R004c capture if persisted; else 1 small fresh capture
  L20 only, ε∈{∞,1024,768,512,384,256}, 160 prompts), vocab-disjoint + shuffle selectivity.
- Metric: τ-selectivity uplift vs ridge AND vs one-shot MLP; Spearman(sel, capPVI/CLUB) over ε.
- Success: C5 + C6. Failure: strong decoder no better than MLP → the limit is information, not estimator
  (would weaken the thesis at depth — honest).
### B6b — novelty isolation (iteration/forward-model vs capacity) — MUST-RUN
- Capacity-matched deep one-shot decoder vs the iterative corrector (same params). Simplicity check:
  does T>1 help over T=1? If pure embedding iteration = capacity, conclude faithful Vec2Text needs the
  forward model (forward-model optimization = future work if it exceeds budget).
### B6c — forward-model optimization (ER-style) — NICE-TO-HAVE (budget-permitting)
- Optimize continuous z to min ‖f_L(z)−Y‖² via backprop through L blocks, snap to token. Strongest but
  costliest; run only on a small prompt subset if B6a leaves GPU budget.

## Run order / budget
| M | Goal | Gate | Cost |
|---|---|---|---|
| M6a | B6a strong+iterative decoder vs ridge/MLP @L20 | C5+C6 | ≤12 min GPU |
| M6b | B6b capacity control + T-sweep | C7 | reuse M6a (~+3 min) |
| M6c | B6c forward-model opt (if budget) | — | ≤5 min, small subset |

**Hard budget: 10–20 min GPU total, ONE GPU process.** Reuse capture; decoders/correctors are small
MLPs on GPU (cheap). Forward-model opt gated to leftover budget.

## Checklist
- [x] capacity control (B6b) isolates iteration vs params
- [x] simplicity (T>1 vs T=1) defended
- [x] threat-model fairness + selectivity controls carried
- [x] no frontier-LLM-component claim (non-frontier)
