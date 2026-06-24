# Experiment Plan — resid-rep2text (Rep2Text on Qwen3 residual stream)

**Problem**: Does a single last-token residual @ L10 of Qwen3-4B leak the full input text,
and does an attack-independent geometry-only probe predict how much it leaks as a function of
sequence length?
**Method Thesis**: A *single* hidden vector is a genuine information **bottleneck**; an
adapter→frozen-decoder inverter (Rep2Text, arXiv 2511.06571) recovers short prompts but its
recovery decays with sequence length L, and the spectral channel-MI ceiling I_G of the
last-token vector ensemble — combined with the length-scaled target entropy H(tokens|L) —
predicts that decay. This is the **complement** of the prior per-position-residual null
([[exp:vec2text-feedback-null]]): per-position resid has no bottleneck (feedback null); the
single last-token vector does (recovery is capacity-limited).
**Date**: 2026-06-24
**Threat model**: WEIGHTS-PUB + aux corpus (adversary has the public model and an auxiliary
text corpus to train the adapter; no access to the target's private corpus).

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| **C1 — Bottleneck leakage with length decay.** A trained Rep2Text adapter recovers text from one L10 last-token residual well above a no-information control for short prompts, and recovery (token-F1 / ROUGE-L) decays monotonically with sequence length L. | Establishes the surface leaks AND that the leak is capacity-limited (distinguishes it from the per-position-resid null). | Rep2Text recovery (token-F1, ROUGE-L) per length bucket >> control (random soft-prompt / shuffled-residual) for short L; monotone decay across buckets; gap to control shrinks with L. | B1, B2 |
| **C2 — Matched probe tracks recovery.** The geometry-only spectral channel-MI ceiling I_G of the last-token residual ensemble, scaled by target entropy (I_G / (L·H_token) rate-distortion proxy), correlates with recovery across the length sweep (Spearman ρ, plaintext→length strata). | Tests the core campaign thesis (attack-independent IT measure predicts attack success) on a NEW channel; reuses the validated spectral probe. | (bits from I_G readout) vs (token-F1 recovery) across L buckets: Spearman ρ ≥ 0.7 ⇒ correlates; report bits canonical + readout both. | B3 |

**Anti-claims to rule out:**
- *"Recovery is just length-prior memorization, not residual leakage"* → controls B2: (a) random/zero
  soft-prompt decoder baseline (decoder's own LM prior), (b) shuffled-residual (break the h↔text pairing),
  (c) mean-residual (no per-prompt info). C1 only holds if real-residual >> all three.
- *"The probe correlation is circular (probe ≈ attack)"* → I_G is computed **geometry-only** from the
  residual covariance spectrum; it never trains a decoder, never sees target text, cannot be computed by
  running the attack. Document this explicitly in the audit.
- *"Decay is a decoder-capacity artifact, not an information bottleneck"* → the I_G ceiling is a property
  of the *source vector ensemble* independent of the decoder; if decay tracks I_G/(L·H) it is information,
  not decoder size.

## Paper Storyline
- **Main paper must prove**: C1 (bottleneck leakage + length decay) and C2 (matched probe tracks it).
- **Appendix can support**: adapter capacity (k soft tokens) ablation; layer choice rationale (L10).
- **Experiments intentionally cut**: full retrain-per-bucket (one adapter, buckets are eval strata);
  multi-seed adapter training (single seed pilot; note variance risk); decoder-size sweep.

## Experiment Blocks

### Block 1 (B1): Rep2Text main recovery + length sweep — **MUST-RUN**
- **Claim tested**: C1.
- **Why this block exists**: the anchor result — does the surface leak, and is the leak length-limited?
- **Dataset / split / task**: corpus stratified by token length into buckets (e.g. [≤12], [13–18], [19–25],
  [26+] tokens). Train adapter on aux split (≈70%), eval on held-out (≈15%), val (≈15%). If
  release-gate-512 (512 prompts, 6–25 words → ~8–35 tokens) is too small / too narrow, **generate a
  length-stratified aux corpus** (~1.5–2k prompts spanning 8–48 tokens) procedurally; document generation.
- **Compared systems**: Rep2Text adapter (real residual) vs the three controls in B2.
- **Metrics** (bits canonical for the probe; per-secret readout here): **token-F1** (primary, order-free
  set overlap in shared Qwen3 token space), **ROUGE-L** (sequence), exact-token top-k as secondary.
- **Setup**: source = Qwen3-4B last-token resid @ L10 (d=2560); adapter = MLP(2560→hidden→k·d_dec) → k
  soft-prompt embeddings (k≈8); frozen decoder = **Qwen3-1.7B** (shared Qwen3 tokenizer; backprop through
  frozen decoder updates only the adapter); loss = teacher-forced next-token CE; bf16; AdamW; ≤ a few
  epochs; eval = autoregressive greedy generation from soft prompts.
- **Success criterion**: real-residual token-F1 >> all controls for short buckets; monotone decay with L.
- **Failure interpretation**: (a) no gap over control ⇒ surface does not leak via this attack (negative,
  first-class — queue stronger adapter); (b) flat with L ⇒ no bottleneck after all (contradicts thesis,
  report).
- **Table/figure target**: Fig 1 (recovery vs L, real vs controls); Table 1 (per-bucket F1/ROUGE/bits).
- **Priority**: MUST-RUN.

### Block 2 (B2): Controls / anti-memorization — **MUST-RUN** (folded into B1 run)
- **Claim tested**: C1 anti-claim (length-prior memorization).
- Systems: (i) **random/zero soft-prompt** (decoder LM prior only), (ii) **shuffled-residual** (adapter
  fed a residual from a *different* prompt — breaks h↔text pairing, same marginal), (iii) **mean-residual**
  (every prompt gets the ensemble-mean vector — no per-prompt info).
- Metric/criterion: real-residual must beat all three by a clear margin in short buckets.
- Priority: MUST-RUN (cheap — reuse the trained adapter; only eval-time changes).

### Block 3 (B3): Matched spectral channel-MI probe vs recovery — **MUST-RUN**
- **Claim tested**: C2.
- **Why**: campaign thesis — attack-independent IT measure predicts attack success on this channel.
- **Probe**: per length bucket, take the last-token residual ensemble {h_i} (d=2560), compute the
  channel-MI ceiling **I_G = ½ Σ_j log₂(1 + λ_j/σ²)** from the covariance eigenspectrum {λ_j} (σ² a fixed
  reference-noise floor; reuse the embed-vec2text spectral-probe convention). Report I_G in **bits**, plus
  the rate-distortion proxy **I_G / (L · H_token)** (H_token = empirical per-token entropy of the corpus).
  Geometry-only: no decoder, no target text → not the attack in disguise.
- **Metric**: Spearman ρ between {bits proxy} and {token-F1 recovery} across buckets (+ across a noise
  sweep on the residual if buckets alone give too few points: add σ_inject ∈ {0, small, large} as extra
  sweep points to populate the (bits, recovery) plane — this is the "plaintext→defense-parameter" sweep).
- **Success**: ρ ≥ 0.7 ⇒ probe tracks recovery (CORRELATES → claim+proof). ρ < 0.7 ⇒ that IS the finding
  (decide weak-attack vs non-matched-probe; self-append follow-up, spawn-depth 1).
- **Table/figure target**: Fig 2 (bits vs recovery scatter across the sweep).
- **Priority**: MUST-RUN.

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Capture L10 last-token resid (Qwen3-4B) for the aux corpus; build length buckets; sanity (shapes, bucket counts) | `capture` | residuals captured, ≥ ~80 eval prompts/bucket | ~2–4 min GPU | low |
| M1 | Adapter overfit sanity on a tiny train subset (does the pipeline learn at all?) | `adapter-smoke` | train CE drops, can echo a few train prompts | ~3–5 min GPU | med (backprop through frozen 1.7B) |
| M2 | Train the real adapter; eval real + 3 controls per length bucket | `adapter-train-eval` | real >> controls for short L | ~15–25 min GPU | med (wall-time; confirm saturation, serial) |
| M3 | Compute spectral channel-MI probe + correlation across buckets/σ-sweep | `probe` (CPU/GPU eigh) | ρ computed; CORRELATES y/n | ~1–2 min | low |
| M4 | result-to-claim → experiment-audit → (proof if C2 holds) → report HTML | skills | gate file written | — | low |

**Run discipline**: every GPU step via `TALENS_SURFACE=resid-rep2text scripts/harness/run_step.sh <name> -- scripts/run_in_rocm.sh python3 …`, serial, one GPU process at a time.

## Compute and Data Budget
- **Total estimated GPU**: ~25–35 min wall (M0+M1+M2 dominate; M2 is the only >10-min step → confirm iGPU
  saturation via perf gate before launch, then PROCEED).
- **Data prep**: procedurally generate a length-stratified aux corpus (~1.5–2k prompts, 8–48 tokens) if
  release-gate-512 is too narrow; else reuse + augment. Deterministic seed; commit the corpus.
- **Human eval**: none.
- **Biggest bottleneck**: adapter training time through the frozen 1.7B decoder; mitigate with bf16,
  gradient checkpointing if needed, small adapter, capped epochs, batched teacher forcing.

## Risks and Mitigations
- **Backprop through frozen 1.7B too slow on iGPU** → use Qwen3-1.7B (not 4B) as decoder; cap steps; if
  still >25 min, drop to a smaller k and fewer epochs; smoke-test M1 first.
- **Too few length buckets → unstable Spearman** → add residual-noise σ sweep points to populate the
  (bits, recovery) plane (also doubles as the plaintext→parameter sweep the measurement loop wants).
- **Corpus length range too narrow (6–25 words)** → generate wider 8–48-token aux corpus.
- **Probe circularity worry** → I_G is covariance-spectrum-only; documented in B3 + audit.
- **Decoder LM prior inflates recovery** → the random/mean/shuffled controls (B2) subtract the prior.

## Final Checklist
- [x] Main paper tables covered (Table 1 recovery×L; Fig 2 bits×recovery)
- [x] Novelty isolated (controls subtract decoder prior + length-prior memorization)
- [x] Simplicity defended (one adapter, buckets = eval strata, no per-bucket retrain)
- [x] Frontier component justified (adapter→frozen-decoder is the Rep2Text method under test, not decoration)
- [x] Nice-to-have (k-ablation, layer sweep) separated from must-run
