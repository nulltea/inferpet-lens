# Experiment Results — resid-rep2text (Rep2Text on Qwen3 L10 residual)

**Date**: 2026-06-24 · **provenance**: `scripts/spikes/rep2text_run.py`,
`refine-logs/resid-rep2text/runs/rep2text_results.json`; corpus
`corpora/rep2text-stratified.txt` (1965 prompts, 6–59 words; built by
`scripts/spikes/rep2text_build_corpus.py` from cached piqa/mmlu/if_eval).
**Setup**: source = Qwen3-4B last-token residual @ L10 (d=2560); attack = Rep2Text adapter
(MLP 2560→2048→2048→8·2048, k=8 soft tokens) → FROZEN Qwen3-1.7B decoder; teacher-forced CE,
10 epochs, 762 train / 138 test; greedy gen; bf16; seed 20260624. Privacy sweep = isotropic
Gaussian noise injected on the raw residual at frac∈{0,.5,1,2,4,8}×rms (rms=0.527). Probe =
geometry-only spectral channel-MI I_G of the FULL residual ensemble at the matched σ (reference
floor σ_ref = rms/√100). Bits canonical + token-F1/ROUGE-L readout.

## Training sanity
CE 3.07 → 2.09 nats over 10 epochs (the adapter learns; the frozen decoder is steered by the
8 soft tokens). Capture cached (0.4 s reuse).

## C1 — recovery vs sequence length (token-F1 @ frac=0; 5-draw shuffled null, paired bootstrap)
N=23 prompts/bucket. mean-residual control ≈ 0.002 (no per-prompt info → ~0): metric well-behaved.
Shuffled null across-draw std ≈ 0.003 (stable). Gap = real − mean-over-draws shuffled.
| bucket | L (tok) | real F1 | shuffled F1 | leakage gap | bootstrap 95% CI | p(gap≤0) |
|---|---|---|---|---|---|---|
| ≤12  | 9  | 0.076 | 0.045 | +0.031 | [+0.016, +0.048] | 0.000 |
| 13–18| 14 | 0.088 | 0.074 | +0.015 | [+0.002, +0.027] | 0.009 |
| 19–24| 21 | 0.146 | 0.105 | +0.041 | [+0.024, +0.059] | 0.000 |
| 25–32| 28 | 0.138 | 0.115 | +0.023 | [+0.010, +0.036] | 0.000 |
| 33–48| 39 | 0.209 | 0.156 | +0.054 | [+0.028, +0.079] | 0.000 |
| 49+  | 59 | 0.244 | 0.155 | **+0.089** | [+0.040, +0.147] | 0.000 |

- Real F1 **exceeds** the shuffled-residual null at **every** bucket and **every** bootstrap 95% CI
  **excludes 0** (p ≤ 0.009) ⇒ genuine, residual-specific leakage is **statistically significant** —
  but **modest** (+0.015 to +0.089 token-F1; most raw F1 is the decoder LM / common-token prior,
  shuffled ≈ 0.108).
- The gap is not strictly monotone but is clearly **largest for the longest prompts** (+0.089 @ L=59
  vs +0.031 @ L=9) and **does not decay** ⇒ **C1 (bottleneck-induced length decay) is REFUTED** at L10.

## C2 — does the matched capacity probe predict recovery?
**Across length (the task's operative test):** rd-proxy = min(I_G, H_X(L))/H_X(L) ≈ **1.000 for
every bucket** at low noise because I_G(plaintext)=**2856 bits ≫** H_X(L≤59)≤1026 bits. Capacity is
**non-binding** (slack >2×; even under an aggressive 8–11 bit/token entropy estimate, H_X(L=59)≈470–650
bits ≪ 2856). At the **plaintext** operating point `r(L)≡1`, so the **length-only** Spearman is **undefined** (constant
predictor; lemma below). Over the **full length×σ grid (n=36)** Spearman(rd-proxy, F1) = **0.176** ⇒ the
probe is **vacuous across length**: it predicts ~no variation where recovery clearly varies. (Entropy
basis: the table uses per-bucket **mean** token length; the capped maximum length is 64 tokens ⇒
H_X(max) ≤ 1102 bits ≪ I_G=2856.)

**Across the noise sweep (capacity made artificially binding):**
| raw σ | I_G (bits) | real F1 | shuffled F1 | leakage gap |
|---|---|---|---|---|
| 0.053 | 2856 | 0.150 | 0.113 | +0.038 |
| 0.269 | 1019 | 0.145 | 0.107 | +0.038 |
| 0.530 | 520  | 0.143 | 0.109 | +0.034 |
| 1.055 | 219  | 0.131 | 0.115 | +0.017 |
| 2.109 | 78   | 0.119 | 0.110 | +0.008 |
| 4.217 | 24   | 0.077 | 0.079 | −0.001 |

Spearman(I_G, overall F1) = **1.0**; Spearman(I_G, leakage gap) = **0.943** — ordinal agreement, but
**badly calibrated**: capacity must be destroyed **>80 %** (2856→520 bits) before the genuine
leakage gap even begins to respond (flat +0.038→+0.034 over that range); the gap only collapses once
I_G < ~220 bits. The probe **orders** recovery only deep in an artificial high-noise regime, far from
the plaintext operating point the task asks about.

## Caveats (from cross-model integrity audit — WARN, no fraud)
- **Scope**: one source/decoder pair (Qwen3-4B→Qwen3-1.7B), layer L10, **single seed, single adapter**,
  N=23/bucket. All conclusions are scoped to *this setup*; multi-seed / decoder / layer sweeps are cut.
- **Shuffled null** is a random permutation, not a strict derangement → a possible fixed point per draw
  (~1/23) pairs a residual with its own text, which *inflates* the null and makes the leakage gap
  **conservative** (understated), not inflated.
- **Standardization** uses the full-ensemble per-dim mean/std (incl. test rows) — negligible, and
  applied identically to real and controls.
- **σ matching**: probe σ = √(injected² + σ_ref²) with a fixed geometry-only floor σ_ref=rms/√100; the
  noise *family* (isotropic raw-space Gaussian) is matched, the scalar is offset by σ_ref (binding only
  at the floor where the attack has no injected noise).
- **H_X = L·log₂(vocab)** is an explicit UPPER entropy proxy; Spearman across length is invariant to this
  constant, and the non-binding conclusion survives an aggressive 8–11 bit/token estimate.

## Headline (negative / refutation — first-class)
For this Qwen3-4B→1.7B L10 setup, on the **plaintext** residual the single last-token vector is **not** a binding
information-capacity bottleneck (I_G ≈ 2856 b ≫ sequence entropy: H_X(max 64 tok) ≤ 1102 b), so Rep2Text
recovery is **extraction-limited, not capacity-limited**. Consequently the geometry-only **spectral
channel-MI (capacity) probe does NOT predict recovery across sequence length** (length-only ρ undefined
since r≡1; full length×σ grid ρ=0.18, vacuous); it
orders recovery only deep in an artificial high-noise regime (across-σ ρ=1.0 / gap-ρ=0.94) reached
only after >80 % of capacity is destroyed. Genuine residual-specific leakage is statistically
significant at every length (all bootstrap CIs exclude 0) but **modest**, and is **largest for the
longest prompts** — it **does not decay**, the opposite of the bottleneck prediction. This refines the
prior per-position-resid null ([[exp:vec2text-feedback-null]]): even the single-vector "bottleneck"
is not capacity-binding at L10.

## Diagnosis & follow-up (measurement-loop step 4: "No")
Most plausible cause = **probe not channel-matched**: a capacity ceiling cannot predict recovery when
capacity is slack (it equals 1 across all lengths). For this setup the data **rule out** capacity as the
active constraint at plaintext (>80 % of capacity is destroyable before genuine leakage responds), but
they do **not** by themselves prove the attack is near-optimal — low absolute recovery could also reflect a
weak adapter, limited training data, decoder mismatch, or the residual simply not encoding exact text
in an extractable form. What *is* forced: the **capacity** probe is the wrong matched probe here; a
matched probe must measure **extractable** information under a bounded decoder (V-information /
𝒱-usable information), not channel capacity. → self-append follow-up `resid-rep2text-v2`
(spawn-depth 1): a matched V-information probe **and** a stronger-attack capacity ablation to separate
attack-weakness from representation limits.
