---
type: research
status: current
created: 2026-06-26
updated: 2026-06-26
tags: [dp, privacy-utility, privacy-funnel, no-free-lunch, split-inference, learned-noise, denoise, generation-vs-downstream, obfuscation, llm, mlm, literature-review]
companion: research-wiki/claims/privacy-funnel-llm-regime-split.md
---

# Smarter DP for confidential LLM inference — the privacy–utility fork

**Decision-grade question.** Our naive defense (isotropic Gaussian noise on the input embedding,
propagated) is bleak: utility collapses *before* recovery does. Is there a smarter DP/obfuscation
scheme that **reduces input recovery faster than it degrades utility** — and is that even possible?

**Verdict.** For **faithful generation** the answer is *no* under perturbation-DP — it is **provably
bounded** (Privacy Funnel + a No-Free-Lunch theorem) and we measured the cliff ourselves. The
literature's wins come from one of two escapes: (a) **reframe utility as a coarse downstream task**
(then add-noise-then-denoise reaches ~95% of non-private), or (b) **leave perturbation-DP for
structural/key-based obfuscation** (covariant-obfuscation / null-space schemes reach 96–101%, because
they are not bound by the perturbation theorem). Chasing perturbation-DP for generation is a losing
fork; the two escapes are the live directions, and (b) is the repo's own motivating threat model.

## Definitions (glossary)

- **LDP / local DP**: noise added on the *user* side before release, so the server never sees clean
  data. Our LocalDP hook (Gaussian on the input embedding) is local DP. ε is the per-release budget;
  σ = C·z/ε with sensitivity C, z = √(2 ln(1.25/δ)).
- **Perturbation-DP**: any mechanism whose privacy comes from *adding randomness* (Gaussian/Laplace/
  exponential/metric noise) to the data or representation. Contrast **structural obfuscation** below.
- **Faithful-generation utility**: the released/processed representation still supports the model
  generating text that is *faithful to the true private input* (next-token fidelity). The
  **funnel-hard** regime.
- **Downstream-task utility**: accuracy on a *coarse function* of the input (sentiment, NLI, ICD-9
  coding, retrieval). The **funnel-easy** regime, because I(rep;task) ≪ I(rep;input).
- **Fluency ≠ faithfulness**: MAUVE / BLEU / coherence measure whether output text is *well-formed*,
  not whether it preserves the *private content*. A perturbed prompt can yield fluent text that says
  something else — high fluency is **not** evidence of input faithfulness.
- **Smashed data**: the intermediate representation a split-inference client sends to the server.
- **Denoise / recovery step**: a client- or server-side module (denoiser, extraction LLM, soft-prompt,
  hidden-state corrector, distillation) that *claws back* utility lost to the noise, using side
  information (clean client input, public models) the adversary lacks.
- **Privacy Funnel**: the log-loss dual of the Information Bottleneck — minimize I(rep;X) subject to
  I(rep;T) ≥ threshold. The formal object of "reduce recovery faster than utility."
- **Structural / key-based obfuscation**: privacy from an invertible-with-a-secret transform of
  data+weights (covariant obfuscation, semantic null-space), *not* from added noise. A different
  threat model (the secret is a key, not an ε); not bound by the perturbation No-Free-Lunch theorem.

## 1. The motivating result — our naive DP is funnel-hard

Utility sweep on Pythia-160M (`exp:dp-utility-vs-eps-160m`, `refine-logs/pythia-depth/dp_utility.json`):
next-token accuracy-retention vs the non-private baseline collapses to **−50% by ε≈110**, while the
input token stays recoverable at ε≤64. The leakage grid (ε∈{64,32,16,8}) sat entirely **below** the
usable band — worst quadrant (input recoverable *and* model broken). The defensible operating band is
ε∈[128,512]. This is not a mechanism failure; §2 shows it is the predicted behaviour.

## 2. The formal frame — perturbation-DP is provably bounded for generation

Two independent results say the same thing:

- **Privacy Funnel** (Makhdoumi, Salamatian, Fawaz, Médard 2014, arXiv:1402.1774;
  [[paper:makhdoumi2014_from_information_bottleneck]]). Under log-loss, privacy leakage and utility
  reduce to I(rep;X) and I(rep;T). The achievable region is governed by the X↔T dependency: when T is
  near-deterministic in X (**generation** — the task *is* the content), the funnel floor is high, so
  I(rep;X) cannot fall without collapsing I(rep;T). The curve is generally non-convex; perfect privacy
  with positive utility requires *private-only* information separable from the utility-relevant part —
  which generation lacks.
- **No-Free-Lunch Theorem for Privacy-Preserving LLM Inference** (Zhang et al. 2024, arXiv:2405.20681;
  [[paper:zhang2024_free_lunch_theorem]]). Proves any prompt-privacy *randomization* incurs an
  unavoidable utility cost — an impossibility result specifically for **perturbation-based** inference
  privacy. Found independently by two of our literature agents → high confidence. (The agents
  misattributed it as "Hu et al. 2025"; authoritative arXiv metadata is Zhang 2024.)

Together: **the "reduce recovery faster than utility" objective is achievable only to the extent the
task is separable from the raw input.** This is exactly `claim:privacy-funnel-llm-regime-split`
(status: drafted).

## 3. The landscape (cross-source: arXiv · OpenAlex · web)

Mechanism families and the **best utility-retention each reports**. Regime tag is the key column.

| Scheme (year, id) | Mechanism | Regime | Best utility retention | DP? |
|---|---|---|---|---|
| **Split-and-Denoise** (ICML'24, 2310.09130) | client-side **denoiser** on noisy embeddings | downstream cls | **~95% of non-private** (4.3–5.3% loss); +10–22% over RAPT | LDP (dχ) ✓ |
| **DP-Forward** (CCS'23, 2309.06746) | analytic **matrix-Gaussian** forward-pass noise | downstream cls | "**almost hits non-private**"; +7.7pp over DP-SGD | LDP ✓ |
| **PPFT** (2026, 2604.06831 — web, verify) | encoder-decoder align + noise-aware adaptation | downstream QA | **95.6%** of noise-free (ε=1–75) | dχ ✓ |
| **Selective-DP** (2021, 2108.12944) | privatize only **sensitive tokens** | downstream | high (bulk unnoised) | DP ✓ |
| **Shredder** (2019, 1905.11814) | **learned** per-feature noise distribution | downstream cls | beats isotropic at matched privacy | DP ✓ |
| **Clinical DP pipelines** (2025, 2511.14936) | KD from DP-trained teacher | downstream (ICD-9) | **63%** of non-private at ε∈{4,6} | central DP ✓ |
| **InferDPT / RANTEXT** (TDSC'25, 2310.12214) | exponential-mech perturbed prompt + **extraction LLM** | **generation** | ε=3 utility / ε=6 privacy | Coherence 0.736 *beats* GPT-4 (0.632); but **MAUVE 0.587 vs 0.671 ≈ 88%** — the faithfulness axis lags (⚠ §4) | DP ✓ |
| **ProSan** (2024, 2406.14318) | LLM-guided **selective token sanitization** (utility-weighted) | QA / summ / code | self-information (no ε) | accuracy drop **0.4–2.3%**; RougeL Δ≤0.012 | **No** (self-info) |
| **HiddenEcho** (ICLR'26, web — verify) | server→client **hidden-state correction** | cls + gen | +46.9% AUC **over DP baseline** (not %-of-non-private) | metric-DP ✓ (⚠) |
| **DEL** (2026, 2602.11513 — web, verify) | DP n-bit quantization + **soft-prompt** restore | generation | ~90.4% **coherence** at ASR 0.02 | μ-GDP ✓ (⚠ §4) |
| **MetaMorphosis** (2023, 2305.07815) | learned encoder + de-correlation loss (vision) | downstream | beats adversarial baselines | DP ✓ |
| **Covariant Obfuscation / AloePri** ([[paper:lin2026_towards_privacypreserving_llm]], = repo's AloePri defense) | **structural**: covariant data+weight transform | inference (671B; Qwen2.5-14B) | **0.0–3.5% loss**, <5% tokens recovered | **No** (key-based) |
| **OSNIP** (2026, 2601.22752 — web, verify) | **structural**: key-dependent semantic null-space | cls + gen | **~101%** of baseline | **No** (encryption) |

## 4. The crucial scrutiny — "generation recovery" claims do not break the bound

Several 2025–26 papers *claim* the generation regime. None is a genuine counterexample:

- **InferDPT, DEL, HiddenEcho** measure **fluency** (MAUVE / BLEU / coherence) or **gains over a DP
  baseline**, not *faithfulness to the true private input* and not *% of non-private*. Fluent text
  from a perturbed prompt is precisely what the funnel permits — it does not show the private content
  survives. (Agents even disagreed on InferDPT's ε: 1–3 vs 6 — pin from the paper before trusting.)
- **OSNIP (~101%), Covariant Obfuscation (96–100%)** are **not perturbation-DP** — key-based
  structural obfuscation with *no ε*. They escape the No-Free-Lunch bound by changing the threat model
  (secret = key, not a noise budget). Their high numbers are the *tell*: the field gets generation-
  regime utility only by **leaving DP**.

**So: no genuine perturbation-DP scheme demonstrates faithful-generation recovery.** Strict
noise→denoise tops out ~95% on **downstream classification**.

## 5. Verdict for our fork

1. **Perturbation-DP for faithful generation is a provably losing fork.** Funnel + No-Free-Lunch +
   our own cliff agree. Do not invest here expecting a smarter noise to win.
2. **Escape A — reframe utility as a coarse downstream task.** Then add-noise-then-denoise (SnD,
   DP-Forward, PPFT) reaches ~95% of non-private. Cheap to test on our models: add a downstream-task
   readout to `dp_utility_sweep.py` and confirm task-accuracy retention stays high where next-token
   perplexity collapses. This would promote `claim:privacy-funnel-llm-regime-split` to verified.
3. **Escape B — structural/key-based obfuscation** (covariant obfuscation / **AloePri** = the repo's
   WEIGHTS-PUB motivating model; OSNIP). The literature's best numbers (96–101%) live here because it
   sidesteps the perturbation bound. **The decisive caveat, triple-confirmed across all three source
   sweeps:** these methods buy their high utility by **abandoning a formal ε-DP guarantee** for
   *empirical* attack-resistance (<5% token recovery, no ε). Within strict ε-DP the tradeoff stays
   binding (DP-Forward ~8-pt RTE drop at ε=1; InferDPT's MAUVE faithfulness gap). So Escape B is not
   "DP done smarter" — it is **a different threat model** (secret = key), and it is the repo's existing
   AloePri/covariant-obfuscation direction. The honest fork is therefore: *formal ε-DP and ~95%
   downstream utility*, **or** *empirical key-based obfuscation and ~99% utility including generation*
   — not both a strong ε-guarantee and faithful-generation utility.
4. **Within perturbation-DP, the only structural lever with independent support is noise *placement*:**
   "Enhancing Accuracy-Privacy Trade-Off in DP Split Learning" (IEEE TETCI 2024) finds **noise in later
   layers** gives the best balance — corroborating our anisotropic / release-point lever (I_G
   eigendirections). Worth the cheap **subspace-gap measurement** (recovery eigendirections vs
   next-token-utility gradient directions) to bound how much Lever 1 can ever buy.

## 6. Mechanism note — DP onto a closed-source text API (embed → noise → unembed → text)

A natural way to make an embedding-level mechanism work against a **text-only API**: client embeds →
adds Gaussian noise → maps each noised vector to its **nearest token** (unembed) → sends the resulting
**text**. This is sound and already named: it is **word-level metric local DP** — Feyisetan 2020
(calibrated multivariate perturbations), SANTEXT/CUSTEXT, and InferDPT's RANTEXT. The nearest-token map
is **post-processing** of the Gaussian mechanism, so it inherits the metric-DP (`d_χ`) guarantee; the
output is text, so any closed-source API accepts it. It is *the* standard formal-DP route for the
closed-source cell.

It does **not** escape the bound, and the quantization makes it slightly worse:
- Still perturbation-DP → funnel + No-Free-Lunch apply. `→ nearest token` **discards the continuous
  noised vector**, paying information twice (noise + rounding to vocab) → strictly *lossier* than
  continuous-embedding noise.
- Privacy comes **only from token substitutions**: positions whose token does not flip leak the
  original word verbatim; enough σ to substitute meaningfully also wrecks meaning — the known MLDP
  utility problem (named explicitly in 1-Diffractor).
- Embedding geometry is frequency-biased / anisotropic → a fixed σ over-protects rare/isolated tokens
  and under-protects frequent/clustered ones; Feyisetan's Mahalanobis calibration is the fix.

Crucially, sending the perturbed prompt leaves the API answering the **perturbed** intent. The SOTA
version adds a **local extraction module** that realigns the output to the true intent — i.e.,
**this scheme ≈ InferDPT minus extraction**, so InferDPT/RANTEXT is its strong form, and it inherits the
faithfulness gap (coherence ≈ GPT-4, **MAUVE ≈ 88%**).

**Takeaway for the closed-source cell:** start from InferDPT/RANTEXT + Mahalanobis-calibrated noise; do
not reinvent token-perturbation. Honest ceiling: *fluent, ~88%-faithful generation at ε≈3–6* — not
faithful recovery.

## 7. Resources & connections

- **Benchmark to mine**: VFLAIR-LLM (KDD'25, `10.1145/3711896.3737411`, github FLAIR-THU/VFLAIR-LLM) —
  9 defenses × 5 attacks × 18 datasets for split-learning LLMs. The systematic frontier, not a single
  paper's self-comparison.
- Wiki: `claim:privacy-funnel-llm-regime-split`; papers ingested — Privacy Funnel, Split-and-Denoise,
  DP-Forward, MetaMorphosis, Shredder, Selective-DP, maximal-correlation. **To ingest**: No-Free-Lunch
  (2405.20681), InferDPT (2310.12214), and the 2026 frontier markers once their IDs are verified.
- Experiment: `exp:dp-utility-vs-eps-160m` (the cliff).

## 8. Provenance & confidence

- Sources swept (4): **arXiv** (fetcher), **OpenAlex** (citation graph + semantic search),
  **Semantic Scholar** (Graph API; SnD = **46 citing**, InferDPT = 8 — note OpenAlex under-counted SnD
  at 4, so the lineage is more active than the citation graph first suggested), **web** (Gemini and
  Exa skills were unavailable — no `gemini` CLI, no `EXA_API_KEY` — both agents fell back to
  WebSearch/WebFetch; S2 was rate-limited, no API key).
- **High confidence** (multi-source): Privacy Funnel + No-Free-Lunch (the bound); SnD ~95% / DP-Forward
  ~92% (ε=1, RTE) / RAPT ~98.8% / TextObfuscator ~99% — all **downstream**; the regime split; and the
  crux that the 96–101% "wins" (OSNIP, AloePri/covariant) **abandon formal ε-DP** for empirical
  attack-resistance (independently stated by the Semantic Scholar and web sweeps).
- **Lower confidence / verify**: the 2026 arXiv IDs (HiddenEcho, DEL, OSNIP, PPFT) are web-reported and
  abstract-level; some are future-dated. InferDPT's exact ε is unresolved across agents. Treat all
  2026 "generation recovery" numbers as claims-pending-audit until the PDFs are read.
