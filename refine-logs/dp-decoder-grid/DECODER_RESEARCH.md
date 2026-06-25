---
type: research
status: current
created: 2026-06-25
updated: 2026-06-25
tags: [decoder, inversion, residual-stream, tuned-lens, vec2text, dp, warm-start, probe-architecture]
companion:
---

# Optimal decoder architecture for residual-stream hidden-state → token-id recovery

**Scope.** Per-position residual vector `r_L ∈ ℝ^2304` (gemma-2-2b) → token identity, WEIGHTS-PUB
threat model (adversary knows weights + embedding table E), with and without input-DP noise.
Small-data regime: a few hundred to ~2k **vocab-disjoint** training rows. Goal = **max recovery**,
not faithfulness. Literature + planning only; no runs.

**Bottom line up front.** For a *single* per-position residual at a layer where the optimal
map is near-linear (e.g. L0), **a regularized affine map is the right baseline and a well-built
non-linear decoder should at best match it, not beat it** — and your data constraint, not the
architecture, is the binding limit. The literature that *does* beat affine for residual→token does
so by exploiting (a) **sequence/pooled context** (vec2text, GEIA, Rep2Text) or (b) **a strong
generative language-model prior** over the output token sequence (BeamClean, GEIA, Rep2Text) — not
by making the per-vector regressor deeper. Concrete ranked recommendation in §6.

---

## 1. Tuned Lens & logit lens — the closest established "residual → token" decoder

- **Logit lens** (nostalgebraist 2020): apply the model's own final LayerNorm + unembedding `E`
  directly to an intermediate residual. **Zero learned parameters** — a fixed linear readout. Works
  but is "brittle" across layers/models.
- **Tuned Lens** (Belrose et al. 2023, [arXiv:2303.08112](https://arxiv.org/abs/2303.08112)):
  learns a per-layer **affine** translator and composes it with the frozen unembedding:
  `TunedLens_ℓ(h_ℓ) = LogitLens(A_ℓ h_ℓ + b_ℓ)`. So `A_ℓ ∈ ℝ^{d×d}`, `b_ℓ ∈ ℝ^d`.

  Key design facts (verbatim from the paper):
  - **Affine, not non-linear.** They explicitly chose *not* to learn a new `|V|×d` unembedding per
    layer à la Alain & Bengio 2016, because that "requires considerably more training steps and a
    larger batch size." The `d×d` affine is the deliberate sweet spot between the parameter-free
    logit lens and a full re-learned head. **No non-linear/MLP lens ablation is reported.**
  - **Warm-start = identity.** "We initialized all translators to the identity transform." This is
    the canonical instance of *guaranteeing ≥ the logit-lens baseline by initialization* — exactly
    the warm-start idea you are already reaching for.
  - **Regularization / optimizer.** SGD + Nesterov, base LR 1.0 (0.25 with final layer), linear
    decay over 250 steps; **weight decay 1e-3**; gradient clipping to norm 1; batch `2^18` tokens.
  - Result: tuned lens is "more predictive, reliable, and unbiased" than logit lens.

**Implication for you.** The single most-validated residual→token decoder in the field is an
**affine map composed with the frozen unembedding, identity-initialized, weight-decayed**. It does
*not* use a hidden non-linearity. This is strong prior evidence that for single-position residual,
affine is the right backbone and non-linearity is unproven upside. The one structural idea you are
*not* yet using and the tuned lens *is* — composing with `E` rather than regressing to a free
2304-dim target — is potentially the bigger lever than depth (see §6, R1).

Repo: `tuned-lens` (EleutherAI), `pip install tuned-lens`. Directly reusable as a reference impl of
the affine-into-unembedding decoder with identity warm-start.

---

## 2. Embedding / hidden-state inversion attacks — what decoders actually win

| Work | Surface | Decoder architecture | What makes it win |
|---|---|---|---|
| **Song & Raghunathan 2020** ([arXiv:2004.00053](https://arxiv.org/abs/2004.00053)) | sentence embedding | white-box: **multi-label MLP** over vocab (BCE, multiset/"which words" prediction); also a **generative LSTM** for ordered recovery | First to frame inversion; MLP predicts *set* of tokens, not a regression to embedding space |
| **GEIA / Li et al. 2023** ([arXiv:2305.03010](https://arxiv.org/abs/2305.03010), [code](https://github.com/HKUST-KnowComp/GEIA)) | sentence embedding | **GPT-2 generative decoder** + a **linear projection module** mapping the victim embedding into the decoder's input-embedding space; decoder is randomly init then trained | Generative LM prior recovers the *whole coherent sentence*; beats classification-style inversion on word metrics |
| **vec2text / Morris et al. 2023** ([arXiv:2310.06816](https://arxiv.org/abs/2310.06816)) | pooled text embedding | **T5 encoder-decoder corrector**, iterative: a base "zero-step" model proposes text, then a corrector conditioned on (target emb, current hypothesis text, hypothesis emb) refines over ~20–50 steps with beam search; embedding projected to a sequence of soft tokens via a small MLP | **Iterative correction + LM prior**: 83.4 BLEU / 60.9% exact at 32 tokens (ada-002). SOTA for pooled-embedding→text |
| **Rep2Text / 2025** ([arXiv:2511.06571](https://arxiv.org/abs/2511.06571)) | **single last-token LLM residual** (closest to you) | **two-layer MLP adapter with gated skip + GELU** (`W1 ∈ ℝ^{d×d_hid}`, expansion f=0.5) projecting the residual to **k=16 soft token embeddings**, fed to a **frozen generative LLM** that autoregressively decodes; adapter-only fine-tune, label smoothing ε=0.075 | Adapter is small/linear-ish; the **frozen LLM decoder is the workhorse**. Peak recovery L10–L15; ROUGE-1 ~0.60 at 8 tokens → ~0.30 at 64 |

**Read across the table.** Every method that beats a plain regression-MLP does it the *same* way:
a **small (often near-linear, GELU+skip) front-end** projects the vector into the input space of a
**generative decoder with a strong language prior**, and recovery comes from that prior + (for
vec2text) **iterative correction**, *not* from a deep feed-forward regressor. Crucially:

- The front-end adapters are deliberately shallow (GEIA: a *linear* projection; Rep2Text: a
  2-layer MLP with **gated skip** and GELU, expansion 0.5 — i.e. *narrower* than input). Nobody
  reports needing a wide deep MLP on the vector itself.
- The wins scale with **how much sequence/output context the decoder gets to condition on**, which
  is why pooled-embedding (vec2text) and multi-token-output (Rep2Text k=16) work better than a
  per-position vector → single argmax token.

---

## 3. Probing-architecture tradeoffs — power vs overfitting on small data

For **faithfulness** probing the field wants *low* power (Hewitt & Liang 2019 control tasks,
[aclanthology 2020.emnlp-main.14](https://aclanthology.org/2020.emnlp-main.14/) Voita & Titov MDL,
Pimentel et al. 2020 Pareto probing [arXiv:2010.02180](https://arxiv.org/abs/2010.02180)). **For an
*attack* you want the opposite — maximum recovery** — so the relevant lessons invert:

- **Hewitt & Liang 2019 (control tasks).** A high-capacity probe can memorize a random labeling
  (high *selectivity gap*). For *attacks* you actively want this capacity — but their finding is the
  warning for your **vocab-disjoint** split: with only hundreds–2k rows, a wide MLP will fit
  spurious structure and your *held-out* recovery (the only number that counts) collapses.
  Vocab-disjoint eval is your control task; it is what protects you from reporting memorization.
- **Voita & Titov 2020 (MDL).** Description length = goodness *and* the cost to extract it. Their
  practical knob — **online/variational coding favors probes that reach good fit with few examples
  and few parameters** — is exactly the small-data regime. The lesson: prefer the lowest-capacity
  decoder that reaches the fit, because extra capacity buys description length (overfit), not
  recovery. ([code](https://github.com/lena-voita/description-length-probing))
- **Pimentel et al. 2020 / Pareto probing.** Probe accuracy and complexity trade off on a Pareto
  front; the honest comparison fixes the front, not a single capacity. For you: sweep capacity and
  report the recovery-vs-capacity curve, don't cherry-pick one width.

**Net.** In the few-hundred-to-2k vocab-disjoint regime, **overfitting is the binding constraint
and capacity is a liability past the point that fits the signal.** That is the central reason your
h=384 and h=1024 MLPs lost to ridge: at L0 the signal is ~linear, so the MLP's extra capacity only
added variance on a tiny disjoint train set. This is predicted, not surprising.

---

## 4. Activation & regularization for inversion MLPs

- **ReLU vs GELU/SiLU.** ReLU's hard zero gives the **dying-neuron** failure: a neuron stuck in the
  negative half-space gets zero gradient forever; with `h` large and few data points many neurons
  initialize dead and never recover, so an h=1024 ReLU layer can be *effectively* far narrower —
  consistent with your widening not helping. GELU/SiLU are smooth, pass gradient through the
  negative region, and avoid this. **Empirically the GELU↔SiLU difference is tiny; the load-bearing
  choice is smooth-vs-ReLU, and it matters more for small/narrow nets where a few dead units is a
  large fraction.** (Rep2Text uses **GELU**; this is the relevant precedent for residual adapters.)
  Standard reference points: GELU (Hendrycks & Gimpel 2016, [arXiv:1606.08415]), SiLU/Swish
  (Ramachandran et al. 2017, [arXiv:1710.05941]).
- **Skip / residual to guarantee ≥ linear.** `pred = Linear(x) + MLP(x)` only guarantees ≥ linear
  *if the linear path is actually as good as ridge* — a from-scratch linear path is **not** the
  ridge solution and SGD may not find it within budget on small data. The fix is structural +
  init: **warm-start the linear path from the closed-form ridge solution and zero-init the MLP
  branch's last layer**, so the decoder *starts exactly at ridge* and the non-linear branch can only
  add signal. This is precisely the tuned-lens identity-init trick (§1) applied to your skip-MLP,
  and Rep2Text's **gated skip with a learnable gate initialized near zero** is the published
  instance of "start at the linear/skip path, let the gate open only if it helps."
- **Regularization that matters most here (ranked):**
  1. **Early stopping on a vocab-disjoint val split** — by far the biggest lever in your regime.
  2. **Weight decay** (tuned lens uses 1e-3) — cheap, always on.
  3. **Zero-init of the residual branch's output layer** (start at the linear baseline).
  4. Dropout — secondary; can help a wide MLP but competes with early stopping.
  5. Spectral norm — overkill for a 1-hidden-layer decoder; skip unless instability appears.

---

## 5. DP / denoise-then-invert

**BeamClean** (2025, [arXiv:2505.13758](https://arxiv.org/abs/2505.13758)) is the directly-relevant
work and it answers the "denoiser+inverter?" question: **it is NOT a learned denoiser→inverter
cascade.** It is a **beam-search decoder over token sequences that fuses (a) a surrogate noise
model π_θ jointly estimated during decode, with (b) a frozen language-model prior** (Llama-3.2-1B):

- score update `s' = s · π_θ(y_t | x(w_1:t)) · p_LM(w_t | w_1:t−1)`; beams pruned top-k; noise
  params θ refined from current beams (joint estimate, no separate denoise stage).
- Threat model **matches yours**: adversary sees obfuscated embeddings + the embedding table, no
  access to the model or the obfuscation mechanism.
- Gains over nearest-neighbor under noise are large and *grow with noise*: Gaussian ε=15 → 74.3% vs
  42.1%; Laplace ε=8.5 → 86% vs 18%; PII recovery 60% vs 1.9%.

**Lesson for DP.** Under input-DP the per-vector signal is degraded, so the marginal value of a
**sequence-level LM prior** rises sharply — exactly where a per-position regressor (linear *or*
MLP) saturates and a prior-using decoder pulls ahead. "Split-and-Denoise"
([arXiv:2310.09130](https://arxiv.org/abs/2310.09130)) is the defense-side dual (denoise before the
server sees it); useful as a baseline DP mechanism, not as your attacker.

---

## 6. Ranked recommendation for OUR setting

Setting: **single per-position residual `r_L ∈ ℝ^2304` → one token id**, ~hundreds–2k vocab-disjoint
rows, ±DP. Ranked best→worst expected recovery *given the data constraint*:

**R1 — Affine-into-unembedding, ridge/identity warm-started (DO THIS FIRST; likely the best
single-position decoder).**
Decode as `logits = E · (A r_L + b)` then argmax, with `E` the **frozen embedding table** (you have
it under WEIGHTS-PUB), `A ∈ ℝ^{2304×2304}` init to identity (or to the layer's tuned-lens map),
`b=0`. Train with weight decay 1e-3, early stop on disjoint val.
*Why:* this is the tuned-lens decoder (§1) — the most-validated residual→token map — and it bakes
in the strongest prior available (the real token geometry of `E`) instead of regressing to a free
2304-dim target. Composing with `E` is the structural lever you are currently *missing*; it is more
likely to move recovery than any MLP width. Ref impl: `tuned-lens`.

**R2 — Linear-skip MLP, ridge-warm-started linear path + zero-init non-linear branch + GELU + gate.**
`pred = A_ridge·x (frozen-init, trainable) + g·MLP_GELU(x)` with last MLP layer zero-init and gate
`g` init ~0; **narrow** hidden (`h ≈ 256–512`, i.e. ≤ input, mirroring Rep2Text's f=0.5), 1 hidden
layer, dropout 0.1 optional, weight decay 1e-3, early stop. Combine with R1 (project the *sum*
through `E`).
*Why:* this is the only honest way to **guarantee ≥ ridge** and let non-linearity add *only if it
helps* — which on small disjoint data it usually won't at L0 but may at deeper layers where the map
is more curved. GELU avoids the dead-ReLU collapse that explains your h=1024 failure. **Do not widen
past input dim**; in your regime width is variance, not signal (§3).

**R3 — Generative-adapter decoder (Rep2Text-style) — only if you can relax data or move to
multi-token output.** 2-layer GELU MLP + gated skip → k soft tokens → frozen LLM autoregressive
decode. Needs *much* more data (Rep2Text: 640K seqs) and pays off mainly when decoding a *span*, not
a single position. Queue this as the "stronger attack" if R1/R2 plateau and you can expand training
data or widen the target to a token window.

**R4 — BeamClean-style LM-prior beam decode — the recommended attacker UNDER DP and for
multi-position recovery.** When you sweep input-DP, the per-vector decoders (R1/R2) will lose signal;
a beam decode that fuses an LM prior + jointly-estimated noise model is the published SOTA for
*exactly your threat model* and degrades far more gracefully (§5). Use R1's affine as the per-token
likelihood term inside it. Ref impl: BeamClean paper; reproduce the π_θ · p_LM scoring.

**Not recommended:** a deep (≥2 hidden layer) or wide (h≫2304) plain MLP regressor with from-scratch
init and ReLU — this is what underperformed ridge, and the literature predicts it will keep doing so.
A from-scratch small transformer on a single 2304-vector has no sequence to attend over and only adds
parameters to overfit.

---

## 7. The honest answer to the headline question

**Is a non-linear decoder even expected to beat regularized affine/ridge for single-position
residual→token?**

- **At a layer where the optimal map is ~linear (L0): no, not reliably.** Tuned lens uses affine on
  purpose; your MLPs lost; probing theory says extra capacity = overfit on small disjoint data. The
  realistic best is **affine-into-unembedding (R1)**, possibly with a *gated, zero-init, ridge-warm-
  started* non-linear branch (R2) that the data may or may not switch on.
- **At deeper layers the map is more curved**, so a *narrow* skip-MLP (R2) has a genuine but modest
  upside — bounded by data, not by architecture.
- **The real recovery gains in the literature come from a different axis entirely:** conditioning a
  **generative LM-prior decoder** on **sequence/pooled context** (vec2text, GEIA, Rep2Text) or
  fusing that prior into a **noise-aware beam decode** (BeamClean) — *not* from a deeper per-vector
  regressor. If you want a step-change over ridge, that is where it lives (R3/R4), and it requires
  either more output context or more data.

**Dominant factor = data.** With hundreds–2k vocab-disjoint rows, overfitting caps you well below
the architecture's ceiling. What relaxes it, in order: (1) **compose with `E`** so the decoder
borrows the full vocab geometry instead of learning it from your rows (R1 — free, do now);
(2) **more / less-disjoint training rows** (the biggest lever for R2/R3); (3) **widen the target to
a token window + generative decode** (R3) so each example carries more supervision;
(4) under DP, **switch attacker class to an LM-prior beam decode** (R4) that supplies the missing
information as a prior rather than learning it.

---

## References

- Belrose et al. 2023, *Eliciting Latent Predictions from Transformers with the Tuned Lens*,
  [arXiv:2303.08112](https://arxiv.org/abs/2303.08112). Repo: EleutherAI `tuned-lens`.
- nostalgebraist 2020, *interpreting GPT: the logit lens* (LessWrong).
- Song & Raghunathan 2020, *Information Leakage in Embedding Models*, CCS '20,
  [arXiv:2004.00053](https://arxiv.org/abs/2004.00053).
- Li, Xu, Song 2023 (GEIA), *Sentence Embedding Leaks More Information than You Expect*, Findings of
  ACL 2023, [arXiv:2305.03010](https://arxiv.org/abs/2305.03010), [code](https://github.com/HKUST-KnowComp/GEIA).
- Morris et al. 2023 (vec2text), *Text Embeddings Reveal (Almost) As Much As Text*, EMNLP 2023,
  [arXiv:2310.06816](https://arxiv.org/abs/2310.06816), repo `jxmorris12/vec2text`.
- Rep2Text 2025, *Decoding Full Text from a Single LLM Token Representation*,
  [arXiv:2511.06571](https://arxiv.org/abs/2511.06571).
- BeamClean 2025, *Language-Aware Embedding Reconstruction*,
  [arXiv:2505.13758](https://arxiv.org/abs/2505.13758).
- Hewitt & Liang 2019, *Designing and Interpreting Probes with Control Tasks*, EMNLP 2019.
- Voita & Titov 2020, *Information-Theoretic Probing with Minimum Description Length*, EMNLP 2020,
  [aclanthology 2020.emnlp-main.14](https://aclanthology.org/2020.emnlp-main.14/),
  [code](https://github.com/lena-voita/description-length-probing).
- Pimentel et al. 2020, *Pareto Probing: Trading Off Accuracy for Complexity*,
  [arXiv:2010.02180](https://arxiv.org/abs/2010.02180).
- Hendrycks & Gimpel 2016, *GELU*, [arXiv:1606.08415]. Ramachandran et al. 2017, *Swish/SiLU*,
  [arXiv:1710.05941].
- Du et al. 2023, *Split-and-Denoise* (DP defense dual), [arXiv:2310.09130](https://arxiv.org/abs/2310.09130).

*Note: edgequake MCP `query` was returning 502s during this session; sourcing is WebSearch/WebFetch.
arXiv IDs verified via search result listings; citation counts/venues drift.*
