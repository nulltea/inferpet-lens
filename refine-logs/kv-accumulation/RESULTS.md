# Results — KV/QKV accumulation / BSS (Task B-1)

Surface: `kv-accumulation` · run_id `b-kv1-accumulation` · dev-24 cached capture (Qwen3-4B,
24 prompts, layers {0,12,20}, kinds {kq, kqv_out, resid_post}) · Identity transform (plaintext),
WEIGHTS-PUB. CPU-only (cached operands; attacks/probe are numpy/BLAS). max_dim=64, max_features=256.

Artifacts: `sanity_bss.json` (Block 0, all pass), `pilot_dev24.json` (Block 1), `analysis_b3.json`
(Block 3 — proper floor + C1 slope + C2 correlation).

## The decisive correction: the proper floor

The shipped `bss.jd_floor` compares a recovered source against **unrelated** Gaussian ground
truth (p95 ≈ 0.155). Under Identity (U == H) that is the wrong control: any demixing B of U yields
rows in the **row-span of U == H**, so the Hungarian p95-cosine against H's own rows is high
regardless of whether the joint-diagonalisation found the right rotation. The apples-to-apples
floor is a **random-orthogonal-demixing** floor: same whitened data + same Hungarian pipeline, but
a random rotation in place of joint-diag. Genuine separation = real-attack p95 **minus** this floor.

| | median |
|---|---|
| jade raw p95-cosine | **0.776** |
| random-demixing proper floor | **0.708** |
| **genuine margin** — median of per-cell (raw − floor) | **0.027** |

Readout note: the genuine margin is the **median of the nine per-cell differences**
`jade_p95 − proper_floor_p95`, **not** `median(raw) − median(floor)`. The latter (0.776 − 0.708 ≈
0.067) is not a valid margin because the two column medians come from different cells; subtracting
them double-counts the cross-cell spread. The per-cell margin (0.027) is the honest readout.

≈ 96–97% of the apparent "recovery" is subspace-membership artifact. Per kind: kq ≈ 0 (L12 = −0.003),
kqv_out ≈ 0.02–0.03, resid_post ≈ 0.06–0.07. Genuine separation is tiny and ordered by kind.

## C1 — accumulation question: FLAT (no accumulation)

jd p95-cosine slope vs log₂T, median over 9 (kind×layer) cells = **+0.009** (flat). Max genuine
margin over the proper floor at any (cell, T) = **0.094** (resid_post L20, T=8). Margins do not grow
monotonically with T. T=16 unavailable at dev-24 (0 disjoint stacks); T axis = {1,2,4,8}.

**Reading.** On plaintext KV/QKV there is no unknown mixing matrix to invert (U == H), so BSS is
structurally ill-posed: the "sources" are the activation rows themselves and recovery cannot
accumulate across fresh-per-prompt observations. A mixing defense (Task 2 KV-CLOAK, Task 5 GELO) is
the next place where BSS could become informative; this baseline makes those sweeps interpretable —
any climb-with-T under a defense would be attributable to mask correlation, not BSS on the plaintext.
(The Gaussian-GT floor p95 ≈ 0.155 is the value reported by the shipped `bss.jd_floor`, logged in
`pilot_dev24.json`.)

## C2 — probe predicts attack: YES for the jade channel, against the genuine margin

| probe → recovery axis | Spearman | Pearson | n |
|---|---|---|---|
| negentropy → **genuine margin** | **0.92** | 0.95 | 9 |
| negentropy → raw p95 (uncorrected) | −0.43 | −0.09 | 9 |
| shared-spectral-capacity(T) → jd p95 | 0.56 | 0.57 | 36 |

The attack-independent negentropy probe (Hyvärinen whitened-row negentropy, geometry-only, never
calls joint-diag) predicts the **genuine** separation margin across layer×kind at ρ = 0.92 — but
anti-correlates with the naive raw readout. **The floor definition is what makes or breaks the
correlation**: the matched IT probe predicts genuine BSS recovery once, and only once, recovery is
graded against the correct random-demixing floor. jd-channel shared-capacity tracks jd recovery
moderately (0.56, below the 0.7 bar) — a known-weaker, accumulation-axis probe.

Caveat: n = 9 cells; the ρ = 0.92 is driven by a clean monotone ordering across kind-families
(kq ≪ kqv_out ≪ resid_post in both negentropy and genuine margin). Thin but structural.

## gram_error baseline

cos_norm_distance = 0.0 and row_gram_spectrum_error = 0.0 at every (kind×layer): trivially, U == H
under Identity ⇒ the row-Gram **is** the fingerprint. Protocol-confirming appendix baseline, not a
recovery claim.

## Not-applicable family members (per WEIGHTS-PUB threat model)

- **sda / tfma** (aloepri): operate on a recovered token-id sequence, not on activations — they do
  not cross the activation boundary. `not_applicable`.
- **ia weight-axis**: needs an obfuscated weight pair (W_plain, W_obf); WEIGHTS-PUB gives the true
  weights ⇒ no weight secret to attack. `not_applicable`.

## GO/NO-GO for the B2 GPU scale-up: NO-GO

Genuine recovery is near-zero and flat in T; the qualitative finding (BSS ill-posed on plaintext;
probe tracks the tiny genuine margin) is decided on CPU. A 512-corpus scale-up would only tighten
the slope CI on a flat line and add T=16/32 stacks — it cannot change the qualitative claim.
C2's correlation is across (kind×layer) cells, which more prompts mostly tighten per-cell rather
than widen n. Conserve the single iGPU. Decision logged to the campaign progress file.
