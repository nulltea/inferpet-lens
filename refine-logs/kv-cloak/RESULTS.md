# Results — KV-CLOAK defense + block-size sweep (Task B-2)

**Date**: 2026-06-24
**Surface**: raw per-head KV-cache (kind `k`), Qwen3-4B, layers {0,12,20}; 48 long prompts
(110–178 tokens). **Defense**: KV-CLOAK `K' = S·P̂·(K+A)·M` (arXiv 2508.09442 eq. 9) and its
channel ablations, implemented as a `talens.transforms.Transform`.
**Attacks** (Task-1 BSS family, transform-aware): `gram_error`, `jade`, `jd`/`jd_floor`.
**Matched probes** (geometry-only): whitened-row `negentropy_bits`, `shared_spectral_capacity_bits`.
**Sweep**: channel ∈ {m, sp, scx, naive, a, full} × b ∈ {16,32,64} × mask-energy α ∈ {0,1,4}
(for A-bearing channels) × seed ∈ {0,1,2}. 273 cells. CPU-only on the cached capture
(`capture-7de5ef8d6e14afe9.pt`); host venv + numba JIT; no GPU.

Metric convention: **bits canonical** (negentropy, spectral capacity) **+ per-secret readout**
(BSS recovery = p95 Hungarian |cosine| of recovered sources vs plaintext key rows; Gram distances
unitless). The random-orthogonal-demixing floor (`jd_floor`) is the subspace-membership null.

## B1 — channel identities (exact, full-feature)

On a full-width operand (single prompt, b=32, seed=0) the decoupling identities hold to float:

| channel | row-Gram cos-dist | Gram-spectrum rel-err |
|---|---|---|
| M-only (`U=KM`)        | 2.1e-9  | 5.7e-10 |
| S·P̂-only (`U=SP̂K`)    | 1.41    | 3.2e-9  |
| SCX / perm-only (`P̂K`)| —       | 1.2e-15 |
| A-only (`U=K+A`, α=1)  | 0.19    | 0.20    |

So: right-orthogonal **M leaves the row-Gram exactly invariant**; left-orthogonal **S·P̂ act as a
similarity** (Gram spectrum invariant, full Gram rotated); the additive **A is the only channel that
moves the Gram spectrum**. `S`/`M` orthogonal to 1e-15. (8/8 unit checks pass; `refine-logs/kv-cloak/sanity.json`.)

## B2/B3 — channel decoupling on the attack + probe (layer 0, mean over b/mask/seed)

| channel | jade recovery (p95cos) | negentropy (bits) | spectral cap (bits) | gram cos-dist | gram spec-err |
|---|---|---|---|---|---|
| identity (plaintext) | **0.626** (floor 0.157) | 1044.7 | 18.24 | 0.000 | — |
| m  (M only)          | **0.126** | 1.47   | 20.94 | 0.124 | — |
| sp (S·P̂ only)        | **0.612** | 1082.2 | 17.92 | 1.398 | — |
| scx (perm only)      | **0.612** | 1072.5 | 18.19 | 0.382 | — |
| naive (S K M, eq.7)  | **0.124** | 1.70   | 20.52 | 1.399 | — |
| a  (mask only)       | 0.581     | 2009.4 | 35.06 | 0.444 | — |
| full (eq. 9)         | **0.126** | 0.72   | 16.32 | 1.392 | — |

Consistent at L12 and L20 (identity jade 0.598/0.632; M → 0.151/0.160; S·P̂ → 0.497/0.517).

**Reading.**
1. **The right-orthogonal feature mix `M` is the sole channel that defeats the BSS recovery
   attack**: every configuration containing `M` (m, naive, full) drops jade recovery from 0.626 to
   ~0.13 (the 0.157 random-demixing floor), and collapses the matched negentropy from ~1045 bits to
   ~1.5 bits (mixing 128 head-dim coordinates gaussianises the key rows). Configurations without
   `M` (sp, scx) leave recovery and negentropy at the plaintext level.
2. **The left token-mix + one-time-pad permutation `S·P̂` — and therefore the block size `b` — are
   recovery-irrelevant.** They rotate the full row-Gram (gram cos-dist 1.40) but leave jade recovery
   (0.61) and the matched probe (~1075 bits) at the plaintext value: the recovered sources still
   span the key-row subspace, so the Hungarian-optimal match to plaintext rows is unchanged. This is
   the subspace-membership invariance established in Task 1, now seen through a published defense.
3. **The additive beacon mask `A` is the only channel that perturbs the Gram spectrum** (B1; and
   `gram_spec_err` rises 0.24 → 0.35 → 4.70 with α = 0,1,4 under `full`), but alone it barely dents
   recovery (0.626 → 0.581).

## C2 — matched probe predicts the attack

Across all 270 non-identity cells:

| pair | Spearman ρ | p | n |
|---|---|---|---|
| **negentropy vs jade recovery** | **0.706** | 5e-42 | 270 |
| spectral-capacity vs jade        | 0.325 | 5e-8 | 270 |
| negentropy vs gram cos-dist      | -0.304 | 3e-7 | 270 |

The geometry-only **negentropy probe predicts BSS recovery at ρ = 0.71** — it is the channel-matched
measure (jade is a non-Gaussianity/ICA attack; negentropy is its separability resource, computed
without running the demixing). Spectral capacity is not matched to this attack (ρ = 0.33), as in
Task 1.

## b-flatness (the block-size knob is inert)

Recovery and the spectral probe vs block size `b ∈ {16,32,64}`, mean over seeds:

| channel | jade by b | spectral-cap spread over b |
|---|---|---|
| m   | 0.145 / 0.145 / 0.145 | 0.000 |
| sp  | 0.549 / 0.539 / 0.538 | 0.264 |
| scx | 0.551 / 0.545 / 0.547 | 0.054 |
| naive | 0.145 / 0.146 / 0.147 | 0.273 |

Both the attack and the probe are flat in `b` for every channel whose privacy comes from the
token-axis operators. **Block size buys no protection against the WEIGHTS-PUB BSS adversary**;
`b=64` (data-unreachable in Task 1) is therefore predictable, not a gap. The combinatorial `b!`
barrier the paper cites defends against a *permutation-recovery* adversary, not against a
subspace-membership / ICA adversary that never needs the token order.

## Is the negentropy↔recovery correlation just channel separation? (within vs between)

The aggregate ρ = 0.71 decomposes: the **channel-mean** correlation is ρ = 0.77 (n=6 channels),
while **within-channel** correlations are weak and sign-inconsistent — m −0.95 (at noise-level
recovery variance 2e-4, i.e. recovery is floored regardless), sp/scx +0.48, a −0.41, full −0.54.
So the probe ranks the channel families correctly (it separates the M-floored channels from the
leaking ones), but it is **not a within-channel oracle**, and under the additive mask it
**anticorrelates**: the beacon outliers inflate whitened-row negentropy without raising recovery.
The negentropy probe is therefore a between-channel leakage diagnostic, not a fine-grained meter.

## JD accumulation (layer 0): M defeats the accumulation attack too

p95 recovery vs observation count T:

| channel | jd T=1 | jd T=4 |
|---|---|---|
| random-demixing floor | 0.157 | 0.168 |
| identity | 0.488 | **0.751** |
| sp | 0.486 | 0.730 |
| scx | 0.488 | 0.725 |
| a  | 0.493 | 0.569 |
| m / naive / full | ~0.126 | **~0.12 (flat, at/below floor)** |

The M-bearing channels sit at or below the chance floor (0.157/0.168) at both ends of the
accumulation axis; identity/sp/scx are far above it. On the raw key surface the accumulation
attack **gains with T** at plaintext (0.49 → 0.75) and under
the token-axis covers (sp, scx) — recovery accumulates with observations. Only the channels
containing the feature mix `M` hold recovery at the floor across T. So `M` is the load-bearing
privacy channel against both the single-observation (jade) and the accumulation (jd) attack;
`S·P̂` and block size are inert against both.

## Synthesis / claims

- **C1 (channel decoupling)** — supported (narrowed), partly proved exactly: KV-CLOAK's three
  channels map onto three orthogonal observables — the right-orthogonal feature mix `M` →
  BSS recovery + row negentropy (the only channel that drives both the single-observation and the
  accumulation attack to the random-demixing floor, and the only one that collapses negentropy);
  the left token-mix + permutation `S·P̂` → full row-Gram rotation only (recovery stays far above
  floor and accumulates with T, flat in block size `b`); the additive beacon mask `A` → Gram
  spectrum only (`gram_spec_err` tracks its energy), with a small, non-floor recovery effect.
  Block size `b` is recovery-inert against this adversary. Not claimed: that `S·P̂` leaves recovery
  at *exactly* the plaintext value (it drops ~15% at L12/L20); that `A` has *zero* recovery effect.
- **C2 (matched probe)** — partial / reframed: the attack-independent, demixing-free negentropy
  probe is a **between-channel** leakage diagnostic — it ranks the channel families by recovery
  (channel-mean ρ = 0.77; aggregate ρ = 0.71) and correctly separates the M-floored channels from
  the leaking ones — but it is **not** a within-channel predictor, and it anticorrelates with
  recovery under the additive mask. Spectral capacity is not matched to this attack (ρ = 0.33).

Caveats: (1) the *exact* M-Gram-invariance (B1, 2e-9) is on the full feature set; the attack's
random 256/1024 feature subsample breaks orthogonality, so the empirical M-only `gram_error` is
0.124 (still the smallest of the matrix channels), not 0. (2) jade uses max_dim=16 subsampled
sources; recovery is graded against the matched random-demixing floor. (3) `b ∈ {16,32,64}` on
110–178-token prompts; the `b`-inertness is established for this adversary over this range.
