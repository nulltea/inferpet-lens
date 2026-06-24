# Experiment plan — embed-sgt (Task 7, Block B)

**Surface:** pooled GTR sentence embedding (`sentence-transformers/gtr-t5-base`, 768-d, mean-pooled).
**Attack:** Vec2Text pretrained `gtr-base` iterative corrector (Morris et al. 2023) — the same
faithful attack used for the Block A isotropic-DP study (`scripts/eval/vec2text_attack.py`).
**Defense:** Stained Glass Transform (SGT) — learned **heteroscedastic** Gaussian noise on the
released embedding with an MI-budget loss (arXiv 2506.09452).
**Probe (attack-independent):** spectral channel-MI `I_G`, generalized from isotropic to a
diagonal noise covariance — geometry-only, closed-form, never sees Vec2Text.

## Why this is the cleanest MI-probe-predicts-attack test

Block A already verified `I_G` as a matched converse probe for **isotropic** DP Gaussian noise
(`claim:spectral-channel-mi-embedding-inversion`, Spearman 1.0 on token-F1/cos over the ε-sweep).
SGT differs from isotropic DP in **exactly one thing: the noise shape** (anisotropic vs σ²I).
So holding `I_G` fixed and varying the shape isolates whether the scalar `I_G` is a *complete*
predictor of Vec2Text recovery, or only a within-shape monotone summary.

## Generalized probe (matched to SGT)

For release `Y = e0 + N`, `N ~ 𝒩(0, D)`, `D = diag(σ²₁..σ²_d)`, `Σ = Cov(e0)`:

    I_G = ½ log₂ det(I + D^{-1/2} Σ D^{-1/2}) = ½ Σ_i log₂(1 + μ_i)

where `μ_i` = eigenvalues of `D^{-1/2} Σ D^{-1/2}`. Reduces to `½Σ log₂(1+λ_i/σ²)` when `D=σ²I`.
This is the Gaussian-channel MI ceiling on `I(e0;Y) = I(X;Y)` (T1 sufficiency carries over).
**Attack-independent:** depends only on `(Σ, D)`.

## Noise shapes (all constructed to hit a *target* I_G budget B, exactly, by bisection)

Work in the PCA basis where `Σ = diag(λ₁≥..≥λ_d)`; per-mode noise variance `v_i`; total
distortion (utility cost) `D_tot = Σ v_i` (basis-invariant). `I_G(v) = ½Σ log₂(1+λ_i/v_i)`.

1. **iso** — isotropic DP: `v_i = σ²` ∀i, σ chosen so `I_G = B`. (the Block A defense)
2. **sgt-opt** — utility-optimal heteroscedastic SGT: minimize `D_tot` s.t. `I_G = B`
   (reverse-water-filling; closed-form Lagrangian, the global optimum of the SGT MI-budget loss —
   "learned" allocation at convergence). Allocates **more** noise to high-λ modes.
3. **sig-preserve** — adversarial shape control: minimize `I_G`... no — *maximize* recovery risk:
   put noise on tail (low-λ) modes, leaving high-signal modes cleaner, while still hitting `I_G=B`.
   A worst-case shape at matched budget — if recovery here ≫ sgt-opt at the same B, `I_G` is shape-blind.
4. **sgt-trained** (robustness add-on, GPU): input-conditioned diagonal noise from a tiny MLP
   `g(e0)→log v(e0)`, Adam on `D_tot + α·I_G` (reparam). Verifies the analytic sgt-opt = learned optimum.

## Sweep

Target budgets `B ∈ {∞(plaintext), B₁, B₂, B₃, B₄}` chosen to span the Block A I_G range
(~1597 → ~68 bits). For each finite B build {iso, sgt-opt, sig-preserve}; plaintext shared.
→ `1 + 4×3 = 13` Vec2Text inversions. Pilot N=48; full N=96, max_tokens=32, num_steps=20.

## Metrics (bits canonical + per-secret readout)

- **bits:** `I_G` (generalized), `d_eff`, `accessible_bit_ceiling=min(H_e0,I_G)`, Fano exact ceiling,
  RD per-token floor (H_X/H_e0 proxies flagged, as Block A).
- **recovery readout:** token-F1, exact-match, positional-token-acc, cosine (embedding readout).
- **utility readout:** `D_tot` (distortion) and mean cosine(e0, clean-task) — SGT's privacy-utility selling point.

## Measurement loop → decision

- **C1 (within-shape monotone):** down each shape column, does `I_G↓ ⇒ recovery↓`?
  Spearman(I_G, recovery) per shape over B. (expected yes, like Block A.)
- **C2 (HEADLINE — matched-I_G shape invariance):** across each row (fixed B), is recovery the
  same across {iso, sgt-opt, sig-preserve}?
  - **invariant (spread ≤ noise)** → `I_G` is a *complete* matched probe: a single geometric scalar
    predicts Vec2Text recovery regardless of noise shape. **Correlates → claim + proof.**
  - **shape-dependent** → `I_G` is shape-blind; that IS the finding. Bound the gap (which shape
    direction the attack exploits), decide weak-attack vs non-matched-probe, queue spawn-depth-1.
- **C3 (defense-utility):** at matched B, does sgt-opt give lower `D_tot` / higher utility than iso
  (SGT's claim) — and does that buy *more* or *less* recovery? (privacy-utility frontier.)

## Perf gate

Inversions dominate (~0.6 s/text × N). Pilot N=48 (~6 min); full N=96 (~13 min, confirm iGPU
saturation). I_G/shape construction is closed-form (ms), no GPU. One GPU process; serial.
Reuse `Vec2TextAttack`, `dp_noise`, `spectral_channel_mi` (generalize for diagonal D).
