# Standardized Results — resid-split / PriPert (Task 6)

Metric convention (CLAUDE.md): **bits canonical** (`I_G` channel-MI, CLUB MI, accessible-bit
ceiling, Fano recovery ceiling) **+ per-secret readout** (token-id → TTRSR top-1/top-10, token
selectivity vs label-shuffle with bootstrap 95% CI, embedding cosine). Secret = token id at a
position on the split residual; threat WEIGHTS-PUB; inverter trained vocab-disjoint (no
memorization). Model Qwen3-4B `resid_post`, corpus `release-gate-512`, cached capture.

σ floor: per-layer `σ = β·meanRMS(plaintext H)` (fixed across ρ). H_X = log2(pool) ≈ 11 bits
(candidate pool ~2048). I_G = ½Σlog2(1+λ_i/σ²) on cov(Sparsify_ρ(H)); β=0 ⇒ σ=0 ⇒ I_G=∞
(noiseless channel, converse vacuous — reported, excluded from rank correlations).

## Headline (full sweep, 32 cells, L∈{0,8,16,24}, ρ∈{1,.25,.05}, β∈{0,.25,.5,1,2})

**C2 — matched probe tracks recovery across the joint sweep (POSITIVE):**

| correlate (across all finite-I_G cells, n=24) | Spearman |
|---|---|
| I_G (bits) vs best-inverter recovery | **0.958** |
| I_G (bits) vs best-inverter selectivity | 0.955 |
| I_G (bits) vs ridge recovery | 0.934 |
| I_G (bits) vs **mlp2 (stronger learned)** recovery | **0.915** |
| CLUB (bits) vs best recovery (secondary probe) | 0.977 |

The matched channel-MI probe predicts inversion recovery — **including the stronger learned
inverter** (0.915) — so **no probe–attack gap is observed for the tested inverters** (it is a
monotone capacity tracker, not a tight predictor); no spawn-depth-1 follow-up is warranted.

**C1 — PriPert suppresses recovery; perturbation is the load-bearing knob; depth-dependent:**

best-inverter TTRSR top-1 (selectivity ≈ same, shuffle floor ≈ 0 under vocab-disjoint split):

| layer | β=0 | β=0.25 | β=0.5 | β=1 | β=2 | (all at ρ=0.25) |
|---|---|---|---|---|---|---|
| L0  | 0.661 | 0.627 | 0.688 | 0.668 | 0.363 | input-embedding residual resists (σ_ref=0.19) |
| L8  | 0.608 | 0.223 | 0.019 | 0.005 | 0.002 | collapses by β=0.5 |
| L16 | 0.475 | 0.184 | 0.017 | 0.012 | 0.005 | collapses by β=0.5 |
| L24 | 0.576 | 0.385 | 0.160 | 0.012 | 0.010 | intermediate |

ρ-axis (at β=0.25) is secondary: e.g. L8 ρ=1→0.283, ρ=0.25→0.223, ρ=0.05→0.046 (sparsification
helps only at the heaviest setting). mlp2 ≫ ridge at moderate β (L8 β=0.25: mlp2 0.283 vs ridge
0.094) — the learned attack is far more perturbation-robust than linear ridge.

**C3 — the converse (Fano ceiling) is SLACK on this surface (honest negative):**

I_G stays ≫ H_X=11 bits across the useful range; the Fano accuracy ceiling = 1.0 (vacuous) in
31/32 cells. It binds (I_G<H_X) in exactly **one** cell — L8, ρ=0.25, β=2 → I_G=10.0 bits,
Fano ceiling=0.986 — where recovery is already floored at 0.002. **0 converse violations**
(empirical top-1 ≤ Fano ceiling everywhere): a valid but loose certificate. The empirical
recovery floor is reached by the perturbation degrading the inverter at β*≈0.5 (L8/16), far
inside where the IT capacity converse would bite (β**≈2). The probe still *tracks* (monotone)
through that slack regime — capacity-slack pattern, cf. rep2text.

## Probe bits vs β (L8, ρ=0.25) — the converse approaching the secret entropy
| β | I_G (bits) | best recovery | Fano ceiling | accessible (bits) |
|---|---|---|---|---|
| 0    | ∞     | 0.608 | 1.000 | 11.0 |
| 0.25 | 166.7 | 0.223 | 1.000 | 11.0 |
| 0.5  | 55.7  | 0.019 | 1.000 | 11.0 |
| 1    | 20.6  | 0.005 | 1.000 | 11.0 |
| 2    | 10.0  | 0.002 | 0.986 | 10.0 |

Artifacts: `runs/pilot/pripert_pilot.json` (9 cells, L8), `runs/sweep/pripert_sweep.json`
(32 cells). Defense `scripts/defenses/pripert.py`; driver `scripts/spikes/pripert_sweep.py`.
