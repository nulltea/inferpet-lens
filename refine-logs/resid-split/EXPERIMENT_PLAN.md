# Experiment Plan — resid-split (Task 6, campaign-B-expand)

**Problem**: Split inference exposes an intermediate residual at a cut layer to an untrusted
server. PriPert (arXiv 2605.23158) defends that split with **activation sparsification** + an
**adversarial perturbation** and proves a formal reconstruction-error converse ("Thm 1"). We test
PriPert on Qwen3-4B `resid_post`: does inversion recovery fall as we sparsify/perturb, and does a
**matched, attack-independent** channel-MI probe (the empirical realization of that converse)
track recovery across the joint defense sweep?
**Method Thesis**: On the split residual, recovery is governed by the surviving channel capacity;
the spectral channel-MI probe `I_G=½Σlog2(1+λ_i/σ²)` (geometry+budget only, never the attack)
both tracks inversion recovery across (split layer × sparsity ρ × perturbation budget β) AND
furnishes a Fano error floor that lower-bounds the empirical reconstruction error — i.e. PriPert's
Thm-1 converse, measured.
**Date**: 2026-06-24
**Threat anchor**: WEIGHTS-PUB (adversary knows weights/embeddings; the inverter is fit on
public-prompt activations under a vocab-disjoint split, so memorization cannot win).

## Definitions
- **split residual**: `resid_post` at split layer ℓ, per token position — the operand the
  untrusted server observes after the defense Transform. Secret = the token id at that position;
  graded recovery target = the token embedding.
- **PriPert Transform** (`scripts/defenses/pripert.py`): `U = Sparsify_ρ(H) + δ`. `Sparsify_ρ`
  keeps the ⌈ρ·d⌉ largest-magnitude coordinates per row and zeros the rest (lossy, deterministic).
  `δ` = additive perturbation with per-row energy budget `β·RMS(row)`. Primary (attack-INDEPENDENT)
  realization: isotropic Gaussian δ — the channel-matched noise the spectral probe is the converse
  of. Stronger arm (optional): `δ` aligned to the top defender-PCA directions of `Sparsify_ρ(H)`
  (energy-matched, still attack-independent — uses no inverter). The faithful fully-adversarial
  optimized-against-an-inverter perturbation is documented as the upper-strength variant, **not**
  the per-position primary (it would couple the defense to a specific attack).
- **ρ (sparsity ratio)**: fraction of coordinates kept; ρ=1.0 = no sparsification.
- **β (perturbation budget)**: δ energy as a fraction of per-row RMS; β=0 = no perturbation.
- **TTRSR** (token-table retrieval success rate): predict embedding from `U`, cosine-match to the
  candidate-pool table, score top-1/top-10; plus token-F1 and embedding cosine.
- **vocab-disjoint split + shuffle control**: train/val/test token vocabularies disjoint (no
  memorization win); `selectivity = real − label-shuffled`, with a bootstrap 95% CI.
- **spectral channel-MI probe** (`talens.measures.spectral_channel_mi`, MATCHED): `I_G` bits on the
  covariance of `Sparsify_ρ(H)` at noise σ=β·RMS; returns `accessible_bit_ceiling=min(H_e0,I_G)`,
  `fano_exact_ceiling` (token error floor under uniform prior), `rd_pertoken_floor`. Geometry+budget
  only ⇒ attack-INDEPENDENT. This is the converse, measured.
- **CLUB**: contrastive log-ratio upper bound on `I(U; token-embedding)` — secondary
  attack-independent MI probe (cross-check; not channel-matched to the Gaussian converse).

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: PriPert monotonically suppresses inversion recovery | reproduces the defense's intended effect on Qwen3-4B split residual | best-inverter vocab-disjoint selectivity falls as ρ↓ and β↑; defended corner (low ρ, high β) selectivity CI includes 0 | B1, B2 |
| C2: the matched channel-MI probe tracks recovery across the joint sweep | the measurement-loop thesis — an attack-independent probe predicts the attack | Spearman(I_G bits, best-recovery) across all (ℓ,ρ,β) cells ≥ 0.6 (sign + magnitude); CLUB cross-check reported | B1, B2, B3 |
| C3: the probe's Fano ceiling lower-bounds the empirical error (Thm-1, measured) | connects empirical floor to the formal converse | per-cell: empirical token-error ≥ `fano_exact_ceiling`-implied floor (no violations); accessible-bit ceiling ≥ realized recovered bits | B1, B2, B3 |
| Anti-claim to rule out | "the probe is the attack reparameterized" / "the suppression is just memorization loss" | I_G is geometry-only (computed without any inverter); vocab-disjoint+shuffle kills memorization; ρ(I_G, recovery) is across an attack-independent axis | all |
| Decision claim | does a stronger inverter (mlp2) re-open recovery where the probe says capacity is gone? | per-cell Δ(mlp2−ridge) selectivity at defended corners; if mlp2 wins where I_G≈0 ⇒ NEGATIVE/gap → spawn-depth-1 follow-up | B2 |

## Paper Storyline
- **Main paper**: C1 (PriPert suppresses recovery on Qwen3-4B split residual) + C2 (matched
  channel-MI probe tracks recovery across the joint defense sweep) + C3 (its Fano ceiling is the
  empirical error floor — Thm-1 realized). If C2 fails the bar, bound the gap and self-append a
  spawn-depth-1 follow-up (stronger attack vs better-matched probe).
- **Appendix**: per-layer ρ-curves; β-sweep at L8; CLUB cross-check; defender-PCA-aligned δ arm.
- **Cut**: the fully-adversarial optimized perturbation (attack-coupled, out-of-scope for a
  per-position TTRSR pipeline — documented as not-run, energy-matched Gaussian/PCA is the proxy);
  end-to-end task-accuracy degradation curve (utility side; the defense's utility claim, not the
  privacy converse — noted as the defender's tradeoff axis, not measured here).

## Experiment Blocks

### Block 1: Sparsity sweep at one split layer (pilot/anchor) — MUST-RUN
- **Claim tested**: C1, C2, C3 (fast).
- **Why**: validate the PriPert Transform + matched probe wiring and the (bits, recovery) loop on a
  single layer before the joint grid.
- **Dataset / split**: `corpora/release-gate-512.txt`, cached `resid_post` L8; vocab-disjoint.
- **Compared systems**: inverters {ridge, nn, mlp2}; probes {I_G (matched), CLUB}.
- **Metrics**: TTRSR top-1/top-10, token-F1, embedding cosine, selectivity+CI (recovery);
  `I_G` bits + `fano_exact_ceiling` + `accessible_bit_ceiling` (probe).
- **Setup**: ρ∈{1.0,0.5,0.25,0.1,0.05} at β=0.5; β∈{0,0.25,0.5,1.0} at ρ=0.25. mlp2 epochs 150.
- **Success**: recovery monotone-decreasing in (ρ↓,β↑); Spearman(I_G, recovery)≥0.6; no Fano
  violation.
- **Failure interpretation**: flat recovery ⇒ defense ineffective at this layer (try deeper);
  non-tracking probe ⇒ decide weak-attack vs non-matched-probe (B2 decides).
- **Table/figure**: Fig 1 (recovery + I_G vs ρ); Table 1 (β-sweep).
- **Priority**: MUST-RUN.

### Block 2: Joint (split layer × ρ) sweep — MUST-RUN
- **Claim tested**: C1, C2, C3, decision claim.
- **Why**: the headline — does the probe track recovery across BOTH defense axes and depth?
- **Dataset / split**: cached `resid_post` L∈{0,8,16,24}; vocab-disjoint.
- **Compared systems**: inverters {ridge, nn, mlp2}; probes {I_G, CLUB}.
- **Metrics**: as B1, per (ℓ,ρ) cell at β=0.5 (the defended operating point) + ρ=1.0,β=0 plaintext anchor.
- **Setup**: layers {0,8,16,24}, ρ∈{1.0,0.5,0.25,0.1,0.05}, β=0.5. 20 cells.
- **Success**: pooled Spearman(I_G, best-recovery) ≥0.6 across all cells; mlp2 does NOT beat ridge
  where I_G≈0 (no probe–attack gap).
- **Failure interpretation**: mlp2 re-opens recovery at I_G≈0 ⇒ probe under-counts extractable info
  → spawn-depth-1 follow-up (matched-probe). Probe doesn't track but attacks agree ⇒ weak-probe.
- **Table/figure**: Fig 2 (heatmap recovery & I_G over ℓ×ρ); Fig 3 (scatter I_G vs recovery, all cells).
- **Priority**: MUST-RUN.

### Block 3: Converse certificate (Thm-1, measured) — MUST-RUN (analysis, no new GPU)
- **Claim tested**: C3.
- **Why**: turn the per-cell probe outputs into a verified converse — empirical recovered bits ≤
  `accessible_bit_ceiling`, empirical token-error ≥ Fano-implied floor.
- **Compared systems**: none (post-hoc over B1/B2 JSON).
- **Metrics**: per-cell `recovered_bits = H_X·(1 − err)` vs `accessible_bit_ceiling`; violation count.
- **Setup**: H_X = log2(candidate-pool size) per row; n_tokens=1, vocab=pool.
- **Success**: zero ceiling violations across all cells (within CI).
- **Failure interpretation**: violation ⇒ probe mis-specified (wrong σ or covariance) — fix the
  channel match.
- **Table/figure**: Table 2 (accessible ceiling vs realized bits per cell).
- **Priority**: MUST-RUN.

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | wiring + B1 pilot (L8 ρ/β) | `pilot` | recovery moves with ρ/β; probe finite; Fano sane | ~2 min GPU | cache miss → recapture (still <5 min) |
| M1 | B2 joint ℓ×ρ sweep | `sweep` | pooled Spearman computed; gap decided | ~4 min GPU | mlp2 epochs cost; capped at 150 |
| M2 | B3 converse certificate | (in-driver) | zero Fano violations | CPU/in-run | probe σ mismatch |

## Compute and Data Budget
- Total estimated GPU time: **~6 min** (capture reused from cache; 25 cells × ~4–8 s + mlp2 fits).
- Data prep: none (corpus + cache exist).
- Human eval: none.
- Biggest bottleneck: mlp2 (learned inverter) fits, capped at 150 epochs; PCA via cov-eigh on GPU.

## Risks and Mitigations
- **Probe not channel-matched** (σ definition vs realized δ): set σ to the exact per-row RMS of δ
  used by the Transform; verify on the β-sweep that I_G falls monotone with β. → if still
  non-tracking, that is the finding (spawn-depth-1).
- **Sparsification breaks Gaussianity of the residual** (I_G is a Gaussian-channel bound): I_G is a
  certified *upper* bound on I regardless of Gaussianity (max-entropy), so the converse direction
  (C3) holds; tracking (C2) is the empirical question.
- **One-GPU**: single serial run via `run_step.sh`; kill stray containers first.

## Final Checklist
- [ ] Main paper tables covered (Fig1/2/3, Table1/2)
- [ ] Novelty isolated (matched probe is geometry-only, not the attack)
- [ ] Simplicity defended (Gaussian δ is the channel-matched proxy; adversarial-optimized cut)
- [ ] Frontier contribution justified (per-position TTRSR proxy for the paper's attack; documented)
- [ ] Nice-to-have (β-sweep, PCA-δ arm, CLUB) separated from must-run
