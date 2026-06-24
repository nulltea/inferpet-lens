# Experiment Plan — resid-depth-inversion (Task 4, campaign-B-expand)

**Problem**: "Depth gives a false sense of privacy" (Dong et al., arXiv 2507.16372) claims deep-layer
LLM internal states are NOT irreversible to inputs. We test this on Qwen3-4B `resid_post` across depth
with a baseline/learned inverter family, and ask whether an **attack-independent** probe predicts
inversion recovery across both depth and inverter strength.
**Method Thesis**: On plaintext residuals, token recovery does not vanish with depth; a capacity-matched
token-id V-information reader (and CLUB MI) tracks recovery across layers, and remains predictive when
the inverter is upgraded from a memorizing baseline → linear ridge → a learned 2-layer head — UNLESS a
stronger learned inverter opens a depth-localized probe–attack gap (the open question from
[[info-efficient-attacks-findings]]: a 250-epoch MLP LOST to ridge at depth under noise; does it also lose
on plaintext?).
**Date**: 2026-06-24

## Definitions
- **resid_post**: residual-stream activation after a transformer block, per token position. Secret = the
  token id at that position; graded recovery target = the token embedding.
- **TTRSR** (token-table retrieval success rate): predict embedding from activation, cosine-match to the
  candidate-pool table, score top-1/top-10. Plus **token-F1** over the recovered vs true token multiset.
- **vocab-disjoint split**: train/val/test token vocabularies are disjoint, so a memorizing inverter
  cannot win on seen tokens — only genuine *generalizing* recovery counts (`docs/dev/control-tasks.md`).
- **shuffle-control selectivity**: `real − shuffled`; the shuffled run permutes labels (breaks X↔Y), giving
  the frequency/chance floor. Reported alongside raw recovery.
- **cap-PVI reader accuracy**: capacity-matched token-id V-information (`vinfo_capacity`, pca_softmax dim 64).
  Attack-independent — uses token-id *classes*, never the embedding table. Canonical unit bits; robust
  readout = reader top-1 accuracy ([[capacity-pvi-findings]]).
- **CLUB**: contrastive log-ratio upper bound on I(resid ; token-embedding); an MI estimator, not the attack.
- **Inverters**: `nn` = cosine nearest-neighbour over train activations (memorizing baseline);
  `isa`/ridge = linear ridge resid→embedding (EXISTS); `ima_paper_like` = learned 2-layer head resid→embedding
  (the tractable per-position analog of the paper's generation/optimization attacks).

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Blocks |
|---|---|---|---|
| C1: recovery does not vanish with depth (plaintext resid_post leaks at every depth) | reproduces 2507.16372 headline on Qwen3-4B; confirms [[sweep-controls-findings]] | best-inverter vocab-disjoint selectivity > shuffle floor (CI excludes 0) at ALL swept depths; non-monotone-or-flat depth curve | B1, B2 |
| C2: an attack-independent probe tracks recovery across depth AND inverter strength | the measurement-loop thesis — probe predicts attack | Spearman(cap-acc, best-recovery) and Spearman(CLUB, best-recovery) across the depth grid; report sign per layer; positive ⇒ predictive | B3 |
| Anti-claim to rule out | "the leak is just memorization" / "the probe is the attack reparameterized" | vocab-disjoint + shuffle subtraction kills memorization; cap-PVI independence ρ(cap, retrieval) < 0.9 | B1, B3 |
| Decision claim | does a stronger learned inverter beat ridge at depth (probe–attack gap)? | per-layer Δ(ima_paper_like − ridge) selectivity; if learned wins at deep layers where probe is flat ⇒ NEGATIVE/gap finding | B2 |

## Paper Storyline
- Main paper must prove: C1 (depth ≠ privacy on Qwen3-4B) and C2 (probe tracks across depth + strength), or
  bound the gap if C2 fails (which inverter, which layer, by how much).
- Appendix: per-inverter depth curves; cap-PVI bits vs reader-accuracy fragility.
- Cut: the paper's white-box two-phase per-sample optimization attack (per-sample gradient descent on
  hidden states; out of scope/compute for a per-position TTRSR pipeline — documented as not-run, the
  learned 2-layer head is the tractable proxy). Black-box transfer attack: not-applicable under WEIGHTS-PUB.

## Experiment Blocks

### Block 1: Depth recovery sweep (anchor) — MUST-RUN
- Claim tested: C1.
- Why: establishes that recovery survives depth and is genuine (not memorized).
- Data/split: Qwen3-4B, `resid_post`, corpora/release-gate-512.txt; vocab-disjoint train/val/test; shuffle control.
- Compared systems: nn, ridge(isa), ima_paper_like — same operand, same split, per layer.
- Metrics: TTRSR top-1/top-10 + token-F1 (recovery); selectivity = real − shuffled; bits-canonical cap-PVI alongside.
- Setup: depth grid every-4 (L0,4,8,…,32 ≈ 9 layers); capture once across those layers, sweep in memory.
- Success: best-inverter selectivity CI excludes 0 at every depth; depth curve flat/non-monotone, not →0.
- Failure interpretation: if selectivity →0 at depth, the paper's claim does not transfer to Qwen3-4B resid_post.
- Table/figure: depth × inverter recovery line chart; selectivity table.

### Block 2: Inverter-strength axis at depth (novelty isolation + decision) — MUST-RUN
- Claim tested: decision claim (stronger inverter vs ridge at depth).
- Why: tests whether ridge under-reads deep-layer leakage; isolates whether learned ≫ linear at depth.
- Compared systems: ima_paper_like vs ridge vs nn; Δ-selectivity per layer with bootstrap CI.
- Metrics: per-layer Δ(best learned − ridge); does the ranking of inverters change with depth?
- Setup: reuse B1 captures; learned head trained on GPU, fixed epochs/width, vocab-disjoint train only.
- Success criterion (for a clean C2): ridge ≈ best at all depths ⇒ probe-tracks-ridge suffices.
- Failure/gap: learned beats ridge at deep layers ⇒ probe must track the stronger attack — test in B3.

### Block 3: Probe-vs-recovery correlation across depth (the measurement loop) — MUST-RUN
- Claim tested: C2.
- Why: the core thesis — does an attack-independent measure predict recovery across the grid.
- Systems: cap-PVI reader accuracy + CLUB bits (probes) vs best-inverter recovery (attack).
- Metrics: Spearman per layer and pooled-rank within the depth sweep; independence ρ(cap, retrieval) < 0.9.
- Setup: probes computed on the SAME captures; pca_softmax dim 64; CLUB fast fidelity.
- Success: positive Spearman across depth ⇒ C2 supported; render Claim→Theory→Report.
- Failure: non-correlation IS the finding — classify weak-attack vs non-matched-probe; bound the gap; append
  a spawn-depth-1 follow-up.

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | impl + sanity (nn, ima_paper_like wired; unit/shape) | host pytest (model-free) + dev-24 pilot 1–2 layers | inverters run, selectivity sane on pilot | ~3 min GPU | new-code bugs |
| M1 | pilot depth sweep | dev-24, every-8 (≈5 layers), all 3 inverters + probes | curves sane, perf-gate pass | ~3 min GPU | undertrained learned head on 24 prompts |
| M2 | full depth sweep | release-gate-512, every-4 (≈9 layers), 3 inverters + cap-PVI + CLUB | B1/B2/B3 data complete | ~10–15 min GPU | wall-time; learned-head epochs |
| M3 | analysis + correlation | calibration.calibrate_records per layer | C2 correlate? | host | — |

## Compute and Data Budget
- Total GPU: one capture (~2–4 min) + in-memory inverter/probe sweep; target < 15 min wall on M2. Single iGPU, serial.
- Data: existing corpora; no new data.
- Bottleneck: learned-head training time per layer × 9 layers — cap epochs (≤150) and width; GPU-batched.

## Risks and Mitigations
- Learned head undertrained / overfit → fixed budget, vocab-disjoint train only, early-stop on val top-1; compare to ridge as sanity.
- nn baseline trivially memorizes under row-split → use vocab-disjoint so nn collapses to floor (intended control).
- cap-PVI bits fragile (log-loss) → report bounded reader **accuracy** as the tracker ([[capacity-pvi-findings]]).
- Wall-time > 15 min → drop to every-8 or fewer learned-head epochs; capture-once already amortizes the GPU cost.

## Final Checklist
- [ ] Main tables covered (depth×inverter recovery; probe-vs-recovery Spearman)
- [ ] Novelty isolated (inverter-strength axis at depth)
- [ ] Simplicity defended (learned head only if it beats ridge; else ridge suffices)
- [ ] Frontier component justified (learned 2-layer head as tractable proxy; white-box opt attack explicitly cut)
- [ ] Nice-to-have separated from must-run
