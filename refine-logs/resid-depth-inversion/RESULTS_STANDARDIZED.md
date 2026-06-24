# Results — resid-depth-inversion (Task 4, campaign-B-expand)

**Model**: Qwen/Qwen3-4B · **surface**: `resid_post` · **corpus**: release-gate-512 (512 prompts,
9469 token-positions/layer captured) · **inverter split** (vocab-disjoint): **n_train 3373 / n_test
413 rows per layer** (remaining positions fall in the disjoint val partition or are excluded so
train/test token vocabularies are disjoint); **bootstrap 95% CIs are over the 413 test rows per
layer**. **depth grid**: every-4, L0..L32 (9 points) · **control**: label-shuffle · **probes**:
capacity-matched token-id V-information (`pca_softmax`, dim 64; ≤2500 rows) + CLUB MI upper bound
(≤2500 rows). Data: `runs/full/depth_sweep.json` (`run.exit`=0). Wall-time 205s.

Metric convention: **bits canonical + per-secret readout**. Secret = token id; recovery readout =
TTRSR top-1 (token retrieval) with selectivity = real − label-shuffle floor; probe bits = CLUB MI
(I(resid;embedding)) and cap-PVI bits, probe readout = reader top-1 accuracy.

## Per-layer table (recovery = selectivity = real − shuffle; bits beside readout)

| layer | ridge sel [95% CI] | mlp2 sel | nn (memorization floor) | cap reader acc | cap shuffle | CLUB bits |
|---|---|---|---|---|---|---|
| L0  | 0.685 [0.639, 0.731] | 0.639 | 0.000 | 0.939 | 0.003 | 3426 |
| L4  | 0.598 [0.550, 0.646] | 0.651 | 0.000 | 0.861 | 0.003 | 3381 |
| L8  | 0.588 [0.540, 0.637] | 0.581 | 0.000 | 0.797 | 0.005 | 3003 |
| L12 | 0.593 [0.547, 0.639] | 0.586 | 0.000 | 0.732 | 0.003 | 2824 |
| L16 | 0.533 [0.482, 0.581] | 0.494 | 0.000 | 0.691 | 0.003 | 2792 |
| L20 | 0.504 [0.455, 0.552] | 0.523 | 0.000 | 0.708 | 0.004 | 2891 |
| L24 | 0.603 [0.552, 0.649] | 0.576 | 0.000 | 0.769 | 0.001 | 3062 |
| L28 | 0.540 [0.492, 0.588] | 0.571 | 0.000 | 0.753 | 0.001 | 3043 |
| L32 | 0.390 [0.341, 0.438] | **0.542** | 0.000 | 0.685 | 0.000 | 2970 |

(nn top-1 reported; nn is scored by direct nearest-train-neighbour id, so under a vocab-disjoint
split it is identically 0 — the memorization floor. cap-PVI bits omitted from the table as the
robust tracker is bounded reader accuracy, per [[capacity-pvi-findings]]; raw `cap_pvi_bits` is in
the JSON.)

## Findings

### C1 — depth does NOT confer privacy (reproduces arXiv 2507.16372 on Qwen3-4B)
Best-inverter vocab-disjoint **selectivity stays 0.39–0.69 across all nine depths**; every
bootstrap 95% CI excludes 0 (lowest is L32 ridge [0.341, 0.438]). The cosine-NN memorization
baseline is **0.000 at every depth**, so the recovery is *genuine generalizing* token inversion,
not train-vocabulary memorization. The depth curve is flat / non-monotone, never collapsing — the
"deep layers are irreversible" assumption fails on the Qwen3-4B residual stream.

### Decision claim — a learned inverter beats linear ridge at the deepest layer
mlp2 (learned 2-layer head) ≈ ridge through mid-network, but at **L32 mlp2 = 0.542 vs ridge =
0.390 (gap +0.153)**; smaller learned wins at L4 (+0.053), L20 (+0.019), L28 (+0.031). Ridge's
late-layer drop is therefore partly a **linear-inverter artifact**: a non-linear inverter recovers
what ridge misses at the deepest layer. This directly answers the open question from
[[info-efficient-attacks-findings]] (a 250-epoch MLP *lost* to ridge at depth **under noise**) —
on *plaintext* depth the learned inverter is at least competitive and wins at the very deepest layer.

### C2 — the attack-independent probe tracks recovery across depth AND inverter strength (POSITIVE)
Across the 9-layer grid, Spearman(cap reader acc, best recovery) = **+0.85**; Spearman(CLUB bits,
best recovery) = **+0.78**. The probe also tracks each inverter: cap-acc vs ridge +0.80, vs mlp2
+0.83. Both probes are attack-independent by construction — cap-PVI reads token-id *classes* (never
the embedding table the attack retrieves against; [[capacity-pvi-findings]] established
ρ(cap, retrieval) ≈ 0.66–0.76 < 0.9), CLUB is an MI estimator. The shuffle floor (cap_acc_shuffle
≈ 0.001–0.005) confirms no label leakage in the reader. **This is the positive measurement-loop
regime: an attack-independent IT measure predicts inversion recovery across depth.**

## Threat-model coverage (per objective)
- aloepri `nn` (cosine-NN), `isa` (deep-layer ridge = `ridge`), `ima_paper_like` (2-layer learned =
  `mlp2`): all three implemented and swept. ✅
- 2507.16372 white-box two-phase per-sample optimization attack: **cut** (per-sample gradient
  descent on hidden states is out of scope for a per-position TTRSR pipeline; the learned `mlp2`
  head is the tractable amortized proxy and already wins at depth). Documented, not run.
- 2507.16372 black-box transfer attack: **not-applicable** under WEIGHTS-PUB (weights are public).
