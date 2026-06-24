# Staged content for the `V_cap` probe page (Task 4 — `docs/html/probe-vcap.html`)

This is the handoff from Task 3 (capacity-pvi restructure). It carries the estimator-repair
methodology and the accuracy-vs-bits rationale that used to masquerade as a surface page
(`resid-capacity-pvi.html`). Task 4 builds `probe-vcap.html` with the canonical sections
**Algorithm / Method / Rationale / Plaintext reference**; the material below maps onto those
sections. Do NOT re-derive — lift these verbatim, only re-wrapping into the probe-page template.

Canonical registry entry (Task 2): symbol `V_cap`, name **capacity-matched predictive
V-information**, quantity = V-information (capacity-bounded reader accuracy). Source module:
`src/talens/measures/vinfo_capacity.py`. Model-free test: (locate in `tests/` for the
`vinfo_capacity` estimator). Diagnostic spike: `scripts/spikes/diag_capacity.py`; input-DP runner:
`scripts/spikes/localdp_runner.py`.

---

## → Algorithm (how `V_cap` is computed: inputs → transforms → output in bits)

The token-identity reader. For a released residual-stream vector `X = resid_post[L] ∈ R^d` whose
secret is the token id behind it:

1. Standardize `X` on training statistics (mean/variance from the train split only).
2. Reduce to `dim < n_val` by a **train-only** principal-component projection (covariance
   eigendecomposition — the capacity-matching step; this is the single change from the original
   failing estimator).
3. Fit a calibrated linear-softmax classifier over the token-identity classes.
4. Read out two quantities:
   - **reader accuracy** — bounded top-1 recovery over the token-identity classes (the robust,
     primary readout).
   - **reader information (bits)** — predictive V-information = log-prob the reader assigns the
     true token minus the prior log-prob, reported as **selectivity** (real minus label-shuffled
     control). Calibration-sensitive; auxiliary.

It **never reads the embedding table** — it predicts a class label, not a vector. That is what makes
it attack-independent w.r.t. the embedding-inversion attack.

Diagram (DIAGRAM-STYLE.md): the two-independent-paths figure already drawn on the old page
(`hidden state X → {inversion attack → recovery}` and `→ {token-identity probe → accuracy + bits}`,
joined by `Spearman(accuracy, recovery)`) is the Method/measurement-loop diagram; for the
Algorithm section draw the internal computation sequence (standardize → PCA dim<n → softmax → two
readouts) per DIAGRAM-STYLE.md.

## → Method (what IT property, of what KIND, on which surface, what it bounds)

- KIND: **V-information** (predictive, capacity-bounded), reported as bounded reader accuracy +
  auxiliary bits.
- Surface(s) it accepts: residual-stream vectors `resid_post[L]` (and at L0 the noised input
  embedding). Layer-defined — has a per-layer reading.
- What it bounds: the capacity-bounded reader accuracy is a realizable lower bound on token-id
  decodability for a reader of that capacity; the bits selectivity estimates predictive
  V-information of the token id given the representation.

### Estimator repair — capacity matching (Results, layer 12, gemma-2-2b, single seed)

The original reader failed as a bits estimator: with more features than validation points it
interpolated, and its information on a label-shuffled control fell to roughly −50 bits at every
depth — an artifact, not a leakage signal. Reducing the feature dimension below the validation
count bounds that control near −1 bit at every depth, at about 0.57× the cost.

| reader family | real information (bits) | shuffle floor (bits) | cost vs original |
|---|---|---|---|
| original (full dimension) | 4.96 | −49.7 | 1× |
| **PCA softmax (selected)** | 5.39 | **−1.9** | 0.57× |
| random-projection softmax | 5.17 | −3.9 | 0.42× |
| Gaussian class-conditional | −12.0 | −32.6 | 0.15× |

The repaired floor holds at every depth: reduced reader shuffle floor = −1.25 / −1.23 / −1.24 /
−1.27 bits at L0 / L5 / L12 / L20, against the original reader's −44.9 / −50.2 / −51.4 / −48.1.

## → Rationale (why it is the attack-INDEPENDENT matched measure; where it tracks, where it is vacuous)

- **Independence:** never reads the embedding table; predicts a class label, not a vector. Its
  correlation with a retrieval-style reader is 0.66–0.76, below the redundancy threshold. The one
  change from the failing version (dimensionality reduction) adds no attack information.
- **Accuracy-primary / bits-auxiliary:** bits has no lower bound over readers (a single
  confidently-wrong assignment sends it to −∞), so it is fragile; bounded accuracy stays in [0,1]
  and is the object that tracks the attack. Report bits as auxiliary.
- **Where it tracks recovery:** under at-layer **representation-space** defenses (PCA-subspace
  ablation, isotropic hidden-state noise) the bounded reader accuracy tracks the ridge inversion
  attack at Spearman ρ 0.80–1.00 at every depth — the positive, attack-predictive regime. (This
  tracking evidence now lives, with the input-DP results, on `resid-dp-attacks.html`; cross-link it
  from the probe page as the "where it tracks" example.)
- **Where it decouples (the canonical example to cite):** under **propagated input-DP**, the
  tracking attenuates monotonically with depth (ρ +0.96 L0 → −0.21 L20), while CLUB attenuates in
  parallel but stays positive (+0.96 → +0.29). This is the matched-probe program's canonical
  decoupling example — cross-link `claim:depth-decoupling-input-dp` and the `resid-dp-attacks.html`
  Results block that now owns the measured tables.

### Why the repair works — the three propositions (Analysis; full proof in the claim record)

- **Lemma 1.** The probe's information in bits is the log-prob the reader assigns the true token
  minus the prior log-prob. Because a probability can approach zero, this quantity has no lower
  bound over readers: a single confidently-wrong assignment sends it to −∞. Accuracy, an indicator
  average, stays in the unit interval — so accuracy cannot blow up while the bits can.
- **Proposition 2.** With more features than validation points the data is separable, and an
  unregularized log-loss fit interpolates: training probabilities → 1. On a shuffled control the
  confident prediction disagrees with the permuted label on ≥1 point, the assigned probability
  there → 0, and the empirical control information (a finite average with one diverging term) falls
  without bound. This is the mechanism of the observed −50-bit floor (not a derivation of its exact
  value).
- **Proposition 3.** If the reader's logits are bounded by a cap fixed independently of the feature
  dimension, every class probability is at least a positive softmax floor, so the information is
  bounded below by a finite, dimension-independent constant. Dimensionality reduction does not
  enter this bound directly; it is the practical device that keeps the fit out of the interpolation
  regime so a modest logit cap is attainable — which is why the repair tracks the dimension rather
  than the regularization strength.

Full proof: `research-wiki/claims/capacity-matched-pvi.md` (estimator lemma verified).

## → Plaintext reference (Task 4 source order)

Use the on-disk depth sweep (`V_cap` depth sweep, present per Task-4 pointers). If a clean-model
plaintext-across-layers reading is not on disk, queue a small emission onto Task 7 — do NOT run GPU
in Task 4.

## Scope / limits (carry onto the page)

Single model (gemma-2-2b), single seed, 7 privacy points per layer, no CIs. The Gaussian
class-conditional reader stays miscalibrated even after dimensionality reduction — capacity
matching is necessary but not sufficient; the reader must also be calibrated. The bits readout is
calibration-sensitive (auxiliary). Named next steps: multiple seeds with intervals, cross-model
replication, sensitivity at a smaller reduced dim, a calibration diagnostic on the bits-fragile
cells.

## Claims / external refs to cite on the probe page

- `research-wiki/claims/capacity-matched-pvi.md` — estimator lemma (verified).
- `research-wiki/claims/depth-decoupling-input-dp.md` — the canonical decoupling example (now
  cited from `resid-dp-attacks.html`).
- Hewitt et al. — conditional V-information (2021); Soudry et al. — implicit bias of GD (2018);
  Lyu & Li — multiclass max-margin (2020); LMs are Injective — arXiv 2510.15511.
