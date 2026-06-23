# perm-cover — consolidated standardized results (bits canonical + per-secret readout)

**Surface:** the **permutation channel** — AloePri column-permutation cover on a real
token-embedding table (`obfuscate_embedding_table`, `scripts/defenses/aloepri.py`). Each token row's
feature dimensions are permuted by a per-table column permutation (the cover), with an optional
Gaussian noise knob `α_e`; the adversary observes the obfuscated table and the plaintext table and
must recover the **row correspondence** (which obfuscated row is which plaintext token).
**Secret kind:** `permutation` (the token-to-token correspondence Π over N rows). Readout =
fraction-correctly-placed recovery rate (`permutation_readout`, `src/talens/report.py`).
**Threat model:** WEIGHTS-PUB — adversary knows both tables; the column permutation is a known
group action, so any column-permutation-*invariant* row statistic (the sorted row, its quantiles, a
Gram) is a known function of the secret.
**Attacks (ground truth):**
- **VMA** (Vocabulary-Matching Attack, `talens.weights.vma`) — match each obfuscated row to its
  nearest plaintext row by a column-perm-invariant signature + Hungarian assignment. Two strengths:
  *RowSort-64* (the AloePri baseline: a 64-quantile sorted signature, cosine) vs *full-sorted-row*
  (the entire sorted row; cosine or negative-Euclidean = the iid-Gaussian MLE on the sorted vector).
- **cover_break** (`talens.attacks.cover_break`) — anchor-ridge row reconstruction from K known
  plaintext/observed anchor pairs; the activation-surface counterpart of the same threat (below).
**Probes — independence audited (`EXPERIMENT_AUDIT.md`, Codex read-only):**
- **CLUB-on-φ** `I(obf-φ ; plain-φ)` (`mi_upper_bound`) is the **clean independent probe** — it runs
  no Hungarian assignment and no nearest-neighbor matching; it uses the true τ only to form positive
  paired samples and estimates MI with a variational CLUB network. This is the probe the thesis
  rests on.
- **retrieval-PVI on φ** (`v_information`) is a **dependent reference — "VMA-in-bits"**, NOT an
  independent probe. It fits an inverter and scores the true candidate under cosine-softmax retrieval
  over the plaintext signatures (`src/talens/measures/_retrieval.py`), which is effectively the
  matching attack rendered in bits (the spike `aloepri_vma_sweep.py` labels it "dependent / the VMA
  in bits"). Its near-perfect correlation with VMA is **partially circular** and is reported only as
  the attack-in-bits readout, never as independent confirmation.

**Metric convention (CLAUDE.md):** canonical **bits** = CLUB-on-φ (`mi_upper_bound`, the independent
probe); retrieval-PVI (`v_info`) travels beside it as the labeled attack-in-bits reference only;
per-secret **readout** =
`recovery_rate` (VMA fraction-correctly-placed) rendered beside it. Rows below were **re-rendered**
from the raw result JSONs through `src/talens/report.py`
(`LeakageReport.from_measure(...)`, `permutation_readout`); bits stored verbatim, legibility by
`report.format_bits`.

Embedding table: gemma-2 family cache (`results/capture_cache/embed-b0c6566474cadb27.pt`, d=2304).
Provenance JSONs under `results/`; numbers verbatim.

---

## R1 — VMA attack strength: full-sorted-row ≫ RowSort-64  (`results/vma_stronger.json`; N=1000, 3 seeds)

`α_e`-sweep of the AloePri permutation cover. The probe (CLUB-on-φ) and three matchers, by `α_e`
(`bits | readout`):

| α_e | CLUB-on-φ | RowSort-64 | full-sort cos | **full-sort euc** | uplift (fe−rs) |
|---|---|---|---|---|---|
| 0.0  | 245 bits | 1.000 | 1.000 | 1.000 | +0.000 |
| 0.1  | 243 bits | 0.977 | 1.000 | 1.000 | +0.023 |
| 0.2  | 240 bits | 0.565 | 0.999 | **0.999** | **+0.434** |
| 0.35 | 235 bits | 0.204 | 0.804 | **0.804** | **+0.600** |
| 0.5  | 230 bits | 0.099 | 0.442 | 0.442 | +0.343 |
| 0.75 | 219 bits | 0.037 | 0.140 | 0.140 | +0.103 |
| 1.0  | 206 bits | 0.023 | 0.054 | 0.054 | +0.031 |

- **RowSort-64 collapses where the leakage has not gone.** At `α_e=0.2` RowSort-64 reads 0.565 while
  the full-sorted matcher recovers **0.999**; at `α_e=0.35`, 0.204 vs **0.804** (+0.60). CLUB-on-φ
  meanwhile barely moves (245→235 bits, −4%): the information is preserved; RowSort's 64-quantile
  binning is a **lossy** compression of the sorted row that cannot use it.
- full-sort cosine and full-sort negative-Euclidean are numerically identical here (the signatures
  are L2-normalized and mean-centered, so cosine ranking = neg-Euclidean ranking up to scale).
- **Both attacks track the probe over the full sweep.** Spearman(τ-recovery, CLUB-on-φ):
  RowSort all-sweep **+1.00**, full-sort-euc **+0.99**; low-noise subset (α_e ≤ 0.5)
  RowSort **+1.00**, full-sort **+0.97** (`recorrelation` block of the JSON). So the *probe* is
  faithful across attack strengths; only the *weak attack's recovery* under-reports the leakage at
  small noise.

This is the **permutation-channel analog of the L0 Bayes-NN result** (memory
`info-efficient-attacks-findings`): weak-attack collapse at small noise is **attack weakness, not
leakage loss**, and the independent probe stays faithful.

## R2 — the independent probe tracks recovery + the keymat cover destroys both  (`results/aloepri_vma_sweep.json`; N=1200, 1 seed)

Hungarian VMA on the sorted-quantile signature vs the independent CLUB-on-φ probe and the dependent
retrieval-PVI (VMA-in-bits) reference, then one **keymat** point (the full invertible Algorithm-1
key matrix, the dense cover):

| point | VMA recovery | **CLUB-on-φ (independent)** | retrieval-PVI (VMA-in-bits, dependent) |
|---|---|---|---|
| α_e=0.0  | 1.0000 | 252 bits | 3.34 bits |
| α_e=0.1  | 0.9600 | 253 bits | 3.33 bits |
| α_e=0.2  | 0.5608 | 250 bits | 3.31 bits |
| α_e=0.35 | 0.2117 | 246 bits | 3.25 bits |
| α_e=0.5  | 0.0883 | 240 bits | 3.17 bits |
| α_e=0.75 | 0.0375 | 229 bits | 2.99 bits |
| α_e=1.0  | 0.0217 | 214 bits | 2.75 bits |
| α_e=1.5  | 0.0067 | 181 bits | 2.27 bits |
| **KEYMAT** | **0.0000** (chance≈1/1200=8.3e-4) | **−2.4 bits (≈0)** | **0.63 mbit (≈0)** |

- Across the α_e sweep, Spearman(measure, VMA): the **independent CLUB-on-φ = +0.976**
  (`spearman_vs_recovery.club_bits`). retrieval-PVI = +1.00 but it is the attack-in-bits, so that
  figure is a circular consistency check, *not* independent confirmation. The faithful statement: the
  **independent probe is a monotone indicator of VMA recovery** across the sweep (ρ +0.976).
- **The keymat cover is the negative control.** When the dense key matrix removes the sorted-quantile
  fingerprint entirely, the attack drops to chance (0.0) **and** the independent CLUB-on-φ reports
  ≈ 0 bits (−2.4, i.e. its estimator noise floor). (retr-PVI → 0.0006 too, but as the attack-in-bits
  that is expected and not independent evidence.) So the independent probe collapses *together with*
  recovery when the cover genuinely destroys the signature — a probe-faithfulness check, not a proof
  of zero information.

## R3 — cover_break: the activation-surface anchor attack (plaintext baseline; matched run deferred)

`cover_break` (anchor ridge, `src/talens/attacks/cover_break.py`) is the second aloepri-resident
attack on this threat family. It fits a linear map on K known anchor pairs `(U[i], H[i])` and
reconstructs non-anchor rows; readout = p95 absolute cosine to the true row (an `embedding`-kind
secret). Under `Identity` (`U=H`, plaintext) it is the **recovery baseline** any cover must defend:

| layer (resid_post) | anchor p95-cosine |
|---|---|
| L0 | 0.917 |
| L2 | 0.909 |
| L4 | 0.901 |
| L6 | 0.822 |
| L8 | 0.780 |
| L10 | 0.812 |

(`results/fullcheck-L0-10.json`, **model Qwen/Qwen3-4B, residual-stream activations** — a *different
surface* from the gemma embedding table of R1/R2; anchor counts K∈{1,4,16}, median over prompts.)
Residual rows are linearly reconstructable from a handful of anchors at plaintext, decaying with
depth. This is the activation-surface plaintext baseline, included only to situate the cover-break
threat; it is not an AloePri-cover result.

**Honest gap (DEAD-END branch):** the *matched* cover-break — `cover_break` run against a genuine
non-identity orthogonal/permutation cover — is the **`fastica_anchor` variant, which is deferred**
(`cover_break.run(..., fastica=True)` raises `NotImplementedError`). So on the permutation cover the
relevant *executed* attack is VMA, not ridge-anchor cover_break; the anchor-ICA cover-break against
an injected cover Transform is a documented follow-up, **not** claimed here. See the negative-result
log `research-wiki/experiments/cover-break-matched-deferred.md`.

---

## Synthesis (the permutation channel = thesis confirmation)

1. **Sufficient-statistic matcher (R1):** the full sorted row is the maximal invariant under the
   column-permutation cover; the full-sorted matcher recovers +0.43–0.60 over RowSort-64 across the
   noise sweep, recovering exactly the leakage CLUB-on-φ says is preserved. RowSort-64's 64-quantile
   binning is DPI-dominated and under-reports leakage at small noise. → keeper claim
   `claim:perm-llr-threshold` (proof inline).
2. **Probe faithfulness (R1+R2):** the **independent** CLUB-on-φ is a monotone indicator of VMA
   recovery across the α_e sweep (Spearman +0.976 on the aloepri sweep; recorrelation +1.00 / +0.99
   in `vma_stronger.json`), and under the keymat cover it collapses *together with* recovery (CLUB
   → ≈0, VMA → chance). retrieval-PVI tracks at ρ=1.0 but it is the attack-in-bits (dependent), so it
   is a circular consistency check, not independent confirmation. Honest claim: the independent probe
   faithfully tracks the permutation attack across this sweep, with a clean keymat negative control —
   single embedding table, R2 single-seed.
3. **cover_break (R3):** plaintext anchor-recovery baseline on Qwen3-4B activations (a different
   surface) documented; the matched anchor-ICA cover-break is a deferred follow-up (negative log),
   not a manufactured claim.

### Provenance / integrity
- All numbers verbatim from `results/vma_stronger.json`, `results/aloepri_vma_sweep.json`,
  `results/fullcheck-L0-10.json`; canonical rows re-rendered through `src/talens/report.py`.
- Probe ≠ attack (audited, `EXPERIMENT_AUDIT.md`): **CLUB-on-φ** runs no Hungarian/NN matching
  (variational MI on τ-paired signatures) → independent; the keymat point shows it collapsing to its
  noise floor when the signature is destroyed. **retrieval-PVI** is the attack-in-bits (inverter +
  cosine-softmax retrieval) → dependent, reported as a labeled reference only.
- Eval type: genuine — VMA recovery is exact row-correspondence match against the known permutation;
  cover_break p95-cosine is against held-out true rows. No synthetic ground truth.
