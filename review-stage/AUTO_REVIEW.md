# Auto Review Log — matched-probe program

Reviewer: gpt-5.5 (Codex MCP, xhigh), difficulty=medium. Artifacts pasted inline
(sandbox can't read repo files). (Prior capacity-PVI thread archived as
`AUTO_REVIEW.capacity-pvi.md`.)

## Round 1 (2026-06-20)

### Assessment (Summary)
- **Score: 6.5/10**
- **Verdict: ALMOST** (meets stop condition: ≥6 AND almost)
- Key criticisms:
  1. **B3 not yet demonstrated** — the off-diagonal decoupling matrix is the headline; without it the paper reads as "correlations among leakage proxies."
  2. **"Law" language overclaims** — call it a channel-specific *calibration principle* / empirical decoupling *protocol* until replicated across defense families.
  3. **Correlations may be driven by one monotone knob** — use a 2D grid (ε × depth) or multiple defense families; add monotone-baseline + shuffled-label controls.
  4. **B2 statistically thin** — ρ=0.976 over 7 α points, 1 seed. Add seeds, densify the 0.2–0.7 transition, bootstrap CIs, a second model width.
  5. **Channel definitions may blur** — add a table: per channel {secret, surface, probe, attack, independence criterion, metric, expected failure mode}.

### Reviewer Raw Response
<details><summary>Full response</summary>

Verdict: ALMOST, not top-venue ready yet. Score 6.5/10. Core idea promising, B2 a useful new channel result, but risks reading as "correlations among several leakage proxies" unless B3 cleanly proves channel specificity with strong controls.

SHARPEST FRAMING — do not call it an IT "law" yet. Frame as: "Confidential-inference leakage is channel-specific: a probe is meaningful only relative to a target secret, observation surface, and attacker family. Matched probes calibrate their paired operational attack, while mismatched probes can anti-calibrate or become statistically irrelevant." Stronger than "different measures correlate with different attacks" because it gives a falsifiable protocol: define channels, require probe/attack independence, test diagonal calibration, test off-diagonal failure, show security conclusions change under scalar aggregation.

B3 MINIMUM DESIGN — use a shared CONDITION INDEX, not a shared metric scale. For each defense setting s, compute every probe P_i(s) and attack metric A_j(s); build M[i,j]=Spearman(P_i(s),A_j(s)). Compare rank correlations with bootstrap CIs (never raw TTRSR vs cosine vs τ). (1) one defense sweep moving ≥3 channels non-identically — input-DP depth/ε grid or split-depth×noise; 2D grid stronger (monotone shared degradation can fake diagonal dominance). (2) ≥12–20 shared settings (7 too few): e.g. ε∈{.25,.5,1,2,4}×depth∈{0,5,10,15}=20. (3) K×K matrix over identical settings: token vs TTRSR, Π vs τ-recovery, embedding vs ridge cosine, attention PID increment vs ISA. (4) diagonal dominance test: per row i, Δ_i=ρ(i,i)−max_{j≠i}ρ(i,j), bootstrap over settings (+seeds/tokens), strong claim = most Δ_i>0 with CIs excluding 0. (5) ≥1 genuine sign flip, preferably 2 across different channels/defenses; a "law" needs repeated flips or near-zero off-diagonals across >1 defense family. (6) negative controls: shuffled defense labels, random probe, attack-derived probe marked dependent, monotone shared-noise baseline showing diagonal dominance is not automatic. Best figure: Spearman heatmap with CI annotations + diagonal-vs-best-offdiag bar plot.

B2/CLUB-on-φ: ρ=0.976 over 7 α promising but thin; enough for go/no-go not a top-venue channel claim. Add 3–5 seeds, 12+ α (densify 0.2–0.7), bootstrap CIs for CLUB and VMA, show CLUB independent of RowSort/Hungarian details, one more model width / subset size. Separate the keymat floor result clearly (perm-core exposes the channel; dense keymat closes it).

Capacity-reader degeneracy: deferring to activations is sound but the bake-off wording is risky — say explicitly "the weight-table Π surface does not admit a meaningful capacity-reader baseline; the activation Π surface is the appropriate place to test reader-style probes." Stop presenting the weight-table reader as a failed competitor.

Ranked weaknesses + min fixes: (1) B3 not demonstrated → one shared 12–20 condition sweep, full probe×attack matrix, bootstrap diagonal dominance, ≥1 sign flip. (2) "law" overclaims → "empirical decoupling protocol / channel-specific calibration principle." (3) correlations may be one monotone knob → 2D grid or multiple defense families + monotone-baseline + shuffled-label controls. (4) B2 low support → more α, seeds, CIs, transition density. (5) channel definitions blur → table {secret, surface, probe, attack, independence criterion, metric, expected failure mode}.

HIGHEST-LEVERAGE NEXT: run B3, not B5. The paper lives or dies on the off-diagonal matrix. Minimum GPU run: 3 channels, 12–20 shared settings, 3 seeds. A clean 3×3 diagonal-dominant decoupling matrix with a sign flip is more publishable than a scattered 4-channel story without off-diagonal rigor.

</details>

### Actions Taken (this round — documentation fixes, no GPU needed)
- Reframed "decoupling law" → **"channel-specific calibration principle / empirical decoupling protocol"** in FINAL_PROPOSAL + EXPERIMENT_PLAN (fix #2).
- Rewrote **B3** with the reviewer's concrete protocol: shared condition index, 2D ε×depth grid (≥16 settings), K×K Spearman matrix, bootstrap Δ_i diagonal-dominance test, ≥1–2 sign-flips, 4 negative controls (fixes #1, #3).
- Added the **channel-definition table** to FINAL_PROPOSAL (fix #5).
- Fixed the **capacity-reader / weight-table** wording in EXPERIMENT_RESULTS + plan (fix; not a "failed competitor").
- Logged the **B2+ firm-up** block (seeds, α density, CIs, 2nd width) (fix #4).

### Status
- **Loop TERMINATED** at round 1: stop condition met (6.5 ≥ 6 AND verdict=almost).
- Next concrete experiment (reviewer #1, highest-leverage): the **B3 off-diagonal GPU run** — unified runner, 3 channels × ε×depth grid × 3 seeds, ROCm container. Needs user go-ahead (GPU discipline: validate optimality + saturation first).

## Method Description
The method decomposes confidential-inference leakage into (target × surface)
**channels** {token-identity, permutation-Π, embedding-geometry, attention-QK/OV}.
Each channel is paired with a **matched independent information-theoretic probe**:
capacity-matched PVI reader accuracy (token-id), CLUB on the sorted-quantile row
signature φ (permutation-Π), CLUB I(rep;emb) (embedding), and MMI-PID
conditional-increment atoms (attention). A probe is *independent* iff it is computed
without the paired attack's fitted map (verified by per-instance collinearity). Data
flow: capture activations / load the weight table → apply a defense (input-DP,
split-depth, AloePri permutation-core / Algorithm-1 key-matrix, Shredder
static/learned Laplace) → run each channel's attack (ridge TTRSR, VMA τ-recovery,
ridge cosine, ISA) and its matched probe over a shared defense-setting grid → build
the K×K probe×attack Spearman matrix and test diagonal dominance (matched pairs
calibrate; mismatched pairs decouple, with documented sign-flips).
