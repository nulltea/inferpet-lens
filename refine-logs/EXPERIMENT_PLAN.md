---
type: plan
status: current
created: 2026-06-20
updated: 2026-06-20
tags: [experiment-plan, matched-probe, leakage-channels, decoupling-law, AloePri, Shredder, PID, cross-scheme-calibration]
companion: [FINAL_PROPOSAL, idea-stage/IDEA_REPORT]
supersedes: [archive-capacity-pvi/EXPERIMENT_PLAN]
---

# Experiment Plan — matched probes × leakage channels × defences

Grounded in `refine-logs/FINAL_PROPOSAL.md`. The program tests **one principle**
(matched independent probe predicts its channel's attack; mismatched pairs
decouple) over **4 channels × 6 defences**, and produces the data that
**adjudicates the headline framing (F-A/B/C) and selects the Π-probe** — both
deferred by the user to data.

## Definitions

| Term | Meaning |
|------|---------|
| **Channel** | A (target × surface) leakage path: {token-id, permutation-Π, embedding-geometry, attention-QK/OV}. |
| **Matched probe `P_c`** | An IT readout with high `I(P_c; target_c)` computed *without* the attack's fitted map (independent by construction). |
| **`m_c`** | Channel `c`'s attack success metric (TTRSR; τ-recovery rate; ridge cosine; ISA recovery). |
| **Diagonal** | Matched pair correlation ρ(P_c, m_c). **Off-diagonal**: ρ(P_c, m_{c′}), c≠c′. |
| **Decoupling law** | Diagonal calibrates (ρ≥0.9); off-diagonal systematically lower / sign-flips under defences (the L20×DP datum, generalised). |
| **Defence knob** | Per-family monotone leakage control: DP ε; AloePri noise α_e; Shredder SNR/cut; split depth. |

## Claim map

| Claim | Statement | Minimum convincing evidence | Blocks |
|-------|-----------|------------------------------|--------|
| **C1 (per-channel matching)** | Each channel has a matched **independent** probe that calibratedly predicts its attack. | ≥1 probe/channel with macro/within-layer ρ(P_c,m_c) ≥ 0.9 across ≥2 defences, **and** per-instance collinearity <0.9 with the channel's attack-in-bits reference. Channel 1 already PASSED. | B1, B2, B5 |
| **C2 (decoupling law — the prediction)** | Mismatched probe↔target pairs decouple; the matrix diagonal dominates the off-diagonal. | A K×K ρ-matrix (K=#channels) whose diagonal mean − off-diagonal mean > 0 with non-overlapping bootstrap CIs, **and** ≥1 off-diagonal cell that *sign-flips* under a defence (extends L20×DP). CLUB-replicates-gradient control rules out "estimator artifact." | B3 |
| **C3 (cross-scheme calibration)** | A channel's matched curve transfers across defence families. | Fit `P_c→m_c` on one defence, predict on a held-out defence: ρ ≥ 0.85, for ≥2 channels across ≥3 of the 6 defences. | B4 |
| **C4 (Π-probe selection)** | Identify the independent Π-probe (resolve the deferred question). | Of {CLUB-on-φ, capacity-reader-on-φ, retrieval-PVI-on-φ}, the winner has ρ(·,τ-recovery)≥0.9 over the AloePri α_e sweep **and** per-instance collinearity <0.9 vs VMA-PVI. If none → Π-channel has no independent probe (a result; weakens F-A, informs F-C). | B2 |
| **C5 (framing verdict)** | Conclude F-A / F-B / F-C from the matrix. | The B6 decision rule (FINAL_PROPOSAL table) applied to B1–B5 outputs. | B6 |
| **C1′ (conditional formal)** | If a channel's matching fails, an identifiability / MI↔success argument predicts the failure. | A condition (e.g. floor ∝ d/n; MI-success gap from de Chérisey) reproducing the observed failure. | B6-formal |

**Anti-claims to rule out**
- **"The probe is the attack."** → per-instance collinearity test in every channel
  (Π: vs VMA-PVI; token-id: vs retrieval-PVI — already done, ρ=0.76).
- **"Decoupling is an estimator artifact."** → CLUB (independent estimator) must
  show the same off-diagonal gradient (it did at L20). Report as the control.
- **"AloePri obfuscation broke the model, not the probe."** → logit-fidelity check
  (B0): obfuscated forward must match plaintext within tolerance.

## Experiment blocks

### B0 — Implement + unit-test (model-free / synthetic) — MUST-RUN
- **Builds:**
  1. **AloePri Alg1 obf-table generator** — `KEYMATGEN`/`INVKEYMATGEN` (orthogonal
     `U`, `B=U+λV`, null-space `C⊥F`, `D⊥E`, `P̂=[B C E]Z`, `Q̂=Zᵀ[B⁻¹ F D]ᵀ`,
     verify `P̂Q̂=I`), + per-row permutation Π + Gaussian α_e on embed/head →
     emits an obfuscated `WeightPair`. (lives in defence-eval / Part 2, not core.)
  2. **AloePri permutation-core Transform** (`Tensor→Tensor`): token + embed-row
     permutation; `name="aloepri_perm"`.
  3. **Shredder static-Laplace Transform**: additive Laplace at a split activation.
  4. **Shredder learned-noise trainer**: noise tensor min `1/SNR`+task-CE at a cut,
     fit Laplace, per-input sample (stub the train loop; tested on a toy net).
  5. **Capacity-reader-on-invariants** Π-probe: `v_information_capacity` adapted to
     predict *row-position* from φ(obf) (the independent bake-off candidate).
  6. **PID QK-vs-OV probe**: redundant/unique/synergistic decomposition of
     `I(tokens; {QK, OV})` (BROJA or CVXPY discrete PID on binned/clustered tokens,
     or a Gaussian-PID approximation — pick cheapest that passes the toy oracle).
- **Tests:** `P̂Q̂=I` to fp tol; AloePri obf preserves logits on a toy linear net;
  permutation Transform is a bijection; Laplace fit recovers known b; PID
  non-negativity + sums to total MI on a synthetic redundant/synergistic source;
  capacity-reader floor≈0 on shuffle. Full suite stays green.
- **Cost:** <20 min CPU. **Gate:** all unit tests pass → B1.

### B1 — Per-channel matched calibration on cheap defences (cached capture) — MUST-RUN
- **Claim:** C1 (channels 1,3; channel 2 in B2; channel 4 in B5).
- **Data:** cached gemma-2-2b capture (`results/capture_cache/…`), layers {0,5,12,20}.
  Defences: input-DP ε-sweep + split-depth (both exist, cheap).
- **Systems:** ch1 capacity-PVI reader-acc (✅) vs ridge TTRSR; ch3 CLUB vs ridge
  cosine. Reference: retrieval-PVI (attack-in-bits), class-PVI (broken baseline).
- **Metrics:** within-layer + macro ρ, partial-ρ|knob, per-instance collinearity.
- **Success:** ch1 ρ≥0.9 (already 0.82–1.0); ch3 CLUB ρ≥0.85.
- **Cost:** ~20–40 min ROCm (cache hit). **Priority:** MUST-RUN.

### B2 — Π-channel: AloePri sweep + the 3-way probe bake-off — MUST-RUN
- **Claim:** C1 (channel 2) + **C4 (resolves the deferred Π-probe)**.
- **Data:** AloePri-obfuscated `WeightPair` from B0, **α_e noise sweep**
  {0, 0.1, 0.2, 0.5, 1.0} (low→high noise = the calibration knob); embed surface
  primary, head surface confirm. VMA τ-recovery (RowSort+Hungarian) = ground truth.
- **Systems:** the 3 candidate probes on φ=sorted_quantile — **CLUB-on-φ** (✅
  `club_mi_weights`), **capacity-reader-on-φ** (B0, independent), **retrieval-PVI-on-φ**
  (✅ `v_information_weights`, the *dependent* reference = VMA in bits).
- **Metrics:** ρ(probe, τ-recovery) over α_e; per-instance collinearity(probe,
  VMA-PVI) — independence iff <0.9; cost.
- **Success (C4):** a probe with ρ≥0.9 **and** collinearity<0.9. **Failure:** only
  retrieval-PVI tracks → Π has no independent probe (result; informs framing).
- **Cost:** ~15–30 min (mostly CPU table algebra + a few CLUB fits). **Priority:** MUST-RUN.

### B3 — The decoupling matrix (the prediction, headline) — MUST-RUN
- **Claim:** C2. **This is the experiment the paper lives or dies on** (round-1 reviewer).
- **Protocol (reviewer-specified — shared CONDITION INDEX, never a shared metric scale):**
  1. **One shared defence grid that moves ≥3 channels non-identically** — a **2D
     `ε × depth` grid** (a single monotone knob can *fake* diagonal dominance):
     ε ∈ {0.25, 0.5, 1, 2, 4} (or the existing {∞,…,256} mapped) × depth/layer ∈
     {0, 5, 12, 20} → **≥16–20 shared settings** `s` (7 is too few). 3 seeds.
  2. At each setting `s`, compute **every probe** `P_i(s)` and **every attack**
     `A_j(s)`; build `M[i,j] = Spearman(P_i(s), A_j(s))` — **rank** correlations
     only (never raw TTRSR vs cosine vs τ-recovery), with bootstrap CIs.
  3. Channels (rows/cols): token-id (probe: cap-PVI acc / attack: ridge TTRSR),
     Π (CLUB-on-φ / VMA τ-recovery), embedding (CLUB I(rep;emb) / ridge cosine);
     attention (MMI-PID increment / ISA) **only if B5 is ready and 3×3 is clean**.
  4. **Diagonal-dominance test:** per row `i`, `Δ_i = ρ(i,i) − max_{j≠i} ρ(i,j)`,
     bootstrap over settings (+seeds/tokens); **strong claim = most `Δ_i > 0` with
     95% CIs excluding 0**.
  5. **≥1 genuine sign-flip, preferably 2** across different channels/defences (the
     L20×DP cell is the seed). Repeated flips / near-zero off-diagonals across >1
     defence family is what would upgrade "principle" → "law" (do not claim yet).
  6. **Negative controls (all four):** shuffled defence labels; a random probe; the
     attack-derived probe (retrieval-PVI) marked *dependent*; a **monotone
     shared-noise baseline** demonstrating diagonal dominance is *not* automatic.
- **Figure:** Spearman heatmap with CI annotations + a diagonal-vs-best-off-diagonal bar plot.
- **Success:** most `Δ_i > 0` (CIs exclude 0) **and** ≥1 sign-flip **and** the
  monotone-baseline control does *not* show diagonal dominance.
- **Cost:** the headline GPU run — unified runner over the ε×depth grid × 3 seeds on
  the ROCm container (~1–2 GPU-hr; validate optimality + saturation first, restrict
  to L0/5/12/20). **Priority:** MUST-RUN (highest leverage; run before B5).

### B2+ — Firm up the Π channel (reviewer fix #4) — SHOULD-RUN (cheap, CPU)
- Extend B2: **3–5 seeds** (perm + noise), **12+ α_e** densifying the 0.2–0.7
  transition, **bootstrap CIs** for CLUB and VMA, confirm CLUB is invariant to
  RowSort/Hungarian details, and **one more model width** (Qwen3-4B embed, d=2560).
- **Success:** ρ(CLUB-on-φ, τ-recovery) ≥ 0.9 with CI lower bound > 0.8, stable across
  seeds + both widths. **Cost:** ~10 min CPU. Promotes B2 from go/no-go to a channel claim.

### B4 — Cross-scheme calibration over all 6 defences — NICE-TO-HAVE (gated on C1)
- **Claim:** C3.
- **Data:** for channels with a matched probe (≥ch1, ch3, and ch2 if C4 passes),
  fit `P_c→m_c` on one defence, predict held-out: matrix over {DP, split-depth,
  AloePri-perm, AloePri-full, Shredder-static, Shredder-learned}.
- **Order (cheapest first):** split-depth → AloePri-perm → Shredder-static →
  AloePri-full → Shredder-learned. **Gate each behind the previous transferring.**
- **Success:** held-out-scheme ρ≥0.85 for ≥2 channels across ≥3 defences.
- **Cost:** ~4–10 GPU-hr (dominated by AloePri-full logit-verify + learned-Shredder
  training). **Priority:** NICE-TO-HAVE; the F-B robustness story.

### B5 — Attention QK/OV PID channel — NICE-TO-HAVE
- **Claim:** C1 (channel 4) + the `softmax(QK^T)`-cover-invariance whitespace.
- **Data:** kq / kqv_out captures; ISA-AttnScore attack; PID(tokens; QK, OV).
- **Probe-target match:** PID *unique-to-QK* tracks kq-leakage; *unique-to-OV*
  tracks kqv_out-leakage. Cover-invariance check: inject a shared rotation
  Transform; confirm `I(tokens; QK)` and ISA recovery are unchanged (the lemma).
- **Success:** PID component ρ≥0.85 with its surface's attack; invariance confirmed.
- **Cost:** ~30–60 min + PID solve. **Priority:** NICE-TO-HAVE; richest if B1–B3 hold.

### B6 — Framing decision gate (+ conditional formal) — MUST-RUN (decision)
- **Claim:** C5 (and C1′ only if a channel's matching failed).
- **Method:** apply the FINAL_PROPOSAL decision rule to B1–B5: count channels with
  matched diagonal ≥0.9; check off-diagonal dominance (B3); check Π-channel sharpness
  (B2). Emit verdict F-A / F-B / F-C with the supporting cells. If any channel
  failed matching, write the identifiability/MI-success argument (C1′).
- **Cost:** analysis only. **Priority:** MUST-RUN — this is what the user deferred.

## Run order & milestones

| M | Goal | Blocks | Decision gate | Cost |
|---|------|--------|---------------|------|
| **M0** | implement + unit-test all new probes/defences | B0 | all tests green (incl. `P̂Q̂=I`, logit-fidelity, PID non-neg) | <20 min CPU |
| **M1** | per-channel matched calibration (cheap defences) | B1 | ch1 ρ≥0.9 (✅), ch3 CLUB ρ≥0.85 | 20–40 min ROCm |
| **M2** | Π bake-off + AloePri sweep → **select Π-probe** | B2 | C4: independent probe found, or "none" | 15–30 min |
| **M3** | **decoupling matrix** | B3 | C2: diagonal>off-diag (CIs), ≥1 sign-flip | ~10 min |
| **M4** | **framing verdict** | B6 | F-A / F-B / F-C declared + Π-probe locked | analysis |
| **M5** | cross-scheme calibration (gated) | B4 | C3 transfer ρ≥0.85 | 4–10 GPU-hr |
| **M6** | attention PID + invariance lemma | B5 | C1 ch4 + invariance | 30–60 min |

**External review:** run `/auto-review-loop` (gpt-5.5 xhigh) **after M3** — once the
matrix exists, per the user's data-first stance (and the Codex-sandbox-can't-read-
files constraint makes pre-data review low-value). Paste matrix tables inline.

## Compute & data budget
- **To the framing verdict (M0–M4):** ~1.5–2.5 GPU-hr — all on cached capture +
  CPU table algebra + the existing DP/split sweeps. **Cheap; decides the headline.**
- **Full program (through M5–M6):** ~6–13 GPU-hr, dominated by AloePri-full
  logit-verification and learned-Shredder training. Heavily gated.
- **Data prep:** none new for M0–M4 (cached capture + synthetic obf tables). M5
  AloePri-full / Shredder-learned need GPU forward passes (+training for Shredder).

## Discipline (from auto-memory)
- Heavy runs via `scripts/run_in_rocm.sh` only; validate GPU saturation; inspect if
  >10 min. PCA = covariance-eigh on GPU (never full SVD). Kill containers by explicit ID.
- Fast loop: `--layers 12 --every-n 2` on the cached capture; capture-fresh only for L0/L20.

## Final checklist
- [x] Main tables: T1 per-channel matching, T2 Π bake-off, T3 decoupling matrix, T4 cross-scheme
- [x] Novelty isolated (B2/B3 independence + the decoupling law)
- [x] Simplicity defended (membership cut; Vec2Text→CLUB; cheapest PID solver)
- [x] No frontier-LLM-primitive claimed (non-frontier method)
- [x] Must-run (B0–B3, B6) vs nice-to-have (B4, B5) separated
- [x] Deferred decisions (framing, Π-probe) routed to explicit gates (B6, B2)
