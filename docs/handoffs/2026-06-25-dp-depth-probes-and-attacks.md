---
type: handoff
status: current
created: 2026-06-25
updated: 2026-06-25
tags: [dp, residual, depth, probes, I_G, CE-logit-lens, decoder, beamclean, measurement-loop]
companion: docs/html/resid-dp-attacks.html
---

# Handoff — DP × depth: results, claims, and unexplained behaviours

Focus for the next session: **current results, the claims they support, and the open/unexplained
behaviours** in the residual-stream-under-local-DP study. All work this session is committed
(`6f6fb5a` … `44df4d8`); read those commit messages + the files they touch rather than re-deriving.

## TL;DR state

Surface = gemma-2-2b residual stream under **local DP** (Gaussian on the input embedding), measured
across depth (L0/5/12/20/25) × ε(∞,512,256,128), 512 prompts, vocab-disjoint split. Eval =
`scripts/evals/dp_leakage_sweep.py` (one orchestrator; attacks in `src/talens/attacks/dp_inversion.py`,
probes in `src/talens/probes/`, defense `scripts/defenses/local_dp.py`). Page =
`docs/html/resid-dp-attacks.html`.

Two settled findings + one confirmed-but-not-yet-claimed finding, below.

## Results (settled, committed)

1. **Per-vector token recovery is affine-saturated.** Single-position residual→token: ridge (≈ tuned
   lens) is the ceiling; no non-linear per-vector decoder beats it. Triangulated two ways —
   cosine-regression gated decoder ≡ ridge (`refine-logs/dp-decoder-r3-fixed/`), and CE logit-lens
   `declens ≈ lens` with CE-retraining the affine even *losing* to closed-form ridge (open-set
   overfit). Claim: `research-wiki/claims/single-position-residual-linearly-saturated.md` (has the full
   audit trail incl. the withdrawn dead-decoder section). Recovery falls **monotonically with depth**
   (clean 0.99→0.42).

2. **Two real implementation bugs found + fixed (both were producing false "saturation"):**
   - `skip_decoder_attack` **dead non-linear branch** — warm-start zeroed *both* gate and MLP tail →
     zero gradient → decoder pinned ≡ ridge by construction. Fixed ReZero-style (`faae98f`).
   - CE logit-lens **dot-product/norm decode bug** — raw-dot logits rank by ‖E‖ (gemma norms are
     heterogeneous) → lens recovered 0.28 vs ridge 0.79. Fixed: cosine-CE + cosine decode + val-top1
     early-stop (`7c32893`). After fix, lens ≈ ridge (still no non-linear gain).
   Guards: `tests/test_dp_inversion.py` (4 tests, run with `.venv/bin/python -m pytest`).

3. **CONFIRMED, not yet a formal claim — the L20 depth/information peak.** Under moderate DP (ε=512),
   token-recoverable information is **non-monotone in depth, peaking at L20**, while hard recovery
   falls monotonically. Verified across **5 independent DP-noise draws** (`--seeds`): L12→L20 error
   bars **separated** for CLUB, PVI, *and* the fitting-free geometry-only **I_G** (so not estimator/
   draw noise, not probe overfitting). **Norm-control passes**: unit-normalized (direction-space) I_G
   keeps the L12→L20 rise → structural, not residual-norm growth. Data: `refine-logs/multiseed-ig/`,
   `refine-logs/multiseed-ig-unit/`; memory `l20-depth-information-peak.md`; shown in page **FIG.04**.
   **Next step (queued): write this up as a research-wiki claim** (single ε=512 so far — widen the ε
   grid to generalize).

## Probes/attacks now in the eval (registries)

- attacks: `ridge`, `decoder` (ReZero gated, settled ≡ridge), `lens`/`declens` (CE logit-lens).
- probes: `club`, `vcap` (PVI), `mdl` (retrieval), `ig` + `ig_unit` (geometry-only channel-MI).
- `--seeds` multi-seed loop varies only the DP-noise draw (split + probe init fixed), hoisting all
  noise-independent work; design note `docs/research/ce-logit-lens-attack.md`.

## Unexplained / open behaviours (the meat for next session)

1. **I_G measures representation-survival, NOT token-survival at depth** — and this is currently a
   half-resolved tension. `I_G = I(clean rep; noised rep)` (channel MI), so it (a) is **∞ at ε=∞**
   (noiseless channel → no plaintext bar in FIG.02), and (b) grows **monotonically** with depth (SNR
   rises as the residual accumulates signal energy vs bounded input noise) — *opposite* to CLUB/PVI's
   L20 peak and recovery's decline. It is a **loose upper bound** on token-MI at depth (rep carries
   context too); only at L0 is it token-matched. **OPEN: the user asked to relabel the FIG.02 I_G
   panel/caption to make "representation-channel MI, not token-MI at depth" explicit — NOT yet done.**
   Consider also whether I_G belongs alongside the token probes at all, or only as the mechanism.

2. **Why the L20 peak specifically, and the ε-dependence.** The information bump is largest at
   *moderate* noise (ε=512), tiny at ε=∞, gone at ε=128 — a depth×noise interaction (deeper layers
   partially "outrun" the input-injected noise). Mechanistic explanation (which gemma layers / what
   makes L20 the token sweet spot) is a hypothesis, not proven. Multi-ε multi-seed would map it.

3. **MDL is circular with ridge as implemented.** `online_code_length_retrieval` = the ridge model's
   prequential NLL, so MDL↔ridge correlation is **not independent** (flagged, see commit `a84f85c`
   discussion + `docs/research/ce-logit-lens-attack.md` neighbours). Canonical class-probe MDL is
   independent but vocab-capped. **Decision pending:** for the "probe predicts recovery" claim use
   geometry-only I_G (token-matched only at L0) or class-probe MDL — neither is both open-set and
   ridge-independent. MDL added little new under the unlimited-data WEIGHTS-PUB adversary (its
   data-efficiency axis is moot); it would matter only under a *data-limited* adversary reframe ("DP
   as a sample-complexity tax").

## Largest queued build: BeamClean (campaign-D Task 5)

Fully spec'd inline in `docs/plans/campaign-D-followups.md` Task 5 (impl-context block). Key points:
noisy-channel sequence decode (emission = L0 Gaussian / cosine logit-lens posterior at depth; prior =
gemma itself; beam + λ); **needs a NEW sequence eval** (the vocab-disjoint per-position bag can't
carry contiguous context) → `scripts/evals/beamclean_seq_sweep.py`; **mandatory prior-only (λ→∞)
control** to separate LM-fill from real leakage. It's the only remaining axis expected to beat ridge
under DP. Multi-hour build.

## Loose end: auto-review-loop on the CE logit-lens

`/auto-review-loop` ran Round 1 (codex, threadId in `review-stage/REVIEW_STATE.json`): score 5/10,
NOT-READY — correctness CLEAN, blockers were all perf. Fixes were applied inline (guards, doc, the
saturation-pilot evidence, the cosine bug fix), but **no formal Round-2 re-review was run** — the loop
is effectively resolved by the subsequent work but its state file still says in_progress. Either run
one closing re-review or mark it completed.

## Suggested skills for next session

- `/result-to-claim` then write `research-wiki/claims/<slug>.md` — for the L20 depth/information-peak
  finding (the one confirmed result without a claim yet).
- `/ponytail` + `/auto-review-loop` — before any BeamClean GPU run (standing rule).
- `/proof-checker` if the L20 claim grows a formal converse.

## Run discipline (carried over, do not skip)

ONE GPU process at a time; heavy commands via `scripts/run_in_rocm.sh`. The launch-time
`pgrep -f dp_leakage_sweep` self-matches the launcher's own argv (false "BUSY") — verify with
`docker ps | grep rocm` (expect exactly one container) instead. Wait on runs by PID or container,
not poll-spin. Sweeps here are ~3–7 min; full grid + club ~7 min.
