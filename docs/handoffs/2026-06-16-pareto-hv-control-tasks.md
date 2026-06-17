---
type: handoff
status: current
created: 2026-06-16
updated: 2026-06-16
tags: [pareto-probing, hypervolume, control-tasks, V-information, PVI, MDL, CLUB, attacker-compute-budget, leakage-measurement]
companion: [it-leakage-estimation-set, mdl-vinfo-inversion-toolkit]
---

# Handoff — Pareto-hypervolume as a measure layer, and its relation to control tasks

## What this session was

An exploratory comparison of **Pareto Probing** (Pimentel et al., EMNLP 2020,
[arXiv:2010.02180](https://arxiv.org/abs/2010.02180); already row **A3** in
[`mdl-vinfo-inversion-toolkit.md`](../research/mdl-vinfo-inversion-toolkit.md))
against the three committed measures (PVI/V-info, MDL, CLUB), to decide whether
its accuracy–complexity **Pareto frontier + hypervolume** adds a perspective
that helps (a) predict attacks, (b) **formalize attackability given attacker
compute budget**, (c) other characteristics. No code or docs were changed —
this is a decision-input handoff. An `AskUserQuestion` on next-step was offered
and **rejected** (user wanted to keep discussing), so the integration decision
is still open.

## Conclusions reached (the substance — don't re-derive)

1. **Framing correction.** The paper is about probe *complexity*, not attacker
   compute. The bridge to our thesis: probe complexity ≈ **bounded attacker
   capacity**. V-information is MI at **one fixed point** on a capacity axis;
   **Pareto sweeps that axis**. That is the only genuinely new perspective vs.
   our current scalars.

2. **Unifying picture.** Leakage is a **surface over (data-budget, probe-
   capacity) → recovery**. MDL = one integrated scalar; SDL (Whitney) = the
   *data*-axis slice; **V-info/PVI = a capacity-axis slice (one family V)**;
   CLUB = capacity-agnostic upper bracket. **Pareto is the only one keeping the
   capacity axis as a curve** — and that axis is "attacker compute budget."

3. **Against the three goals.**
   - *Predict attacks*: marginal as a regressor; more expensive (sweep, not one
     probe); still correlational (causal localisation is DAS's job). Only real
     gain = **frontier crossings** (two reps with equal MDL/PVI can cross — one
     leaks more to a weak attacker, the other to a strong one; a scalar can't
     express this).
   - *Formalize attackability-vs-budget*: **this is the win.** Turns the
     headline scalar into a **function**: recovery-vs-budget = a Pareto
     frontier; **hypervolume = "total attackability across all budgets"** as one
     defensible number. Lets you say "invertible *for an attacker spending ≥ c*."
   - *Other*: composes with **SDL** (swap capacity axis for attacker-data-budget
     axis); answers "which layer is *robustly* leakiest" (largest dominated
     area) vs. "leakiest at one capacity" — relevant to sensitive-layer-exclusion
     / `tee_direct`.

4. **The nesting subtlety (load-bearing for the V-family sweep idea).** User's
   intended budget axis = sweep V = {linear (current), MLP, …}. But families
   **nest** ⇒ V-info is **monotone non-decreasing** in expressivity, so
   "MLP recovers more than linear" is near-tautological and uninformative. The
   principled object is **not** "does MLP recover more" (always yes) but **"how
   much attacker capacity must be *paid* for the extra recovery, and where does
   it saturate"**. Pareto **prices recovery against capacity**; frontier
   crossings distinguish *cheap* leak (weak attacker already wins) from
   *expensive* leak (only high-capacity attacker wins) — a real cover-design
   distinction (a linear cover defeats linear probes, not MLPs).

5. **What Pareto adds beyond "just run V-info at linear and MLP".**
   - (a) **A family-agnostic complexity coordinate.** V-info gives the y-value
     only; heterogeneous families ("linear" vs "MLP") have **no shared x-axis**.
     Pareto supplies one (nuclear norm within a family; or non-parametric
     **memorization = shuffled-label accuracy** across families). Without it the
     sweep is an unordered list; with it, an integrable curve.
   - (b) **Gaming/confound invariance.** A high-capacity probe can hit accuracy
     by reading the *input distribution* (memorizing), inflating "leakage".
     Recovery-*at-fixed-complexity* is not gameable that way.
   - **Caveat:** for only 2–3 discrete families, Pareto-HV is overkill — just
     report both V-info numbers + flag crossings. HV earns its keep for (i) a
     single cross-layer-comparable scalar over 36 layers, (ii) a genuine
     *continuum* (reg strength / MLP width / rank), or (iii) high-capacity
     probes where memorization is a real confound.

6. **Control-tasks ↔ Pareto (the last exchange).** Same disease — a probe wins
   by memorizing the input→label map, not via the representation. Lineage of
   increasingly principled cures, **all of which we should treat as one family**:

   | Year | Mechanism | Confound removal |
   |---|---|---|
   | Control tasks (Hewitt–Liang '19) | `selectivity = real_acc − control_acc` | random-label baseline; **fixed weight-1 subtraction** |
   | Pareto (Pimentel '20) | frontier / hypervolume | **no committed weighting**; complexity axis *is* the control task (shuffled-label acc) |
   | **Conditional V-info** (Hewitt '21, toolkit **B2**) | leakage **above baseline B** | baseline is a *real representation*, not random labels |

   **Punchline / current standing recommendation:** our committed primary
   measure — **conditional V-info above the `WEIGHTS-PUB` baseline B** — is
   already the modern successor to control-task selectivity and does the
   confound removal better. So the **control-task / memorization-complexity axis
   of Pareto is already subsumed** for us. The part of Pareto worth taking is
   **only the capacity-axis frontier** (pricing recovery vs. attacker capacity =
   the budget question). Its complexity axis (= the control task) we can skip.

## Concrete recommendation left on the table

Add Pareto as a **reporting/aggregation layer**, *not* a fourth competing
scalar. For graded-recovery attacks (#1 hidden-state inversion, #4 embedding
inversion in [`it-leakage-estimation-set.md`](../plans/it-leakage-estimation-set.md)):
sweep a budget knob and emit a **recovery-vs-budget frontier + hypervolume**,
reusing probes already built. Natural budget axis in *this* repo = the V-info
**family capacity** (the user's instinct, not data/epochs) — making Pareto the
explicit-sweep generalization of the PVI already committed. It does **not**
replace CLUB (bracketing) or DAS (causal). Defense reading: a good cover should
**flatten the low-capacity end of the frontier** (recovery ≈ 0 until the
attacker pays for nonlinearity); Pareto-HV with a capacity x-axis measures that.

Licensing: `rycolab/pareto-probing` is **GPL-3.0** — reimplement (small: a
sweep harness + a 2D hypervolume integral, reference-point frozen once).

## Open decision for the next session

The rejected selector's options still capture the live choices:
1. **Fold into plan docs** — promote A3 from a one-line lineage entry to a
   worked subsection in `it-leakage-estimation-set.md` + `mdl-vinfo-inversion-toolkit.md`
   (capacity-axis/budget framing, the (data,capacity)→recovery surface, the
   nesting/pricing argument, the control-task→conditional-V-info subsumption).
2. **Spec the recovery-vs-budget probe** — design note: budget knob choice
   (V-capacity vs params vs SDL-data), sweep harness, 2D HV integral,
   reference-point choice, reuse of existing PVI/MDL probes.
3. **Prototype on attack #1 or #4** — minimal frontier + HV wired into
   `calibrate()`. Heavier; capture is a GPU job (not yet run on real Qwen3 —
   see plan doc "Status").
4. Leave as analysis only.

User's framing in the args ("Pareto hypervolume as a measure **on top of**
control tasks, to *formalize*") leans toward option 1/2: write up Pareto as the
weighting-free formalization sitting above control-task selectivity, with the
caveat from §6 that conditional-V-info already supersedes the control-task axis,
so the formalization should be built on the **capacity axis**, not the
memorization axis.

## Pointers

- Plan / attack×measure matrix: [`docs/plans/it-leakage-estimation-set.md`](../plans/it-leakage-estimation-set.md)
- Measure lineage + toolkit (A3 = Pareto, B2 = conditional probing): [`docs/research/mdl-vinfo-inversion-toolkit.md`](../research/mdl-vinfo-inversion-toolkit.md)
- IT bridge survey: [`docs/research/interpretability-leakage-bridge.md`](../research/interpretability-leakage-bridge.md)
- Measure code: `src/talens/measures/` (`vinfo.py`, `mdl.py`, `club.py`, `_probe.py`, `_retrieval.py`); calibration in `src/talens/calibration.py`.
- Key external refs not yet in lineage: **Hewitt & Liang, Designing and
  Interpreting Probes with Control Tasks** (EMNLP 2019) — the control-task /
  selectivity origin; pair with A3 and B2 when writing up.

## Suggested skills for next session

- `research` — if pulling the Hewitt–Liang control-tasks paper details / any
  follow-on selectivity critiques into the lineage doc.
- `docs-tidy` — if folding the comparison into the two research/plan docs (keeps
  frontmatter + folder conventions correct).
