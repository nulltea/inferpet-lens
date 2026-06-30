# transformer-attacks-lens

Information-theoretic, interpretability-grounded study of confidential-inference attacks on
transformers: measure how invertible a representation is, run attacks on it, and test whether an
attack-**independent** IT measure predicts attack success. Map: `README.md` (premise),
`docs/plans/it-leakage-estimation-set.md` (thesis + attack×measure matrix),
`docs/plans/component-topology.md` (attack/probe/defense schema).

## GPU — run heavy workflows in the host `.venv`

The host `.venv` **runs on the GPU directly** — no container. Use it for **all** heavy workflows
(capture, PVI/MDL/CLUB probes, inversion attacks, `talens.cli`, `calibrate_capture`) and for
`pytest`:

```bash
.venv/bin/python -m talens.cli --corpus corpora/dev-24.txt --control all --out results/run.json
# sanity: .venv/bin/python -c 'import torch; print(torch.cuda.is_available())'   # -> True
```

One AMD Strix Halo iGPU (gfx1151). **One GPU process at a time**: don't launch a second GPU run while
one is live; wait on long runs, never poll-spin.

GPU-torch setup, troubleshooting, and enabling another venv → **`~/docs/torch-gpu.md`** (installer:
`~/scripts/install_rocm_torch.sh`). Don't delete the `rocm/pytorch` base image — it's the torch source.

## Scheme-agnostic core

The library asserts nothing about any defense — it measures leakage and runs attacks on whatever
it is handed. A defense is an external `talens.transforms.Transform` (`Tensor→Tensor`); only
`Identity` ships. Keep covers / noise / threat models in callers, tests, or `scripts/defenses/` —
never in the core. `WEIGHTS-PUB` (adversary knows weights + embeddings, so norms / Grams /
`softmax(QKᵀ)` are known functions of the secret) is the default *motivating* threat model, not a
library invariant. Secrets of interest: activations/hidden states, Q·K·V, attention scores,
KV-cache, and the tokens behind them.

## Repo structure (code layout) — enforce this

Reusable components live in fixed homes; the runnable scripts only *call* them.

| Kind | Home | Notes |
|---|---|---|
| **attacks** | `src/talens/attacks/` | e.g. `dp_inversion.py` (ridge, linear-skip MLP decoder, BeamClean). Array-interface attacks for evals + the CaptureSet API (`hidden_state.py`, `_inversion.py`). |
| **probes** | `src/talens/probes/` | the leakage measures (CLUB, V_cap, spectral channel-MI, MDL/SDL, …). **Renamed from `measures/`** — import `talens.probes.*`. |
| **defenses** | `scripts/defenses/` | external `Transform`/mechanism modules (LocalDP, PriPert, GELO, KV-Cloak, Shredder, AloePri, SGT). Never in the core. |
| **runnable evals** | `scripts/evals/` | config-driven sweeps that ONLY import + orchestrate attacks/probes/defenses + capture. |
| **spikes** | `scripts/spikes/` | **temporary exception only** — for testing a *new* probe/attack/defense. As soon as it is confirmed, MOVE the reusable logic into the appropriate home above and have the spike/eval call it. Do not let logic accrete in spikes. |

Rule of thumb: if a function would be reused by a second script, it belongs in `src/talens/attacks|probes` or `scripts/defenses`, not inline in an eval or spike.

## Research operating model

- **Probe ≠ attack (integrity-critical).** A probe is an *independent* measure that *correlates
  with* attack recovery — never the attack reporting its bits instead of its recovery rate. If you
  could not compute it *without running the attack*, it is the attack in disguise and any
  correlation is circular. Prefer geometry-only / channel-matched probes.
- **Metric: bits canonical + per-secret readout.** Store **bits** (MI / V-info / capacity / SDL —
  one comparable scale; fix "1/100 of a bit" illegibility in the readout, not the stored value).
  Render beside it: token-id→perplexity+top-k; text→token-F1/ROUGE; permutation→recovery-rate/τ;
  embedding→token-F1/cosine; membership→AUC. Tables show both axes.
- **Missing data in tables — always check and address.** Before presenting any results table, audit
  every cell. A bare **`—` means data NOT collected** (a real gap: either fill it by running the cell,
  or flag it explicitly as not-run). If a cell **cannot have a value by definition**, write **`—*`**
  with a footnote **`* value missing because …`**. Never leave an unexplained blank — the reader must
  always be able to tell "not run" from "undefined".

### The measurement loop — per (surface × attack × probe)

This is the core method. Follow it for every cell:

1. **Run the attack** on the surface → graded **recovery** (token-F1 / top-k / BLEU / AUC / cosine).
2. **Run the probe** (attack-*independent*) on the same surface → **bits** (+ readout).
3. **Sweep** from plaintext through the defense's privacy parameter(s); collect `(bits, recovery)`
   pairs across the sweep.
4. **Do bits and recovery correlate across the sweep?**
   - **Yes** → the probe predicts the attack. Draft the claim, prove it, render the page
     (Claim → Theory → Report recipes below).
   - **No** → that *is* the finding. Identify which:
     - *attack too weak / ill-equipped* → a stronger attack should re-correlate — queue it;
     - *probe not channel-matched* → it measures the wrong thing — design/queue a matched probe.
     Bound or explain the gap in theory, report it, and queue the follow-up.
5. **Negative results are first-class** — record them; never hide a gap or manufacture a claim.

### Performance gate — before any GPU run

A run must pass the perf gate before launch:

- **Optimal scope** — the smallest run that answers the question. Fast-iterate on one layer /
  `--every-n` before a full sweep; no redundant sweep points.
- **Max GPU utilization** — every component that *can* run on GPU does (capture, probe nets,
  attack fits); PCA via cov-eigh on GPU, not full SVD; batch sizes saturate the iGPU.

Refine the run plan with `/auto-review-loop` against the standardized perf prompt
(`scripts/harness/perf_gate.md`) until it passes. One GPU process at a time; estimate wall-time,
and if it exceeds 10 min, confirm saturation first.

## Skills & cadence

**Doctrine — the outer loop drives, never acquits** (`.aris/shared-references/external-cadence.md`).
You decide *when/whether* a phase runs; every *correctness or quality verdict* belongs to an ARIS skill's own
cross-model jury. **Never self-certify** a result, proof, or claim. Verdict-bearing skills
(`/auto-review-loop`, `/proof-checker`, `/result-to-claim`, `/experiment-audit`) run **inline**
inside a phase, hold their own thread memory, and resume from their `*_STATE.json` — do not
re-spawn them per tick. Skills are invoked by `/name` only (they are not auto-enumerated).

Run the recipe for the work at hand:

| Stage | Skill chain |
|---|---|
| Literature / discovery | `/research-lit` · `/arxiv` · `/semantic-scholar` · `/openalex` (+ edgequake MCP) → `/research-wiki` (register paper nodes) |
| Experiment | `/experiment-plan` → `/experiment-bridge` → `/result-to-claim` → `/experiment-audit` → `/auto-review-loop` |
| Theory | `/formula-derivation` (optional) → `/proof-writer` → `/proof-checker` |
| Claim | `/result-to-claim` (judge support) → write `research-wiki/claims/<slug>.md` **in full, proof inline** → `/research-wiki` (register) → `/novelty-check` (if claiming novelty) |
| Report | build `docs/html/<surface>.html` per `docs/html/STYLE.md` + `/figure-spec` / `/paper-figure` diagrams → `/auto-review-loop` |
| Paper (later) | `/paper-plan` → `/paper-write` → `/paper-figure` → `/citation-audit` + `/paper-claim-audit` → `/kill-argument` (pre-submission hardening) → `/paper-compile` |

**Notes.** A research claim is a **research-wiki node** (Claim row), never a patent claim.
`/experiment-bridge` deploys via `/run-experiment` — keep it in **local** mode (runs directly in the
GPU-enabled host `.venv`); on the single iGPU run sweep points **serially** (`max_parallel=1`,
no `/experiment-queue` fan-out). Off-domain skills (patent, remote-GPU, Feishu) were pruned from
this project's install.

Campaign + harness design: `scripts/harness/README.md`.

## Deliverable locations

| Artifact | Path |
|---|---|
| Claims **with full proofs** | `research-wiki/claims/<slug>.md` |
| Experiment logs | `research-wiki/experiments/` |
| Working notes / runs | `refine-logs/<surface>/` (archive old `EXPERIMENT_*` into named subdirs first) |
| HTML report | `docs/html/<surface>.html` (reuse `css/site.css`; template `vec2text.html`; serve via `scripts/harness/serve_docs.sh`) |

## Docs (`docs/**/*.md`)

Frontmatter required: `type` (handoff·plan·prototype-note·research·theory·dev-log·reference),
`status` (current·partial·stale), `created`, `updated`, `tags`; optional `superseded_by`,
`supersedes`, `companion`, `archive_reason`. Folder by `type`: handoff→`docs/handoffs/YYYY-MM-DD-<slug>.md`,
plan|reference→`docs/plans/`, research|theory→`docs/research/`, prototype-note→`docs/dev/prototype/`,
dev-log→`docs/dev/logs/`. Archive inactive handoffs to `docs/archive/handoffs/`; stale plans stay
put with `status: stale` + `archive_reason`. `companion:` references repo-local docs only.
`docs/research/` docs need a Definitions glossary (the IT vocabulary is dense and cross-community).

## Voice

State what is known / measured / decided and *why*. Comparative tables over prose lists. Cite
arXiv/DOI inline; note that citation counts drift.
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 55 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/home/timo/repos/Auto-claude-code-research-in-sleep`).
Update with: `bash /home/timo/repos/Auto-claude-code-research-in-sleep/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
