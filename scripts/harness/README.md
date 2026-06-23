# talens autonomous-research harness (ralphex + ARIS)

Long-horizon, unattended research driver. **ralphex** is the outer fire-control loop
(schedules phases, fresh `claude` per phase, commits, retries); **ARIS skills** are the
inner verdict loops (experiment-plan↔bridge, proof-writer↔checker, auto-review-loop) that
run *inline* inside each phase and terminate in a cross-model jury. This harness is the
glue: a doctrine-wired task prompt, an acceptance gate, Telegram alerting, silent-death /
stall detection, and a circuit-breaker.

Design rests on `Auto-claude-code-research-in-sleep/skills/shared-references/external-cadence.md`:
**outer cadence may DRIVE, never ACQUIT.** ralphex decides *when/whether* a phase runs; the
ARIS jury inside the phase decides *whether the work is good*. The gate below is how ralphex
reads that verdict without making one.

## Components

| File | Role |
|---|---|
| `ralphex-config/config` | tracked ralphex config (reviews off, retry=1, timeouts, custom-notify, commit trailer) |
| `ralphex-config/prompts/task.txt` | the per-phase prompt: heartbeat→recipe→stall/escalation→**gate**→complete. Recipes: `consolidate` (harvest existing spikes), `full` (new surface: experiment→theory→report), `experiment`, `theory` |
| `preflight.sh` | one-shot pre-campaign check (Codex `doctor` auth, GPU, ralphex, skills); wired into `run_campaign.sh` so a phase never blocks mid-run. Skip with `TALENS_SKIP_PREFLIGHT=1` |
| `perf_gate.md` | standardized `/auto-review-loop` prompt gating every GPU run (optimal scope + max GPU utilization) |
| `serve_docs.sh` | `start\|stop\|status` — expose `docs/html/` at one tailnet URL via `tailscale serve` |
| `gate_check.sh` | exit 0 iff the ARIS verdict file records acceptance (review/proof/runstate/jq/marker) |
| `notify_telegram.sh` | out-of-band Telegram curl (works even when a session is dead); no-ops if unconfigured |
| `ralphex_notify.sh` | ralphex `notify_custom_script` → parses Result JSON → `notify_telegram.sh` |
| `watchdog_poller.sh` | runs ARIS `watchdog.py` daemon + forwards STALE/DEAD alerts to Telegram |
| `run_campaign.sh` | outer wrapper: skip-and-continue past a stuck phase, halt after N consecutive no-progress rounds |
| `systemd/*.service` | `talens-research` (campaign, Restart=on-failure) + `talens-watchdog` (always-on) |

## Campaign plan format (ralphex-native markdown)

One `### Task N:` per **experimentation surface**. Each carries metadata lines and exactly
**one checkbox** `- [ ] run-phase: <run_id>` (the agent flips it to `[x]` only after the gate
passes; the wrapper flips it to `[x] … SKIPPED-gate-not-met` to advance past a stuck phase).
`recipe:` ∈ {`consolidate`, `full`, `experiment`, `theory`}; a `full` phase that doesn't
correlate self-appends a follow-up Task tagged `spawn-depth:` (capped at 2). The live campaign
is split in two: `docs/plans/campaign-A-consolidate.md` (Block A consolidate = supervised
foreground) and `docs/plans/campaign-B-expand.md` (Block B full = unattended via systemd, after A).

```markdown
# Plan: <campaign name>

## Overview
<one paragraph: the campaign thesis>

### Task 1: capacity-PVI depth sweep
recipe: experiment
gpu: true
surface: capacity-pvi-sweep
run_id: cap-pvi-01
gate: review refine-logs/capacity-pvi-sweep/REVIEW_STATE.json
objective: does capacity-matched class-PVI track attack recovery across depth under input-DP?
- [ ] run-phase: cap-pvi-01

### Task 2: BNN error-bound proof
recipe: theory
gpu: false
surface: bnn-bounds
run_id: bnn-proof-01
gate: proof refine-logs/bnn-bounds/PROOF_AUDIT.json
objective: prove the union-Bhattacharyya + Fano L0 error bounds as stated.
- [ ] run-phase: bnn-proof-01
```

**Dependencies:** ralphex runs Task sections top-to-bottom, so encode ordering by position
(put prerequisites first). There is no declarative `depends_on`; the wrapper skips a failed
phase and continues, and a dependent of a skipped phase will itself fail its gate and be
skipped in turn (cascading skips trip the circuit-breaker if ≥N in a row).

### Gate specs (`gate:` line, passed verbatim to `gate_check.sh`)
- `review <REVIEW_STATE.json>` — auto-review positive STOP (score≥6 ∧ verdict∈{ready,almost})
- `proof <PROOF_AUDIT.json>` — proof-checker verdict PASS
- `runstate <run_state.json> <phase>` — that phase `accepted`
- `jq <file.json> <jqpath> <expected>` — generic
- `marker <file>` — file contains "accepted" (testing)

Point the gate at a **surface-scoped** path (`refine-logs/<surface>/…`) so successive phases
don't read each other's stale verdict files. Aligning ARIS skill output paths to these is part
of authoring a real campaign (see "Deferred / to-verify").

## Running

Manual (foreground, for a first supervised run):
```bash
scripts/harness/run_campaign.sh docs/plans/<campaign>.md
```

Unattended (systemd user services):
```bash
cp scripts/harness/telegram.env.example  ~/.config/talens-harness/telegram.env   # fill in token+chat
cp scripts/harness/campaign.env.example  ~/.config/talens-harness/campaign.env   # set PLAN=...
ln -s "$PWD"/scripts/harness/systemd/talens-watchdog.service ~/.config/systemd/user/
ln -s "$PWD"/scripts/harness/systemd/talens-research.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now talens-watchdog
systemctl --user start talens-research
journalctl --user -u talens-research -f
```

## Telegram (what you must provide)

1. **Bot token + chat id** → `~/.config/talens-harness/telegram.env` (used by `notify_telegram.sh`
   for stuck/failed/complete/dead alerts). Create a bot via @BotFather; get the chat id from
   `https://api.telegram.org/bot<token>/getUpdates` after messaging it. Until set, alerts no-op
   harmlessly. (Optionally mirror the same token/chat into `ralphex-config/config`
   `notify_telegram_*` if you'd rather use ralphex's native telegram than the custom script.)
2. **Two-way MCP channel** (optional) → `/telegram:configure <token>` then `/telegram:access`.
   Lets you *monitor/steer* from your phone. **Limitation (Q5):** ralphex runs each phase as a
   one-shot headless `claude -p`, which **cannot block mid-task waiting for a reply**. So an
   "agent needs a decision" case is an **async escalation**: the phase sends a `stuck` alert,
   emits `TASK_FAILED`, and stops; you resolve it later by editing the plan and re-running. The
   two-way channel answers *future* invocations, not the in-flight one.

## Failure & escalation policy (Q8)

- transient crash → ralphex retries once (`task_retry_count=1`) + `idle_timeout`/`session_timeout` guards
- clean-but-not-accepted (gate fails after the recipe's internal rounds) → `TASK_FAILED` + alert;
  wrapper skips the phase and continues with independent phases
- **circuit-breaker**: after **2 consecutive** no-progress rounds, the wrapper halts (exit 0 →
  systemd won't relaunch). Tune via `TALENS_CB_LIMIT`.
- silent death / hung (heartbeat mtime stale > 6h) → watchdog STALE → `dead` alert (detect-only)
- stall (≥2 zero-finding inner ticks) → forced structural pivot; ≥4 → human escalation

## Validated (2026-06-22)
- ralphex's headless `claude` loads the gitignored, symlinked ARIS skills (via `/skill-name`).
- `--config-dir` picks up our `config` + `task.txt`; `--tasks-only` disables reviews; one
  Task = one fresh `claude` invocation.
- Gate seam end-to-end: gate pass → `[x]` + commit + `ALL_TASKS_DONE`; gate fail → `TASK_FAILED`,
  checkbox stays `[ ]`, ralphex aborts the run (→ wrapper skip handles it).
- Offline: all `gate_check` predicates, notify formatting, and wrapper skip/circuit-breaker (15/15).

## Deferred / to-verify (before/at first real campaign)
- **Full production `task.txt` with a real ARIS recipe** end-to-end (needs GPU + Codex MCP +
  hours) — validate on ONE small real phase before trusting an overnight run.
- **ARIS skill output paths vs gate paths**: confirm `experiment-bridge`/`auto-review-loop`/
  `proof-checker` write `REVIEW_STATE.json`/`PROOF_AUDIT.json` where the `gate:` line points
  (surface-scoped), or adjust the recipe invocation to direct them there.
- **claude OAuth under systemd user service** (HOME/keychain reachable).
- **`run_in_rocm.sh` container teardown** between serial phases (no stray GPU container).
- ARIS Feishu integration stays **off** (we use Telegram).
