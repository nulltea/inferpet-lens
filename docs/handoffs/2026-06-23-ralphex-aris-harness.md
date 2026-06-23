---
type: handoff
status: current
created: 2026-06-23
updated: 2026-06-23
tags: [autonomous-research, ralphex, aris, harness, orchestration]
companion: []
---

# Handoff — ralphex + ARIS autonomous-research harness

Goal: one-time multiphase planning, then run research **unattended** — ralphex (outer
fire-control) drives phases, ARIS skills (inner verdict loops) do the work, Telegram alerts
on stuck/done. Full design + validated facts live in project memory
`autonomous-research-harness.md`; operational detail in `scripts/harness/README.md`. This
handoff is **state + what's left**, not a re-derivation.

## Decisions (locked, via /grill-me — 12 forks)
Recorded in memory `autonomous-research-harness.md`. Headlines: ralphex outer / ARIS inner;
one phase = one experimentation surface; **fully serial** (no worktrees — ARIS skills are
gitignored symlinks that vanish in a worktree; single iGPU forbids concurrent GPU anyway);
gate = ARIS state-file `accepted`; ralphex reviews **off** (`--tasks-only`); skip-and-continue
+ circuit-breaker at 2 consecutive failures; Telegram = curl (out-of-band) + optional two-way
MCP; autonomy = self-resolve routine / async-escalate load-bearing; persistence = systemd user
service `Restart=on-failure` (clean stops exit 0). Doctrine spine: ARIS `external-cadence.md`
— outer cadence may DRIVE, never ACQUIT.

## Done (built + tested 2026-06-22)
- **ralphex v1.5.1 installed** at `~/.local/bin/ralphex` via `go install` (Go 1.26.4 user-local
  at `~/.local/go`).
- **Integration validated end-to-end** (two throwaway worktrees, since torn down):
  - ralphex's headless `claude` loads the gitignored, symlinked ARIS skills — only via
    `/skill-name` (skills are NOT auto-enumerated; never pass `--disable-slash-commands`).
  - `--config-dir` picks up our tracked `config` + `task.txt`; `--tasks-only` disables reviews;
    one `### Task N:` = one fresh `claude` invocation.
  - Gate seam: gate pass → `[x]` + commit + `ALL_TASKS_DONE`; gate fail → `TASK_FAILED`,
    checkbox stays `[ ]`. **ralphex aborts the run on `TASK_FAILED` after its one retry (exit 1)**
    → confirms skip-and-continue must live in the wrapper.
  - Offline 15/15: all `gate_check` predicates, notify formatting, wrapper skip + circuit-breaker.
- **Harness written** (all under `scripts/harness/`, tracked but **uncommitted**): see
  `scripts/harness/README.md` for the file map, plan-format schema, gate specs, run commands.
  `bash -n` clean on all scripts.

## Left to do (in priority order)
1. **USER provides Telegram bot token + chat id** → `~/.config/talens-harness/telegram.env`
   (template at `scripts/harness/telegram.env.example`). Until then alerts no-op. Optionally
   `/telegram:configure <token>` for the two-way MCP channel.
2. **Validate the production `task.txt` on ONE small REAL phase** (needs GPU + Codex MCP + time —
   not done; only trivial GPU-free runs were tested). This is the last real risk before trusting
   an overnight run. Watch: ARIS skills writing `REVIEW_STATE.json`/`PROOF_AUDIT.json` to the
   **surface-scoped** path the phase's `gate:` line points at (may need to direct skill outputs
   into `refine-logs/<surface>/`).
3. **Author the one-time campaign plan** (ralphex-native markdown; enumerate surfaces, order,
   gates). Candidate surfaces from current work: capacity-PVI sweep, BNN error-bound proof,
   DP-stronger-attacks, spectral-channel-MI probe, vec2text. Use the schema in the README.
4. **Confirm under systemd**: claude OAuth reachable in a `--user` service; `run_in_rocm.sh`
   container teardown between serial phases (honor `one-gpu-process-at-a-time`).
5. Decide whether to **commit** `scripts/harness/` (currently uncommitted on branch
   `capacity-matched-pvi`). ARIS Feishu integration stays **off** (we use Telegram).

## Gotchas for the next agent
- Bare `ralphex` is **not** a dry-run — it auto-discovers a plan in `docs/plans/` and starts a
  loop. Always pass an explicit plan + `--config-dir scripts/harness/ralphex-config --tasks-only`
  (or just use `scripts/harness/run_campaign.sh <plan>`).
- ralphex's branch diff-stat ("N files, +M lines") is **vs `main`** (whole-branch divergence),
  not the agent's per-task work — don't be alarmed.
- **Q5 limitation**: each phase is a one-shot headless `claude -p` and CANNOT block for a Telegram
  reply. "Agent needs a decision" = async (alert + `TASK_FAILED` + stop; human edits plan + reruns).
- Test only in **worktrees** with a recreated `.claude/skills` symlink + copied `scripts/harness/`
  (the harness is uncommitted, so a clean worktree won't have it). Tear down with
  `git worktree remove --force` + `git branch -D`.
- A leftover gitignored `.ralphex/` (config/progress) sits in the main repo from an earlier run —
  harmless. The unrelated worktree `.claude/worktrees/jaunty-roaming-pond` is pre-existing; leave it.

## Suggested skills for next session
- `experiment-plan` / `experiment-bridge`, `proof-writer` / `proof-checker`, `auto-review-loop`,
  `result-to-claim` — the inner recipe being orchestrated; exercise one on the real-phase test (#2).
- `telegram:configure` / `telegram:access` — wire the two-way channel (#1).
- `grill-me` — if authoring the campaign plan (#3) needs the surfaces/ordering stress-tested.
