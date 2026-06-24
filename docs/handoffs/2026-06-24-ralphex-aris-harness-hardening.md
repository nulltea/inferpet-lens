---
type: handoff
status: current
created: 2026-06-24
updated: 2026-06-24
tags: [ralphex, aris, harness, autonomous-research, campaign, known-issues]
companion: [2026-06-23-ralphex-aris-harness]
---

# Handoff — ralphex+ARIS auto-research: runtime hardening, known issues, follow-ups

Design + decisions are in memory `autonomous-campaign-decisions.md` and `autonomous-research-harness.md`,
and `scripts/harness/README.md`. This handoff is **what changed this session, what broke, and what's left** — not a re-derivation.

## What happened this session

The campaign actually ran. Block A consolidation + most of Block B committed real claims/experiments/HTML — see `git log` (e.g. `065738a` defenses, `4513f57` site, `cf9e7f2` kv-accumulation, `40a498f` kv-cloak, `1ae1e09` rep2text, `cbca688` depth-inversion, `2b33026` gelo, `c06a33c` split). It also **self-spawned follow-ups** (non-correlation → new `### Task` with `spawn-depth:1`) — see Tasks 9–10 + `embed-sgt-v2` in `docs/plans/campaign-B-expand.md`. Block B state: **7 `[x]`, 4 `[ ]`** (Task 8 attn-cover-invariance + spawned follow-ups). The user stopped the campaign at the Task-7 boundary; **nothing is running now**.

Most of the session was **hardening the harness** against failures found while it ran, plus an **HTML-quality overhaul**.

## Bugs found + fixes (all in `scripts/harness/` + `.claude/skills/` + `docs/html/STYLE.md`)

1. **claude-code-warp PostToolUse hook hung ~600s/tool** in headless executors. Cause: the user's `~/.bashrc` `claude()` wrapper `export`ed `WARP_*` vars, inherited by ralphex's headless claude; the hook then blocked (no TTY) until Claude Code's 600s hook timeout. **Fix:** bashrc wrapper rewritten (prefix-assign, `[ -t 1 ]`, no export); `run_campaign.sh` `unset`s `WARP_CLI_AGENT_PROTOCOL_VERSION WARP_CLIENT_VERSION TERM_PROGRAM`.
2. **Background-yield churn**: the agent backgrounded a pilot and ended its turn to "wait" → headless turn ends → ralphex starts a fresh memory-less session → restarts the phase. **Fix:** `scripts/harness/run_step.sh` (durable, resumable, idempotent, one-GPU-safe run wrapper — launch detached, attach-and-block, re-attach on re-entry); `task.txt` "DURABLE RUNS" rule routes heavy runs through it; STEP-0 "RECOVER PRIOR CONTEXT" reads `.ralphex/progress/progress-<plan>.txt` + surface dir to resume.
3. **Stop-path bug**: ralphex absorbs SIGINT and exits "normally", so `run_campaign` relaunched the next task instead of halting. **Fix (verified):** `run_campaign.sh` now backgrounds ralphex via process-substitution (`$!` = ralphex's real PID), `wait`s on it, and an INT/TERM trap sets `STOP` + kills ralphex + `pkill`s the executor; loop checks `STOP` → exit 130. `kill -INT -<pgid>` now halts deterministically.
4. **5h usage window** would fail the phase: `wait_on_limit` defaulted to 0 (fatal). **Fix:** `config` sets `wait_on_limit = 30m` + `claude_limit_patterns` (ride out the window, retry loop).
5. **session id now in the progress log**: patched ralphex (`scripts/harness/rebuild_ralphex.sh`, reversible) to print `claude session: <id>` from the stream-json init event — replaces the sidecar approach. **Re-run `rebuild_ralphex.sh` after any `go install …ralphex@VER` upgrade.**
6. **Output**: `run_campaign.sh` strips ralphex's hardcoded date prefix to time-only via streaming `sed` (no ralphex flag exists).
7. **`run_bg.sh`** added: detached campaign launcher with a one-GPU guard (refuses if any campaign/ralphex is live), logs to `refine-logs/_campaign-logs/`.
8. **`serve_docs.sh`**: tailscale-served docs at one URL, now `no-store` (no stale-CSS caching), port-based start/stop, `http|https` + arbitrary port.

### HTML quality (the user's biggest pain)
`docs/html/STYLE.md` was overhauled — **read it before generating any page**: academic register only (no colloquial/figurative, no em-dashes), **no project/process jargon** (`R5`,`T1`,`Codex`,`PARTIAL`,run-ids), standard IMRaD section titles (Introduction/Method/Results/…, Method has a **mandatory** diagram), terse subtitle, **epistemic-status labels** (Established/Supported/Preliminary/Speculative/Contested) instead of review verdicts. `css/site.css`: text is full-width (user's choice), `.colophon` capped, tables must not sit in `.diagram-frame`. New non-interactive cleanup skills **`humanize` / `proofread` / `term-audit`** (`.claude/skills/`, HTML-aware, rule-set merged from Wikipedia-AI-cleanup + Strunk&White + `writing_quality_check.md`); the `task.txt` report stage runs `/humanize → /proofread → /term-audit` before the gate.

## Known issues / gotchas

- **Config + `task.txt` load once at ralphex launch** → edits apply only on the **next ralphex invocation** (next `run_campaign` round, or a restart). The current invocation keeps the cached prompt.
- **Phase commits do `git add -A`** ("commit everything"), so unrelated working files get swept into `feat(<surface>)` commits — that's how this session's harness edits landed (mixed into research commits). History is messy; consider scoping commits.
- **Task 7 (SGT) artifacts are UNCOMMITTED** though `b-e1-sgt` is `[x]`: `scripts/defenses/sgt.py`, `scripts/eval/sgt_attack_eval.py`, `docs/html/embed-sgt.html`, `research-wiki/claims/sgt-…md`, `research-wiki/experiments/embed-sgt-…md`, `refine-logs/embed-sgt*/`. The stop caught it just before its commit landed. **Decide: commit them, or reset `b-e1-sgt`→`[ ]` and rerun.**
- **Existing HTML pages predate the STYLE overhaul** (rep2text/gelo/etc. may still have colloquial titles, `R5/T1`/Codex jargon, AI tells). They need a retro cleanup pass.
- **GPU container under detached campaign**: a Task-1 pilot logged "container died without output" and fell back to CPU. Confirm `run_in_rocm.sh` works for the GPU-heavy phases before trusting them headless.
- Currently uncommitted (besides Task-7 SGT): `run_campaign.sh` (stop-path fix), `docs/plans/campaign-B-expand.md`, `research-wiki/{index,log,graph,query_pack}`, `review-stage/*`, `src/talens/measures/spectral_channel_mi.py` (+test, pre-existing).

## Follow-up work (suggested order)

1. **Resolve Task-7 SGT** (commit or reset), then **commit `run_campaign.sh` + plan + wiki** cleanly.
2. **Resume Block B** (4 open): `scripts/harness/run_bg.sh docs/plans/campaign-B-expand.md` (fresh ralphex picks up all the fixes). Confirm a GPU phase spawns the container.
3. **Retro-clean existing HTML** to `STYLE.md`: run `/humanize /proofread /term-audit` per page + fix titles/jargon/diagrams.
4. Consider lowering `--max-iterations` in `run_campaign.sh` so config/prompt edits apply per-round without manual restarts.
5. Consider scoping phase commits (drop `git add -A`).
6. Deferred decision: interactive HTML diagrams (inline SVG+vanilla-JS vs a D3/Plotly skill) — see the web-research options discussed; `/figure-spec` covers static SVG now.
7. systemd unattended path + watchdog still un-exercised end-to-end (handoff `2026-06-23` items #2/#4). Telegram is wired (chat `344249100`).

## Skills for the next session
`experiment-bridge` · `result-to-claim` · `experiment-audit` · `auto-review-loop` (the campaign recipe, run inline by phases); `humanize` · `proofread` · `term-audit` (HTML cleanup); `figure-spec` (Method diagrams); `research-wiki` (claim/experiment nodes). Do **not** use `/claims-drafting` (patent) for research claims.
