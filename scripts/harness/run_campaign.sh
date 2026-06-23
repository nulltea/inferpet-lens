#!/usr/bin/env bash
# run_campaign.sh <plan-file> [max_iterations]
# Outer fire-control for an autonomous research campaign. Drives ralphex over a phase
# plan, applies the Q8 failure policy (skip-and-continue past a stuck phase, halt after
# N consecutive no-progress rounds), and sends Telegram on terminal states. It NEVER
# judges research quality — that stays with each phase's ARIS cross-model gate.
#
# Phase-checkbox convention (one per `### Task N:`):  - [ ] run-phase: <run_id>
#   the agent flips it to [x] only after gate_check passes; this wrapper flips it to
#   "[x] ... SKIPPED-gate-not-met" to advance past a stuck phase.
#
# Exit 0 on clean completion OR clean circuit-breaker halt (so systemd Restart=on-failure
# does NOT relaunch a deliberate stop). Non-zero only on genuine wrapper error.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"

plan="${1:?usage: run_campaign.sh <plan-file> [max_iterations]}"
max_iter="${2:-50}"
cfg="${TALENS_CFG_DIR:-scripts/harness/ralphex-config}"
cb_limit="${TALENS_CB_LIMIT:-2}"
ralphex_bin="${RALPHEX_BIN:-ralphex}"

export ARIS_REPO="${ARIS_REPO:-/home/timo/repos/Auto-claude-code-research-in-sleep}"
export PATH="$HOME/.local/bin:$PATH"

count_open() { grep -cE '^[[:space:]]*-[[:space:]]+\[ \][[:space:]]*run-phase:' "$plan" 2>/dev/null || echo 0; }
skip_first_open() {
  awk '!done && $0 ~ /^[[:space:]]*-[[:space:]]+\[ \][[:space:]]*run-phase:/ {
         sub(/\[ \]/,"[x]"); $0=$0"  SKIPPED-gate-not-met"; done=1 } {print}' \
      "$plan" > "$plan.tmp" && mv "$plan.tmp" "$plan"
}

command -v "$ralphex_bin" >/dev/null || { echo "ralphex not found: $ralphex_bin"; exit 3; }
[ -f "$plan" ] || { echo "plan not found: $plan"; exit 3; }

# Preflight once: Codex auth, GPU, ralphex, skills (so a phase never blocks mid-run).
# Skip with TALENS_SKIP_PREFLIGHT=1 (e.g. a GPU-free pilot dry-run).
if [ "${TALENS_SKIP_PREFLIGHT:-0}" != "1" ]; then
  bash "$here/preflight.sh" || { echo "[run_campaign] preflight failed — aborting"; exit 4; }
fi

consec=0
round=0
while :; do
  open_before="$(count_open)"
  if [ "$open_before" -eq 0 ]; then
    bash "$here/notify_telegram.sh" complete "campaign '$plan': all phases resolved"
    echo "[run_campaign] all phases resolved — done"; exit 0
  fi

  round=$((round+1))
  echo "[run_campaign] round $round: $open_before phase(s) open; launching ralphex"
  out="$(mktemp)"
  "$ralphex_bin" "$plan" --tasks-only --config-dir "$cfg" --max-iterations "$max_iter" 2>&1 | tee "$out"
  grep -q '<<<RALPHEX:ALL_TASKS_DONE>>>' "$out" && { rm -f "$out"; \
    bash "$here/notify_telegram.sh" complete "campaign '$plan': ALL_TASKS_DONE"; \
    echo "[run_campaign] ALL_TASKS_DONE"; exit 0; }
  rm -f "$out"

  open_after="$(count_open)"
  if [ "$open_after" -lt "$open_before" ]; then
    consec=0                      # a phase was accepted -> progress
    echo "[run_campaign] progress: $open_before -> $open_after open"
    continue
  fi

  # no phase accepted this round -> a phase failed / escalated
  consec=$((consec+1))
  echo "[run_campaign] no progress (consecutive=$consec)"
  if [ "$consec" -ge "$cb_limit" ]; then
    bash "$here/notify_telegram.sh" failed "campaign '$plan': circuit-breaker — $consec consecutive no-progress rounds, halting"
    echo "[run_campaign] circuit-breaker halt"; exit 0
  fi
  skip_first_open
  bash "$here/notify_telegram.sh" stuck "campaign '$plan': skipped a stuck phase, continuing with independent phases"
done
