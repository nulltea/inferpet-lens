#!/usr/bin/env bash
# run_bg.sh <plan-file> — launch a campaign DETACHED (survives terminal close), logging to a file.
# Thin wrapper over run_campaign.sh (which does preflight, the ralphex loop, skip-and-continue,
# circuit-breaker, Telegram, and the clean-stop SIGINT trap). Use for the UNATTENDED Block B; for
# the SUPERVISED Block A, run run_campaign.sh in the foreground so you can watch it.
#
#   scripts/harness/run_bg.sh docs/plans/campaign-B-expand.md
#
# setsid makes the campaign its own process group, so the stop command can signal the whole tree
# (run_campaign + ralphex + claude) at once with a clean SIGINT.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"; cd "$repo"

plan="${1:?usage: run_bg.sh <plan-file>}"
[ -f "$plan" ] || { echo "plan not found: $plan"; exit 3; }

base="$(basename "$plan" .md)"
logdir="$repo/refine-logs/_campaign-logs"; mkdir -p "$logdir"
log="$logdir/$base-$(date +%Y%m%d-%H%M%S).log"
pidf="$logdir/$base.pid"

if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
  echo "campaign '$base' already running (pgid $(cat "$pidf")). Stop it first:"
  echo "    kill -INT -$(cat "$pidf")"
  exit 1
fi

# Hard one-GPU-process-at-a-time guard: refuse to launch if ANY campaign/ralphex is already live
# (a second campaign would spawn a concurrent GPU executor and collide).
other="$(pgrep -af 'run_campaign\.sh|ralphex ' | grep -v "$$" | grep -v 'pgrep' || true)"
if [ -n "$other" ]; then
  echo "REFUSING: another campaign/ralphex is already running (one GPU process at a time):" >&2
  echo "$other" | sed 's/^/    /' >&2
  echo "  stop it first (clean): kill -INT -<its-pgid>   or   systemctl --user stop talens-research" >&2
  exit 5
fi

open="$(grep -cE '^[[:space:]]*-[[:space:]]+\[ \][[:space:]]*run-phase:' "$plan" 2>/dev/null || echo 0)"
echo "launching '$base' detached — $open open phase(s)"

setsid bash "$here/run_campaign.sh" "$plan" >"$log" 2>&1 </dev/null &
pgid=$!
echo "$pgid" > "$pidf"
sleep 1
echo "  pid/pgid: $pgid"
echo "  log:      tail -f $log"
echo "  status:   grep -E 'round|PHASE (START|END)|progress|halt|DONE|FAILED' $log | tail -20"
echo "  stop:     kill -INT -$pgid     # clean stop; in-progress phase stays [ ], no SKIP"
