#!/usr/bin/env bash
# run_step.sh <name> -- <cmd...>
# Persistent, resumable, idempotent heavy-run wrapper for the autonomous harness. Decouples a long
# run from the ephemeral one-shot ralphex `claude` session: the run is launched DETACHED (survives
# session death) and is NEVER duplicated, so a fresh session re-attaches instead of relaunching
# (one-GPU-safe). The wrapper ALWAYS blocks until the run finishes, streaming its log — which also
# keeps ralphex's idle-timer alive — and returns the run's real exit code.
#
# State dir: refine-logs/$TALENS_SURFACE/runs/<name>/  (TALENS_SURFACE defaults to 'misc')
#   cmd  run.log  run.pid (pgid)  run.exit (present == finished)  run.lock
#
# Re-entry semantics:
#   finished -> print tail, return cached exit code
#   active   -> attach + block on run.log until run.exit appears, return it
#   absent   -> launch detached, then attach as above
#
# Usage in a phase:  export TALENS_SURFACE=<surface>
#                    scripts/harness/run_step.sh kv-pilot -- scripts/run_in_rocm.sh python3 ...
set -uo pipefail
name="${1:?usage: run_step.sh <name> -- <cmd...>}"; shift
[ "${1:-}" = "--" ] && shift
[ "$#" -ge 1 ] || { echo "run_step: no command given" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"; cd "$repo"
surface="${TALENS_SURFACE:-misc}"
dir="refine-logs/$surface/runs/$name"; mkdir -p "$dir"
log="$dir/run.log"; pidf="$dir/run.pid"; exitf="$dir/run.exit"; lockf="$dir/run.lock"

attach() {
  echo "[run_step] attach '$name' (surface=$surface) — streaming $log"
  tail -n +1 -f "$log" 2>/dev/null & local tp=$!
  local p; p="$(cat "$pidf" 2>/dev/null || true)"
  while [ ! -f "$exitf" ]; do
    sleep 5
    if [ -n "${p:-}" ] && ! kill -0 "$p" 2>/dev/null; then
      sleep 3                                   # grace for the exit marker to land
      [ -f "$exitf" ] || { echo 137 > "$exitf"; echo "[run_step] '$name' vanished without exit marker -> 137" >&2; }
      break
    fi
  done
  kill "$tp" 2>/dev/null || true
  local rc; rc="$(cat "$exitf" 2>/dev/null || echo 1)"
  echo "[run_step] '$name' done, exit=$rc"
  return "$rc"
}

# fast path: already finished
if [ -f "$exitf" ]; then echo "[run_step] '$name' already complete (exit $(cat "$exitf"))"; tail -8 "$log" 2>/dev/null; exit "$(cat "$exitf")"; fi

# serialize the launch decision so two sessions can't both start it
exec 9>"$lockf"; flock 9
if [ -f "$exitf" ]; then flock -u 9; exit "$(cat "$exitf")"; fi
if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
  echo "[run_step] '$name' already running (pgid $(cat "$pidf")) — attaching"
else
  : > "$log"; rm -f "$exitf"; printf '%q ' "$@" > "$dir/cmd"
  RS_LOG="$log" RS_EXIT="$exitf" setsid bash -c '"$@" >>"$RS_LOG" 2>&1; echo $? >"$RS_EXIT"' _ "$@" </dev/null &
  echo $! > "$pidf"
  echo "[run_step] launched '$name' detached (pgid $(cat "$pidf"))"
fi
flock -u 9
attach
