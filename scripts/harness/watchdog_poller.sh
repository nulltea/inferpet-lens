#!/usr/bin/env bash
# watchdog_poller.sh [interval_seconds]
# Runs the ARIS watchdog daemon (detects a silently-dead/hung phase via heartbeat-file
# mtime) and forwards its alerts to Telegram. DETECT-ONLY: never restarts anything
# (recovery is a human/systemd decision, per external-cadence.md). Meant to run as a
# long-lived systemd service from the repo root.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
interval="${1:-300}"
base="${ARIS_WATCHDOG_BASE:-/tmp/aris-watchdog}"

wd=".aris/tools/watchdog.py"
[ -f "$wd" ] || wd="${ARIS_REPO:-/home/timo/repos/Auto-claude-code-research-in-sleep}/tools/watchdog.py"

# start the daemon if not already running (it writes $base/{summary.txt,alerts.log,status/})
if ! pgrep -f "watchdog.py --interval" >/dev/null 2>&1; then
  python3 "$wd" --interval "$interval" &
  echo "[watchdog_poller] started daemon (interval=${interval}s, base=$base)"
fi

alerts="$base/alerts.log"
for _ in $(seq 1 30); do [ -f "$alerts" ] && break; sleep 1; done
mkdir -p "$base"; : > "${alerts}.poller_seen" 2>/dev/null || true

# forward STALE/DEAD/MISSING/STALLED alert lines (each is "[ts] task: STATUS — msg")
last=""
tail -n0 -F "$alerts" 2>/dev/null | while IFS= read -r line; do
  case "$line" in
    *STALE*|*DEAD*|*MISSING*|*STALLED*)
      [ "$line" = "$last" ] && continue
      last="$line"
      bash "$here/notify_telegram.sh" dead "watchdog: ${line}"
      ;;
  esac
done
