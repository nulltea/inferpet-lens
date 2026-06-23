#!/usr/bin/env bash
# serve_docs.sh [start|stop|status]  — expose docs/html/ at one persistent tailnet URL.
# Live multi-file research site (all surface pages, shared css/site.css), updates as phases
# commit. Private to your tailnet; not a public claude.ai link (by design — see
# autonomous-campaign-decisions).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
docroot="$repo/docs/html"
port="${TALENS_DOCS_PORT:-8099}"
pidf="/tmp/talens-docs-http.$port.pid"
cmd="${1:-start}"

case "$cmd" in
  start)
    command -v tailscale >/dev/null || { echo "tailscale not found"; exit 3; }
    [ -d "$docroot" ] || { echo "docroot missing: $docroot"; exit 3; }
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
      echo "http server already running (pid $(cat "$pidf"))"
    else
      ( cd "$docroot" && nohup python3 -m http.server "$port" --bind 127.0.0.1 >/tmp/talens-docs-http.$port.log 2>&1 & echo $! > "$pidf" )
      sleep 1; echo "http server on 127.0.0.1:$port (pid $(cat "$pidf")) serving $docroot"
    fi
    tailscale serve --bg "$port" >/dev/null 2>&1 || tailscale serve --bg "http://127.0.0.1:$port" >/dev/null 2>&1 || true
    url="https://$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || echo '<tailnet-host>')"
    echo "serving at: $url   (docs/html/ — open /index.html)"
    ;;
  stop)
    tailscale serve --https="$port" off 2>/dev/null || tailscale serve reset 2>/dev/null || true
    [ -f "$pidf" ] && kill "$(cat "$pidf")" 2>/dev/null && rm -f "$pidf" && echo "stopped" || echo "no http server pid"
    ;;
  status)
    tailscale serve status 2>/dev/null || true
    [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null && echo "http server up (pid $(cat "$pidf"))" || echo "http server down"
    ;;
  *) echo "usage: serve_docs.sh [start|stop|status]"; exit 2 ;;
esac
