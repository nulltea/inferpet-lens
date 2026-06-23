#!/usr/bin/env bash
# serve_docs.sh [start|stop|status] [pub_port] [http|https]
#   Expose docs/html/ at one persistent tailnet URL (live multi-file site; updates as phases
#   commit). Private to your tailnet; not a public claude.ai link (see autonomous-campaign-decisions).
#
# Ports / scheme:
#   local http.server : TALENS_DOCS_PORT (default 8099, bound to 127.0.0.1 only — internal)
#   tailnet-facing    : pub_port (2nd arg) + scheme (3rd arg or TALENS_DOCS_SCHEME; default https)
#       https → TLS (needs tailnet HTTPS/MagicDNS certs enabled); any port, default 443
#       http  → plaintext, any port, default 8080
# Examples:
#   serve_docs.sh start                 # https://<host>/         (443)
#   serve_docs.sh start 8099 http       # http://<host>:8099/
#   serve_docs.sh start 8443            # https://<host>:8443/
#   serve_docs.sh stop 8099 http        # tear that mapping down
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
docroot="$repo/docs/html"
port="${TALENS_DOCS_PORT:-8099}"
cmd="${1:-start}"
pub_port="${2:-}"
scheme="${3:-${TALENS_DOCS_SCHEME:-https}}"
[ -z "$pub_port" ] && { [ "$scheme" = http ] && pub_port=8080 || pub_port=443; }
pidf="/tmp/talens-docs-http.$port.pid"

hostname_ts() { tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || echo '<tailnet-host>'; }

case "$cmd" in
  start)
    command -v tailscale >/dev/null || { echo "tailscale not found"; exit 3; }
    [ -d "$docroot" ] || { echo "docroot missing: $docroot"; exit 3; }
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
      echo "http server already running (pid $(cat "$pidf")) on 127.0.0.1:$port"
    else
      ( cd "$docroot" && nohup python3 -m http.server "$port" --bind 127.0.0.1 >/tmp/talens-docs-http.$port.log 2>&1 & echo $! > "$pidf" )
      sleep 1; echo "http server on 127.0.0.1:$port (pid $(cat "$pidf")) serving $docroot"
    fi
    if ! tailscale serve --bg "--$scheme=$pub_port" "$port"; then
      echo "ERROR: 'tailscale serve --bg --$scheme=$pub_port $port' failed." >&2; exit 4
    fi
    if [ "$scheme" = https ] && [ "$pub_port" = 443 ]; then sfx=""; else sfx=":$pub_port"; fi
    echo "serving at: $scheme://$(hostname_ts)$sfx/   (open /index.html)"
    ;;
  stop)
    tailscale serve "--$scheme=$pub_port" off 2>/dev/null || tailscale serve reset 2>/dev/null || true
    [ -f "$pidf" ] && kill "$(cat "$pidf")" 2>/dev/null && rm -f "$pidf" && echo "stopped ($scheme=$pub_port + http server)" || echo "no http server pid (cleared $scheme=$pub_port)"
    ;;
  status)
    tailscale serve status 2>/dev/null || true
    [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null && echo "http server up (pid $(cat "$pidf")) on 127.0.0.1:$port" || echo "http server down"
    ;;
  *) echo "usage: serve_docs.sh [start|stop|status] [pub_port] [http|https]"; exit 2 ;;
esac
