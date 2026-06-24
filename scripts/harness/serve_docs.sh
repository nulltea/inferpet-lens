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
# pid of the LOCAL (127.0.0.1) http server on $port — robust kill/check without pidf tracking.
# (Do NOT match tailscaled's own :port proxy listeners on the tailnet address.)
port_pid() { ss -ltnp 2>/dev/null | grep -E "127\.0\.0\.1:$port " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2; }

case "$cmd" in
  start)
    command -v tailscale >/dev/null || { echo "tailscale not found"; exit 3; }
    [ -d "$docroot" ] || { echo "docroot missing: $docroot"; exit 3; }
    if [ -n "$(port_pid)" ]; then
      echo "http server already running on 127.0.0.1:$port (pid $(port_pid))"
    else
      ( nohup python3 -c "import http.server,socketserver,functools
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-store, max-age=0'); super().end_headers()
socketserver.TCPServer.allow_reuse_address=True
# Bind the document root ABSOLUTELY (not via cwd): SimpleHTTPRequestHandler falls back to
# os.getcwd() per request when directory is None, which throws FileNotFoundError (-> empty
# reply -> tailscale 502) if the process's cwd inode is later removed/replaced (e.g. a git
# checkout or dir rewrite under it).
handler=functools.partial(H, directory='$docroot')
with socketserver.TCPServer(('127.0.0.1',$port),handler) as s: s.serve_forever()" >/tmp/talens-docs-http.$port.log 2>&1 & )
      sleep 1; echo "http server (no-store) on 127.0.0.1:$port (pid $(port_pid)) serving $docroot"
    fi
    if ! tailscale serve --bg "--$scheme=$pub_port" "$port"; then
      echo "ERROR: 'tailscale serve --bg --$scheme=$pub_port $port' failed." >&2; exit 4
    fi
    if [ "$scheme" = https ] && [ "$pub_port" = 443 ]; then sfx=""; else sfx=":$pub_port"; fi
    echo "serving at: $scheme://$(hostname_ts)$sfx/   (open /index.html)"
    ;;
  stop)
    # Refuse to guess the port. 'stop' with no arg used to default to 443; the
    # off-then-`reset` fallback below then wiped ALL tailnet Service serves
    # (postmortem 2026-06-23). Require the exact published port.
    [ -z "${2:-}" ] && { echo "refuse: 'stop' requires an explicit port, e.g. 'serve_docs.sh stop 8099' (or 'stop 443 https'). Not guessing — guessing 443 previously wiped every Service serve." >&2; exit 2; }
    # Targeted teardown ONLY. Never fall back to `tailscale serve reset`: reset
    # clears all node AND service serve config, not just this docs mapping.
    if ! tailscale serve "--$scheme=$pub_port" off; then
      echo "WARNING: 'tailscale serve --$scheme=$pub_port off' failed (nothing to turn off, or an error). Leaving all other serve config untouched." >&2
    fi
    p="$(port_pid)"; if [ -n "$p" ]; then kill "$p" 2>/dev/null; echo "stopped local http server (pid $p) + cleared $scheme=$pub_port"; else echo "no local http server on $port (cleared $scheme=$pub_port)"; fi
    rm -f "$pidf" 2>/dev/null
    ;;
  status)
    tailscale serve status 2>/dev/null || true
    p="$(port_pid)"; [ -n "$p" ] && echo "http server up (pid $p) on 127.0.0.1:$port" || echo "http server down"
    ;;
  *) echo "usage: serve_docs.sh [start|stop|status] [pub_port] [http|https]"; exit 2 ;;
esac
