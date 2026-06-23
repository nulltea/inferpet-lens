#!/usr/bin/env bash
# Rebuild ralphex with a local patch that writes the claude session id into the progress log
# (.ralphex/progress/*.txt), so each phase's session is recorded by ralphex itself — no agent
# prompt echo, no sidecar log. Re-run this after any `go install github.com/umputun/ralphex@VER`
# upgrade (a fresh install reverts the patch). Idempotent.
#
# Patch: pkg/executor/executor.go — add a SessionID field to streamEvent, and on the first
# stream-json event carrying session_id (the `init` event) push one "claude session: <id>" line
# through the existing OutputHandler (which the progress logger consumes).
set -euo pipefail
VER="${RALPHEX_VER:-v1.5.1}"
export PATH="$HOME/.local/go/bin:$HOME/go/bin:$PATH"
export GOPATH="${GOPATH:-$HOME/go}"
command -v go >/dev/null || { echo "go not found (expected ~/.local/go/bin/go)"; exit 1; }

mod="$GOPATH/pkg/mod/github.com/umputun/ralphex@$VER"
work="$HOME/.local/src/ralphex-$VER-talens"
[ -d "$mod" ] || { echo "module not cached: $mod — run: go install github.com/umputun/ralphex@$VER"; exit 1; }

mkdir -p "$(dirname "$work")"
rm -rf "$work"; cp -r "$mod" "$work"; chmod -R u+w "$work"
f="$work/pkg/executor/executor.go"

# 1) SessionID field as the first member of streamEvent
perl -0pi -e 's/(type streamEvent struct \{\n)/${1}\tSessionID string `json:"session_id"`\n/' "$f"
# 2) declare the once-only flag just before the read loop's recentBlocks
perl -0pi -e 's/(\tvar recentBlocks \[recentBlockCount\]string\n)/\tvar sessionLogged bool\n${1}/' "$f"
# 3) emit the session id once, right before extractText, via output + OutputHandler
perl -0pi -e 's/(\t\ttext := e\.extractText\(&event\)\n)/\t\tif event.SessionID != "" \&\& !sessionLogged {\n\t\t\tsessionLogged = true\n\t\t\toutput.WriteString("claude session: " + event.SessionID + "\\n")\n\t\t\tif e.OutputHandler != nil {\n\t\t\t\te.OutputHandler("claude session: " + event.SessionID + "\\n")\n\t\t\t}\n\t\t}\n${1}/' "$f"

# verify all three edits landed before building
grep -q 'SessionID string `json:"session_id"`' "$f" \
  && grep -q 'var sessionLogged bool' "$f" \
  && grep -q 'sessionLogged = true' "$f" \
  || { echo "PATCH FAILED to apply (upstream source may have drifted) — inspect $f"; exit 2; }

cd "$work"
go build -ldflags "-X main.revision=$VER+talens-sessionid" -o "$HOME/.local/bin/ralphex" ./cmd/ralphex
echo "OK: rebuilt ralphex $VER with session-id-in-progress patch -> $HOME/.local/bin/ralphex"
"$HOME/.local/bin/ralphex" --version 2>/dev/null || true
