#!/usr/bin/env bash
# notify_telegram.sh <severity> <message...>
# Out-of-band Telegram alert via the Bot API. Works even when a task session has died
# (it's a plain curl, not the in-session MCP tool). Best-effort and NEVER load-bearing:
# if the token/chat is unset, it no-ops with exit 0 (doctrine: cadence/notify degrades gracefully).
#
# Reads credentials from $TALENS_TELEGRAM_ENV (default ~/.config/talens-harness/telegram.env):
#   TELEGRAM_BOT_TOKEN=123456:ABC...
#   TELEGRAM_CHAT_ID=123456789
# Set DRY_RUN=1 to print instead of send (used by the unit test).
set -uo pipefail

sev="${1:-info}"; shift || true
msg="$*"

env_file="${TALENS_TELEGRAM_ENV:-$HOME/.config/talens-harness/telegram.env}"
[ -f "$env_file" ] && . "$env_file"

case "$sev" in
  stuck)    icon="🟠 STUCK" ;;
  failed)   icon="🔴 FAILED" ;;
  complete) icon="🟣 COMPLETE" ;;
  dead)     icon="💀 DEAD" ;;
  ok)       icon="🟢 OK" ;;
  *)        icon="ℹ️ INFO" ;;
esac
host="$(hostname 2>/dev/null || echo host)"
text="${icon} [talens@${host}] ${msg}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[notify_telegram DRY_RUN] $text"
  exit 0
fi

if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] || [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
  echo "[notify_telegram] no token/chat configured ($env_file) — skipping: $text" >&2
  exit 0
fi

curl -sS --max-time 15 \
  "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
  --data-urlencode "text=${text}" \
  >/dev/null 2>&1 || echo "[notify_telegram] send failed (best-effort): $text" >&2
exit 0
