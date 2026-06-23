#!/usr/bin/env bash
# ralphex_notify.sh — ralphex notify_custom_script.
# ralphex pipes a Result JSON on stdin (exit 0 = success). We parse it best-effort and
# forward a one-line summary to Telegram via notify_telegram.sh. Schema is parsed
# defensively (fields vary by ralphex version); unknown shapes fall back to a raw snippet.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
payload="$(cat)"

read -r sev summary < <(
  printf '%s' "$payload" | python3 - <<'PY'
import json,sys
raw=sys.stdin.read()
try: d=json.loads(raw)
except Exception:
    print("info "+(raw[:160].replace("\n"," ") or "ralphex event")); sys.exit()
def g(*ks):
    for k in ks:
        if isinstance(d,dict) and d.get(k) not in (None,""): return d[k]
    return None
success=g("success","ok","completed")
err=g("error","failure","reason")
plan=g("plan","plan_file","plan_name") or "campaign"
stats=g("summary","message","status") or ""
if err and not success: sev="failed"; msg=f"{plan}: {err} {stats}".strip()
elif success is True or str(success).lower()=="true": sev="complete"; msg=f"{plan}: {stats or 'all tasks done'}".strip()
else: sev="info"; msg=f"{plan}: {stats or 'ralphex event'}".strip()
print(sev+" "+msg.replace("\n"," ")[:300])
PY
)

bash "$here/notify_telegram.sh" "${sev:-info}" "${summary:-ralphex event}"
exit 0
