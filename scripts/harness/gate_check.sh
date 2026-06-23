#!/usr/bin/env bash
# gate_check.sh <kind> <file> [extra...]
# Exit 0 iff the ARIS cross-model verdict file records ACCEPTANCE for this phase.
# This is the seam that keeps ralphex as fire-control: the agent may mark a phase done
# ONLY when this returns 0, and the verdict here was written by ARIS's cross-model jury,
# NOT by the executing agent (anti-self-acquittal, per external-cadence.md / acceptance-gate.md).
#
# Supported kinds (the plan's `gate:` line is passed verbatim as these args):
#   review   <REVIEW_STATE.json>           -> last_score>=6 AND last_verdict in {ready,almost}
#   proof    <PROOF_AUDIT.json>            -> verdict == PASS
#   runstate <run_state.json> <phase>      -> phases[phase].status == accepted
#   jq       <file.json> <jqpath> <expect> -> generic equality check
#   marker   <file>                        -> file exists and contains "accepted" (unit-test only)
set -uo pipefail

kind="${1:-}"; file="${2:-}"
if [ -z "$kind" ] || [ -z "$file" ]; then echo "gate_check: usage: <kind> <file> [extra]" >&2; exit 2; fi
if [ ! -f "$file" ]; then echo "gate_check: NOT-ACCEPTED ($kind: file absent: $file)" >&2; exit 1; fi

case "$kind" in
  review)
    python3 - "$file" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception as e: print("review: unreadable:",e,file=sys.stderr); sys.exit(1)
score=float(d.get("last_score",d.get("score",0)) or 0)
verdict=str(d.get("last_verdict",d.get("verdict",""))).strip().lower()
ok = score>=6 and verdict in ("ready","almost")
print(f"review: score={score} verdict={verdict} -> {'ACCEPTED' if ok else 'NOT-ACCEPTED'}",file=sys.stderr)
sys.exit(0 if ok else 1)
PY
    ;;
  proof)
    python3 - "$file" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception as e: print("proof: unreadable:",e,file=sys.stderr); sys.exit(1)
v=str(d.get("verdict","")).strip().upper()
ok = v=="PASS"
print(f"proof: verdict={v} -> {'ACCEPTED' if ok else 'NOT-ACCEPTED'}",file=sys.stderr)
sys.exit(0 if ok else 1)
PY
    ;;
  runstate)
    phase="${3:-}"; [ -z "$phase" ] && { echo "gate_check runstate: need <phase>" >&2; exit 2; }
    python3 - "$file" "$phase" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); want=sys.argv[2]
st=next((p.get("status") for p in d.get("phases",[]) if p.get("phase")==want),None)
ok = st=="accepted"
print(f"runstate[{want}]: status={st} -> {'ACCEPTED' if ok else 'NOT-ACCEPTED'}",file=sys.stderr)
sys.exit(0 if ok else 1)
PY
    ;;
  jq)
    expr="${3:-}"; expect="${4:-}"
    got="$(jq -r "$expr" "$file" 2>/dev/null)"
    if [ "$got" = "$expect" ]; then echo "jq: $expr=$got -> ACCEPTED" >&2; exit 0; fi
    echo "jq: $expr=$got (want $expect) -> NOT-ACCEPTED" >&2; exit 1
    ;;
  marker)
    if grep -q "accepted" "$file" 2>/dev/null; then echo "marker: ACCEPTED" >&2; exit 0; fi
    echo "marker: NOT-ACCEPTED" >&2; exit 1
    ;;
  *)
    echo "gate_check: unknown kind '$kind'" >&2; exit 2 ;;
esac
