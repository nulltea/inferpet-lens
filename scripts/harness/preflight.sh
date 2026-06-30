#!/usr/bin/env bash
# preflight.sh — verify everything an unattended campaign needs BEFORE the first phase,
# so a phase never blocks mid-run on a missing dependency (esp. Codex MCP auth, which the
# ARIS verdict skills call and which cannot be authenticated headless). Run once by
# run_campaign.sh; alerts via Telegram and exits non-zero on any failure.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"; cd "$repo"
export PATH="$HOME/.local/bin:$PATH"
fail=0
note() { printf '[preflight] %s\n' "$*"; }
bad()  { printf '[preflight] FAIL: %s\n' "$*" >&2; fail=1; }

# 1. Codex reviewer backend MUST be authenticated (the verdict skills' cross-model jury).
if command -v codex >/dev/null; then
  if codex doctor >/tmp/codex_doctor.$$ 2>&1 && [ -f "$HOME/.codex/auth.json" ]; then
    note "codex: authenticated (auth.json present, doctor OK)"
  else
    bad "codex not healthy/authenticated — run 'codex login' (verdict skills will block). doctor: $(tail -3 /tmp/codex_doctor.$$ 2>/dev/null)"
  fi
  rm -f /tmp/codex_doctor.$$
else
  bad "codex CLI missing on PATH"
fi

# 2. ralphex outer loop present.
command -v ralphex >/dev/null && note "ralphex: $(ralphex --version 2>/dev/null || echo present)" || bad "ralphex not on PATH"

# 3. Host venv sees the GPU (the campaign is GPU-bound; honor one-GPU-process-at-a-time).
#    GPU torch now runs directly in the host .venv (shared gfx1151 build; see CLAUDE.md).
if .venv/bin/python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  note "rocm: .venv torch.cuda.is_available() == True"
else
  bad ".venv torch cannot see the GPU (CPU-only torch? re-point the shared-torch .pth — see CLAUDE.md)"
fi

# 4. ARIS skills + shared-references reachable (verdict skills resolve via /name; cadence ref).
[ -e .claude/skills/auto-review-loop ] && note "aris skills: present" || bad "ARIS skills missing (.claude/skills/) — run install_aris.sh"
[ -f .aris/shared-references/external-cadence.md ] && note "shared-references: present" || bad ".aris/shared-references missing"

# 5. Telegram (non-fatal — alerts just no-op if unset).
if [ -f "$HOME/.config/talens-harness/telegram.env" ]; then note "telegram: configured"; else note "telegram: NOT configured (alerts will no-op)"; fi

if [ "$fail" -ne 0 ]; then
  bash "$here/notify_telegram.sh" failed "preflight FAILED — campaign not started (see journal)" 2>/dev/null || true
  note "PREFLIGHT FAILED — fix the above before starting the campaign."; exit 1
fi
note "preflight OK"; exit 0
