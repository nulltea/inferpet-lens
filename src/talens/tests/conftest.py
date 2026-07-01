"""Make the ``scripts/`` directory importable so model-free tests can reach the
defense-eval package (``defenses.aloepri`` / ``defenses.shredder``). Defenses
live outside ``src/talens`` core by the scheme-agnostic rule; tests are an
allowed home for them per ``CLAUDE.md``."""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"  # src/talens/tests/ → repo root
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
