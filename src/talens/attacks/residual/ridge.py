"""ISA-HiddenState · ridge (linear obs→embedding, nearest-token)."""
from __future__ import annotations

import numpy as np

from .._common import nearest_token, ridge_W


def ridge_attack(Xtr, Etr, Xte, pool_emb, pool_ids, *, alpha=1.0, **_):
    """Linear (ridge) obs→embedding map, then nearest token. The strong single-position baseline."""
    return nearest_token(Xte @ ridge_W(Xtr, Etr, alpha), pool_emb, pool_ids)
