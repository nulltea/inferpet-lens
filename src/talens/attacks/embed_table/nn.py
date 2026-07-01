"""IMA/NN · training-free cosine nearest-neighbour attack."""
from __future__ import annotations

import numpy as np
import torch.nn.functional as F

from .._common import nearest_token


def nn_attack(Xtr, Etr, Xte, pool_emb, pool_ids, **_):
    """NN — Nearest-Neighbour attack (AloePri paper §F.1 / Table 1). TRAINING-FREE: cosine-match each
    observed hidden-state row directly to the nearest candidate token-embedding row. Recovers at L0 /
    plaintext (residual ≈ embedding) but needs the observation to live in the embedding space — under
    AloePri the released residual is in the secret P̂-basis (and a different width), so a cross-space
    match is undefined and recovery is chance (paper reports AloePri NN = 0%). Xtr/Etr are unused."""
    if Xte.shape[1] != pool_emb.shape[1]:        # obf basis (d+2h) ≠ embedding space → cannot match
        return np.full(Xte.shape[0], pool_ids[0], dtype=pool_ids.dtype)   # degenerate ⇒ ~chance
    return nearest_token(Xte, pool_emb, pool_ids)
