"""ISA-AttnValue · known-plaintext rotation-recovery via orthogonal Procrustes (kqv_out)."""
from __future__ import annotations

import numpy as np

from .._common import nearest_token, ridge_W, orthogonal_procrustes_R


def rotation_recovery_attack(Xp_align, Xd_align, Xp_tr, ytr, Xd_te, pool_emb, pool_ids, *,
                             table, alpha=1.0, R_estimator=orthogonal_procrustes_R, **_):
    """Known-plaintext rotation-recovery inversion (claim:aloepri-kqvout-basis-alignment). A KNOWN attack
    (orthogonal-Procrustes known-plaintext recovery of a linear/orthogonal obfuscation — cf. cross-lingual
    embedding alignment MUSE/VecMap, Hill-cipher KPA, and the orthogonal-obfuscation LLM-inference line
    arXiv:2606.16461 / 2603.01499), instantiated on AloePri Alg2's per-head value rotation of kqv_out.

    AloePri Alg2 rotates kqv_out by a SECRET but context-independent (block-)orthogonal map R
    (deployment = plaintext·R). A keyless self-generated inverter (§05) collapses under Alg2 because it
    cannot reproduce R. This attack spends a harvest on recovering R ONLY — not on the token map:

      1. estimate R̂ from aligned (plaintext, deployment) anchor pairs (`Xp_align`, `Xd_align`) via
         `R_estimator` (orthogonal_procrustes_R global, or blockwise_procrustes_R per-head);
      2. un-rotate the deployment reps into the plaintext basis (`Xd_te · R̂ᵀ`);
      3. decode with a self-generated ridge fit on the attacker's own plaintext reps (`Xp_tr → emb[ytr]`).

    Keyless: R̂ from the (threat-model-legitimate) anchors, token map from unlimited self-generation."""
    R = R_estimator(Xp_align, Xd_align)                      # Xp·R ≈ Xd
    W = ridge_W(Xp_tr, table[ytr], alpha)                    # self-gen inverter, plaintext basis
    return nearest_token((Xd_te @ R.T) @ W, pool_emb, pool_ids)
