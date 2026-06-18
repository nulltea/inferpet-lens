"""VMA — Vocabulary-Matching Attack (τ-recovery) on a weight pair.

Recovers the secret token permutation Π by matching each **obfuscated**
row's signature to the nearest **plaintext** row signature (AloePri §F.1 /
Table 8; RowSort + sorted-quantile features). Reported metric is the
per-row **permutation-recovery rate** — the fraction of obfuscated rows
mapped back to their correct plaintext token — placed in the ``ttrsr_top1``
slot so it flows through the existing calibration unchanged.

Two matching modes:

* ``"nn"`` — independent 1-NN: each obfuscated row → its single nearest
  plaintext row (rows may collide on the same plaintext token).
* ``"hungarian"`` — a global one-to-one assignment maximising total
  similarity (a permutation is one-to-one, so this is the faithful
  objective; usually higher recovery).

``top10`` is always reported from the NN ranking, regardless of mode.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..metrics import AttackResult, classify_risk_level
from .features import get_feature, sorted_quantile
from .types import WeightPair


def run(
    pair: WeightPair,
    *,
    feature: str = "sorted_quantile",
    bins: int = 64,
    match: str = "hungarian",
    topk: int = 10,
) -> AttackResult:
    if feature == "sorted_quantile":
        feat = lambda W: sorted_quantile(W, bins=bins)  # noqa: E731
    else:
        feat = get_feature(feature)

    A = feat(pair.plain)                 # (N, f)  plaintext signatures (index = plain token i)
    B = feat(pair.obf)                   # (N, f)  obfuscated signatures (index = obf row j)
    n = A.shape[0]
    k = min(topk, n)

    # Features are L2-normalised, so the inner product is cosine similarity.
    sims = A @ B.T                       # (N_plain, N_obf)
    order = np.argsort(-sims, axis=0)    # plaintext indices ranked per obf column
    topk_plain = order[:k].T             # (N_obf, k)

    if match == "hungarian":
        plain_idx, obf_idx = linear_sum_assignment(-sims)
        pred_plain = np.empty(n, dtype=np.int64)
        pred_plain[obf_idx] = plain_idx
    elif match == "nn":
        pred_plain = order[0]            # (N_obf,)
    else:
        raise ValueError(f"match must be 'hungarian' or 'nn', got {match!r}")

    true_plain = pair.inverse_perm()     # true plaintext index of each obf row
    true_tok = pair.token_ids[true_plain]
    top1 = float((pair.token_ids[pred_plain] == true_tok).mean())
    top10 = float(
        (topk_plain == true_plain[:, None]).any(axis=1).mean()
    )

    return AttackResult(
        attack="vma_tau_recovery",
        transform="identity",
        model_id=pair.model_id,
        kind=pair.kind,
        layer=0,
        n_prompts=n,
        n_train=0,
        n_test=n,
        ttrsr_top1=top1,
        ttrsr_top10=top10,
        risk_level=classify_risk_level(top1),
        primary_metric_value=top1,
        primary_metric_name="permutation_recovery_rate",
        extra={"feature": feature, "match": match, "bins": bins},
    )
