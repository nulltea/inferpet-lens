"""Retrieval predictive family — resolution **B** for the split-regime
mismatch (see ``docs/research/attacks_setting.md``).

The class-probe family (``_probe.py``) cannot score vocab-disjoint test
tokens, so PVI/MDL built on it are forced into row-split. This module
provides the alternative: a probabilistic model ``q(y|x)`` that *is* the
inversion attack — fit the ridge map ``X→embedding``, then define a
softmax over cosine similarity to a candidate token-embedding pool::

    q(y | x) ∝ exp( cos(ŷ(x), E[y]) / τ )

Because recovery flows through embedding geometry, this **generalises to
unseen token ids**, so PVI/MDL built on it run vocab-disjoint — the same
regime as the honest attack. Used by ``v_information_retrieval`` and
``online_code_length_retrieval``.
"""

from __future__ import annotations

import numpy as np
import torch

from ..ridge import fit_ridge, predict_ridge

_DEFAULT_TEMPERATURES: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


def fit_inverter(
    X_train: np.ndarray,
    y_train_ids: np.ndarray,
    embed_table: torch.Tensor,
    *,
    ridge_alpha: float,
) -> dict[str, torch.Tensor]:
    """Fit the ridge ``X→embedding`` map (the attack's inverter)."""
    y_emb = embed_table[torch.from_numpy(y_train_ids)].to(torch.float32)
    return fit_ridge(torch.from_numpy(X_train).to(torch.float32), y_emb, ridge_alpha=ridge_alpha)


def predict_embeddings(model: dict[str, torch.Tensor], X: np.ndarray) -> torch.Tensor:
    return predict_ridge(model, torch.from_numpy(X).to(torch.float32))


def retrieval_log_softmax(
    pred_emb: torch.Tensor,
    candidate_ids: torch.Tensor,
    embed_table: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """``log q(·|x)`` over the candidate pool: log-softmax of cosine
    similarity / temperature. Returns ``(n, n_candidates)`` natural log.
    """
    cand = embed_table[candidate_ids]
    pn = pred_emb / pred_emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cn = cand / cand.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sims = pn @ cn.T
    return torch.log_softmax(sims / temperature, dim=1)


def log_prob_true(
    pred_emb: torch.Tensor,
    true_ids: torch.Tensor,
    candidate_ids: torch.Tensor,  # sorted ascending
    embed_table: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Natural log ``q(y_true | x)`` for each row. ``candidate_ids`` must
    be sorted and contain every id in ``true_ids``.
    """
    logq = retrieval_log_softmax(pred_emb, candidate_ids, embed_table, temperature)
    cols = torch.searchsorted(candidate_ids, true_ids)
    return logq[torch.arange(true_ids.shape[0]), cols]


def pick_temperature(
    pred_emb: torch.Tensor,
    true_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    embed_table: torch.Tensor,
    temperatures: tuple[float, ...] = _DEFAULT_TEMPERATURES,
) -> float:
    """Choose τ minimising validation NLL of the retrieval softmax — a
    proper calibration of the probabilistic model.
    """
    best_t, best_nll = temperatures[0], float("inf")
    for t in temperatures:
        nll = float(-log_prob_true(pred_emb, true_ids, candidate_ids, embed_table, t).mean())
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


def build_candidate_pool(
    *id_arrays: np.ndarray,
    vocab_size: int,
    pool_size: int,
    seed: int,
) -> torch.Tensor:
    """Sorted unique candidate ids: the union of the supplied id arrays
    (the scored tokens) plus a random vocab sample, capped at ``pool_size``.
    """
    rng = np.random.default_rng(seed)
    sample = rng.choice(vocab_size, size=min(pool_size, vocab_size), replace=False)
    must = np.concatenate([np.asarray(a, dtype=np.int64) for a in id_arrays] + [sample])
    return torch.from_numpy(np.unique(must)).to(torch.long)
