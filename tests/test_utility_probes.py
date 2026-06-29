"""CPU unit tests for the standardized utility probes (pure functions; teacher_forced_pass needs a
model and is exercised by the eval GPU runs, not here)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talens.probes.utility import (  # noqa: E402
    TokenPass,
    UtilityResult,
    embedding_fidelity,
    embedding_recovery,
    next_token_accuracy,
    output_agreement,
    perplexity,
    retention_thresholds,
)


def _pass(ce, pred, tgt):
    return TokenPass(np.array(ce, float), np.array(pred), np.array(tgt))


def test_next_token_accuracy_retention():
    tgt = [1, 2, 3, 4]
    clean = _pass([0.1] * 4, [1, 2, 3, 9], tgt)        # 3/4 correct
    defended = _pass([0.5] * 4, [1, 2, 9, 9], tgt)     # 2/4 correct
    r = next_token_accuracy(defended, clean)
    assert isinstance(r, UtilityResult) and r.metric == "next_token_accuracy"
    assert r.clean == 0.75 and r.defended == 0.5
    assert abs(r.retention - (0.5 / 0.75)) < 1e-9 and r.higher_is_better


def test_next_token_accuracy_requires_alignment():
    with pytest.raises(ValueError):
        next_token_accuracy(_pass([0.1], [1], [1]), _pass([0.1], [1], [2]))


def test_perplexity_retention_and_degradation():
    clean = _pass([0.0, 0.0], [0, 0], [0, 0])          # ppl = 1
    defended = _pass([1.0, 1.0], [0, 0], [0, 0])       # ppl = e
    r = perplexity(defended, clean)
    assert abs(r.clean - 1.0) < 1e-9 and abs(r.defended - np.e) < 1e-6
    assert abs(r.retention - 1.0 / np.e) < 1e-6 and not r.higher_is_better
    assert abs(r.extra["degradation"] - (np.e - 1.0)) < 1e-6


def test_output_agreement():
    clean = _pass([0.1] * 4, [1, 2, 3, 4], [0, 0, 0, 0])
    defended = _pass([0.1] * 4, [1, 2, 9, 9], [0, 0, 0, 0])   # agrees on 2/4
    r = output_agreement(defended, clean)
    assert r.clean == 1.0 and r.defended == 0.5 and r.retention == 0.5


def test_embedding_fidelity_and_recovery():
    rng = np.random.default_rng(0)
    e_c = rng.standard_normal((12, 8)).astype(np.float32)
    e_n = e_c + rng.standard_normal((12, 8)).astype(np.float32)
    fid = embedding_fidelity(e_c.copy(), e_c)
    assert fid.metric == "embedding_cosine" and fid.retention > 0.999 and fid.extra["mse"] < 1e-9
    perfect = embedding_recovery(e_c, e_n, e_c.copy())          # denoised == clean
    assert perfect["recovery_cos"] > 0.99 and perfect["recovery_mse"] > 0.99
    none = embedding_recovery(e_c, e_n, e_n.copy())             # denoised == noised
    assert abs(none["recovery_cos"]) < 1e-5 and abs(none["recovery_mse"]) < 1e-5


def test_retention_thresholds_crossing():
    xs = [5000, 2000, 1000, 500]                  # descending budget
    ret = [0.99, 0.92, 0.70, 0.40]                # descending retention
    thr = retention_thresholds(xs, ret, targets=(0.90, 0.50))
    assert 1000 < thr["retention_90pct"] < 2000   # 0.90 crossed between η=2000 and 1000
    assert 500 < thr["retention_50pct"] < 1000    # 0.50 crossed between η=1000 and 500
    assert retention_thresholds(xs, ret, targets=(0.999,))["retention_100pct"] is None  # never crossed (max ret 0.99)


def test_as_dict_flattens_extra():
    d = next_token_accuracy(_pass([0.1], [1], [1]), _pass([0.1], [1], [1])).as_dict()
    assert d["metric"] == "next_token_accuracy" and d["retention"] == 1.0 and d["n_tokens"] == 1
