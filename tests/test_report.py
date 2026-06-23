"""Tests for the unified reporting layer (bits canonical + per-secret readout).

Model-free / numpy-only — runs in the host CPU venv. Covers: bits legibility (the
"1/100 of a bit" fix), canonical-bits extraction from every measure's real output
schema, per-secret readout builders, and the end-to-end LeakageReport.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from talens.report import (
    LeakageReport,
    Readout,
    canonical_bits,
    embedding_readout,
    format_bits,
    membership_readout,
    permutation_readout,
    perplexity_from_bits,
    text_readout,
    token_f1,
    token_id_readout,
)

# ---------------------------------------------------------------------------
# format_bits — the legibility contract: nonzero never renders as "0".
# ---------------------------------------------------------------------------


def test_format_bits_small_value_not_zero():
    # The illegibility CLAUDE.md warns about: 0.01 bit must not collapse to "0.00".
    s = format_bits(0.01)
    assert "0" != s and s != "0.00 bits"
    assert "mbit" in s  # rendered in millibits
    # Parse the magnitude back out and confirm it is nonzero and ~10 mbit.
    assert abs(float(s.split()[0]) - 10.0) < 1e-6


@pytest.mark.parametrize(
    "x,expect_sub",
    [(3.14159, "3.14 bits"), (0.0, "0 bits"), (1.0, "1 bits"), (-0.005, "mbit")],
)
def test_format_bits_examples(x, expect_sub):
    assert expect_sub in format_bits(x)


def test_format_bits_infinite_and_none():
    assert format_bits(None) == "n/a"
    assert "∞" in format_bits(float("inf"))
    assert "−∞" in format_bits(float("-inf"))


def test_format_bits_does_not_mutate_value():
    x = 0.0123456
    _ = format_bits(x)
    assert x == 0.0123456  # rendering only; stored value untouched


# ---------------------------------------------------------------------------
# perplexity / token_f1 helpers
# ---------------------------------------------------------------------------


def test_perplexity_from_bits():
    assert perplexity_from_bits(0.0) == 1.0
    assert perplexity_from_bits(1.0) == 2.0
    assert perplexity_from_bits(3.0) == 8.0
    assert perplexity_from_bits(None) is None


def test_token_f1_exact_and_disjoint():
    a = np.array([1, 2, 3, 4])
    assert token_f1(a, a) == 1.0
    assert token_f1(np.array([1, 2]), np.array([3, 4])) == 0.0


def test_token_f1_partial_overlap():
    # pred {1,2,3}, gt {2,3,4}: overlap 2 -> P=R=2/3 -> F1=2/3.
    f1 = token_f1(np.array([1, 2, 3]), np.array([2, 3, 4]))
    assert abs(f1 - 2 / 3) < 1e-9


def test_token_f1_multiplicity():
    # pred has 1 twice, gt once -> overlap counts the min multiplicity (1).
    f1 = token_f1(np.array([1, 1, 2]), np.array([1, 2, 2]))
    # overlap = min(2,1)+min(1,2) = 1+1 = 2; P=2/3, R=2/3 -> 2/3
    assert abs(f1 - 2 / 3) < 1e-9


def test_token_f1_both_empty():
    assert token_f1(np.array([]), np.array([])) == 1.0


# ---------------------------------------------------------------------------
# canonical_bits — extract the comparable scalar from each REAL measure schema.
# ---------------------------------------------------------------------------


def test_canonical_bits_vinfo():
    res = {"v_information_bits": 0.42, "h_y_given_x_bits": 1.0, "num_classes": 8}
    bits, kind = canonical_bits("v_information", res)
    assert bits == 0.42 and kind == "v_info"


def test_canonical_bits_club_mdl_pid_spectral():
    assert canonical_bits("club", {"club_mi_bits": 1.7, "club_mi_nats": 1.18})[0] == 1.7
    assert canonical_bits("mdl", {"surplus_description_length_bits": 5.0})[0] == 5.0
    assert canonical_bits("pid", {"i_joint_bits": 2.5})[0] == 2.5
    assert canonical_bits("spectral_channel_mi", {"i_g_bits": 12.3})[0] == 12.3
    # the semantic kinds are distinct so a mixed table stays interpretable
    kinds = {
        canonical_bits("club", {"club_mi_bits": 1.0})[1],
        canonical_bits("mdl", {"surplus_description_length_bits": 1.0})[1],
        canonical_bits("spectral_channel_mi", {"i_g_bits": 1.0})[1],
    }
    assert kinds == {"mi_upper_bound", "sdl", "channel_mi_upper_bound"}


def test_canonical_bits_capacity_distinct_from_plain_vinfo():
    # capacity-matched PVI shares the dict key but is a DIFFERENT bits kind.
    _, kind = canonical_bits("v_information_capacity", {"v_information_bits": -1.5})
    assert kind == "capacity_v_info"


def test_canonical_bits_declined_measure_is_none():
    bits, _ = canonical_bits("v_information", {"v_information_bits": None, "note": "too few rows"})
    assert bits is None


def test_canonical_bits_unknown_measure_raises():
    with pytest.raises(KeyError):
        canonical_bits("not_a_measure", {})


# ---------------------------------------------------------------------------
# per-secret readouts
# ---------------------------------------------------------------------------


def test_token_id_readout_has_perplexity_and_topk():
    r = token_id_readout(top1=0.3, topk=0.6, k=10, h_bits=2.0)
    assert r.secret_kind == "token_id"
    assert r.primary_value == 0.3
    assert r.fields["top10"] == 0.6
    assert r.fields["perplexity"] == 4.0


def test_each_secret_kind_builder():
    assert text_readout(token_f1=0.5).secret_kind == "text"
    assert permutation_readout(recovery_rate=0.9, kendall_tau=0.8).fields["kendall_tau"] == 0.8
    assert embedding_readout(cosine=0.95, token_f1=0.4).fields["token_f1"] == 0.4
    assert membership_readout(auc=0.7).primary_value == 0.7


def test_readout_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Readout("not_a_secret", "x", 0.0)


def test_readout_render():
    r = token_id_readout(top1=0.25, topk=0.5, k=10)
    s = r.render()
    assert "token_top1_recovery_rate=0.25" in s and "top10=0.5" in s


# ---------------------------------------------------------------------------
# end-to-end LeakageReport
# ---------------------------------------------------------------------------


def test_leakage_report_from_measure_pairs_bits_and_readout():
    res = {"v_information_bits": 0.013}
    rep = LeakageReport.from_measure(
        "v_information", res,
        readout=token_id_readout(top1=0.02, topk=0.05),
        surface="resid-capacity-pvi", layer=20, sigma=1.0,
    )
    assert rep.bits == 0.013 and rep.bits_kind == "v_info"
    # legible rendering surfaces the small magnitude as millibits, not 0.
    assert "mbit" in rep.bits_legible()
    line = rep.render()
    assert "token_top1_recovery_rate=0.02" in line and "mbit" in line


def test_leakage_report_no_attack_row():
    rep = LeakageReport.from_measure("spectral_channel_mi", {"i_g_bits": 30.0})
    assert rep.readout is None
    assert "no-attack" in rep.render()


def test_leakage_report_to_dict_roundtrip_serializable():
    rep = LeakageReport.from_measure(
        "club", {"club_mi_bits": 1.25}, readout=membership_readout(auc=0.66), sigma=0.5,
    )
    d = rep.to_dict()
    assert d["bits"] == 1.25 and d["bits_legible"] == "1.25 bits"
    assert d["readout"]["secret_kind"] == "membership"
    # JSON-serializable (no numpy / dataclass leftovers)
    import json

    json.dumps(d)


def test_declined_bits_render_na():
    rep = LeakageReport.from_measure("v_information", {"v_information_bits": None})
    assert rep.bits is None
    assert rep.bits_legible() == "n/a"
