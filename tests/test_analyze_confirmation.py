"""Tests for the confirmation-run aggregator.

The statistics here decide what gets reported as a finding, so they are tested
against hand-computable cases rather than against their own output.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import pytest

from experiments.analyze_confirmation import (
    arm_endpoint,
    check_integrity,
    holm,
    load_manifests,
    sign_flip_p,
)


# ---------------------------------------------------------------------------
# sign_flip_p
# ---------------------------------------------------------------------------


def test_all_positive_differences_give_the_smallest_attainable_p():
    """With every seed agreeing, only the all-same-sign flips are as extreme.

    Two of the 2^10 assignments (all +, all -) reach the observed mean, so the
    exact two-sided p is 2/1024. This is the FLOOR of the test: no result with
    10 seeds can ever be more significant than this.
    """

    p = sign_flip_p([0.05] * 10)
    assert p == pytest.approx(2 / 1024)


def test_the_floor_is_a_real_constraint_not_an_artifact_of_equal_values():
    unequal = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    assert sign_flip_p(unequal) == pytest.approx(2 / 1024)


def test_a_symmetric_sample_is_maximally_unsurprising():
    assert sign_flip_p([0.05, -0.05]) == pytest.approx(1.0)


def test_p_is_computed_by_hand_for_three_observations():
    """Enumerated exhaustively by hand for diffs [3, 1, -1], mean 1.0.

    The 8 sign assignments give these means (denominators of 3 omitted):
      +3+1-1 = +3 -> 1.000   -3+1-1 = -3 -> -1.000
      +3-1-1 = +1 -> 0.333   -3-1-1 = -5 -> -1.667
      +3+1+1 = +5 -> 1.667   -3+1+1 = -1 -> -0.333
      +3-1+1 = +3 -> 1.000   -3-1+1 = -3 -> -1.000
    Six of the eight have |mean| >= 1.0; only the two at 0.333 do not.
    """

    diffs = [3.0, 1.0, -1.0]
    by_hand = [1.0, 1.0 / 3, 5.0 / 3, 1.0, -1.0, -5.0 / 3, -1.0 / 3, -1.0]
    assert sorted(by_hand) == sorted(
        st.mean([d if mask >> i & 1 else -d for i, d in enumerate(diffs)])
        for mask in range(8)
    )
    assert sign_flip_p(diffs) == pytest.approx(6 / 8)


def test_sign_flip_is_two_sided():
    """A consistently NEGATIVE effect is just as significant as a positive one."""

    assert sign_flip_p([-0.05] * 10) == pytest.approx(sign_flip_p([0.05] * 10))


def test_magnitude_matters_not_just_sign():
    """One seed disagreeing weakly is less damning than disagreeing strongly."""

    weak = sign_flip_p([0.05, 0.05, 0.05, 0.05, -0.001])
    strong = sign_flip_p([0.05, 0.05, 0.05, 0.05, -0.20])
    assert weak < strong


# ---------------------------------------------------------------------------
# holm
# ---------------------------------------------------------------------------


def test_holm_scales_the_smallest_p_by_the_family_size():
    adjusted = holm([0.01, 0.5, 0.5, 0.5])
    assert adjusted[0] == pytest.approx(0.04)


def test_holm_is_step_down_not_bonferroni():
    """The second-smallest p is multiplied by m-1, not by m."""

    adjusted = holm([0.01, 0.02, 0.5, 0.5])
    assert adjusted[1] == pytest.approx(0.06)


def test_holm_enforces_monotonicity():
    """An adjusted p can never fall below an adjusted p that ranked before it.

    Without the running maximum, [0.01, 0.04] over 2 tests would give 0.02 and
    0.04; but a case where the raw order would inverse after scaling must be
    clamped instead of reported as more significant.
    """

    adjusted = holm([0.01, 0.0101, 0.0102])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_holm_preserves_input_order():
    adjusted = holm([0.5, 0.01, 0.5])
    assert adjusted[1] < adjusted[0]
    assert adjusted[1] == pytest.approx(0.03)


def test_holm_caps_at_one():
    assert all(value <= 1.0 for value in holm([0.5, 0.6, 0.9, 0.99]))


def test_holm_on_the_real_family_size_leaves_almost_no_room():
    """24 comparisons at n=10 seeds: the floor p barely clears 0.05.

    This is the design constraint the analysis has to live with. The smallest
    p the exact test can produce is 2/1024; over 24 comparisons Holm scales it
    to 0.047. A 26-comparison family could not produce ANY significant result,
    however large the effect.
    """

    floor = 2 / 1024
    adjusted = holm([floor] + [0.9] * 23)
    assert adjusted[0] == pytest.approx(24 * floor)
    assert adjusted[0] < 0.05

    too_many = holm([floor] + [0.9] * 25)
    assert too_many[0] > 0.05


# ---------------------------------------------------------------------------
# integrity gate
# ---------------------------------------------------------------------------


def _manifest(**overrides):
    base = {
        "protocol_hash": "abc",
        "task_fingerprint": "xyz",
        "capacity_scope": "local",
        "protocol": {
            "steps_per_task": 120,
            "tasks": ["a"] * 10,
            "precision": "fp32",
        },
        "families": {},
    }
    base.update(overrides)
    return base


def test_integrity_passes_on_a_consistent_set():
    assert check_integrity({10: _manifest(), 11: _manifest()})["ok"]


def test_integrity_rejects_mixed_protocol_hashes():
    manifests = {10: _manifest(), 11: _manifest(protocol_hash="different")}
    report = check_integrity(manifests)
    assert not report["ok"]
    assert not report["protocol_hash_unico"]


def test_integrity_rejects_mixed_step_budgets():
    """The 30-step and 120-step runs must never be pooled."""

    old = _manifest()
    old["protocol"] = dict(old["protocol"], steps_per_task=30)
    assert not check_integrity({10: _manifest(), 11: old})["ok"]


def test_integrity_rejects_fp16():
    bad = _manifest()
    bad["protocol"] = dict(bad["protocol"], precision="fp16")
    assert not check_integrity({10: bad})["ok"]


def test_integrity_rejects_mixed_task_fingerprints():
    manifests = {10: _manifest(), 11: _manifest(task_fingerprint="other")}
    assert not check_integrity(manifests)["ok"]


# ---------------------------------------------------------------------------
# manifest reading
# ---------------------------------------------------------------------------


def test_arm_endpoint_skips_discarded_arms():
    manifest = {
        "families": {
            "0.5": {
                "arms": [
                    {
                        "arm": {"name": "iso_b0.5"},
                        "discarded": True,
                        "reason": "inalcancavel",
                    },
                ]
            }
        }
    }
    assert arm_endpoint(manifest, "0.5", "iso_b0.5", "final_average_accuracy") is None


def test_arm_endpoint_reads_the_named_arm():
    manifest = {
        "families": {
            "0.75": {
                "arms": [
                    {
                        "arm": {"name": "vanilla"},
                        "discarded": False,
                        "endpoints": {"final_average_accuracy": 0.4},
                    },
                    {
                        "arm": {"name": "hard_b0.75"},
                        "discarded": False,
                        "endpoints": {"final_average_accuracy": 0.46},
                    },
                ]
            }
        }
    }
    got = arm_endpoint(manifest, "0.75", "hard_b0.75", "final_average_accuracy")
    assert got == pytest.approx(0.46)


def test_load_manifests_fails_loudly_on_a_missing_seed(tmp_path: Path):
    (tmp_path / "s10").mkdir()
    (tmp_path / "s10" / "manifest.json").write_text(json.dumps(_manifest()))
    with pytest.raises(SystemExit):
        load_manifests(tmp_path, [10, 11], "s")
