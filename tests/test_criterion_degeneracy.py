"""Cross-arm protection overlap, the second mandatory metric of Section F.

goals/protocol_importance_criterion_ablation.md Section F declares TWO
obligatory diagnostics per arm: the normalized ranking variance AND the
Jaccard overlap of the protected top-k sets between `magnitude` and `random`,
and between `magnitude` and `functional`. The frozen interpretation rule reads
the first pair: overlap > 0.8 declares the `magnitude` arm degenerate and turns
any criterion tie into an inconclusive result rather than evidence.

The 2026-09-28 run emitted only the variance -- `aggregate_ranking_degeneracy`
is called without a `reference`, so `top_k_overlap` is never produced and the
published artefacts cannot answer the falsifier's second half. This module
recovers the missing number from the saved checkpoints, which store the
`slow_heat` buffers the mask actually used.
"""

from __future__ import annotations

import pytest
import torch

from experiments.criterion_degeneracy import (
    DEGENERACY_OVERLAP_THRESHOLD,
    arm_overlap,
    protected_masks,
    summarize_degeneracy_overlap,
)


def _state(**trackers: torch.Tensor) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, heat in trackers.items():
        state[f"{name}.slow_heat"] = heat
        state[f"{name}.importance_memory"] = torch.ones_like(heat)
    # Weights and unrelated buffers must be ignored by the extractor.
    state["bert.encoder.layer.0.output.dense.weight"] = torch.randn(4, 4)
    state["residual_trackers.0.slow_heat"] = torch.zeros(8)
    return state


def test_protected_masks_reads_only_ffn_and_attention_slow_heat():
    state = _state(
        **{
            "ffn_trackers.0": torch.tensor([0.0, 0.5, 0.0, 1.0]),
            "attention_trackers.0": torch.tensor([0.0, 0.2]),
        }
    )

    masks = protected_masks(state)

    assert set(masks) == {"ffn_trackers.0", "attention_trackers.0"}
    assert masks["ffn_trackers.0"].tolist() == [False, True, False, True]
    assert masks["attention_trackers.0"].tolist() == [False, True]


def test_protected_masks_rejects_a_state_without_trackers():
    with pytest.raises(ValueError):
        protected_masks({"bert.pooler.dense.weight": torch.randn(2, 2)})


def test_arm_overlap_of_identical_protection_is_one():
    heat = torch.tensor([0.0, 0.3, 0.9, 0.0])
    left = _state(**{"ffn_trackers.0": heat})
    right = _state(**{"ffn_trackers.0": heat.clone()})

    report = arm_overlap(left, right)

    assert report["mean_jaccard"] == pytest.approx(1.0)
    assert report["per_tracker"]["ffn_trackers.0"] == pytest.approx(1.0)


def test_arm_overlap_of_disjoint_protection_is_zero():
    left = _state(**{"ffn_trackers.0": torch.tensor([1.0, 1.0, 0.0, 0.0])})
    right = _state(**{"ffn_trackers.0": torch.tensor([0.0, 0.0, 1.0, 1.0])})

    assert arm_overlap(left, right)["mean_jaccard"] == pytest.approx(0.0)


def test_arm_overlap_averages_over_trackers_not_over_units():
    """One huge tracker must not drown a small one, and vice versa.

    Section F ranks per-tracker: `_apply_capacity_budget` runs per state, so a
    unit-weighted average would hide a degenerate small tracker behind a large
    healthy one.
    """

    left = _state(
        **{
            "ffn_trackers.0": torch.cat(
                [torch.ones(100), torch.zeros(100)]
            ),
            "attention_trackers.0": torch.tensor([1.0, 0.0]),
        }
    )
    right = _state(
        **{
            "ffn_trackers.0": torch.cat(
                [torch.ones(100), torch.zeros(100)]
            ),
            "attention_trackers.0": torch.tensor([0.0, 1.0]),
        }
    )

    report = arm_overlap(left, right)

    assert report["per_tracker"]["ffn_trackers.0"] == pytest.approx(1.0)
    assert report["per_tracker"]["attention_trackers.0"] == pytest.approx(0.0)
    assert report["mean_jaccard"] == pytest.approx(0.5)


def test_arm_overlap_reports_family_capacity_so_pairing_is_checked_not_assumed():
    """Capacity is matched per family, not per tracker.

    BERT consolidates with a hierarchical scope, so two arms can protect
    different counts inside each tracker while the family total is identical.
    A per-tracker count difference is therefore NOT evidence of unmatched
    capacity -- only the total is. The report must carry both so the check is
    visible instead of assumed.
    """

    left = _state(
        **{
            "ffn_trackers.0": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "ffn_trackers.1": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }
    )
    right = _state(
        **{
            "ffn_trackers.0": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "ffn_trackers.1": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        }
    )

    report = arm_overlap(left, right)

    assert report["left_protected_total"] == 3
    assert report["right_protected_total"] == 3
    assert report["capacity_matched"] is True
    assert report["left_protected_per_tracker"]["ffn_trackers.0"] == 2


def test_arm_overlap_flags_unmatched_capacity():
    left = _state(**{"ffn_trackers.0": torch.tensor([1.0, 1.0, 0.0, 0.0])})
    right = _state(**{"ffn_trackers.0": torch.tensor([1.0, 0.0, 0.0, 0.0])})

    report = arm_overlap(left, right)

    assert report["capacity_matched"] is False


def test_arm_overlap_rejects_mismatched_tracker_sets():
    left = _state(**{"ffn_trackers.0": torch.tensor([1.0, 0.0])})
    right = _state(**{"ffn_trackers.1": torch.tensor([1.0, 0.0])})

    with pytest.raises(ValueError):
        arm_overlap(left, right)


def test_summarize_applies_the_frozen_degeneracy_rule():
    """Section F: overlap(magnitude, random) > 0.8 means the arm degenerated."""

    assert DEGENERACY_OVERLAP_THRESHOLD == 0.8

    # Two seeds where magnitude and random protect almost the same units.
    def near_identical(flip: int):
        heat = torch.zeros(100)
        heat[:75] = 1.0
        other = heat.clone()
        other[:flip] = 0.0
        other[75 : 75 + flip] = 1.0
        return heat, other

    by_seed = {}
    for seed, flip in ((1, 1), (2, 2)):
        heat, other = near_identical(flip)
        by_seed[seed] = {
            "random": _state(**{"ffn_trackers.0": other}),
            "magnitude": _state(**{"ffn_trackers.0": heat}),
            "functional": _state(**{"ffn_trackers.0": heat.clone()}),
        }

    summary = summarize_degeneracy_overlap(by_seed)

    assert summary["seeds"] == [1, 2]
    assert summary["pairs"]["magnitude_vs_random"]["mean"] > 0.8
    assert summary["degenerate"] is True
    assert summary["verdict"].startswith("inconclusive")


def test_summarize_passes_the_falsifier_when_rankings_disagree():
    by_seed = {}
    for seed in (1, 2, 3):
        protected = torch.zeros(100)
        protected[:75] = 1.0
        random_heat = torch.zeros(100)
        random_heat[25:] = 1.0  # 50 shared, union 100 -> Jaccard 0.5
        by_seed[seed] = {
            "random": _state(**{"ffn_trackers.0": random_heat}),
            "magnitude": _state(**{"ffn_trackers.0": protected}),
            "functional": _state(**{"ffn_trackers.0": protected.clone()}),
        }

    summary = summarize_degeneracy_overlap(by_seed)

    assert summary["pairs"]["magnitude_vs_random"]["mean"] == pytest.approx(0.5)
    assert summary["pairs"]["magnitude_vs_functional"]["mean"] == pytest.approx(1.0)
    assert summary["degenerate"] is False
    assert summary["verdict"].startswith("exercised")
    assert summary["pairs"]["magnitude_vs_random"]["by_seed"]["1"] == pytest.approx(0.5)


def test_summarize_reads_the_magnitude_arm_for_the_frozen_verdict():
    """The rule is about `magnitude` vs `random`, not about any pair of arms.

    Kills the mutation that points `magnitude_vs_random` at the functional arm:
    with three mutually distinct protection sets, reading the wrong left arm
    changes the number the frozen threshold is compared against.
    """

    protected = torch.zeros(100)
    protected[:75] = 1.0            # magnitude: units 0..74
    functional = torch.zeros(100)
    functional[10:85] = 1.0         # functional: units 10..84
    random_heat = torch.zeros(100)
    random_heat[25:] = 1.0          # random: units 25..99

    by_seed = {
        seed: {
            "random": _state(**{"ffn_trackers.0": random_heat.clone()}),
            "magnitude": _state(**{"ffn_trackers.0": protected.clone()}),
            "functional": _state(**{"ffn_trackers.0": functional.clone()}),
        }
        for seed in (1, 2)
    }

    summary = summarize_degeneracy_overlap(by_seed)

    # magnitude vs random: |∩| = 50 (25..74), |∪| = 100.
    assert summary["pairs"]["magnitude_vs_random"]["mean"] == pytest.approx(0.5)
    # magnitude vs functional: |∩| = 65 (10..74), |∪| = 85 (0..84).
    assert summary["pairs"]["magnitude_vs_functional"]["mean"] == pytest.approx(
        65 / 85
    )
    # functional vs random would be 60/90 -- a different number, so pointing
    # the declared pair at the wrong arm cannot pass unnoticed.
    assert summary["pairs"]["magnitude_vs_random"]["mean"] != pytest.approx(60 / 90)


def test_arm_overlap_min_and_max_bracket_the_mean_across_trackers():
    """`min_jaccard` must be the worst tracker, not a second copy of the mean.

    Section F's falsifier is about a degenerate ranking; one collapsed tracker
    inside an otherwise healthy model is exactly the case a mean hides.
    """

    left = _state(
        **{
            "ffn_trackers.0": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "ffn_trackers.1": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "attention_trackers.0": torch.tensor([1.0, 0.0]),
        }
    )
    right = _state(
        **{
            "ffn_trackers.0": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "ffn_trackers.1": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            "attention_trackers.0": torch.tensor([0.0, 1.0]),
        }
    )

    report = arm_overlap(left, right)

    assert report["max_jaccard"] == pytest.approx(1.0)
    assert report["min_jaccard"] == pytest.approx(0.0)
    assert report["mean_jaccard"] == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)
    assert report["min_jaccard"] < report["mean_jaccard"] < report["max_jaccard"]


def test_arm_overlap_of_two_unprotected_trackers_is_one_by_convention():
    """Empty ∩ empty is total agreement, matching `capacity_calibration.jaccard`.

    A tracker that protects nothing in both arms says the two rankings agree
    there; scoring it 0.0 would push the mean down and could flip the frozen
    verdict toward "not degenerate" for the wrong reason.
    """

    left = _state(**{"ffn_trackers.0": torch.zeros(4)})
    right = _state(**{"ffn_trackers.0": torch.zeros(4)})

    assert arm_overlap(left, right)["mean_jaccard"] == pytest.approx(1.0)


def test_arm_overlap_rejects_trackers_of_different_widths():
    """Same tracker name, different unit count, means the runs are not comparable.

    Without this check the boolean `|` broadcasts and silently produces a
    number for two models that never shared an architecture.
    """

    left = _state(**{"ffn_trackers.0": torch.tensor([1.0, 0.0, 1.0, 0.0])})
    right = _state(**{"ffn_trackers.0": torch.tensor([1.0, 0.0])})

    with pytest.raises(ValueError):
        arm_overlap(left, right)


def test_summarize_requires_the_three_declared_arms():
    by_seed = {1: {"magnitude": _state(**{"ffn_trackers.0": torch.ones(4)})}}

    with pytest.raises(ValueError):
        summarize_degeneracy_overlap(by_seed)


def test_summarize_rejects_an_empty_run():
    with pytest.raises(ValueError):
        summarize_degeneracy_overlap({})
