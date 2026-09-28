"""Contract tests for the LoRA sweep aggregator.

The aggregator turns per-seed manifests into paired contrasts. Its failure
modes are silent: a wrong pairing still produces a plausible p-value, so the
properties below are asserted directly.
"""

from __future__ import annotations

import pytest

from experiments.qwen_lora_sweep import _aggregate


def _manifest(seed: int, values: dict[str, tuple[float, float]]) -> dict:
    """values: arm -> (final_average_accuracy, average_forgetting)."""

    return {
        "config": {"seed": seed},
        "results": [
            {
                "method": arm,
                "final_average_accuracy": faa,
                "average_forgetting": forg,
                "backward_transfer": -forg,
                "wall_clock_seconds": 1.0,
                "peak_memory_mib": 100.0,
                "train_tokens": 1000,
                "stages": [{"effective_plasticity": 0.85}],
            }
            for arm, (faa, forg) in values.items()
        ],
    }


def test_pairing_follows_seed_not_list_position() -> None:
    """A reordered manifest list must not change any paired difference.

    This is the aggregator's most dangerous silent failure: pairing by
    position would compare one seed's treatment against another seed's
    reference and still return a well-formed p-value.
    """

    a = _manifest(11, {"vanilla": (0.50, 0.40), "exact": (0.60, 0.30)})
    b = _manifest(22, {"vanilla": (0.70, 0.20), "exact": (0.75, 0.15)})

    forward = _aggregate([a, b])["paired_vs_vanilla"]["exact"]
    reversed_ = _aggregate([b, a])["paired_vs_vanilla"]["exact"]

    assert (
        forward["final_average_accuracy"]["per_seed_differences"]
        == reversed_["final_average_accuracy"]["per_seed_differences"]
    )
    # Both seeds improve, so every difference is positive regardless of order.
    assert all(
        d > 0 for d in forward["final_average_accuracy"]["per_seed_differences"]
    )
    assert forward["final_average_accuracy"]["seeds"] == [11, 22]


def test_contrast_against_the_control_arm_is_reported() -> None:
    """The mechanism-isolating contrast must exist whenever the control ran."""

    manifests = [
        _manifest(
            seed,
            {
                "vanilla": (0.50, 0.40),
                "lr_control": (0.45, 0.45),
                "exact": (0.60, 0.30),
            },
        )
        for seed in (11, 22, 33)
    ]
    report = _aggregate(manifests)

    assert "paired_vs_lr_control" in report
    against_control = report["paired_vs_lr_control"]["exact"]
    # exact beats lr_control by 0.15 on accuracy in every seed.
    assert against_control["final_average_accuracy"]["mean_difference"] == pytest.approx(
        0.15
    )
    assert "vanilla" in report["paired_vs_lr_control"]


def test_control_contrast_is_absent_without_the_control_arm() -> None:
    """No control arm means the isolating contrast must not be fabricated."""

    manifests = [
        _manifest(seed, {"vanilla": (0.50, 0.40), "exact": (0.60, 0.30)})
        for seed in (11, 22)
    ]
    assert "paired_vs_lr_control" not in _aggregate(manifests)


def test_only_seeds_present_in_both_arms_are_paired() -> None:
    """A partially failed sweep must pair the intersection, not pad it.

    This is the aggregator's most dangerous silent failure. If the treatment
    arm crashed on seed 22 but the reference did not, pairing by "seeds where
    the treatment ran" is correct, while pairing by "all seeds" or by list
    position silently compares seed 11's treatment against seed 22's
    reference and still returns a well-formed p-value.
    """

    complete = _manifest(11, {"vanilla": (0.50, 0.40), "exact": (0.60, 0.30)})
    # exact crashed on this seed; vanilla's numbers are deliberately far from
    # seed 11's so that a mispairing produces a visibly wrong difference.
    partial = _manifest(22, {"vanilla": (0.90, 0.05)})

    report = _aggregate([complete, partial])
    paired = report["paired_vs_vanilla"]["exact"]["final_average_accuracy"]

    assert paired["seeds"] == [11]
    # Correct pairing: 0.60 - 0.50 on seed 11 only.
    assert paired["per_seed_differences"] == pytest.approx([0.10])
    assert paired["mean_difference"] == pytest.approx(0.10)
    # The unpaired seed still counts toward the per-arm summary.
    assert report["per_arm"]["vanilla"]["final_average_accuracy"]["n"] == 2
    assert report["per_arm"]["exact"]["final_average_accuracy"]["n"] == 1


def test_a_reference_missing_on_one_seed_shrinks_the_pairing() -> None:
    """Symmetric case: the reference is the arm that failed."""

    complete = _manifest(11, {"vanilla": (0.50, 0.40), "exact": (0.60, 0.30)})
    partial = _manifest(22, {"exact": (0.95, 0.02)})  # vanilla crashed

    paired = _aggregate([complete, partial])["paired_vs_vanilla"]["exact"]
    assert paired["final_average_accuracy"]["seeds"] == [11]
    assert paired["final_average_accuracy"]["per_seed_differences"] == pytest.approx(
        [0.10]
    )
