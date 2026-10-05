"""Seed band and design guards for the long-sequence criterion extension.

`goals/protocol_long_sequence_criterion.md`, sections S2, S4 and E.1. A seed
reused from a band already spent couples the new pass to the particular noise
of a published run, and a sequence length that silently drifts would make the
new result incomparable with the T=2 ablation it extends.

This file is a pre-registration guard. When it goes red, the fix is never to
update the expected value: the protocol is frozen and the code is what moved.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from experiments.bert_slowheat_diagnostic import (
    IMPORTANCE_CRITERION_ABLATION_SEEDS,
    LONG_SEQUENCE_CRITERION_SEEDS,
    LONG_SEQUENCE_TASK_LIMIT,
    criterion_ablation_conditions,
)


def test_seed_band_has_ten_distinct_seeds() -> None:
    """S4: ten seeds, the same count as the T=2 ablation it extends."""

    assert len(LONG_SEQUENCE_CRITERION_SEEDS) == 10
    assert len(set(LONG_SEQUENCE_CRITERION_SEEDS)) == 10


def test_seed_band_matches_the_frozen_values() -> None:
    """E.1, pinned literally. These numbers are the pre-registration."""

    assert LONG_SEQUENCE_CRITERION_SEEDS == (
        12_000_017,
        12_025_031,
        12_050_033,
        12_075_059,
        12_100_063,
        12_125_083,
        12_150_107,
        12_175_117,
        12_200_129,
        12_225_149,
    )


def test_seed_band_is_disjoint_from_the_two_task_ablation() -> None:
    """E.2: the published T=2 band must not be reused."""

    overlap = set(LONG_SEQUENCE_CRITERION_SEEDS) & set(
        IMPORTANCE_CRITERION_ABLATION_SEEDS
    )
    assert not overlap, f"reusa seeds da ablação T=2: {sorted(overlap)}"


def test_seed_band_is_disjoint_from_the_earlier_bert_seeds() -> None:
    """The BERT seeds already spent, hardcoded from their artefacts."""

    spent = {2, 9, 10, 28, 30, 32, 57, 67, 2005, 2012, 11, 22, 33}
    overlap = set(LONG_SEQUENCE_CRITERION_SEEDS) & spent
    assert not overlap, f"reusa seeds de runs BERT anteriores: {sorted(overlap)}"


def test_seed_band_is_disjoint_from_every_other_band_in_the_project() -> None:
    """Checked by import, not by inspection."""

    from experiments.confirmatory_split_mnist import (
        CONFIRMATORY_SEEDS,
        DERPP_CONFIRMATORY_SEEDS,
        LAMBDA_SWEEP_SEEDS,
        PENALTY_PASS2_SEEDS,
        PENALTY_REEVALUATION_SEEDS,
        SGD_PLASTICITY_SEEDS,
    )
    from experiments.qwen_lora_slowheat import (
        EXACT_DECOMPOSITION_SEEDS,
        PLASTICITY_MATCHED_SEEDS,
    )
    from experiments.replay_selection_sweep import (
        REPLAY_SELECTOR_CONFIRMATORY_SEEDS,
    )

    seeds = set(LONG_SEQUENCE_CRITERION_SEEDS)
    for name, other in (
        ("confirmatory split-mnist", CONFIRMATORY_SEEDS),
        ("derpp", DERPP_CONFIRMATORY_SEEDS),
        ("penalty reevaluation", PENALTY_REEVALUATION_SEEDS),
        ("sgd plasticity", SGD_PLASTICITY_SEEDS),
        ("lambda sweep", LAMBDA_SWEEP_SEEDS),
        ("penalty pass 2", PENALTY_PASS2_SEEDS),
        ("replay selector", REPLAY_SELECTOR_CONFIRMATORY_SEEDS),
        ("exact decomposition", EXACT_DECOMPOSITION_SEEDS),
        ("plasticity matched", PLASTICITY_MATCHED_SEEDS),
    ):
        overlap = seeds & set(other)
        assert not overlap, f"reusa seeds de {name}: {sorted(overlap)}"


def test_calibration_seed_is_outside_the_confirmatory_band() -> None:
    """Section I. The calibration runs the full matrix and writes accuracy.

    Spending a band seed on it would mean reading one confirmatory seed's FAA
    before the other nine exist, so any later decision about the regime would
    carry information from the band. Measuring wall-clock does not require a
    band seed.
    """

    from experiments.bert_slowheat_diagnostic import LONG_SEQUENCE_CALIBRATION_SEED

    assert LONG_SEQUENCE_CALIBRATION_SEED not in LONG_SEQUENCE_CRITERION_SEEDS
    assert LONG_SEQUENCE_CALIBRATION_SEED not in IMPORTANCE_CRITERION_ABLATION_SEEDS
    assert LONG_SEQUENCE_CALIBRATION_SEED == 11_900_003


def test_frozen_sequence_length_is_five() -> None:
    """S2. Section F explains why not ten; changing this needs a new document."""

    assert LONG_SEQUENCE_TASK_LIMIT == 5


def test_the_arms_are_the_same_four_as_the_two_task_ablation() -> None:
    """S3: the extension varies sequence length and nothing else.

    If an arm were added or its hyperparameters moved, the new pass would not
    be comparable with the published T=2 result it is meant to extend.
    """

    conditions = criterion_ablation_conditions()

    assert [c.name for c in conditions] == [
        "vanilla",
        "slowheat_random_hard",
        "slowheat_magnitude_hard",
        "slowheat_hard",
    ]
    protected = [c for c in conditions if c.method != "vanilla"]
    assert all(c.slow_strength == 3.0 for c in protected), "S10"
    assert all(c.mask_mode in ("hard", "random_hard") for c in protected), "S3"
    by_name = {c.name: c.importance_criterion for c in conditions}
    assert by_name["slowheat_magnitude_hard"] == "magnitude"
    assert by_name["slowheat_hard"] == "functional"
