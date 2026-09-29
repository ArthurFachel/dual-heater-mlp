"""Seed band and arm set for the plasticity-matched protocol.

goals/protocol_plasticity_matched.md, sections P1 and P2. A seed reused from a
band already spent breaks the paired design's independence from prior analysis,
and an arm added to the matrix after the fact is an undeclared condition inside
a pre-registered run.
"""

from __future__ import annotations

import pytest


def test_seed_band_is_ten_distinct_seeds() -> None:
    from experiments.qwen_lora_slowheat import PLASTICITY_MATCHED_SEEDS

    assert len(PLASTICITY_MATCHED_SEEDS) == 10
    assert len(set(PLASTICITY_MATCHED_SEEDS)) == 10


def test_seed_band_is_disjoint_from_every_band_already_spent() -> None:
    from experiments.bert_slowheat_diagnostic import (
        IMPORTANCE_CRITERION_ABLATION_SEEDS,
    )
    from experiments.confirmatory_split_mnist import DERPP_CONFIRMATORY_SEEDS
    from experiments.qwen_lora_slowheat import (
        EXACT_DECOMPOSITION_SEEDS,
        PLASTICITY_MATCHED_SEEDS,
    )
    from experiments.replay_selection_sweep import (
        REPLAY_SELECTOR_CONFIRMATORY_SEEDS,
    )

    seeds = set(PLASTICITY_MATCHED_SEEDS)
    for name, other in (
        ("exact decomposition", EXACT_DECOMPOSITION_SEEDS),
        ("criterion ablation", IMPORTANCE_CRITERION_ABLATION_SEEDS),
        ("replay selector", REPLAY_SELECTOR_CONFIRMATORY_SEEDS),
        ("derpp", DERPP_CONFIRMATORY_SEEDS),
    ):
        overlap = seeds & set(other)
        assert not overlap, f"reusa seeds de {name}: {sorted(overlap)}"


def test_seed_band_is_disjoint_from_the_lora_confirmation_seeds() -> None:
    """The band this protocol re-tests, hardcoded from its artifacts."""

    from experiments.qwen_lora_slowheat import PLASTICITY_MATCHED_SEEDS

    confirmation = {
        700_001, 725_009, 750_019, 775_037, 800_053,
        825_059, 850_061, 875_089, 900_089, 925_097,
    }
    assert not set(PLASTICITY_MATCHED_SEEDS) & confirmation


def test_the_protocol_declares_exactly_three_arms() -> None:
    """P1. `exact` is deliberately absent: the decomposition closed it."""

    from scripts.run_plasticity_matched import ARMS

    assert ARMS == ("vanilla", "lr_control", "frozen_a_control")
    assert "exact" not in ARMS


def test_the_runner_passes_the_frozen_learning_rate_scale() -> None:
    """P8. The scale is the freezing fraction, not a tuned value."""

    from dual_heater.lora_slowheat import FROZEN_A_LR_SCALE
    from scripts.run_plasticity_matched import LEARNING_RATE_SCALE

    assert LEARNING_RATE_SCALE == pytest.approx(FROZEN_A_LR_SCALE, abs=1e-12)
    assert LEARNING_RATE_SCALE == pytest.approx(4_214_016 / 6_769_920, abs=1e-12)


def test_the_runner_command_carries_the_scale_and_the_frozen_scenario() -> None:
    """The flag must be in the command the runner actually builds."""

    from scripts.run_plasticity_matched import build_command

    command = build_command(seed=6_000_003, destination="results/x/seed_6000003")
    assert "--learning-rate-scale" in command
    index = command.index("--learning-rate-scale")
    assert float(command[index + 1]) == pytest.approx(4_214_016 / 6_769_920)

    # Scenario copied verbatim from the confirmation manifest (section E).
    for flag, value in (
        ("--tasks", "10"),
        ("--rank", "16"),
        ("--alpha", "16.0"),
        ("--train-per-class", "50"),
        ("--eval-per-class", "20"),
        ("--epochs-per-task", "3"),
        ("--batch-size", "8"),
        ("--max-length", "48"),
    ):
        assert flag in command, flag
        assert command[command.index(flag) + 1] == value, flag

    # No masked arm exists here, so the masked-arm knob must NOT be passed:
    # it would take precedence in older runners and silently move the control.
    assert "--target-plasticity" not in command


def test_the_runner_requests_exactly_the_declared_arms() -> None:
    from scripts.run_plasticity_matched import build_command

    command = build_command(seed=6_000_003, destination="results/x/seed_6000003")
    start = command.index("--arms") + 1
    requested = []
    for token in command[start:]:
        if token.startswith("--"):
            break
        requested.append(token)
    assert tuple(requested) == ("vanilla", "lr_control", "frozen_a_control")


def test_sharding_partitions_the_band_exactly_once() -> None:
    """Every seed runs exactly once across the shards, or the run is invalid."""

    from experiments.qwen_lora_slowheat import PLASTICITY_MATCHED_SEEDS
    from scripts.run_plasticity_matched import shard_seeds

    for shards in (1, 2, 3, 4):
        covered: list[int] = []
        for shard in range(shards):
            covered.extend(
                shard_seeds(PLASTICITY_MATCHED_SEEDS, shard=shard, shards=shards)
            )
        assert sorted(covered) == sorted(PLASTICITY_MATCHED_SEEDS), shards
        assert len(covered) == len(set(covered)), f"seed duplicada em {shards} shards"


def test_sharding_keeps_a_whole_seed_on_one_device() -> None:
    """Splitting a seed's arms across GPUs would break the pairing."""

    from experiments.qwen_lora_slowheat import PLASTICITY_MATCHED_SEEDS
    from scripts.run_plasticity_matched import shard_seeds

    shard = shard_seeds(PLASTICITY_MATCHED_SEEDS, shard=0, shards=3)
    assert set(shard) <= set(PLASTICITY_MATCHED_SEEDS)
    # Three GPUs, ten seeds: 4/3/3.
    sizes = [
        len(shard_seeds(PLASTICITY_MATCHED_SEEDS, shard=index, shards=3))
        for index in range(3)
    ]
    assert sorted(sizes) == [3, 3, 4]


def test_sharding_rejects_an_out_of_range_shard() -> None:
    from experiments.qwen_lora_slowheat import PLASTICITY_MATCHED_SEEDS
    from scripts.run_plasticity_matched import shard_seeds

    with pytest.raises(ValueError):
        shard_seeds(PLASTICITY_MATCHED_SEEDS, shard=3, shards=3)
    with pytest.raises(ValueError):
        shard_seeds(PLASTICITY_MATCHED_SEEDS, shard=0, shards=0)
