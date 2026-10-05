"""Guards for the pre-endpoint verifier of the long-sequence extension.

scripts/verify_long_sequence_criterion.py is a GATE: it runs before anyone
looks at an endpoint, and a gate that cannot fail is decoration. Each test here
builds a run that violates exactly one check and asserts the gate catches it.

The near-chance gate (V7) and the degeneracy rule (V6) are the two that exist
because the project already paid for not having them: pass 2 (6970361) produced
p = 0.00049 with every arm near chance, and the 2026-09-28 criterion ablation
shipped without the cross-arm overlap its own Section F declared mandatory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from experiments.bert_slowheat_diagnostic import (
    LONG_SEQUENCE_CRITERION_SEEDS,
    LONG_SEQUENCE_TASK_LIMIT,
    criterion_ablation_conditions,
)
from experiments.criterion_degeneracy import ABLATION_ARM_PATHS
from scripts.verify_long_sequence_criterion import (
    NEAR_CHANCE_FACTOR,
    check_long_sequence_run,
)

ARMS = [condition.name for condition in criterion_ablation_conditions()]
HEALTHY_FAA = 0.40
HEALTHY_VARIANCE = 2.9e-03


def _stat(mean):
    return {"n": 10, "mean": mean, "sample_std": 0.0}


def _summary(
    *,
    seeds=LONG_SEQUENCE_CRITERION_SEEDS,
    task_count=LONG_SEQUENCE_TASK_LIMIT,
    faa=HEALTHY_FAA,
    variance=HEALTHY_VARIANCE,
    protected=2_364_672.0,
    arms=None,
):
    names = ARMS if arms is None else arms
    return {
        "schema_version": 1,
        "endpoint_source": "validation",
        "task_count": task_count,
        "seeds": sorted(seeds),
        "conditions": {
            name: {
                "final_average_accuracy": _stat(faa),
                "ranking_variance": _stat(None if name == "vanilla" else variance),
                "protected_count": _stat(0.0 if name == "vanilla" else protected),
            }
            for name in names
        },
    }


def _write_checkpoints(root: Path, seeds, *, magnitude_like_random: bool):
    """Write the per-arm slow_heat buffers the overlap metric reads.

    `functional` and `magnitude` protect units 0..74; `random` protects 25..99,
    giving Jaccard 0.5 -- comfortably below the 0.8 degeneracy threshold.
    When `magnitude_like_random` is set, `magnitude` is moved onto the random
    arm's units so the frozen rule must fire.
    """

    protected = torch.zeros(100)
    protected[:75] = 1.0
    random_heat = torch.zeros(100)
    random_heat[25:] = 1.0

    heats = {
        "functional": protected,
        "random": random_heat,
        "magnitude": random_heat.clone() if magnitude_like_random else protected.clone(),
    }
    for seed in seeds:
        for arm, (condition, method) in ABLATION_ARM_PATHS.items():
            directory = root / f"seed_{seed}" / condition / method
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": {
                        "ffn_trackers.0.slow_heat": heats[arm],
                        "ffn_trackers.0.importance_memory": torch.ones(100),
                    }
                },
                directory / "checkpoint.pt",
            )


def _build_run(tmp_path: Path, summary: dict, *, magnitude_like_random=False) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True, exist_ok=True)
    (root / "diagnostic_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    _write_checkpoints(
        root, summary["seeds"], magnitude_like_random=magnitude_like_random
    )
    return root


def test_a_sound_run_passes_every_check(tmp_path):
    failures, report = check_long_sequence_run(_build_run(tmp_path, _summary()))

    assert failures == []
    assert report["task_limit"] == 5
    assert report["chance"] == pytest.approx(1 / 75)
    assert report["degeneracy_overlap"]["degenerate"] is False
    assert report["degeneracy_overlap"]["pairs"]["magnitude_vs_random"][
        "mean"
    ] == pytest.approx(0.5)


def test_v1_catches_a_seed_band_that_is_not_the_pre_registered_one(tmp_path):
    seeds = list(LONG_SEQUENCE_CRITERION_SEEDS[:-1]) + [999_983]
    root = _build_run(tmp_path, _summary(seeds=seeds))

    failures, _ = check_long_sequence_run(root)

    assert any(f.startswith("V1") for f in failures)


def test_v2_catches_a_run_at_the_wrong_sequence_length(tmp_path):
    root = _build_run(tmp_path, _summary(task_count=2))

    failures, _ = check_long_sequence_run(root)

    assert any(f.startswith("V2") for f in failures)


def test_v3_catches_an_undeclared_arm(tmp_path):
    root = _build_run(tmp_path, _summary(arms=[*ARMS, "slowheat_beta_30"]))

    failures, _ = check_long_sequence_run(root)

    assert any(f.startswith("V3") for f in failures)


def test_v4_catches_unmatched_capacity_between_arms(tmp_path):
    summary = _summary()
    summary["conditions"]["slowheat_magnitude_hard"]["protected_count"] = _stat(
        1_000_000.0
    )
    root = _build_run(tmp_path, summary)

    failures, _ = check_long_sequence_run(root)

    assert any(f.startswith("V4") for f in failures)


def test_v5_catches_a_flat_ranking(tmp_path):
    """The variance falsifier of roadmap item 3.3. Zero variance means the
    top-k was chosen by noise and the criterion was never exercised."""

    summary = _summary()
    summary["conditions"]["slowheat_magnitude_hard"]["ranking_variance"] = _stat(0.0)
    root = _build_run(tmp_path, summary)

    failures, _ = check_long_sequence_run(root)

    assert any(f.startswith("V5") for f in failures)


def test_v5_catches_a_missing_variance(tmp_path):
    """Absent is not the same as fine: without the number the tie is not
    reportable at all (roadmap 3.3)."""

    summary = _summary()
    summary["conditions"]["slowheat_hard"]["ranking_variance"] = _stat(None)
    root = _build_run(tmp_path, summary)

    failures, _ = check_long_sequence_run(root)

    assert any(f.startswith("V5") for f in failures)


def test_v6_fires_the_frozen_degeneracy_rule(tmp_path):
    """Section F: overlap(magnitude, random) > 0.8 makes a tie inconclusive."""

    root = _build_run(tmp_path, _summary(), magnitude_like_random=True)

    failures, report = check_long_sequence_run(root)

    assert any(f.startswith("V6") for f in failures)
    assert report["degeneracy_overlap"]["degenerate"] is True
    assert "INCONCLUSIVO" in " ".join(failures)


def test_v6_reports_a_missing_checkpoint_instead_of_skipping_the_check(tmp_path):
    root = _build_run(tmp_path, _summary())
    target = (
        root
        / f"seed_{LONG_SEQUENCE_CRITERION_SEEDS[0]}"
        / ABLATION_ARM_PATHS["magnitude"][0]
        / ABLATION_ARM_PATHS["magnitude"][1]
        / "checkpoint.pt"
    )
    target.unlink()

    failures, _ = check_long_sequence_run(root)

    assert any(f.startswith("V6") for f in failures)


def test_v7_fires_the_near_chance_gate(tmp_path):
    """Section G.1. Pass 2's lesson: a significant contrast between arms that
    barely learned does not separate consolidation from "changed less"."""

    chance = 1 / 75
    root = _build_run(tmp_path, _summary(faa=chance * 2.0))

    failures, report = check_long_sequence_run(root)

    assert any(f.startswith("V7") for f in failures)
    assert "INCONCLUSIVA" in " ".join(failures)
    assert report["near_chance_threshold"] == pytest.approx(
        NEAR_CHANCE_FACTOR * chance
    )


def test_v7_lets_a_healthy_regime_through(tmp_path):
    chance = 1 / 75
    root = _build_run(tmp_path, _summary(faa=chance * NEAR_CHANCE_FACTOR + 1e-6))

    failures, _ = check_long_sequence_run(root)

    assert not any(f.startswith("V7") for f in failures)


def test_v7_reads_the_best_arm_not_the_worst(tmp_path):
    """Section G.1 asks whether ANY arm learned, not whether every arm did.

    `vanilla` is expected to collapse in class-IL -- that is the baseline, not
    a degenerate regime. Reading the worst arm would declare every healthy run
    inconclusive and make the gate useless.
    """

    chance = 1 / 75
    summary = _summary(faa=chance * 10.0)
    summary["conditions"]["vanilla"]["final_average_accuracy"] = _stat(chance * 1.2)
    root = _build_run(tmp_path, summary)

    failures, report = check_long_sequence_run(root)

    assert report["best_faa"] == pytest.approx(chance * 10.0)
    assert not any(f.startswith("V7") for f in failures), (
        "um vanilla colapsado não pode disparar o gate de regime"
    )


def test_v7_fires_when_even_the_best_arm_is_near_chance(tmp_path):
    """The mirror case: the best arm barely beats a collapsed vanilla."""

    chance = 1 / 75
    summary = _summary(faa=chance * 2.5)
    summary["conditions"]["vanilla"]["final_average_accuracy"] = _stat(chance * 1.1)
    root = _build_run(tmp_path, summary)

    failures, report = check_long_sequence_run(root)

    assert report["best_faa"] == pytest.approx(chance * 2.5)
    assert any(f.startswith("V7") for f in failures)


def test_v7_counts_vanilla_as_an_arm_for_the_regime_check(tmp_path):
    """G.1 says "the best arm", with no exclusion.

    `vanilla` learning while the protected arms collapse would be a strange
    run, but it is not a degenerate REGIME -- the benchmark is clearly
    learnable under these settings, so the right verdict is "look at the
    mechanism", not "the measurement means nothing". Excluding vanilla from the
    maximum would silently rewrite the frozen rule.
    """

    chance = 1 / 75
    summary = _summary(faa=chance * 2.0)
    summary["conditions"]["vanilla"]["final_average_accuracy"] = _stat(chance * 12.0)
    root = _build_run(tmp_path, summary)

    failures, report = check_long_sequence_run(root)

    assert report["best_faa"] == pytest.approx(chance * 12.0)
    assert not any(f.startswith("V7") for f in failures)


def test_a_missing_summary_fails_instead_of_passing_vacuously(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()

    failures, _ = check_long_sequence_run(empty)

    assert failures and failures[0].startswith("V1")
