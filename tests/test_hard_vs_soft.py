"""Tests for the hard-versus-soft protection suite.

These cover the design invariants that make the comparison interpretable, not
just that the code runs: iso-scope comparators, a shared capacity budget, a
protocol frozen before training, and refusal of combinations that have no
versioned config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.confirmatory_split_mnist import CONFIRMATORY_SEEDS
from experiments.hard_vs_soft import (
    HARD,
    HARD_REPLAY,
    METHOD_PAIRS,
    PAIRED_METHODS,
    SOFT,
    SOFT_REPLAY,
    suite_config,
    suite_protocol,
)
from experiments.split_mnist import (
    SUPPORTED_METHODS,
    _method_budget,
    _method_spec,
    _protects_output,
    _uses_replay,
)


def test_hard_freeze_replay_is_registered() -> None:
    spec = _method_spec(HARD_REPLAY)
    assert spec is not None
    assert spec.slowheat and spec.replay
    assert HARD_REPLAY in SUPPORTED_METHODS


def test_hard_and_soft_arms_share_protection_scope() -> None:
    """The contrast must isolate the regime, not mix in the protection scope."""
    for method in (HARD, HARD_REPLAY, SOFT, SOFT_REPLAY):
        assert _protects_output(method), f"{method} deve proteger a saída"


def test_hard_and_soft_arms_share_capacity_budget() -> None:
    for method in (HARD, HARD_REPLAY, SOFT, SOFT_REPLAY):
        assert _method_budget(method, 0.25) == 0.25


def test_replay_arms_are_paired_with_replay_arms() -> None:
    assert _uses_replay(HARD_REPLAY) and _uses_replay(SOFT_REPLAY)
    assert not _uses_replay(HARD) and not _uses_replay(SOFT)


def test_primary_contrast_is_hard_versus_soft() -> None:
    primary = METHOD_PAIRS[0]
    assert primary.candidate == HARD
    assert primary.reference == SOFT


def test_every_paired_method_is_supported() -> None:
    for method in PAIRED_METHODS:
        assert _method_spec(method) is not None, method


def test_suite_config_refuses_unversioned_combination() -> None:
    with pytest.raises(ValueError, match="não existe config versionado"):
        suite_config("split_mnist", "cnn")


def test_suite_config_returns_requested_backbone() -> None:
    assert suite_config("split_cifar10", "cnn").backbone == "cnn"
    assert suite_config("split_cifar10", "mlp").backbone == "mlp"


def test_protocol_refuses_confirmatory_seeds() -> None:
    config = suite_config("split_mnist", "mlp")
    reserved = [int(next(iter(CONFIRMATORY_SEEDS)))]
    with pytest.raises(ValueError, match="reservadas"):
        suite_protocol(config, reserved, dataset="split_mnist", backbone="mlp")


def test_protocol_refuses_duplicate_seeds() -> None:
    config = suite_config("split_mnist", "mlp")
    with pytest.raises(ValueError, match="duplicatas"):
        suite_protocol(config, [3, 3], dataset="split_mnist", backbone="mlp")


def test_protocol_records_the_declared_question_and_endpoint() -> None:
    config = suite_config("split_mnist", "mlp")
    protocol = suite_protocol(
        config, [11, 12], dataset="split_mnist", backbone="mlp"
    )
    assert protocol["status"] == "exploratory_frozen_before_execution"
    assert protocol["primary_endpoint"] == "final_average_accuracy"
    assert protocol["primary_contrast"] == "Hard vs Soft"
    assert protocol["seeds"] == [11, 12]
    assert "Holm" in protocol["multiplicity"]
    # The protocol must be serializable: it is written before training starts.
    json.dumps(protocol)


def test_protocol_declares_expectations_before_seeing_results() -> None:
    config = suite_config("split_mnist", "mlp")
    protocol = suite_protocol(
        config, [11, 12], dataset="split_mnist", backbone="mlp"
    )
    assert protocol["declared_expectations"], "o protocolo deve declarar a expectativa"
    joined = " ".join(protocol["declared_expectations"]).lower()
    assert "null or reversed" in joined, "um resultado nulo deve ser reportável"


def test_run_target_refuses_a_different_protocol(tmp_path: Path) -> None:
    from experiments.artifacts import write_json_atomic
    from experiments.hard_vs_soft import run_target

    output = tmp_path / "split_mnist_mlp"
    write_json_atomic(output / "hard_vs_soft_protocol.json", {"schema_version": 0})
    with pytest.raises(ValueError, match="protocolo diferente"):
        run_target(
            dataset="split_mnist",
            backbone="mlp",
            seeds=[11, 12],
            data_dir=tmp_path / "data",
            output_dir=output,
            download=False,
        )


def _write_manifest(destination: Path, backbone: str) -> None:
    from experiments.artifacts import write_json_atomic

    write_json_atomic(
        destination / "multi_seed_config.json",
        {
            "seeds": [11, 12],
            "base_config": {
                "backbone": backbone,
                "methods": list(PAIRED_METHODS),
                "optimizer_state_policy": "follow_update",
            },
        },
    )


def test_pair_report_refuses_a_backbone_outside_the_allowed_set(
    tmp_path: Path,
) -> None:
    """The gate exists so a CNN run is never summarized as if it were an MLP."""
    from experiments.dualheat_pairs import summarize_pair_results

    _write_manifest(tmp_path, "cnn")
    with pytest.raises(ValueError, match="backbone=cnn"):
        summarize_pair_results(tmp_path, output_dir=tmp_path)


def test_pair_report_accepts_the_backbone_the_caller_declares(
    tmp_path: Path,
) -> None:
    """A CNN suite may summarize CNN results; it still refuses an MLP source."""
    from experiments.dualheat_pairs import summarize_pair_results

    _write_manifest(tmp_path, "mlp")
    with pytest.raises(ValueError, match="backbone=mlp"):
        summarize_pair_results(tmp_path, output_dir=tmp_path, allowed_backbones=("cnn",))


def test_report_header_names_the_architecture_actually_trained() -> None:
    """A CNN result labelled 'MLP [1]' would misreport where the numbers came from."""
    from experiments.dualheat_pairs import _architecture_line

    cnn = _architecture_line({"backbone": "cnn", "cnn_channels": [32, 64], "hidden_dims": [1]})
    assert "CNN" in cnn and "32" in cnn and "MLP" not in cnn
    assert _architecture_line({"backbone": "mlp", "hidden_dims": [256, 128]}).startswith("MLP")
