from __future__ import annotations

import json
from pathlib import Path

import pytest

import show_cache_results as cli

CANDIDATE = "slowheat_replay_hidden_beta_30_budget_0.25"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _summary(mean: float, half_width: float = 0.01, seeds: int = 2):
    return {
        "mean": mean,
        "std": 0.02,
        "ci95_normal_half_width": half_width,
        "values": [{"seed": index, "value": mean} for index in range(seeds)],
    }


def _method_summary(accuracy: float, forgetting: float):
    return {
        "final_average_accuracy": _summary(accuracy),
        "average_forgetting": _summary(forgetting),
    }


@pytest.fixture
def sweep_report(tmp_path: Path) -> Path:
    root = tmp_path / "sweep"
    _write_json(
        root / "sweep_report.json",
        {
            "seeds": [11, 22],
            "summaries": {
                "split_cifar10": {
                    "vanilla": {"none": _method_summary(0.1, 0.9)},
                    "replay": {
                        "first": _method_summary(0.70, 0.20),
                        "hybrid": _method_summary(0.75, 0.15),
                    },
                    CANDIDATE: {
                        "first": _method_summary(0.72, 0.18),
                        "hybrid": _method_summary(0.78, 0.12),
                    },
                },
                "split_mnist": {
                    "replay": {"hybrid": _method_summary(0.90, 0.05)},
                    CANDIDATE: {"hybrid": _method_summary(0.92, 0.04)},
                },
            },
        },
    )
    return root


def test_cli_without_filters_shows_all_cache_results_only(sweep_report, capsys):
    assert cli.main([str(sweep_report)]) == 0

    captured = capsys.readouterr()
    assert "split_cifar10" in captured.out
    assert "split_mnist" in captured.out
    assert "Replay" in captured.out
    assert "SlowHeat+Replay" in captured.out
    assert "vanilla" not in captured.out.lower()
    assert "75.00 [74.00, 76.00]" in captured.out
    assert captured.err == ""


def test_cli_filters_cache_across_benchmarks(sweep_report, capsys):
    assert cli.main([str(sweep_report), "--cache", "hybrid"]) == 0

    output = capsys.readouterr().out
    assert "split_cifar10" in output
    assert "split_mnist" in output
    assert "first" not in output


def test_cli_filters_benchmark_and_cache(sweep_report, capsys):
    assert cli.main(
        [
            str(sweep_report),
            "--benchmark",
            "split_cifar10",
            "--cache",
            "first",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "split_cifar10" in output
    assert "split_mnist" not in output
    assert output.count("first") == 2


def test_accuracy_high_selects_best_cache_per_benchmark_and_learner(
    sweep_report, capsys
):
    assert cli.main([str(sweep_report), "--accuracy", "--high"]) == 0

    output = capsys.readouterr().out
    assert output.count("hybrid") == 4
    assert "first" not in output
    assert "78.00 [77.00, 79.00]" in output
    assert "92.00 [91.00, 93.00]" in output


def test_forget_low_selects_smallest_forgetting(sweep_report, capsys):
    assert cli.main(
        [str(sweep_report), "--benchmark", "split_cifar10", "--forget", "--low"]
    ) == 0

    output = capsys.readouterr().out
    assert output.count("hybrid") == 2
    assert "first" not in output
    assert "12.00 [11.00, 13.00]" in output


def test_metric_and_direction_must_be_used_together(sweep_report, capsys):
    assert cli.main([str(sweep_report), "--accuracy"]) == 2
    assert "junto com --high ou --low" in capsys.readouterr().err

    assert cli.main([str(sweep_report), "--low"]) == 2
    assert "junto com --high ou --low" in capsys.readouterr().err


def test_benchmark_and_cache_directories_are_inferred(tmp_path, capsys):
    benchmark = tmp_path / "split_cifar10_cnn"
    run_dir = benchmark / "loss"
    _write_json(
        run_dir / "aggregate.json",
        {
            "methods": {
                "replay": _method_summary(0.65, 0.25),
                CANDIDATE: _method_summary(0.68, 0.22),
                "vanilla": _method_summary(0.1, 0.9),
            }
        },
    )
    _write_json(run_dir / "multi_seed_config.json", {"seeds": [1, 2, 3]})

    assert cli.main([str(benchmark)]) == 0
    benchmark_output = capsys.readouterr().out
    assert "split_cifar10_cnn" in benchmark_output
    assert "loss" in benchmark_output
    assert "  3  " in benchmark_output

    assert cli.main([str(run_dir)]) == 0
    cache_output = capsys.readouterr().out
    assert "split_cifar10_cnn" in cache_output
    assert "loss" in cache_output


def test_direct_aggregate_defaults_to_first_cache(tmp_path, capsys):
    benchmark = tmp_path / "split_cifar100"
    _write_json(
        benchmark / "aggregate.json",
        {"methods": {"replay": _method_summary(0.55, 0.35)}},
    )
    _write_json(
        benchmark / "multi_seed_config.json",
        {"base_config": {"methods": ["replay"]}, "seeds": [7, 9]},
    )

    assert cli.main([str(benchmark)]) == 0

    output = capsys.readouterr().out
    assert "split_cifar100" in output
    assert "first" in output


def test_partial_seeds_are_aggregated_with_warning(tmp_path, capsys):
    run_dir = tmp_path / "permuted_mnist" / "representative"
    for seed, accuracy, forgetting in ((1, 0.7, 0.1), (2, 0.9, 0.3)):
        _write_json(
            run_dir / f"seed_{seed}" / "results.json",
            {
                "replay": {
                    "metrics": {
                        "final_average_accuracy": accuracy,
                        "average_forgetting": forgetting,
                    }
                }
            },
        )

    assert cli.main([str(run_dir)]) == 0

    captured = capsys.readouterr()
    assert "80.00 [60.40, 99.60]" in captured.out
    assert "20.00 [0.40, 39.60]" in captured.out
    assert "resultados parciais" in captured.err


@pytest.mark.parametrize("problem", ["missing", "invalid_json", "no_cache"])
def test_cli_reports_invalid_or_missing_results(tmp_path, capsys, problem):
    root = tmp_path / "results"
    if problem == "invalid_json":
        root.mkdir()
        (root / "sweep_report.json").write_text("{broken", encoding="utf-8")
    elif problem == "no_cache":
        _write_json(
            root / "aggregate.json",
            {"methods": {"vanilla": _method_summary(0.5, 0.5)}},
        )

    assert cli.main([str(root)]) == 2

    error = capsys.readouterr().err
    assert error.startswith("erro:")


def test_unknown_benchmark_lists_available_values(sweep_report, capsys):
    assert cli.main([str(sweep_report), "--benchmark", "unknown"]) == 2

    error = capsys.readouterr().err
    assert "benchmark desconhecido" in error
    assert "split_cifar10" in error
    assert "split_mnist" in error
