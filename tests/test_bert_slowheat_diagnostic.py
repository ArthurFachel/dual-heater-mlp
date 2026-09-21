import json

import pytest

pytest.importorskip("transformers")

from experiments import bert_slowheat_diagnostic as diagnostic  # noqa: E402
from experiments.split_clinc150 import SplitCLINC150Config  # noqa: E402


def test_diagnostic_matrix_has_exactly_the_predeclared_six_conditions():
    conditions = diagnostic.diagnostic_conditions()

    assert [condition.name for condition in conditions] == [
        "vanilla",
        "slowheat_beta_3",
        "slowheat_beta_10",
        "slowheat_beta_30",
        "slowheat_hard",
        "slowheat_random_hard",
    ]
    assert [condition.method for condition in conditions] == [
        "vanilla",
        "slowheat_ffn_attention",
        "slowheat_ffn_attention",
        "slowheat_ffn_attention",
        "slowheat_ffn_attention",
        "slowheat_ffn_attention",
    ]
    assert [condition.slow_strength for condition in conditions] == [
        3.0,
        3.0,
        10.0,
        30.0,
        3.0,
        3.0,
    ]
    assert [condition.mask_mode for condition in conditions] == [
        "soft",
        "soft",
        "soft",
        "soft",
        "hard",
        "random_hard",
    ]


def _fake_result(
    task1_after=0.8,
    task1_retained=0.6,
    task2=0.9,
    drift=0.0,
    elapsed=1.0,
    peak=1024,
):
    return {
        "validation_accuracy_matrix": [
            [task1_after, None],
            [task1_retained, task2],
        ],
        "validation_metrics": {
            "final_average_accuracy": (task1_retained + task2) / 2,
            "average_forgetting": task1_after - task1_retained,
            "backward_transfer": task1_retained - task1_after,
            "forward_transfer": None,
            "per_task_forgetting": [task1_after - task1_retained, 0.0],
        },
        "parameter_drift_history": [
            {
                "stage": 0,
                "protected_count": 0,
                "protected_rms": None,
                "protected_max_abs": None,
                "plastic_count": 0,
                "plastic_rms": None,
                "plastic_max_abs": None,
            },
            {
                "stage": 1,
                "protected_count": 4,
                "protected_rms": drift,
                "protected_max_abs": drift,
                "plastic_count": 8,
                "plastic_rms": drift + 0.1,
                "plastic_max_abs": drift + 0.2,
            },
        ],
        "elapsed_seconds": elapsed,
        "tokens_processed": 1_000,
        "replay_memory_bytes": 2 * 1024 * 1024,
        "peak_memory": {
            "peak_memory_bytes": peak,
            "peak_cuda_reserved_bytes": peak + 1024,
        },
    }


def test_run_diagnostic_builds_paired_two_task_configs(monkeypatch, tmp_path):
    calls = []

    def fake_run(config, tasks, **kwargs):
        calls.append((config, kwargs["output_dir"]))
        return {config.methods[0]: _fake_result()}

    monkeypatch.setattr(diagnostic, "run_split_clinc150", fake_run)
    monkeypatch.setattr(diagnostic, "write_environment_manifest", lambda *a, **k: None)
    base = SplitCLINC150Config(device="cpu")
    raw = diagnostic.run_diagnostic(
        base,
        tasks=[object()] * 10,
        metadata={},
        seeds=[11, 22],
        output_dir=tmp_path,
    )

    assert len(calls) == 12
    assert set(raw) == {11, 22}
    assert all(config.task_limit == 2 for config, _ in calls)
    assert all(config.evaluate_test is False for config, _ in calls)
    assert calls[0][1] == tmp_path / "seed_11" / "vanilla"
    assert calls[-1][1] == tmp_path / "seed_22" / "slowheat_random_hard"
    assert (tmp_path / "diagnostic_summary.json").is_file()
    assert (tmp_path / "diagnostic_table.md").is_file()


def test_summarize_diagnostic_reports_retention_drift_time_and_total_memory():
    names = [condition.name for condition in diagnostic.diagnostic_conditions()]
    raw = {
        11: {
            name: _fake_result(0.8, 0.6, 0.9, 0.0, 10.0, 1024 * 1024)
            for name in names
        },
        22: {
            name: _fake_result(0.9, 0.7, 0.8, 0.0, 14.0, 2 * 1024 * 1024)
            for name in names
        },
    }

    summary = diagnostic.summarize_diagnostic(raw)
    vanilla = summary["conditions"]["vanilla"]

    assert vanilla["task1_acquisition"]["mean"] == pytest.approx(0.85)
    assert vanilla["task1_retention"]["mean"] == pytest.approx(0.65)
    assert vanilla["task1_forgetting"]["mean"] == pytest.approx(0.20)
    assert vanilla["task2_acquisition"]["mean"] == pytest.approx(0.85)
    assert vanilla["final_average_accuracy"]["mean"] == pytest.approx(0.75)
    assert vanilla["backward_transfer"]["mean"] == pytest.approx(-0.20)
    assert vanilla["elapsed_seconds"]["mean"] == pytest.approx(12.0)
    assert vanilla["tokens_processed"]["mean"] == pytest.approx(1_000.0)
    assert vanilla["tokens_per_second"]["mean"] == pytest.approx(
        (100.0 + 1000.0 / 14.0) / 2.0
    )
    assert vanilla["replay_memory_mib"]["mean"] == pytest.approx(2.0)
    assert vanilla["peak_memory_mib"]["mean"] == pytest.approx(1.5)
    assert "paired_vs_vanilla" in summary
    assert "raw_by_seed" in summary
    assert set(summary["paired_contrasts"]) == {
        "learned_hard_minus_random_hard",
        "hard_minus_beta_3",
    }
    assert summary["paired_vs_vanilla"]["slowheat_beta_3"][
        "task1_retention"
    ]["by_seed"] == {"11": 0.0, "22": 0.0}


def test_summarize_diagnostic_accepts_cpu_memory_without_reserved_peak():
    names = [condition.name for condition in diagnostic.diagnostic_conditions()]
    result = _fake_result()
    result["peak_memory"]["peak_cuda_reserved_bytes"] = None
    raw = {11: {name: result for name in names}}

    summary = diagnostic.summarize_diagnostic(raw)

    assert summary["conditions"]["vanilla"]["peak_reserved_mib"] == {
        "n": 0,
        "mean": None,
        "sample_std": None,
    }


def test_diagnostic_markdown_shows_final_average_and_scientific_drift():
    names = [condition.name for condition in diagnostic.diagnostic_conditions()]
    raw = {
        11: {
            name: _fake_result(
                task1_after=0.8,
                task1_retained=0.6,
                task2=0.9,
                drift=8.038814675802855e-4,
                elapsed=10.0,
                peak=1024 * 1024,
            )
            for name in names
        }
    }

    rendered = diagnostic.diagnostic_markdown(
        diagnostic.summarize_diagnostic(raw)
    )

    assert "Final average (%)" in rendered
    assert "75.00 ± 0.00" in rendered
    assert "8.039e-04 ± 0.000e+00" in rendered


def test_load_completed_diagnostic_reads_nested_raw_results(tmp_path):
    for seed in (11, 22):
        for condition in diagnostic.diagnostic_conditions():
            method_dir = tmp_path / f"seed_{seed}" / condition.name / condition.method
            method_dir.mkdir(parents=True)
            (method_dir / "results.json").write_text(
                json.dumps(_fake_result()), encoding="utf-8"
            )

    raw = diagnostic.load_completed_diagnostic(tmp_path)

    assert set(raw) == {11, 22}
    assert set(raw[11]) == {
        condition.name for condition in diagnostic.diagnostic_conditions()
    }


def test_summarize_from_does_not_load_dataset(monkeypatch, tmp_path):
    for seed in (11,):
        for condition in diagnostic.diagnostic_conditions():
            method_dir = tmp_path / f"seed_{seed}" / condition.name / condition.method
            method_dir.mkdir(parents=True)
            (method_dir / "results.json").write_text(
                json.dumps(_fake_result()), encoding="utf-8"
            )
    monkeypatch.setattr(
        diagnostic,
        "load_clinc150_tasks",
        lambda *args, **kwargs: pytest.fail("dataset must not be loaded"),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["bert_slowheat_diagnostic", "--summarize-from", str(tmp_path)],
    )

    diagnostic.main()

    assert (tmp_path / "diagnostic_summary.json").is_file()
    assert (tmp_path / "diagnostic_table.md").is_file()


def test_diagnostic_cli_defaults_to_the_predeclared_protocol():
    args = diagnostic.build_parser().parse_args([])

    assert args.seeds == [11, 22, 33]
    assert args.task_limit == 2
    assert args.batch_size == 2
    assert args.evaluate_test is False


def test_diagnostic_cli_cannot_open_test_split():
    parser = diagnostic.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--evaluate-test"])


def test_diagnostic_main_loads_only_train_and_validation(monkeypatch, tmp_path):
    received = []
    monkeypatch.setattr(
        diagnostic,
        "load_clinc150_tasks",
        lambda config, *, include_test: (
            received.append(include_test) or ([object()] * 10, {})
        ),
    )
    monkeypatch.setattr(diagnostic, "run_diagnostic", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "sys.argv",
        ["bert_slowheat_diagnostic", "--output-dir", str(tmp_path)],
    )

    diagnostic.main()

    assert received == [False]