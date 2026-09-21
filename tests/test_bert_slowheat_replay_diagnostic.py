import pytest

pytest.importorskip("transformers")

from experiments import bert_slowheat_replay_diagnostic as followup  # noqa: E402
from experiments.split_clinc150 import SplitCLINC150Config  # noqa: E402


def _fake_result():
    return {
        "validation_accuracy_matrix": [[0.8, None], [0.6, 0.9]],
        "validation_metrics": {
            "final_average_accuracy": 0.75,
            "average_forgetting": 0.2,
            "backward_transfer": -0.2,
            "forward_transfer": None,
            "per_task_forgetting": [0.2, 0.0],
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
                "protected_rms": 0.0,
                "protected_max_abs": 0.0,
                "plastic_count": 8,
                "plastic_rms": 0.1,
                "plastic_max_abs": 0.2,
            },
        ],
        "elapsed_seconds": 10.0,
        "tokens_processed": 1_000,
        "replay_memory_bytes": 0,
        "peak_memory": {
            "peak_memory_bytes": 1024 * 1024,
            "peak_cuda_reserved_bytes": 2 * 1024 * 1024,
        },
    }


def test_replay_diagnostic_is_the_predeclared_two_by_two_factorial():
    conditions = followup.replay_diagnostic_conditions()

    assert [condition.name for condition in conditions] == [
        "vanilla",
        "replay",
        "slowheat_hard",
        "slowheat_hard_replay",
    ]
    assert [condition.method for condition in conditions] == [
        "vanilla",
        "replay",
        "slowheat_ffn_attention",
        "slowheat_ffn_attention_replay",
    ]
    assert [condition.mask_mode for condition in conditions] == [
        "soft",
        "soft",
        "hard",
        "hard",
    ]


def test_run_replay_diagnostic_builds_paired_validation_only_configs(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run(config, tasks, **kwargs):
        calls.append((config, kwargs["output_dir"]))
        return {config.methods[0]: _fake_result()}

    monkeypatch.setattr(followup, "run_split_clinc150", fake_run)
    monkeypatch.setattr(followup, "write_environment_manifest", lambda *a, **k: None)

    raw = followup.run_replay_diagnostic(
        SplitCLINC150Config(device="cpu"),
        tasks=[object()] * 10,
        metadata={},
        seeds=[11, 22],
        output_dir=tmp_path,
    )

    assert len(calls) == 8
    assert set(raw) == {11, 22}
    assert all(config.task_limit == 2 for config, _ in calls)
    assert all(config.evaluate_test is False for config, _ in calls)
    assert calls[0][1] == tmp_path / "seed_11" / "vanilla"
    assert calls[-1][1] == tmp_path / "seed_22" / "slowheat_hard_replay"
    assert (tmp_path / "diagnostic_summary.json").is_file()
    assert (tmp_path / "diagnostic_table.md").is_file()


def test_replay_summary_contains_predeclared_paired_contrasts():
    names = [condition.name for condition in followup.replay_diagnostic_conditions()]
    raw = {11: {name: _fake_result() for name in names}}

    summary = followup.summarize_replay_diagnostic(raw)

    assert summary["primary_endpoint"] == "final_average_accuracy"
    assert set(summary["paired_contrasts"]) == {
        "replay_minus_vanilla",
        "hard_minus_vanilla",
        "hard_replay_minus_replay",
        "hard_replay_minus_hard",
    }
    assert summary["paired_contrasts"]["hard_replay_minus_replay"][
        "final_average_accuracy"
    ]["by_seed"] == {"11": 0.0}


def test_replay_markdown_reports_resources_and_predeclared_decision_gate():
    names = [condition.name for condition in followup.replay_diagnostic_conditions()]
    raw = {11: {name: _fake_result() for name in names}}

    rendered = followup.replay_diagnostic_markdown(
        followup.summarize_replay_diagnostic(raw)
    )

    assert "Final average (%)" in rendered
    assert "Replay memory (MiB)" in rendered
    assert "Tokens/s" in rendered
    assert "hard_replay_minus_replay" in rendered
    assert "drops by more than 2 percentage points" in rendered


def test_replay_diagnostic_cli_defaults_are_frozen():
    args = followup.build_parser().parse_args([])

    assert args.seeds == [11, 22, 33]
    assert args.batch_size == 2
    assert args.replay_batch_size == 2
    assert args.replay_per_class == 20
    assert args.task_limit == 2
    assert args.evaluate_test is False


def test_replay_diagnostic_cli_rejects_test_evaluation():
    with pytest.raises(SystemExit):
        followup.build_parser().parse_args(["--evaluate-test"])
