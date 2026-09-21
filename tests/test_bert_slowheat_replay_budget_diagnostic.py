import pytest

pytest.importorskip("transformers")

from experiments import bert_slowheat_replay_budget_diagnostic as budget  # noqa: E402
from experiments.split_clinc150 import SplitCLINC150Config  # noqa: E402


def _fake_result(*, replay_memory_bytes: int = 1_000):
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
        "replay_memory_bytes": replay_memory_bytes,
        "peak_memory": {
            "peak_memory_bytes": 2**20,
            "peak_cuda_reserved_bytes": 2 * 2**20,
        },
    }


def test_budget_diagnostic_has_exactly_the_predeclared_six_conditions():
    conditions = budget.replay_budget_conditions()

    assert [(item.name, item.method, item.replay_per_class) for item in conditions] == [
        ("replay_memory_1", "replay", 1),
        ("slowheat_hard_replay_memory_1", "slowheat_ffn_attention_replay", 1),
        ("replay_memory_5", "replay", 5),
        ("slowheat_hard_replay_memory_5", "slowheat_ffn_attention_replay", 5),
        ("replay_memory_10", "replay", 10),
        ("slowheat_hard_replay_memory_10", "slowheat_ffn_attention_replay", 10),
    ]
    assert [item.mask_mode for item in conditions] == [
        "soft",
        "hard",
        "soft",
        "hard",
        "soft",
        "hard",
    ]


def test_run_budget_diagnostic_builds_paired_validation_only_configs(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run(config, tasks, **kwargs):
        calls.append((config, kwargs["output_dir"]))
        return {
            config.methods[0]: _fake_result(
                replay_memory_bytes=config.replay_per_class * 1_000
            )
        }

    monkeypatch.setattr(budget, "run_split_clinc150", fake_run)
    monkeypatch.setattr(budget, "write_environment_manifest", lambda *a, **k: None)

    raw = budget.run_replay_budget_diagnostic(
        SplitCLINC150Config(device="cpu"),
        tasks=[object()] * 10,
        metadata={},
        seeds=[11, 22],
        output_dir=tmp_path,
    )

    assert len(calls) == 12
    assert set(raw) == {11, 22}
    assert [config.replay_per_class for config, _ in calls[:6]] == [
        1,
        1,
        5,
        5,
        10,
        10,
    ]
    assert all(config.task_limit == 2 for config, _ in calls)
    assert all(config.replay_batch_size == 2 for config, _ in calls)
    assert all(config.evaluate_test is False for config, _ in calls)
    assert calls[0][1] == tmp_path / "seed_11" / "replay_memory_1"
    assert calls[-1][1] == (
        tmp_path / "seed_22" / "slowheat_hard_replay_memory_10"
    )
    assert (tmp_path / "diagnostic_summary.json").is_file()
    assert (tmp_path / "diagnostic_table.md").is_file()


def test_budget_summary_declares_primary_budget_and_three_fixed_contrasts():
    raw = {}
    for seed in (11, 22):
        raw[seed] = {}
        for condition in budget.replay_budget_conditions():
            raw[seed][condition.name] = _fake_result(
                replay_memory_bytes=condition.replay_per_class * 1_000
            )

    summary = budget.summarize_replay_budget_diagnostic(raw)

    assert summary["primary_endpoint"] == "final_average_accuracy"
    assert summary["primary_budget_per_class"] == 1
    assert summary["primary_contrast"] == "hard_replay_minus_replay_memory_1"
    assert set(summary["paired_contrasts"]) == {
        "hard_replay_minus_replay_memory_1",
        "hard_replay_minus_replay_memory_5",
        "hard_replay_minus_replay_memory_10",
    }
    assert summary["paired_contrasts"][
        "hard_replay_minus_replay_memory_1"
    ]["final_average_accuracy"]["by_seed"] == {"11": 0.0, "22": 0.0}


def test_budget_summary_rejects_unequal_memory_within_a_pair():
    raw = {
        11: {
            condition.name: _fake_result(
                replay_memory_bytes=condition.replay_per_class * 1_000
            )
            for condition in budget.replay_budget_conditions()
        }
    }
    raw[11]["slowheat_hard_replay_memory_1"]["replay_memory_bytes"] += 1

    with pytest.raises(ValueError, match="memória de replay desigual"):
        budget.summarize_replay_budget_diagnostic(raw)


def test_budget_markdown_labels_primary_and_secondary_budgets():
    raw = {
        11: {
            condition.name: _fake_result(
                replay_memory_bytes=condition.replay_per_class * 1_000
            )
            for condition in budget.replay_budget_conditions()
        }
    }

    rendered = budget.replay_budget_markdown(
        budget.summarize_replay_budget_diagnostic(raw)
    )

    assert "Primary budget: 1 replay example per class" in rendered
    assert "Secondary budgets: 5 and 10" in rendered
    assert "Final average (%)" in rendered
    assert "Actual replay memory (MiB)" in rendered
    assert "hard_replay_minus_replay_memory_1" in rendered
    assert "Do not select the best-looking secondary budget" in rendered


def test_budget_cli_defaults_are_frozen():
    args = budget.build_parser().parse_args([])

    assert args.seeds == [11, 22, 33]
    assert args.batch_size == 2
    assert args.replay_batch_size == 2
    assert args.task_limit == 2
    assert args.evaluate_test is False
    assert not hasattr(args, "replay_per_class")


def test_budget_cli_rejects_protocol_escape_hatches():
    parser = budget.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--evaluate-test"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--replay-per-class", "20"])


def test_budget_main_loads_only_train_and_validation(monkeypatch, tmp_path):
    received = []
    monkeypatch.setattr(
        budget,
        "load_clinc150_tasks",
        lambda config, *, include_test: (
            received.append(include_test) or ([object()] * 10, {})
        ),
    )
    monkeypatch.setattr(
        budget,
        "run_replay_budget_diagnostic",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["bert_slowheat_replay_budget_diagnostic", "--output-dir", str(tmp_path)],
    )

    budget.main()

    assert received == [False]
