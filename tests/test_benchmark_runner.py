import json

import run_all_tests
from experiments.split_mnist_suite import SLOWHEAT_DERPP


def test_default_plan_covers_every_confirmatory_notebook_section():
    args = run_all_tests.parse_args(["--dry-run"])
    plan = run_all_tests.build_run_plan(args)

    assert tuple(plan["sections"]) == run_all_tests.SECTION_NAMES
    assert len(plan["sections"]["confirmation"]["seeds"]) == 20
    for name, section in plan["sections"].items():
        if name != "confirmation":
            assert len(section["seeds"]) == 10
    assert SLOWHEAT_DERPP in plan["sections"]["slowheat-derpp"]["methods"]
    assert plan["sections"]["ablations"]["replay_memory_per_class"] == [
        5,
        10,
        20,
        50,
        100,
    ]
    assert len(plan["sections"]["split-mnist-generalization"]["class_orders"]) == 5
    core50 = plan["sections"]["core50"]
    assert core50["seeds"] == list(range(10))
    assert core50["methods"] == list(run_all_tests.SLOWHEAT_DERPP_METHODS)
    assert core50["protocol"] == {
        "scenario": "new_classes_class_incremental",
        "official_runs": list(range(10)),
        "task_count": 9,
        "task_class_counts": [10, 5, 5, 5, 5, 5, 5, 5, 5],
        "class_count": 50,
        "inference_task_id": False,
        "primary_evaluation": "class_il_seen_classes",
        "secondary_evaluation": "task_il_diagnostic",
        "paired_references": ["replay", "derpp"],
    }


def test_core50_is_recorded_as_skipped_without_local_dataset(tmp_path):
    return_code = run_all_tests.main(
        [
            "--sections",
            "core50",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert return_code == 0
    index = json.loads((tmp_path / "benchmark_index.json").read_text())
    assert index["status"] == "completed"
    assert index["sections"]["core50"]["status"] == "skipped"
    assert "--core50-dir" in index["sections"]["core50"]["reason"]


def test_section_runner_enables_resume_and_forwards_secondary_seeds(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(run_all_tests, "run_slowheat_derpp_test", fake_run)
    args = run_all_tests.parse_args(
        [
            "--sections",
            "slowheat-derpp",
            "--baseline-seeds",
            "7",
            "9",
        ]
    )

    result = run_all_tests._run_section(
        "slowheat-derpp",
        args,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "results",
    )

    assert result == {"ok": True}
    assert captured["seeds"] == [7, 9]
    assert captured["resume"] is True
    assert captured["output_dir"].name == "slowheat_derpp_exploratory"


def test_core50_section_uses_all_ten_official_runs(tmp_path, monkeypatch):
    captured = {}

    def fake_run(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(run_all_tests, "run_visual_generalization", fake_run)
    args = run_all_tests.parse_args(
        ["--sections", "core50", "--core50-dir", str(tmp_path / "core50")]
    )

    result = run_all_tests._run_section(
        "core50",
        args,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "results",
    )

    assert result == {"ok": True}
    assert captured["name"] == "core50"
    assert captured["seeds"] == list(range(10))
    assert captured["download"] is False
    assert captured["data_dir"] == (tmp_path / "core50").resolve()
