import json

import run_all_tests
from experiments.split_mnist_suite import SLOWHEAT_DERPP


def test_default_plan_covers_every_confirmatory_notebook_section():
    args = run_all_tests.parse_args(["--dry-run"])
    plan = run_all_tests.build_run_plan(args)

    assert tuple(plan["sections"]) == run_all_tests.SECTION_NAMES
    assert len(plan["sections"]["confirmation"]["seeds"]) == 20
    assert SLOWHEAT_DERPP in plan["sections"]["slowheat-derpp"]["methods"]
    assert plan["sections"]["ablations"]["replay_memory_per_class"] == [
        5,
        10,
        20,
        50,
        100,
    ]
    assert len(plan["sections"]["split-mnist-generalization"]["class_orders"]) == 5


def test_tiny_imagenet_is_recorded_as_skipped_without_local_dataset(tmp_path):
    return_code = run_all_tests.main(
        [
            "--sections",
            "tiny-imagenet",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert return_code == 0
    index = json.loads((tmp_path / "benchmark_index.json").read_text())
    assert index["status"] == "completed"
    assert index["sections"]["tiny-imagenet"]["status"] == "skipped"
    assert "--tiny-imagenet-dir" in index["sections"]["tiny-imagenet"]["reason"]


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
