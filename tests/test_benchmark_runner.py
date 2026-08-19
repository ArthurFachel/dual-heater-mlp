import run_all_tests
from experiments.split_mnist_suite import ALL_VISUAL_METHODS, SLOWHEAT_DERPP


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
    assert plan["sections"]["split-cifar10"]["protocol"] == {
        "scenario": "class_incremental",
        "task_count": 5,
        "classes_per_task": 2,
        "class_count": 10,
        "inference_task_id": False,
    }
    assert plan["sections"]["split-cifar100"]["protocol"] == {
        "scenario": "class_incremental",
        "task_count": 10,
        "classes_per_task": 10,
        "class_count": 100,
        "inference_task_id": False,
    }
    assert plan["sections"]["split-cifar10"]["methods"] == list(
        ALL_VISUAL_METHODS
    )
    assert plan["sections"]["split-cifar100"]["methods"] == list(
        ALL_VISUAL_METHODS
    )


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


def test_cifar_sections_use_visual_generalization_runner(tmp_path, monkeypatch):
    captured = []

    def fake_run(name, **kwargs):
        captured.append((name, kwargs))
        return {"ok": True}

    monkeypatch.setattr(run_all_tests, "run_visual_generalization", fake_run)
    args = run_all_tests.parse_args(
        ["--sections", "split-cifar10", "split-cifar100"]
    )

    for name in ("split-cifar10", "split-cifar100"):
        result = run_all_tests._run_section(
            name,
            args,
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "results",
        )
        assert result == {"ok": True}

    assert [name for name, _ in captured] == ["split_cifar10", "split_cifar100"]
    assert all(kwargs["download"] is True for _, kwargs in captured)
