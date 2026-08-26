import pytest

import run_all_tests
from experiments.dualheat_pairs import METHOD_PAIRS, PAIRED_METHODS
from experiments.split_mnist_suite import ALL_VISUAL_METHODS, SLOWHEAT_DERPP
from experiments.synthetic_cl import SYNTHETIC_METHODS
from experiments.visual_generalization import CNN_SWEEP_METHODS, CNN_VISUAL_METHODS


def test_dualheat_pairs_section_is_opt_in_and_routes_to_dedicated_runner(tmp_path, monkeypatch):
    assert "dualheat-pairs" not in run_all_tests.DEFAULT_SECTION_NAMES
    args = run_all_tests.parse_args(
        ["--num-seeds", "2", "--sections", "dualheat-pairs", "--no-download"]
    )
    section = run_all_tests.build_run_plan(args)["sections"]["dualheat-pairs"]
    assert section["methods"] == list(PAIRED_METHODS)
    assert section["pairs"] == [
        {"reference": pair.reference, "candidate": pair.candidate}
        for pair in METHOD_PAIRS
    ]
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(run_all_tests, "run_dualheat_pairs", fake_run)
    assert run_all_tests._run_section(
        "dualheat-pairs", args, data_dir=tmp_path, output_dir=tmp_path,
    ) == {"ok": True}
    assert captured["seeds"] == args.baseline_seeds
    assert captured["download"] is False
    assert captured["output_dir"].name == "dualheat_pairs"


def test_dualheat_dedicated_cli_previews_all_mlp_datasets_without_writes(tmp_path, capsys):
    import json

    from experiments.dualheat_pairs import DATASETS, main

    output = tmp_path / "not_created"
    assert main([
        "--datasets", *DATASETS, "--num-seeds", "2", "--dry-run",
        "--output-dir", str(output),
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert set(plan) == set(DATASETS)
    assert plan["split_cifar100"]["config"]["input_dim"] == 3072
    assert plan["permuted_mnist"]["config"]["scenario"] == "domain_incremental"
    assert not output.exists()


def test_dualheat_plan_rejects_reserved_seeds_before_execution():
    args = run_all_tests.parse_args([
        "--num-seeds", "1", "--sections", "dualheat-pairs",
        "--baseline-seeds", str(run_all_tests.CONFIRMATORY_SEEDS[0]),
    ])
    with pytest.raises(ValueError, match="reservadas"):
        run_all_tests.build_run_plan(args)


def test_default_plan_covers_every_confirmatory_notebook_section():
    args = run_all_tests.parse_args(["--num-seeds", "10", "--dry-run"])
    plan = run_all_tests.build_run_plan(args)

    assert tuple(plan["sections"]) == run_all_tests.DEFAULT_SECTION_NAMES
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
    assert "class_orders" not in plan["sections"]["split-mnist-generalization"]
    assert plan["sections"]["split-mnist-generalization"]["architectures"] == [
        [256, 128],
        [512, 256],
        [512, 512, 256],
    ]
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
    assert "split-cifar10-cnn" not in plan["sections"]
    assert "split-cifar10-cnn-sweep" not in plan["sections"]


def test_cnn_cifar_plan_declares_real_image_backbone():
    args = run_all_tests.parse_args(
        ["--num-seeds", "3", "--sections", "split-cifar10-cnn", "--dry-run"]
    )
    plan = run_all_tests.build_run_plan(args)
    section = plan["sections"]["split-cifar10-cnn"]

    assert section["methods"] == list(CNN_VISUAL_METHODS)
    assert section["protocol"] == {
        "scenario": "class_incremental",
        "task_count": 5,
        "classes_per_task": 2,
        "class_count": 10,
        "inference_task_id": False,
        "backbone": "cnn",
        "image_shape": [3, 32, 32],
        "channels": [32, 64],
        "pooled_size": [2, 2],
        "epochs_per_task": 5,
    }


def test_cnn_sweep_plan_declares_preselected_grid():
    args = run_all_tests.parse_args(
        [
            "--num-seeds",
            "10",
            "--sections",
            "split-cifar10-cnn-sweep",
            "--dry-run",
        ]
    )
    plan = run_all_tests.build_run_plan(args)
    section = plan["sections"]["split-cifar10-cnn-sweep"]

    assert section["methods"] == list(CNN_SWEEP_METHODS)
    assert len(section["seeds"]) == 10
    assert section["protocol"]["backbone"] == "cnn"
    assert section["output_dir"].endswith("split_cifar10_cnn_sweep")


def test_all_datasets_all_methods_plan_covers_every_compatible_cross_section():
    args = run_all_tests.parse_args(
        ["--num-seeds", "10", "--all-datasets-all-methods", "--dry-run"]
    )
    plan = run_all_tests.build_run_plan(args)

    assert tuple(plan["sections"]) == run_all_tests.ALL_DATASET_METHOD_SECTIONS
    assert plan["sections"]["synthetic-all-methods"]["methods"] == list(
        SYNTHETIC_METHODS
    )
    for name in (
        "split-mnist-all-methods",
        "permuted-mnist",
        "split-cifar10",
        "split-cifar100",
    ):
        assert plan["sections"][name]["methods"] == list(ALL_VISUAL_METHODS)


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
            "--num-seeds",
            "2",
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
        [
            "--sections",
            "split-cifar10",
            "split-cifar100",
            "--num-seeds",
            "10",
        ]
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


def test_cnn_cifar_section_uses_visual_generalization_runner(tmp_path, monkeypatch):
    captured = {}

    def fake_run(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(run_all_tests, "run_visual_generalization", fake_run)
    args = run_all_tests.parse_args(
        ["--sections", "split-cifar10-cnn", "--num-seeds", "1"]
    )

    result = run_all_tests._run_section(
        "split-cifar10-cnn",
        args,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "results",
    )

    assert result == {"ok": True}
    assert captured["name"] == "split_cifar10_cnn"
    assert captured["output_dir"].name == "split_cifar10_cnn"


def test_cnn_sweep_section_uses_its_own_output_directory(tmp_path, monkeypatch):
    captured = {}

    def fake_run(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(run_all_tests, "run_visual_generalization", fake_run)
    args = run_all_tests.parse_args(
        ["--sections", "split-cifar10-cnn-sweep", "--num-seeds", "10"]
    )

    result = run_all_tests._run_section(
        "split-cifar10-cnn-sweep",
        args,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "results",
    )

    assert result == {"ok": True}
    assert captured["name"] == "split_cifar10_cnn_sweep"
    assert captured["output_dir"].name == "split_cifar10_cnn_sweep"


def test_synthetic_all_methods_section_uses_secondary_seeds(tmp_path, monkeypatch):
    captured = {}

    def fake_run(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(run_all_tests, "run_multi_seed", fake_run)
    args = run_all_tests.parse_args(
        [
            "--sections",
            "synthetic-all-methods",
            "--num-seeds",
            "2",
            "--baseline-seeds",
            "7",
            "9",
        ]
    )

    result = run_all_tests._run_section(
        "synthetic-all-methods",
        args,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "results",
    )

    assert result == {"ok": True}
    assert tuple(captured["config"].methods) == SYNTHETIC_METHODS
    assert captured["seeds"] == [7, 9]
    assert captured["output_dir"].name == "synthetic_all_methods"


def test_num_seeds_generates_distinct_reproducible_random_seeds():
    first = run_all_tests.parse_args(["--num-seeds", "10", "--dry-run"])
    second = run_all_tests.parse_args(["--num-seeds", "10", "--dry-run"])

    assert len(first.baseline_seeds) == 10
    assert len(set(first.baseline_seeds)) == 10
    assert first.baseline_seeds == second.baseline_seeds
    assert first.baseline_seeds != [10]


def test_num_seeds_is_required():
    with pytest.raises(SystemExit):
        run_all_tests.parse_args(["--dry-run"])


def test_custom_seed_count_must_match_num_seeds():
    with pytest.raises(SystemExit):
        run_all_tests.parse_args(
            ["--num-seeds", "3", "--baseline-seeds", "7", "9"]
        )
