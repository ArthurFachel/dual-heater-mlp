from dataclasses import replace

import torch

import experiments.visual_generalization as visual
from experiments.split_mnist_suite import SLOWHEAT_DERPP_METHODS


def test_generalization_configs_preserve_scenario_semantics():
    configs = visual.generalization_configs()

    assert set(configs) == {"permuted_mnist", "core50"}
    assert configs["permuted_mnist"].scenario == "domain_incremental"
    assert configs["permuted_mnist"].task_count == 5
    core50 = configs["core50"]
    assert core50.scenario == "class_incremental"
    assert core50.input_dim == 3 * 64 * 64
    assert core50.task_count == 9
    assert core50.task_class_counts == visual.CORE50_TASK_CLASS_COUNTS
    assert len(core50.class_order) == 50
    for config in configs.values():
        assert config.methods == SLOWHEAT_DERPP_METHODS
        config.validate()


def test_core50_loader_uses_official_nc_batches_and_fixed_test(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "core50_128x128"
    run_dir = dataset_root / "batches_filelists" / "NC_inc" / "Run0"
    run_dir.mkdir(parents=True)
    test_lines: list[str] = []
    label_offset = 0
    for task_index, class_count in enumerate(visual.CORE50_TASK_CLASS_COUNTS):
        train_lines: list[str] = []
        for label in range(label_offset, label_offset + class_count):
            for sample in range(2):
                relative = f"s1/o{label + 1}/train_{label}_{sample}.png"
                image_path = dataset_root / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.touch()
                train_lines.append(f"{relative} {label}\n")
            test_relative = f"s3/o{label + 1}/test_{label}.png"
            test_path = dataset_root / test_relative
            test_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.touch()
            test_lines.append(f"{test_relative} {label}\n")
        (run_dir / f"train_batch_{task_index:02d}_filelist.txt").write_text(
            "".join(train_lines), encoding="utf-8"
        )
        label_offset += class_count
    (run_dir / "test_filelist.txt").write_text(
        "".join(test_lines), encoding="utf-8"
    )

    monkeypatch.setattr(visual, "_core50_transform", lambda: object())

    def fake_materialize(entries, *, transform):
        del transform
        return (
            torch.zeros(len(entries), 3 * 64 * 64),
            torch.tensor([label for _, label in entries]),
        )

    monkeypatch.setattr(visual, "_materialize_core50", fake_materialize)
    config = replace(
        visual.generalization_configs()["core50"],
        train_per_class=1,
        validation_per_class=1,
        test_per_class=1,
    )

    tasks = visual.load_core50_nc(config, data_dir=dataset_root, download=False)

    assert len(tasks) == 9
    assert tuple(len(task.classes) for task in tasks) == (
        visual.CORE50_TASK_CLASS_COUNTS
    )
    assert tasks[0].classes == tuple(range(10))
    assert tasks[-1].classes == tuple(range(45, 50))
    assert all(len(task.train_y) == len(task.classes) for task in tasks)
    assert all(len(task.validation_y) == len(task.classes) for task in tasks)
    assert all(len(task.test_y) == len(task.classes) for task in tasks)
