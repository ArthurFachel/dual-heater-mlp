from dataclasses import replace

import pytest
import torch

import experiments.visual_generalization as visual
from experiments.split_mnist import SUPPORTED_METHODS
from experiments.split_mnist_suite import ALL_VISUAL_METHODS, SLOWHEAT_DERPP_METHODS


def test_generalization_configs_preserve_scenario_semantics():
    configs = visual.generalization_configs()

    assert set(configs) == {
        "permuted_mnist",
        "split_cifar10",
        "split_cifar100",
    }
    assert configs["permuted_mnist"].scenario == "domain_incremental"
    assert configs["permuted_mnist"].task_count == 5
    assert configs["split_cifar10"].scenario == "class_incremental"
    assert configs["split_cifar10"].task_count == 5
    assert configs["split_cifar10"].classes_per_task == 2
    assert configs["split_cifar100"].scenario == "class_incremental"
    assert configs["split_cifar100"].task_count == 10
    assert configs["split_cifar100"].classes_per_task == 10
    assert configs["permuted_mnist"].methods == SLOWHEAT_DERPP_METHODS
    assert configs["split_cifar10"].methods == ALL_VISUAL_METHODS
    assert configs["split_cifar100"].methods == ALL_VISUAL_METHODS
    assert len(ALL_VISUAL_METHODS) == len(set(ALL_VISUAL_METHODS)) == 31
    assert SUPPORTED_METHODS <= set(ALL_VISUAL_METHODS)
    for config in configs.values():
        config.validate()


@pytest.mark.parametrize(
    ("dataset_name", "class_count", "loader_name"),
    [
        ("CIFAR10", 10, "load_split_cifar10"),
        ("CIFAR100", 100, "load_split_cifar100"),
    ],
)
def test_split_cifar_loaders_create_disjoint_class_il_tasks(
    tmp_path, monkeypatch, dataset_name, class_count, loader_name
):
    class FakeCIFAR:
        def __init__(self, *, root, train, download):
            del root, download
            examples_per_class = 3 if train else 1
            self.targets = [
                label
                for label in range(class_count)
                for _ in range(examples_per_class)
            ]
            self.data = torch.arange(
                len(self.targets) * 32 * 32 * 3,
                dtype=torch.int64,
            ).remainder(256).to(torch.uint8).reshape(-1, 32, 32, 3)

    monkeypatch.setattr(f"torchvision.datasets.{dataset_name}", FakeCIFAR)
    config_key = dataset_name.lower().replace("cifar", "split_cifar")
    config = replace(
        visual.generalization_configs()[config_key],
        train_per_class=1,
        validation_per_class=1,
        test_per_class=1,
    )

    tasks = getattr(visual, loader_name)(config, data_dir=tmp_path)

    assert len(tasks) == config.task_count
    assert all(len(task.classes) == config.classes_per_task for task in tasks)
    assert tuple(label for task in tasks for label in task.classes) == tuple(
        range(class_count)
    )
    assert all(task.train_x.shape[1] == 3 * 32 * 32 for task in tasks)
    assert all(len(task.train_y) == config.classes_per_task for task in tasks)
    assert all(len(task.validation_y) == config.classes_per_task for task in tasks)
    assert all(len(task.test_y) == config.classes_per_task for task in tasks)
