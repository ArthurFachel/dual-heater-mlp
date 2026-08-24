from dataclasses import replace

import pytest
import torch

import experiments.visual_generalization as visual
from experiments.split_mnist import (
    SUPPORTED_METHODS,
    MNISTTask,
    SplitMNISTConfig,
    build_paired_models,
    run_split_mnist,
)
from experiments.split_mnist_suite import ALL_VISUAL_METHODS


def test_generalization_configs_preserve_scenario_semantics():
    configs = visual.generalization_configs()

    assert set(configs) == {
        "permuted_mnist",
        "split_cifar10",
        "split_cifar10_cnn",
        "split_cifar10_cnn_sweep",
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
    assert configs["split_cifar10_cnn"].backbone == "cnn"
    assert configs["split_cifar10_cnn"].image_shape == (3, 32, 32)
    assert configs["split_cifar10_cnn"].methods == visual.CNN_VISUAL_METHODS
    assert (
        configs["split_cifar10_cnn_sweep"].methods
        == visual.CNN_SWEEP_METHODS
    )
    assert configs["permuted_mnist"].methods == ALL_VISUAL_METHODS
    assert configs["split_cifar10"].methods == ALL_VISUAL_METHODS
    assert configs["split_cifar100"].methods == ALL_VISUAL_METHODS
    assert len(ALL_VISUAL_METHODS) == len(set(ALL_VISUAL_METHODS)) == 32
    assert SUPPORTED_METHODS <= set(ALL_VISUAL_METHODS)
    for config in configs.values():
        config.validate()


def test_split_cifar_cnn_loader_preserves_nchw_shape(tmp_path, monkeypatch):
    class FakeCIFAR:
        def __init__(self, *, root, train, download):
            del root, download
            examples_per_class = 3 if train else 1
            self.targets = [
                label
                for label in range(10)
                for _ in range(examples_per_class)
            ]
            self.data = torch.zeros(
                len(self.targets), 32, 32, 3, dtype=torch.uint8
            )

    monkeypatch.setattr("torchvision.datasets.CIFAR10", FakeCIFAR)
    config = replace(
        visual.generalization_configs()["split_cifar10_cnn"],
        train_per_class=1,
        validation_per_class=1,
        test_per_class=1,
    )

    tasks = visual.load_split_cifar10(config, data_dir=tmp_path)

    assert tasks[0].train_x.shape == (2, 3, 32, 32)
    assert tasks[0].validation_x.shape == (2, 3, 32, 32)
    assert tasks[0].test_x.shape == (2, 3, 32, 32)


def test_cnn_backbone_runs_paired_vanilla_and_slowheat_methods():
    config = SplitMNISTConfig(
        class_order=(0, 1),
        classes_per_task=2,
        input_dim=64,
        hidden_dims=(1,),
        backbone="cnn",
        image_shape=(1, 8, 8),
        cnn_channels=(2, 3),
        cnn_pooled_size=(1, 1),
        batch_size=2,
        epochs_per_task=1,
        methods=(
            "vanilla",
            "slowheat_none",
            "slowheat_unidirectional",
            "slowheat",
            "hard_freeze",
            "slowheat_unidirectional_beta_3_budget_0.50",
            "slowheat_hidden_beta_10_budget_0.75",
        ),
    )
    inputs = torch.randn(4, 1, 8, 8)
    targets = torch.tensor([0, 1, 0, 1])
    tasks = [
        MNISTTask(
            classes=(0, 1),
            train_x=inputs,
            train_y=targets,
            validation_x=inputs[:2],
            validation_y=targets[:2],
            test_x=inputs[:2],
            test_y=targets[:2],
        )
    ]
    models = build_paired_models(config)
    reference = dict(models["vanilla"].named_parameters())
    protected = dict(models["slowheat"].named_parameters())

    assert reference.keys() == protected.keys()
    assert all(torch.equal(reference[name], protected[name]) for name in reference)
    unidirectional = models["slowheat_unidirectional_beta_3_budget_0.50"]
    assert all(layer.slow_strength == 3.0 for layer in unidirectional.get_slow_layers())
    assert all(
        layer.plasticity_budget == pytest.approx(0.5)
        for layer in unidirectional.get_slow_layers()
    )
    hidden = models["slowheat_hidden_beta_10_budget_0.75"]
    assert len(hidden.get_slow_layers()) == 2
    assert isinstance(hidden.classifier, torch.nn.Linear)
    assert all(layer.slow_strength == 10.0 for layer in hidden.get_slow_layers())

    results = run_split_mnist(config, tasks)

    assert tuple(results) == config.methods
    assert results["slowheat_none"]["capacity_history"] == []
    assert len(results["slowheat"]["capacity_history"]) == 1
    assert len(results["hard_freeze"]["capacity_history"]) == 1
    assert len(
        results["slowheat_unidirectional_beta_3_budget_0.50"]["capacity_history"]
    ) == 1
    assert all(
        result["cost"]["estimated_total_flops"] > 0
        for result in results.values()
    )


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
