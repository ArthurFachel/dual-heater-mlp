"""Dataset adapters for visual continual-learning streams.

The adapters materialize deterministic research subsets as tensors so they can
reuse the same paired baseline engine as Split-MNIST. No dataset is downloaded
or evaluated at import time.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from experiments.split_mnist import (
    MNISTTask,
    SplitMNISTConfig,
    _classes_for_task,
    _normalized_images,
    _select_class_indices,
    run_split_mnist_multi_seed,
)
from experiments.split_mnist_suite import ALL_VISUAL_METHODS


def load_permuted_mnist(
    config: SplitMNISTConfig,
    *,
    data_dir: str | Path,
    download: bool = True,
) -> list[MNISTTask]:
    """Create domain-incremental tasks with fixed, seed-paired permutations."""

    config.validate()
    if config.scenario != "domain_incremental" or len(config.class_order) != 10:
        raise ValueError("Permuted-MNIST requer 10 classes domain-incremental")
    if config.input_dim != 784:
        raise ValueError("Permuted-MNIST requer input_dim=784")
    try:
        from torchvision.datasets import MNIST
    except ImportError as error:
        raise RuntimeError("torchvision é necessário para Permuted-MNIST") from error

    train_dataset = MNIST(root=str(data_dir), train=True, download=download)
    test_dataset = MNIST(root=str(data_dir), train=False, download=download)
    train_images = _normalized_images(train_dataset)
    test_images = _normalized_images(test_dataset)
    train_targets = train_dataset.targets
    test_targets = test_dataset.targets

    train_parts: list[Tensor] = []
    train_labels: list[Tensor] = []
    validation_parts: list[Tensor] = []
    validation_labels: list[Tensor] = []
    test_parts: list[Tensor] = []
    test_labels: list[Tensor] = []
    for label in config.class_order:
        all_train = _select_class_indices(
            train_targets,
            label,
            count=None,
            seed=config.seed * 1_003 + label,
        )
        validation_indices = all_train[: config.validation_per_class]
        remaining = all_train[config.validation_per_class :]
        train_indices = (
            remaining
            if config.train_per_class is None
            else remaining[: config.train_per_class]
        )
        test_indices = _select_class_indices(
            test_targets,
            label,
            count=config.test_per_class,
            seed=config.seed * 2_003 + label,
        )
        train_parts.append(train_images[train_indices])
        train_labels.append(train_targets[train_indices])
        validation_parts.append(train_images[validation_indices])
        validation_labels.append(train_targets[validation_indices])
        test_parts.append(test_images[test_indices])
        test_labels.append(test_targets[test_indices])

    base_train = torch.cat(train_parts)
    base_train_y = torch.cat(train_labels)
    base_validation = torch.cat(validation_parts)
    base_validation_y = torch.cat(validation_labels)
    base_test = torch.cat(test_parts)
    base_test_y = torch.cat(test_labels)
    tasks: list[MNISTTask] = []
    for domain in range(config.task_count):
        permutation = torch.randperm(
            784,
            generator=torch.Generator().manual_seed(
                config.seed * 100_003 + domain * 7_919
            ),
        )
        tasks.append(
            MNISTTask(
                classes=tuple(config.class_order),
                train_x=base_train[:, permutation],
                train_y=base_train_y.clone(),
                validation_x=base_validation[:, permutation],
                validation_y=base_validation_y.clone(),
                test_x=base_test[:, permutation],
                test_y=base_test_y.clone(),
            )
        )
    return tasks


def _normalized_cifar_images(dataset: object) -> Tensor:
    images = torch.as_tensor(dataset.data, dtype=torch.float32).div_(255.0)
    mean = images.new_tensor((0.4914, 0.4822, 0.4465))
    std = images.new_tensor((0.2470, 0.2435, 0.2616))
    return images.sub_(mean).div_(std).flatten(1)


def _load_split_cifar(
    config: SplitMNISTConfig,
    *,
    data_dir: str | Path,
    dataset_name: str,
    download: bool,
) -> list[MNISTTask]:
    """Load CIFAR and create deterministic disjoint Class-IL tasks."""

    config.validate()
    class_counts = {"cifar10": 10, "cifar100": 100}
    if dataset_name not in class_counts:
        raise ValueError(f"dataset CIFAR desconhecido: {dataset_name}")
    class_count = class_counts[dataset_name]
    if config.scenario != "class_incremental" or len(config.class_order) != class_count:
        raise ValueError(
            f"Split-{dataset_name.upper()} requer {class_count} classes "
            "class-incremental"
        )
    if config.input_dim != 3 * 32 * 32:
        raise ValueError(f"Split-{dataset_name.upper()} requer input_dim=3072")

    try:
        from torchvision.datasets import CIFAR10, CIFAR100
    except ImportError as error:
        raise RuntimeError(
            "torchvision é necessário para Split-CIFAR; "
            "instale com: pip install -e '.[research]'"
        ) from error

    dataset_class = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    train_dataset = dataset_class(
        root=str(data_dir), train=True, download=download
    )
    test_dataset = dataset_class(
        root=str(data_dir), train=False, download=download
    )
    train_images = _normalized_cifar_images(train_dataset)
    test_images = _normalized_cifar_images(test_dataset)
    train_targets = torch.as_tensor(train_dataset.targets, dtype=torch.long)
    test_targets = torch.as_tensor(test_dataset.targets, dtype=torch.long)
    tasks: list[MNISTTask] = []

    for task_index in range(config.task_count):
        classes = _classes_for_task(config, task_index)
        train_parts: list[Tensor] = []
        train_labels: list[Tensor] = []
        validation_parts: list[Tensor] = []
        validation_labels: list[Tensor] = []
        test_parts: list[Tensor] = []
        test_labels: list[Tensor] = []
        for label in classes:
            all_train = _select_class_indices(
                train_targets,
                label,
                count=None,
                seed=config.seed * 1_003 + label,
            )
            validation_indices = all_train[: config.validation_per_class]
            remaining = all_train[config.validation_per_class :]
            train_indices = (
                remaining
                if config.train_per_class is None
                else remaining[: config.train_per_class]
            )
            test_indices = _select_class_indices(
                test_targets,
                label,
                count=config.test_per_class,
                seed=config.seed * 2_003 + label,
            )
            train_parts.append(train_images[train_indices])
            train_labels.append(train_targets[train_indices])
            validation_parts.append(train_images[validation_indices])
            validation_labels.append(train_targets[validation_indices])
            test_parts.append(test_images[test_indices])
            test_labels.append(test_targets[test_indices])

        tasks.append(
            MNISTTask(
                classes=tuple(classes),
                train_x=torch.cat(train_parts),
                train_y=torch.cat(train_labels),
                validation_x=torch.cat(validation_parts),
                validation_y=torch.cat(validation_labels),
                test_x=torch.cat(test_parts),
                test_y=torch.cat(test_labels),
            )
        )
    return tasks


def load_split_cifar10(
    config: SplitMNISTConfig,
    *,
    data_dir: str | Path,
    download: bool = True,
) -> list[MNISTTask]:
    """Load Split-CIFAR-10 as five two-class Class-IL tasks."""

    return _load_split_cifar(
        config,
        data_dir=data_dir,
        dataset_name="cifar10",
        download=download,
    )


def load_split_cifar100(
    config: SplitMNISTConfig,
    *,
    data_dir: str | Path,
    download: bool = True,
) -> list[MNISTTask]:
    """Load Split-CIFAR-100 as ten ten-class Class-IL tasks."""

    return _load_split_cifar(
        config,
        data_dir=data_dir,
        dataset_name="cifar100",
        download=download,
    )


def generalization_configs(device: str = "cpu") -> dict[str, SplitMNISTConfig]:
    common = {
        "epochs_per_task": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "slow_strength": 30.0,
        "plasticity_budget": 0.25,
        "replay_per_class": 20,
        "replay_batch_size": 64,
        "device": device,
    }
    return {
        "permuted_mnist": SplitMNISTConfig(
            class_order=tuple(range(10)),
            classes_per_task=10,
            scenario="domain_incremental",
            domain_task_count=5,
            input_dim=784,
            hidden_dims=(512, 256),
            methods=ALL_VISUAL_METHODS,
            **common,
        ),
        "split_cifar10": SplitMNISTConfig(
            class_order=tuple(range(10)),
            classes_per_task=2,
            scenario="class_incremental",
            input_dim=3 * 32 * 32,
            hidden_dims=(1024, 512),
            train_per_class=4_000,
            validation_per_class=500,
            test_per_class=1_000,
            methods=ALL_VISUAL_METHODS,
            **common,
        ),
        "split_cifar100": SplitMNISTConfig(
            class_order=tuple(range(100)),
            classes_per_task=10,
            scenario="class_incremental",
            input_dim=3 * 32 * 32,
            hidden_dims=(1024, 512),
            train_per_class=400,
            validation_per_class=50,
            test_per_class=100,
            methods=ALL_VISUAL_METHODS,
            **common,
        ),
    }


def run_visual_generalization(
    name: str,
    *,
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = False,
) -> dict[str, object]:
    configs = generalization_configs(device)
    loaders = {
        "permuted_mnist": load_permuted_mnist,
        "split_cifar10": load_split_cifar10,
        "split_cifar100": load_split_cifar100,
    }
    if name not in configs:
        raise ValueError(f"benchmark desconhecido: {name}")
    return run_split_mnist_multi_seed(
        replace(configs[name], device=device),
        seeds=seeds,
        data_dir=data_dir,
        output_dir=output_dir,
        download=download,
        verbose=verbose,
        paired_references=("replay", "derpp"),
        task_loader=loaders[name],
        resume=resume,
    )
