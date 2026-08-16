"""Dataset adapters for Permuted-MNIST and harder visual streams.

The adapters materialize deterministic research subsets as tensors so they can
reuse the exact same paired baseline engine as Split-MNIST. No dataset is
downloaded or evaluated at import time.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from experiments.split_mnist import (
    MNISTTask,
    SplitMNISTConfig,
    _normalized_images,
    _select_class_indices,
    run_split_mnist_multi_seed,
)
from experiments.split_mnist_suite import CANDIDATE


def _split_tensor_dataset(
    config: SplitMNISTConfig,
    *,
    train_images: Tensor,
    train_targets: Tensor,
    test_images: Tensor,
    test_targets: Tensor,
) -> list[MNISTTask]:
    tasks: list[MNISTTask] = []
    for task_index in range(config.task_count):
        start = task_index * config.classes_per_task
        classes = config.class_order[start : start + config.classes_per_task]
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


def load_split_cifar100(
    config: SplitMNISTConfig,
    *,
    data_dir: str | Path,
    download: bool = True,
) -> list[MNISTTask]:
    """Load deterministic Split CIFAR-100 tasks as normalized flat tensors."""

    config.validate()
    if config.scenario != "class_incremental" or len(config.class_order) != 100:
        raise ValueError("Split CIFAR-100 requer 100 classes class-incremental")
    if config.input_dim != 3 * 32 * 32:
        raise ValueError("Split CIFAR-100 requer input_dim=3072")
    try:
        from torchvision.datasets import CIFAR100
    except ImportError as error:
        raise RuntimeError("torchvision é necessário para Split CIFAR-100") from error

    train_dataset = CIFAR100(root=str(data_dir), train=True, download=download)
    test_dataset = CIFAR100(root=str(data_dir), train=False, download=download)

    def normalized(data: Any) -> Tensor:
        images = torch.as_tensor(data).permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.tensor((0.5071, 0.4867, 0.4408)).view(1, 3, 1, 1)
        std = torch.tensor((0.2675, 0.2565, 0.2761)).view(1, 3, 1, 1)
        return images.sub_(mean).div_(std).flatten(1)

    return _split_tensor_dataset(
        config,
        train_images=normalized(train_dataset.data),
        train_targets=torch.tensor(train_dataset.targets),
        test_images=normalized(test_dataset.data),
        test_targets=torch.tensor(test_dataset.targets),
    )


def _materialize_imagefolder(dataset: Any, indices: Tensor) -> tuple[Tensor, Tensor]:
    images: list[Tensor] = []
    labels: list[int] = []
    for index in indices.tolist():
        image, label = dataset[index]
        images.append(image.flatten())
        labels.append(label)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def load_split_tiny_imagenet(
    config: SplitMNISTConfig,
    *,
    data_dir: str | Path,
    download: bool = False,
) -> list[MNISTTask]:
    """Load TinyImageNet from ImageFolder-compatible ``train`` and ``val`` dirs."""

    config.validate()
    if download:
        raise ValueError("TinyImageNet deve ser obtido separadamente; download=False")
    if config.scenario != "class_incremental" or len(config.class_order) != 200:
        raise ValueError("TinyImageNet requer 200 classes class-incremental")
    if config.input_dim != 3 * 64 * 64:
        raise ValueError("TinyImageNet requer input_dim=12288")
    try:
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
    except ImportError as error:
        raise RuntimeError("torchvision é necessário para TinyImageNet") from error

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4802, 0.4481, 0.3975),
                std=(0.2302, 0.2265, 0.2262),
            ),
        ]
    )
    root = Path(data_dir)
    train_dir = root / "train"
    validation_dir = root / "val"
    missing = [path for path in (train_dir, validation_dir) if not path.is_dir()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "TinyImageNet não encontrado ou não preparado. Diretórios ausentes: "
            f"{formatted}. Defina TINY_IMAGENET_DIR com a raiz real; train/ e "
            "val/ devem conter uma subpasta por classe (formato ImageFolder)."
        )
    train_dataset = ImageFolder(train_dir, transform=transform)
    test_dataset = ImageFolder(validation_dir, transform=transform)
    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError(
            "train/ e val/ devem usar a mesma estrutura ImageFolder por classe"
        )
    train_targets = torch.tensor(train_dataset.targets)
    test_targets = torch.tensor(test_dataset.targets)
    tasks: list[MNISTTask] = []
    for task_index in range(config.task_count):
        start = task_index * config.classes_per_task
        classes = config.class_order[start : start + config.classes_per_task]
        parts: dict[str, list[Tensor]] = {
            "train_x": [], "train_y": [], "validation_x": [],
            "validation_y": [], "test_x": [], "test_y": [],
        }
        for label in classes:
            all_train = _select_class_indices(
                train_targets, label, count=None, seed=config.seed * 1_003 + label
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
            train_x, train_y = _materialize_imagefolder(train_dataset, train_indices)
            validation_x, validation_y = _materialize_imagefolder(
                train_dataset, validation_indices
            )
            test_x, test_y = _materialize_imagefolder(test_dataset, test_indices)
            for key, value in {
                "train_x": train_x, "train_y": train_y,
                "validation_x": validation_x, "validation_y": validation_y,
                "test_x": test_x, "test_y": test_y,
            }.items():
                parts[key].append(value)
        tasks.append(
            MNISTTask(
                classes=tuple(classes),
                train_x=torch.cat(parts["train_x"]),
                train_y=torch.cat(parts["train_y"]),
                validation_x=torch.cat(parts["validation_x"]),
                validation_y=torch.cat(parts["validation_y"]),
                test_x=torch.cat(parts["test_x"]),
                test_y=torch.cat(parts["test_y"]),
            )
        )
    return tasks


def generalization_configs(device: str = "cpu") -> dict[str, SplitMNISTConfig]:
    common = {
        "epochs_per_task": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "slow_strength": 30.0,
        "plasticity_budget": 0.25,
        "replay_per_class": 20,
        "replay_batch_size": 64,
        "methods": ("replay", CANDIDATE),
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
            **common,
        ),
        "split_cifar100": SplitMNISTConfig(
            class_order=tuple(range(100)),
            classes_per_task=10,
            input_dim=3 * 32 * 32,
            hidden_dims=(1024, 512),
            train_per_class=400,
            validation_per_class=100,
            test_per_class=100,
            **common,
        ),
        "tiny_imagenet": SplitMNISTConfig(
            class_order=tuple(range(200)),
            classes_per_task=10,
            input_dim=3 * 64 * 64,
            hidden_dims=(1024, 512),
            train_per_class=400,
            validation_per_class=100,
            test_per_class=50,
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
) -> dict[str, Any]:
    configs = generalization_configs(device)
    loaders = {
        "permuted_mnist": load_permuted_mnist,
        "split_cifar100": load_split_cifar100,
        "tiny_imagenet": load_split_tiny_imagenet,
    }
    if name not in configs:
        raise ValueError(f"benchmark desconhecido: {name}")
    if name == "tiny_imagenet" and download:
        raise ValueError("use download=False para TinyImageNet local")
    return run_split_mnist_multi_seed(
        replace(configs[name], device=device),
        seeds=seeds,
        data_dir=data_dir,
        output_dir=output_dir,
        download=download,
        verbose=verbose,
        paired_references=("replay",),
        task_loader=loaders[name],
    )
