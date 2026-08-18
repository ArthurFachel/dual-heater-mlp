"""Dataset adapters for Permuted-MNIST and the native CORe50 CL stream.

The adapters materialize deterministic research subsets as tensors so they can
reuse the same paired baseline engine as Split-MNIST. No dataset is downloaded
or evaluated at import time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
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
from experiments.split_mnist_suite import SLOWHEAT_DERPP_METHODS

CORE50_RUNS = tuple(range(10))
CORE50_TASK_CLASS_COUNTS = (10, 5, 5, 5, 5, 5, 5, 5, 5)


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


def _core50_run_dir(dataset_root: Path, run: int) -> Path:
    run_names = (f"Run{run}", f"run{run}")
    parents = (
        dataset_root / "batches_filelists" / "NC_inc",
        dataset_root / "filelists" / "NC_inc",
        dataset_root / "NC_inc",
        dataset_root.parent / "batches_filelists" / "NC_inc",
        dataset_root.parent / "filelists" / "NC_inc",
    )
    candidates = tuple(parent / name for parent in parents for name in run_names)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    formatted = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "filelists oficiais NC_inc do CORe50 não encontrados. Locais aceitos:\n"
        f"  - {formatted}"
    )


def _resolve_core50_image(dataset_root: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    candidates: list[Path] = []
    if relative.is_absolute():
        candidates.append(relative)
    else:
        candidates.extend((dataset_root / relative, dataset_root.parent / relative))
        for index, part in enumerate(relative.parts):
            if part.startswith("s") and part[1:].isdigit():
                candidates.append(dataset_root.joinpath(*relative.parts[index:]))
                break
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"imagem referenciada pelo filelist não encontrada: {raw_path}"
    )


def _read_core50_filelist(
    filelist_path: Path,
    *,
    dataset_root: Path,
) -> list[tuple[Path, int]]:
    entries: list[tuple[Path, int]] = []
    with filelist_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw_path, raw_label = line.rsplit(maxsplit=1)
                label = int(raw_label)
            except ValueError as error:
                raise ValueError(
                    f"filelist inválido em {filelist_path}:{line_number}"
                ) from error
            entries.append((_resolve_core50_image(dataset_root, raw_path), label))
    if not entries:
        raise ValueError(f"filelist vazio: {filelist_path}")
    return entries


def _select_core50_entries(
    entries: list[tuple[Path, int]],
    *,
    count: int | None,
    seed: int,
) -> list[tuple[Path, int]]:
    order = torch.randperm(
        len(entries), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    selected = [entries[index] for index in order]
    return selected if count is None else selected[:count]


def _core50_transform() -> Callable[[object], Tensor]:
    try:
        from torchvision import transforms
    except ImportError as error:
        raise RuntimeError("torchvision é necessário para CORe50") from error
    return transforms.Compose(
        [
            transforms.Resize((64, 64), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _materialize_core50(
    entries: list[tuple[Path, int]],
    *,
    transform: Callable[[object], Tensor],
) -> tuple[Tensor, Tensor]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow é necessário para CORe50") from error
    images: list[Tensor] = []
    labels: list[int] = []
    for path, label in entries:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")).flatten())
        labels.append(label)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def load_core50_nc(
    config: SplitMNISTConfig,
    *,
    data_dir: str | Path,
    download: bool = False,
) -> list[MNISTTask]:
    """Load one official CORe50 NC incremental run in Class-IL mode.

    ``config.seed`` selects one of the ten official run directories ``Run0``
    through ``Run9``. Labels in those filelists are already remapped so the
    first experience introduces labels 0..9 and the next eight introduce five
    consecutive labels each.
    """

    config.validate()
    if download:
        raise ValueError("CORe50 deve ser obtido separadamente; use download=False")
    if config.seed not in CORE50_RUNS:
        raise ValueError("CORe50 NC requer seeds/run IDs inteiros de 0 a 9")
    if config.scenario != "class_incremental" or len(config.class_order) != 50:
        raise ValueError("CORe50 NC requer 50 classes class-incremental")
    if config.task_class_counts != CORE50_TASK_CLASS_COUNTS:
        raise ValueError("CORe50 NC requer tarefas (10, 5, 5, 5, 5, 5, 5, 5, 5)")
    if config.input_dim != 3 * 64 * 64:
        raise ValueError("CORe50 NC requer imagens RGB redimensionadas para 64x64")

    dataset_root = Path(data_dir)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"diretório CORe50 não encontrado: {dataset_root}")
    run_dir = _core50_run_dir(dataset_root, config.seed)
    train_filelists = sorted(run_dir.glob("train_batch_*_filelist.txt"))
    if len(train_filelists) != len(CORE50_TASK_CLASS_COUNTS):
        raise ValueError(
            f"{run_dir} deve conter 9 train_batch_*_filelist.txt; "
            f"encontrados {len(train_filelists)}"
        )
    test_entries = _read_core50_filelist(
        run_dir / "test_filelist.txt", dataset_root=dataset_root
    )
    test_by_label: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for entry in test_entries:
        test_by_label[entry[1]].append(entry)

    transform = _core50_transform()
    tasks: list[MNISTTask] = []
    for task_index, filelist_path in enumerate(train_filelists):
        expected_classes = _classes_for_task(config, task_index)
        train_entries = _read_core50_filelist(
            filelist_path, dataset_root=dataset_root
        )
        train_by_label: dict[int, list[tuple[Path, int]]] = defaultdict(list)
        for entry in train_entries:
            train_by_label[entry[1]].append(entry)
        if set(train_by_label) != set(expected_classes):
            raise ValueError(
                f"classes inesperadas em {filelist_path}: "
                f"esperado={list(expected_classes)}, "
                f"obtido={sorted(train_by_label)}"
            )

        selected_train: list[tuple[Path, int]] = []
        selected_validation: list[tuple[Path, int]] = []
        selected_test: list[tuple[Path, int]] = []
        for label in expected_classes:
            ordered_train = _select_core50_entries(
                train_by_label[label],
                count=None,
                seed=config.seed * 100_003 + label * 1_003,
            )
            minimum_train = config.validation_per_class + (
                config.train_per_class or 0
            )
            if len(ordered_train) < minimum_train:
                raise ValueError(
                    f"classe {label} possui {len(ordered_train)} imagens de treino; "
                    f"o protocolo requer {minimum_train}"
                )
            selected_validation.extend(
                ordered_train[: config.validation_per_class]
            )
            remaining = ordered_train[config.validation_per_class :]
            selected_train.extend(
                remaining
                if config.train_per_class is None
                else remaining[: config.train_per_class]
            )
            if label not in test_by_label:
                raise ValueError(f"classe {label} ausente do test_filelist.txt")
            chosen_test = _select_core50_entries(
                test_by_label[label],
                count=config.test_per_class,
                seed=config.seed * 200_003 + label * 2_003,
            )
            if (
                config.test_per_class is not None
                and len(chosen_test) < config.test_per_class
            ):
                raise ValueError(
                    f"classe {label} possui apenas {len(chosen_test)} imagens de teste"
                )
            selected_test.extend(chosen_test)

        train_x, train_y = _materialize_core50(
            selected_train, transform=transform
        )
        validation_x, validation_y = _materialize_core50(
            selected_validation, transform=transform
        )
        test_x, test_y = _materialize_core50(selected_test, transform=transform)
        tasks.append(
            MNISTTask(
                classes=expected_classes,
                train_x=train_x,
                train_y=train_y,
                validation_x=validation_x,
                validation_y=validation_y,
                test_x=test_x,
                test_y=test_y,
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
        "methods": SLOWHEAT_DERPP_METHODS,
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
        "core50": SplitMNISTConfig(
            seed=0,
            class_order=tuple(range(50)),
            classes_per_task=5,
            task_class_counts=CORE50_TASK_CLASS_COUNTS,
            scenario="class_incremental",
            input_dim=3 * 64 * 64,
            hidden_dims=(1024, 512),
            train_per_class=400,
            validation_per_class=100,
            test_per_class=100,
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
        "core50": load_core50_nc,
    }
    if name not in configs:
        raise ValueError(f"benchmark desconhecido: {name}")
    if name == "core50" and download:
        raise ValueError("use download=False para o CORe50 local")
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
