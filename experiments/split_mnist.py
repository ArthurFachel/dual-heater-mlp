"""Paired Split-MNIST class-incremental benchmark for Functional SlowHeat."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dual_heater import SlowHeatAdamW, SlowHeatMLP, compute_cl_metrics
from experiments.synthetic_cl import make_batch_schedule

SUPPORTED_METHODS = {
    "vanilla",
    "slowheat",
    "slowheat_adaptive",
    "slowheat_native_state",
    "slowheat_unidirectional",
    "slowheat_unbudgeted",
    "slowheat_none",
}


@dataclass(frozen=True)
class SplitMNISTConfig:
    seed: int = 42
    class_order: tuple[int, ...] = tuple(range(10))
    classes_per_task: int = 2
    hidden_dims: tuple[int, ...] = (256, 128)
    batch_size: int = 128
    epochs_per_task: int = 2
    train_per_class: int | None = 1_000
    validation_per_class: int = 200
    test_per_class: int | None = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    slow_strength: float = 3.0
    plasticity_budget: float = 0.25
    optimizer_state_policy: str = "follow_update"
    adaptive_target_accuracy: float = 0.90
    adaptive_rate: float = 0.20
    adaptive_minimum: float = 0.10
    adaptive_maximum: float = 0.80
    methods: tuple[str, ...] = (
        "vanilla",
        "slowheat",
        "slowheat_adaptive",
        "slowheat_native_state",
        "slowheat_unidirectional",
    )
    device: str = "cpu"

    @property
    def task_count(self) -> int:
        return len(self.class_order) // self.classes_per_task

    def validate(self) -> None:
        if sorted(self.class_order) != list(range(10)):
            raise ValueError("class_order deve ser uma permutação das classes 0..9")
        if self.classes_per_task < 1 or 10 % self.classes_per_task != 0:
            raise ValueError("classes_per_task deve ser um divisor positivo de 10")
        integers = {
            "batch_size": self.batch_size,
            "epochs_per_task": self.epochs_per_task,
            "validation_per_class": self.validation_per_class,
        }
        for name, value in integers.items():
            if value < 1:
                raise ValueError(f"{name} deve ser >= 1")
        for name, value in {
            "train_per_class": self.train_per_class,
            "test_per_class": self.test_per_class,
        }.items():
            if value is not None and value < 1:
                raise ValueError(f"{name} deve ser >= 1 ou None")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims deve conter dimensões positivas")
        finite_values = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "slow_strength": self.slow_strength,
            "plasticity_budget": self.plasticity_budget,
            "adaptive_target_accuracy": self.adaptive_target_accuracy,
            "adaptive_rate": self.adaptive_rate,
            "adaptive_minimum": self.adaptive_minimum,
            "adaptive_maximum": self.adaptive_maximum,
        }
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} deve ser finito")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate deve ser > 0 e weight_decay >= 0")
        if self.slow_strength < 0.0:
            raise ValueError("slow_strength deve ser >= 0")
        if not 0.0 <= self.plasticity_budget <= 1.0:
            raise ValueError("plasticity_budget deve estar em [0, 1]")
        if self.optimizer_state_policy not in {"native", "follow_update"}:
            raise ValueError("optimizer_state_policy inválido")
        if not 0.0 <= self.adaptive_target_accuracy <= 1.0:
            raise ValueError("adaptive_target_accuracy deve estar em [0, 1]")
        if self.adaptive_rate < 0.0:
            raise ValueError("adaptive_rate deve ser >= 0")
        if not 0.0 <= self.adaptive_minimum <= self.adaptive_maximum <= 1.0:
            raise ValueError("limites adaptativos inválidos")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods deve ser não vazio e sem duplicatas")
        unknown = set(self.methods) - SUPPORTED_METHODS
        if unknown:
            raise ValueError(f"métodos desconhecidos: {sorted(unknown)}")


@dataclass(frozen=True)
class MNISTTask:
    classes: tuple[int, ...]
    train_x: Tensor
    train_y: Tensor
    validation_x: Tensor
    validation_y: Tensor
    test_x: Tensor
    test_y: Tensor


def _normalized_images(dataset: Any) -> Tensor:
    images = dataset.data.to(dtype=torch.float32).div_(255.0)
    return images.sub_(0.1307).div_(0.3081).flatten(1)


def _select_class_indices(
    targets: Tensor,
    label: int,
    *,
    count: int | None,
    seed: int,
) -> Tensor:
    indices = torch.nonzero(targets == label, as_tuple=False).flatten()
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
    selected = indices[order]
    return selected if count is None else selected[:count]


def load_split_mnist(
    config: SplitMNISTConfig,
    *,
    data_dir: str | Path,
    download: bool = True,
) -> list[MNISTTask]:
    """Download/load MNIST and create deterministic disjoint class tasks."""

    config.validate()
    try:
        from torchvision.datasets import MNIST
    except ImportError as error:
        raise RuntimeError(
            "torchvision é necessário; instale com: pip install -e '.[research]'"
        ) from error

    root = str(Path(data_dir))
    train_dataset = MNIST(root=root, train=True, download=download)
    test_dataset = MNIST(root=root, train=False, download=download)
    train_images = _normalized_images(train_dataset)
    test_images = _normalized_images(test_dataset)
    train_targets = train_dataset.targets
    test_targets = test_dataset.targets
    tasks: list[MNISTTask] = []

    for task_index in range(config.task_count):
        start = task_index * config.classes_per_task
        classes = config.class_order[start : start + config.classes_per_task]
        train_parts: list[Tensor] = []
        train_label_parts: list[Tensor] = []
        validation_parts: list[Tensor] = []
        validation_label_parts: list[Tensor] = []
        test_parts: list[Tensor] = []
        test_label_parts: list[Tensor] = []

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
            train_label_parts.append(train_targets[train_indices])
            validation_parts.append(train_images[validation_indices])
            validation_label_parts.append(train_targets[validation_indices])
            test_parts.append(test_images[test_indices])
            test_label_parts.append(test_targets[test_indices])

        tasks.append(
            MNISTTask(
                classes=tuple(classes),
                train_x=torch.cat(train_parts),
                train_y=torch.cat(train_label_parts),
                validation_x=torch.cat(validation_parts),
                validation_y=torch.cat(validation_label_parts),
                test_x=torch.cat(test_parts),
                test_y=torch.cat(test_label_parts),
            )
        )
    return tasks


def _vanilla_mlp(dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for input_dim, output_dim in pairwise(dims[:-1]):
        layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def build_paired_models(config: SplitMNISTConfig) -> dict[str, nn.Module]:
    """Build all methods with byte-identical trainable initialization."""

    config.validate()
    dims = (784, *config.hidden_dims, 10)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        reference = _vanilla_mlp(dims)
        reference_parameters = {
            name: parameter.detach().clone()
            for name, parameter in reference.named_parameters()
        }
        models: dict[str, nn.Module] = {}
        for method in config.methods:
            torch.manual_seed(config.seed)
            if method == "vanilla":
                model: nn.Module = _vanilla_mlp(dims)
            else:
                budget = (
                    0.0
                    if method == "slowheat_unbudgeted"
                    else config.plasticity_budget
                )
                model = SlowHeatMLP(
                    *dims,
                    act="relu",
                    slow_strength=config.slow_strength,
                    plasticity_budget=budget,
                    protect_output=True,
                )
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(reference_parameters[name])
            models[method] = model.to(config.device)
    return models


def _build_optimizer(
    method: str,
    model: nn.Module,
    config: SplitMNISTConfig,
) -> torch.optim.Optimizer:
    if method == "vanilla":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    state_policy = (
        "native"
        if method == "slowheat_native_state"
        else config.optimizer_state_policy
    )
    optimizer = SlowHeatAdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        state_policy=state_policy,
    )
    assert isinstance(model, SlowHeatMLP)
    if method == "slowheat_unidirectional":
        for layer in model.get_slow_layers():
            optimizer.register_slow_heat_module(layer)
    else:
        optimizer.register_slow_heat_model(model)
    return optimizer


@torch.no_grad()
def _accuracy(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    seen_class_count: int,
    device: str,
) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(inputs), 1_024):
        batch_x = inputs[start : start + 1_024].to(device)
        batch_y = targets[start : start + 1_024].to(device)
        predictions = model(batch_x)[..., :seen_class_count].argmax(dim=-1)
        correct += int((predictions == batch_y).sum().item())
    return correct / len(inputs)


def _stage_curves(matrix: np.ndarray) -> tuple[list[float], list[float]]:
    average_accuracy: list[float] = []
    average_forgetting: list[float] = []
    for stage in range(matrix.shape[0]):
        average_accuracy.append(float(np.mean(matrix[stage, : stage + 1])))
        if stage == 0:
            average_forgetting.append(0.0)
            continue
        forgetting = [
            float(np.max(matrix[task : stage + 1, task]) - matrix[stage, task])
            for task in range(stage)
        ]
        average_forgetting.append(float(np.mean(forgetting)))
    return average_accuracy, average_forgetting


def _json_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in matrix
    ]


def run_split_mnist(
    config: SplitMNISTConfig,
    tasks: list[MNISTTask],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run paired class-incremental methods on preconstructed MNIST tasks."""

    config.validate()
    if len(tasks) != config.task_count:
        raise ValueError("quantidade de tasks incompatível com a configuração")
    schedules = []
    for task_index, task in enumerate(tasks):
        steps_per_epoch = math.ceil(len(task.train_x) / config.batch_size)
        schedules.append(
            make_batch_schedule(
                sample_count=len(task.train_x),
                batch_size=config.batch_size,
                steps=steps_per_epoch * config.epochs_per_task,
                seed=config.seed + task_index,
            )
        )
    models = build_paired_models(config)
    results: dict[str, dict[str, Any]] = {}

    for method, model in models.items():
        optimizer = _build_optimizer(method, model, config)
        matrix = np.full((config.task_count, config.task_count), np.nan)
        baseline_scores = np.asarray(
            [
                _accuracy(
                    model,
                    task.test_x,
                    task.test_y,
                    seen_class_count=(index + 1) * config.classes_per_task,
                    device=config.device,
                )
                for index, task in enumerate(tasks)
            ]
        )
        pretrain_scores = np.empty(config.task_count, dtype=np.float64)
        training_losses: list[list[float]] = []
        validation_acquisition: list[float] = []
        capacity_history: list[list[dict[str, float]]] = []
        started = time.perf_counter()

        for stage, task in enumerate(tasks):
            seen_class_count = (stage + 1) * config.classes_per_task
            pretrain_scores[stage] = _accuracy(
                model,
                task.test_x,
                task.test_y,
                seen_class_count=seen_class_count,
                device=config.device,
            )
            model.train()
            stage_losses: list[float] = []
            for indices in schedules[stage]:
                batch_x = task.train_x[indices].to(config.device)
                batch_y = task.train_y[indices].to(config.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)[..., :seen_class_count]
                loss = F.cross_entropy(logits, batch_y)
                loss.backward()
                optimizer.step()
                stage_losses.append(float(loss.detach().item()))
            training_losses.append(stage_losses)

            acquisition = _accuracy(
                model,
                task.validation_x,
                task.validation_y,
                seen_class_count=seen_class_count,
                device=config.device,
            )
            validation_acquisition.append(acquisition)
            if isinstance(model, SlowHeatMLP) and method != "slowheat_none":
                if method == "slowheat_adaptive":
                    model.adapt_capacity(
                        acquisition_score=acquisition,
                        target_score=config.adaptive_target_accuracy,
                        adaptation_rate=config.adaptive_rate,
                        minimum=config.adaptive_minimum,
                        maximum=config.adaptive_maximum,
                    )
                model.consolidate(strategy="max")
                capacity_history.append(
                    [layer.capacity_metrics() for layer in model.get_slow_layers()]
                )

            for task_index in range(stage + 1):
                matrix[stage, task_index] = _accuracy(
                    model,
                    tasks[task_index].test_x,
                    tasks[task_index].test_y,
                    seen_class_count=seen_class_count,
                    device=config.device,
                )

        elapsed = time.perf_counter() - started
        metrics = compute_cl_metrics(
            matrix,
            pretrain_scores=pretrain_scores,
            baseline_scores=baseline_scores,
        )
        average_accuracy, average_forgetting = _stage_curves(matrix)
        results[method] = {
            "accuracy_matrix": _json_matrix(matrix),
            "stage_average_accuracy": average_accuracy,
            "stage_average_forgetting": average_forgetting,
            "validation_acquisition": validation_acquisition,
            "training_losses": training_losses,
            "baseline_scores": baseline_scores.tolist(),
            "pretrain_scores": pretrain_scores.tolist(),
            "capacity_history": capacity_history,
            "elapsed_seconds": elapsed,
            "metrics": asdict(metrics),
        }

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2, sort_keys=True)
        with (output_path / "results.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True, allow_nan=False)
        with (output_path / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fields = [
                "method",
                "final_average_accuracy",
                "average_forgetting",
                "backward_transfer",
                "forward_transfer",
                "elapsed_seconds",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for method, result in results.items():
                metrics = result["metrics"]
                writer.writerow(
                    {
                        "method": method,
                        **{
                            key: metrics[key]
                            for key in (
                                "final_average_accuracy",
                                "average_forgetting",
                                "backward_transfer",
                                "forward_transfer",
                            )
                        },
                        "elapsed_seconds": result["elapsed_seconds"],
                    }
                )
    return results
