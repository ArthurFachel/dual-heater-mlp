"""Paired Split-MNIST class-incremental benchmark for Functional SlowHeat."""

from __future__ import annotations

import csv
import json
import math
import re
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
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
    "hard_freeze",
    "replay",
    "distillation",
    "slowheat_replay",
    "slowheat_distillation",
}

_STRUCTURED_METHOD = re.compile(
    r"slowheat"
    r"(?:_(?P<auxiliary>replay|distillation))?"
    r"(?:_(?P<scope>hidden))?"
    r"_beta_(?P<beta>\d+(?:\.\d+)?)"
    r"(?:_budget_(?P<budget>\d+(?:\.\d+)?))?$"
)


def _structured_match(method: str) -> re.Match[str] | None:
    return _STRUCTURED_METHOD.fullmatch(method)


def _is_slowheat(method: str) -> bool:
    return method.startswith("slowheat_") or method in {"slowheat", "hard_freeze"}


def _uses_replay(method: str) -> bool:
    match = _structured_match(method)
    return method in {"replay", "slowheat_replay"} or (
        match is not None and match.group("auxiliary") == "replay"
    )


def _uses_distillation(method: str) -> bool:
    match = _structured_match(method)
    return method in {"distillation", "slowheat_distillation"} or (
        match is not None and match.group("auxiliary") == "distillation"
    )


def _method_strength(method: str, default: float) -> float:
    match = _structured_match(method)
    return float(match.group("beta")) if match else default


def _method_budget(method: str, default: float) -> float:
    match = _structured_match(method)
    if match is None or match.group("budget") is None:
        return default
    return float(match.group("budget"))


def _protects_output(method: str) -> bool:
    match = _structured_match(method)
    return match is None or match.group("scope") != "hidden"


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
    replay_per_class: int = 20
    replay_batch_size: int = 64
    distillation_strength: float = 1.0
    distillation_temperature: float = 2.0
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
            "replay_per_class": self.replay_per_class,
            "replay_batch_size": self.replay_batch_size,
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
            "distillation_strength": self.distillation_strength,
            "distillation_temperature": self.distillation_temperature,
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
        if self.distillation_strength < 0.0:
            raise ValueError("distillation_strength deve ser >= 0")
        if self.distillation_temperature <= 0.0:
            raise ValueError("distillation_temperature deve ser > 0")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods deve ser não vazio e sem duplicatas")
        unknown = {
            method
            for method in self.methods
            if method not in SUPPORTED_METHODS and _structured_match(method) is None
        }
        if unknown:
            raise ValueError(f"métodos desconhecidos: {sorted(unknown)}")
        invalid_budgets = {
            method
            for method in self.methods
            if not 0.0 <= _method_budget(method, self.plasticity_budget) <= 1.0
        }
        if invalid_budgets:
            raise ValueError(
                f"budgets embutidos fora de [0, 1]: {sorted(invalid_budgets)}"
            )


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
            if not _is_slowheat(method):
                model: nn.Module = _vanilla_mlp(dims)
            else:
                budget = (
                    0.0
                    if method == "slowheat_unbudgeted"
                    else _method_budget(method, config.plasticity_budget)
                )
                model = SlowHeatMLP(
                    *dims,
                    act="relu",
                    slow_strength=_method_strength(method, config.slow_strength),
                    plasticity_budget=budget,
                    protect_output=_protects_output(method),
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
    if not _is_slowheat(method):
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    state_policy = (
        "native" if method == "slowheat_native_state" else config.optimizer_state_policy
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
        optimizer.register_slow_heat_model(model, hard=method == "hard_freeze")
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


@torch.no_grad()
def _task_aware_accuracy(
    model: nn.Module,
    task: MNISTTask,
    *,
    device: str,
) -> float:
    """Evaluate within-task discrimination with the task classes supplied."""

    model.eval()
    classes = torch.tensor(task.classes, device=device)
    correct = 0
    for start in range(0, len(task.test_x), 1_024):
        batch_x = task.test_x[start : start + 1_024].to(device)
        batch_y = task.test_y[start : start + 1_024].to(device)
        local = model(batch_x).index_select(-1, classes).argmax(dim=-1)
        predictions = classes[local]
        correct += int((predictions == batch_y).sum().item())
    return correct / len(task.test_x)


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


def _replay_memory(
    tasks: list[MNISTTask],
    *,
    before_stage: int,
    samples_per_class: int,
) -> tuple[Tensor, Tensor] | None:
    if before_stage == 0:
        return None
    inputs: list[Tensor] = []
    targets: list[Tensor] = []
    for task in tasks[:before_stage]:
        for label in task.classes:
            indices = torch.nonzero(task.train_y == label, as_tuple=False).flatten()
            selected = indices[:samples_per_class]
            inputs.append(task.train_x[selected])
            targets.append(task.train_y[selected])
    return torch.cat(inputs), torch.cat(targets)


def _distillation_loss(
    student_logits: Tensor,
    teacher: nn.Module,
    inputs: Tensor,
    *,
    old_class_count: int,
    temperature: float,
) -> Tensor:
    with torch.no_grad():
        teacher_logits = teacher(inputs)[..., :old_class_count]
        targets = F.softmax(teacher_logits / temperature, dim=-1)
    predictions = F.log_softmax(
        student_logits[..., :old_class_count] / temperature,
        dim=-1,
    )
    return F.kl_div(predictions, targets, reduction="batchmean") * temperature**2


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
    replay_memories = [
        _replay_memory(
            tasks,
            before_stage=stage,
            samples_per_class=config.replay_per_class,
        )
        for stage in range(config.task_count)
    ]
    replay_schedules = [
        (
            make_batch_schedule(
                sample_count=len(memory[0]),
                batch_size=config.replay_batch_size,
                steps=len(schedules[stage]),
                seed=config.seed + 10_000 + stage,
            )
            if memory is not None
            else None
        )
        for stage, memory in enumerate(replay_memories)
    ]
    models = build_paired_models(config)
    results: dict[str, dict[str, Any]] = {}

    for method, model in models.items():
        optimizer = _build_optimizer(method, model, config)
        matrix = np.full((config.task_count, config.task_count), np.nan)
        task_aware_matrix = np.full((config.task_count, config.task_count), np.nan)
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
        teacher: nn.Module | None = None
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
            replay_schedule = replay_schedules[stage]
            replay_memory = replay_memories[stage]
            for step_index, indices in enumerate(schedules[stage]):
                current_x = task.train_x[indices].to(config.device)
                current_y = task.train_y[indices].to(config.device)
                train_x = current_x
                train_y = current_y
                if _uses_replay(method) and replay_memory is not None:
                    assert replay_schedule is not None
                    replay_indices = replay_schedule[step_index]
                    replay_x = replay_memory[0][replay_indices].to(config.device)
                    replay_y = replay_memory[1][replay_indices].to(config.device)
                    train_x = torch.cat((current_x, replay_x))
                    train_y = torch.cat((current_y, replay_y))
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_x)[..., :seen_class_count]
                loss = F.cross_entropy(logits, train_y)
                if _uses_distillation(method) and teacher is not None:
                    loss = loss + config.distillation_strength * _distillation_loss(
                        logits[: len(current_x)],
                        teacher,
                        current_x,
                        old_class_count=stage * config.classes_per_task,
                        temperature=config.distillation_temperature,
                    )
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
                task_aware_matrix[stage, task_index] = _task_aware_accuracy(
                    model,
                    tasks[task_index],
                    device=config.device,
                )

            if _uses_distillation(method):
                teacher = deepcopy(model).eval()
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)

        elapsed = time.perf_counter() - started
        metrics = compute_cl_metrics(
            matrix,
            pretrain_scores=pretrain_scores,
            baseline_scores=baseline_scores,
        )
        task_aware_metrics = compute_cl_metrics(task_aware_matrix)
        average_accuracy, average_forgetting = _stage_curves(matrix)
        results[method] = {
            "accuracy_matrix": _json_matrix(matrix),
            "task_aware_accuracy_matrix": _json_matrix(task_aware_matrix),
            "stage_average_accuracy": average_accuracy,
            "stage_average_forgetting": average_forgetting,
            "validation_acquisition": validation_acquisition,
            "training_losses": training_losses,
            "baseline_scores": baseline_scores.tolist(),
            "pretrain_scores": pretrain_scores.tolist(),
            "capacity_history": capacity_history,
            "elapsed_seconds": elapsed,
            "metrics": asdict(metrics),
            "task_aware_metrics": asdict(task_aware_metrics),
            "classifier_gap": (
                task_aware_metrics.final_average_accuracy
                - metrics.final_average_accuracy
            ),
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
                "task_aware_final_accuracy",
                "task_aware_forgetting",
                "classifier_gap",
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
                        "task_aware_final_accuracy": result["task_aware_metrics"][
                            "final_average_accuracy"
                        ],
                        "task_aware_forgetting": result["task_aware_metrics"][
                            "average_forgetting"
                        ],
                        "classifier_gap": result["classifier_gap"],
                        "elapsed_seconds": result["elapsed_seconds"],
                    }
                )
    return results


AGGREGATE_METRICS = {
    "final_average_accuracy": ("metrics", "final_average_accuracy"),
    "average_forgetting": ("metrics", "average_forgetting"),
    "backward_transfer": ("metrics", "backward_transfer"),
    "forward_transfer": ("metrics", "forward_transfer"),
    "task_aware_final_accuracy": ("task_aware_metrics", "final_average_accuracy"),
    "task_aware_forgetting": ("task_aware_metrics", "average_forgetting"),
    "classifier_gap": (None, "classifier_gap"),
    "elapsed_seconds": (None, "elapsed_seconds"),
}


def _aggregate_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_normal_half_width": (
            float(1.96 * std / np.sqrt(len(array))) if len(array) > 1 else 0.0
        ),
    }


def _result_metric(result: dict[str, Any], metric: str) -> float:
    section, key = AGGREGATE_METRICS[metric]
    value = result[key] if section is None else result[section][key]
    if value is None:
        raise ValueError(f"métrica {metric} não está disponível")
    return float(value)


def run_split_mnist_multi_seed(
    base_config: SplitMNISTConfig,
    *,
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    download: bool = True,
    verbose: bool = False,
    paired_references: tuple[str, ...] = ("vanilla", "replay"),
) -> dict[str, Any]:
    """Run paired Split-MNIST experiments and aggregate repeated seeds."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds deve ser não vazio e conter valores únicos")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw: dict[int, dict[str, dict[str, Any]]] = {}
    for index, seed in enumerate(seeds):
        if verbose:
            print(f"[Split-MNIST] seed {index + 1}/{len(seeds)}: {seed}", flush=True)
        config = replace(base_config, seed=seed)
        tasks = load_split_mnist(
            config,
            data_dir=data_dir,
            download=download if index == 0 else False,
        )
        raw[seed] = run_split_mnist(
            config,
            tasks,
            output_dir=output_path / f"seed_{seed}",
        )

    aggregate: dict[str, Any] = {
        "seeds": seeds,
        "methods": {},
    }
    csv_rows: list[dict[str, Any]] = []
    for method in base_config.methods:
        aggregate["methods"][method] = {}
        row: dict[str, Any] = {"method": method}
        for metric in AGGREGATE_METRICS:
            values = [_result_metric(raw[seed][method], metric) for seed in seeds]
            summary = _aggregate_summary(values)
            aggregate["methods"][method][metric] = summary
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_std"] = summary["std"]
            row[f"{metric}_ci95"] = summary["ci95_normal_half_width"]
        csv_rows.append(row)

    for reference in paired_references:
        if reference not in base_config.methods:
            continue
        comparison_key = f"paired_differences_vs_{reference}"
        aggregate[comparison_key] = {}
        for method in base_config.methods:
            if method == reference:
                continue
            aggregate[comparison_key][method] = {}
            for metric in AGGREGATE_METRICS:
                differences = [
                    _result_metric(raw[seed][method], metric)
                    - _result_metric(raw[seed][reference], metric)
                    for seed in seeds
                ]
                aggregate[comparison_key][method][metric] = _aggregate_summary(
                    differences
                )

    with (output_path / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True, allow_nan=False)
    with (output_path / "multi_seed_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"base_config": asdict(base_config), "seeds": seeds},
            handle,
            indent=2,
            sort_keys=True,
        )
    with (output_path / "aggregate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return aggregate


def run_split_mnist_epoch_sweep(
    base_config: SplitMNISTConfig,
    *,
    epochs: list[int],
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    download: bool = True,
    verbose: bool = False,
    paired_references: tuple[str, ...] = ("replay",),
) -> dict[str, Any]:
    """Compare methods at multiple training budgets using identical seeds."""

    if (
        not epochs
        or len(set(epochs)) != len(epochs)
        or any(value < 1 for value in epochs)
    ):
        raise ValueError("epochs deve conter inteiros positivos e únicos")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sweep: dict[str, Any] = {"epochs": epochs, "results": {}}
    rows: list[dict[str, Any]] = []

    for index, epoch_count in enumerate(epochs):
        if verbose:
            print(
                f"[Split-MNIST] épocas {index + 1}/{len(epochs)}: {epoch_count}",
                flush=True,
            )
        config = replace(base_config, epochs_per_task=epoch_count)
        aggregate = run_split_mnist_multi_seed(
            config,
            seeds=seeds,
            data_dir=data_dir,
            output_dir=output_path / f"epochs_{epoch_count}",
            download=download if index == 0 else False,
            verbose=verbose,
            paired_references=paired_references,
        )
        sweep["results"][str(epoch_count)] = aggregate
        for method in config.methods:
            row: dict[str, Any] = {"epochs": epoch_count, "method": method}
            for metric in AGGREGATE_METRICS:
                summary = aggregate["methods"][method][metric]
                row[f"{metric}_mean"] = summary["mean"]
                row[f"{metric}_std"] = summary["std"]
                row[f"{metric}_ci95"] = summary["ci95_normal_half_width"]
            rows.append(row)

    with (output_path / "epoch_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(sweep, handle, indent=2, sort_keys=True, allow_nan=False)
    with (output_path / "epoch_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sweep
