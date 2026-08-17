"""Paired Split-MNIST class-incremental benchmark for Functional SlowHeat."""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections.abc import Callable
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
from experiments.confirmatory_statistics import (
    PRIMARY_ENDPOINT,
    paired_confirmatory_summary,
)
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
    "derpp",
    "slowheat_derpp_hidden_beta_30_budget_0.25",
    "er_ace",
    "agem",
    "ewc",
    "si",
    "lwf_calibrated",
    "replay_balanced",
    "replay_more_epochs",
    "replay_early_stopping",
    "replay_global_lr_reduction",
    "slowheat_replay_hidden_adaptive_beta_30_budget_0.25",
    "slowheat_replay_partial_output_beta_30_budget_0.25",
    "slowheat_replay_hidden_beta_30_budget_0.25_calibrated",
}

REPLAY_METHODS = {
    "replay",
    "derpp",
    "slowheat_derpp_hidden_beta_30_budget_0.25",
    "er_ace",
    "agem",
    "replay_balanced",
    "replay_more_epochs",
    "replay_early_stopping",
    "replay_global_lr_reduction",
    "slowheat_replay_hidden_adaptive_beta_30_budget_0.25",
    "slowheat_replay_partial_output_beta_30_budget_0.25",
    "slowheat_replay_hidden_beta_30_budget_0.25_calibrated",
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
    return method in REPLAY_METHODS | {"slowheat_replay"} or (
        match is not None and match.group("auxiliary") == "replay"
    )


def _uses_derpp(method: str) -> bool:
    return method in {"derpp", "slowheat_derpp_hidden_beta_30_budget_0.25"}


def _uses_distillation(method: str) -> bool:
    match = _structured_match(method)
    return method in {"distillation", "slowheat_distillation", "lwf_calibrated"} or (
        match is not None and match.group("auxiliary") == "distillation"
    )


def _method_strength(method: str, default: float) -> float:
    if method in {
        "slowheat_derpp_hidden_beta_30_budget_0.25",
        "slowheat_replay_hidden_adaptive_beta_30_budget_0.25",
        "slowheat_replay_partial_output_beta_30_budget_0.25",
        "slowheat_replay_hidden_beta_30_budget_0.25_calibrated",
    }:
        return 30.0
    match = _structured_match(method)
    return float(match.group("beta")) if match else default


def _method_budget(method: str, default: float) -> float:
    if method in {
        "slowheat_derpp_hidden_beta_30_budget_0.25",
        "slowheat_replay_hidden_adaptive_beta_30_budget_0.25",
        "slowheat_replay_partial_output_beta_30_budget_0.25",
        "slowheat_replay_hidden_beta_30_budget_0.25_calibrated",
    }:
        return 0.25
    match = _structured_match(method)
    if match is None or match.group("budget") is None:
        return default
    return float(match.group("budget"))


def _protects_output(method: str) -> bool:
    if method in {
        "slowheat_derpp_hidden_beta_30_budget_0.25",
        "slowheat_replay_hidden_adaptive_beta_30_budget_0.25",
        "slowheat_replay_hidden_beta_30_budget_0.25_calibrated",
    }:
        return False
    match = _structured_match(method)
    return match is None or match.group("scope") != "hidden"


@dataclass(frozen=True)
class SplitMNISTConfig:
    seed: int = 42
    class_order: tuple[int, ...] = tuple(range(10))
    classes_per_task: int = 2
    input_dim: int = 784
    scenario: str = "class_incremental"
    domain_task_count: int | None = None
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
    derpp_alpha: float = 0.5
    derpp_beta: float = 0.5
    ewc_lambda: float = 100.0
    ewc_decay: float = 1.0
    si_lambda: float = 1.0
    si_epsilon: float = 0.1
    lwf_old_class_weight: float = 1.0
    replay_more_epochs: int = 20
    early_stopping_max_epochs: int = 30
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    global_lr_reduction: float = 1.0 / 31.0
    partial_output_slow_strength: float = 3.0
    max_train_examples_per_task: int | None = None
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20_260_815
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
        if self.scenario == "domain_incremental":
            assert self.domain_task_count is not None
            return self.domain_task_count
        return len(self.class_order) // self.classes_per_task

    def validate(self) -> None:
        if not self.class_order:
            raise ValueError("class_order não pode ser vazio")
        if sorted(self.class_order) != list(range(len(self.class_order))):
            raise ValueError("class_order deve ser uma permutação 0..C-1")
        if self.scenario not in {"class_incremental", "domain_incremental"}:
            raise ValueError("scenario deve ser class_incremental ou domain_incremental")
        if self.classes_per_task < 1:
            raise ValueError("classes_per_task deve ser positivo")
        if self.scenario == "class_incremental":
            if len(self.class_order) % self.classes_per_task != 0:
                raise ValueError("classes_per_task deve dividir o número de classes")
            if self.domain_task_count is not None:
                raise ValueError("domain_task_count só é válido em domain_incremental")
        elif (
            self.domain_task_count is None
            or self.domain_task_count < 1
            or self.classes_per_task != len(self.class_order)
        ):
            raise ValueError(
                "domain_incremental requer domain_task_count >= 1 e todas as classes"
            )
        integers = {
            "batch_size": self.batch_size,
            "input_dim": self.input_dim,
            "epochs_per_task": self.epochs_per_task,
            "validation_per_class": self.validation_per_class,
            "replay_per_class": self.replay_per_class,
            "replay_batch_size": self.replay_batch_size,
            "replay_more_epochs": self.replay_more_epochs,
            "early_stopping_max_epochs": self.early_stopping_max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "bootstrap_resamples": self.bootstrap_resamples,
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
            "derpp_alpha": self.derpp_alpha,
            "derpp_beta": self.derpp_beta,
            "ewc_lambda": self.ewc_lambda,
            "ewc_decay": self.ewc_decay,
            "si_lambda": self.si_lambda,
            "si_epsilon": self.si_epsilon,
            "lwf_old_class_weight": self.lwf_old_class_weight,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "global_lr_reduction": self.global_lr_reduction,
            "partial_output_slow_strength": self.partial_output_slow_strength,
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
        nonnegative = {
            "derpp_alpha": self.derpp_alpha,
            "derpp_beta": self.derpp_beta,
            "ewc_lambda": self.ewc_lambda,
            "si_lambda": self.si_lambda,
            "lwf_old_class_weight": self.lwf_old_class_weight,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "partial_output_slow_strength": self.partial_output_slow_strength,
        }
        if any(value < 0.0 for value in nonnegative.values()):
            raise ValueError(f"parâmetros devem ser >= 0: {nonnegative}")
        if not 0.0 <= self.ewc_decay <= 1.0:
            raise ValueError("ewc_decay deve estar em [0, 1]")
        if self.si_epsilon <= 0.0 or not 0.0 < self.global_lr_reduction <= 1.0:
            raise ValueError("si_epsilon deve ser > 0 e global_lr_reduction em (0, 1]")
        if self.max_train_examples_per_task is not None and self.max_train_examples_per_task < 1:
            raise ValueError("max_train_examples_per_task deve ser >= 1 ou None")
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
    if config.scenario != "class_incremental" or len(config.class_order) != 10:
        raise ValueError("load_split_mnist requer 10 classes class-incremental")
    if config.input_dim != 784:
        raise ValueError("load_split_mnist requer input_dim=784")
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
    dims = (config.input_dim, *config.hidden_dims, len(config.class_order))
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
                    output_slow_strength=(
                        config.partial_output_slow_strength
                        if method
                        == "slowheat_replay_partial_output_beta_30_budget_0.25"
                        else None
                    ),
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
    learning_rate = config.learning_rate
    if method == "replay_global_lr_reduction":
        learning_rate *= config.global_lr_reduction
    if not _is_slowheat(method):
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.weight_decay,
        )
    state_policy = (
        "native" if method == "slowheat_native_state" else config.optimizer_state_policy
    )
    optimizer = SlowHeatAdamW(
        model.parameters(),
        lr=learning_rate,
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
    seen_classes: tuple[int, ...],
    device: str,
    logit_bias: Tensor | None = None,
) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(inputs), 1_024):
        batch_x = inputs[start : start + 1_024].to(device)
        batch_y = targets[start : start + 1_024].to(device)
        logits = model(batch_x)
        if logit_bias is not None:
            logits = logits + logit_bias.to(device)
        logits = _mask_unseen_logits(logits, seen_classes)
        predictions = logits.argmax(dim=-1)
        correct += int((predictions == batch_y).sum().item())
    return correct / len(inputs)


@torch.no_grad()
def _task_aware_accuracy(
    model: nn.Module,
    task: MNISTTask,
    *,
    device: str,
    logit_bias: Tensor | None = None,
) -> float:
    """Evaluate within-task discrimination with the task classes supplied."""

    model.eval()
    classes = torch.tensor(task.classes, device=device)
    correct = 0
    for start in range(0, len(task.test_x), 1_024):
        batch_x = task.test_x[start : start + 1_024].to(device)
        batch_y = task.test_y[start : start + 1_024].to(device)
        logits = model(batch_x)
        if logit_bias is not None:
            logits = logits + logit_bias.to(device)
        local = logits.index_select(-1, classes).argmax(dim=-1)
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
    old_classes: tuple[int, ...],
    temperature: float,
) -> Tensor:
    with torch.no_grad():
        class_indices = torch.tensor(old_classes, device=inputs.device)
        teacher_logits = teacher(inputs).index_select(-1, class_indices)
        targets = F.softmax(teacher_logits / temperature, dim=-1)
    predictions = F.log_softmax(
        student_logits.index_select(-1, class_indices) / temperature,
        dim=-1,
    )
    return F.kl_div(predictions, targets, reduction="batchmean") * temperature**2


def _classes_through_stage(
    config: SplitMNISTConfig,
    stage: int,
) -> tuple[int, ...]:
    if config.scenario == "domain_incremental":
        return config.class_order
    end = (stage + 1) * config.classes_per_task
    return config.class_order[:end]


def _mask_unseen_logits(logits: Tensor, seen_classes: tuple[int, ...]) -> Tensor:
    """Keep global labels valid while excluding classes not yet observed."""

    mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    mask[list(seen_classes)] = False
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _parameter_penalty(
    model: nn.Module,
    importance: dict[str, Tensor],
    anchors: dict[str, Tensor],
) -> Tensor:
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, parameter in model.named_parameters():
        if name in importance:
            penalty = penalty + (
                importance[name] * (parameter - anchors[name]).square()
            ).sum()
    return penalty


def _gradient_vector(model: nn.Module) -> Tensor:
    parts = [
        (
            parameter.grad.detach().flatten()
            if parameter.grad is not None
            else torch.zeros_like(parameter).flatten()
        )
        for parameter in model.parameters()
    ]
    return torch.cat(parts)


def _set_gradient_vector(model: nn.Module, vector: Tensor) -> None:
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        gradient = vector[offset : offset + count].view_as(parameter)
        if parameter.grad is None:
            parameter.grad = gradient.clone()
        else:
            parameter.grad.copy_(gradient)
        offset += count


def _linear_forward_flops(model: nn.Module) -> int:
    return sum(
        2 * module.weight.numel()
        for module in model.modules()
        if isinstance(module, nn.Linear)
        or module.__class__.__name__ == "SlowHeatLinear"
    )


@torch.no_grad()
def _validation_average(
    model: nn.Module,
    tasks: list[MNISTTask],
    *,
    through_stage: int,
    seen_classes: tuple[int, ...],
    device: str,
    logit_bias: Tensor | None = None,
) -> float:
    scores = [
        _accuracy(
            model,
            task.validation_x,
            task.validation_y,
            seen_classes=seen_classes,
            device=device,
            logit_bias=logit_bias,
        )
        for task in tasks[: through_stage + 1]
    ]
    return float(np.mean(scores))


@torch.no_grad()
def _calibrate_old_class_bias(
    model: nn.Module,
    tasks: list[MNISTTask],
    *,
    stage: int,
    config: SplitMNISTConfig,
) -> Tensor:
    """Select one old-vs-new logit offset using only balanced validation data."""

    bias = torch.zeros(len(config.class_order), device=config.device)
    if stage == 0 or config.scenario == "domain_incremental":
        return bias
    old_classes = _classes_through_stage(config, stage - 1)
    seen_classes = _classes_through_stage(config, stage)
    best_score = -math.inf
    # Fixed grid is part of the declared procedure, not tuned on test outcomes.
    for offset in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
        candidate = torch.zeros_like(bias)
        candidate[list(old_classes)] = offset
        score = _validation_average(
            model,
            tasks,
            through_stage=stage,
            seen_classes=seen_classes,
            device=config.device,
            logit_bias=candidate,
        )
        if score > best_score:
            best_score = score
            bias = candidate
    return bias


def _new_cost_record(model: nn.Module) -> dict[str, float | int]:
    return {
        "current_examples": 0,
        "replay_examples": 0,
        "teacher_forward_examples": 0,
        "learner_forward_examples": 0,
        "learner_backward_examples": 0,
        "optimizer_steps": 0,
        "optimizer_step_seconds": 0.0,
        "consolidation_seconds": 0.0,
        "head_calibration_seconds": 0.0,
        "slowheat_hook_flops": 0,
        "mask_application_flops": 0,
        "regularizer_flops": 0,
        "consolidation_flops": 0,
        "replay_memory_bytes": 0,
        "stored_logits_bytes": 0,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def _finalize_cost(model: nn.Module, cost: dict[str, float | int]) -> dict[str, Any]:
    forward_flops = _linear_forward_flops(model)
    learner = int(cost["learner_forward_examples"])
    teacher = int(cost["teacher_forward_examples"])
    backward = int(cost["learner_backward_examples"])
    core = forward_flops * (learner + teacher + 2 * backward)
    overhead = sum(
        int(cost[key])
        for key in (
            "slowheat_hook_flops",
            "mask_application_flops",
            "regularizer_flops",
            "consolidation_flops",
        )
    )
    record = dict(cost)
    record.update(
        {
            "learner_examples_processed": int(cost["current_examples"])
            + int(cost["replay_examples"]),
            "total_model_examples_processed": learner + teacher,
            "estimated_core_flops": core,
            "estimated_overhead_flops": overhead,
            "estimated_total_flops": core + overhead,
            "flop_convention": (
                "Linear forward=2 FLOPs/weight/example; backward=2x forward. "
                "Hooks, masks, regularizers and consolidation are reported "
                "separately as operation-count approximations."
            ),
        }
    )
    return record


def run_split_mnist(
    config: SplitMNISTConfig,
    tasks: list[MNISTTask],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run paired methods, including replay and regularization baselines.

    The function deliberately remains serial. Every method receives identical
    initialization, current-data schedules and replay indices for a given seed.
    """

    config.validate()
    if len(tasks) != config.task_count:
        raise ValueError("quantidade de tasks incompatível com a configuração")

    largest_epoch_budget = max(
        config.epochs_per_task,
        config.replay_more_epochs,
        config.early_stopping_max_epochs,
    )
    schedules: list[list[Tensor]] = []
    for task_index, task in enumerate(tasks):
        steps_per_epoch = math.ceil(len(task.train_x) / config.batch_size)
        step_count = steps_per_epoch * largest_epoch_budget
        if config.max_train_examples_per_task is not None:
            step_count = max(
                step_count,
                math.ceil(config.max_train_examples_per_task / config.batch_size) + 1,
            )
        schedules.append(
            make_batch_schedule(
                sample_count=len(task.train_x),
                batch_size=config.batch_size,
                steps=step_count,
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
                    seen_classes=_classes_through_stage(config, index),
                    device=config.device,
                )
                for index, task in enumerate(tasks)
            ]
        )
        pretrain_scores = np.empty(config.task_count, dtype=np.float64)
        training_losses: list[list[float]] = []
        validation_acquisition: list[float] = []
        validation_history: list[list[float]] = []
        completed_epochs: list[int] = []
        capacity_history: list[list[dict[str, float]]] = []
        teacher: nn.Module | None = None
        der_logits_parts: list[Tensor] = []
        ewc_importance: dict[str, Tensor] = {}
        ewc_anchors: dict[str, Tensor] = {}
        si_importance: dict[str, Tensor] = {}
        si_anchors: dict[str, Tensor] = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        logit_bias = torch.zeros(len(config.class_order), device=config.device)
        cost = _new_cost_record(model)
        remembered_samples = config.replay_per_class * len(config.class_order)
        if _uses_replay(method):
            # Inputs are materialized as float32 and labels as int64.
            cost["replay_memory_bytes"] = remembered_samples * (
                config.input_dim * 4 + 8
            )
        if _uses_derpp(method):
            cost["stored_logits_bytes"] = (
                remembered_samples * len(config.class_order) * 4
            )
        started = time.perf_counter()

        for stage, task in enumerate(tasks):
            seen_classes = _classes_through_stage(config, stage)
            old_classes = (
                _classes_through_stage(config, stage - 1) if stage > 0 else ()
            )
            pretrain_scores[stage] = _accuracy(
                model,
                task.test_x,
                task.test_y,
                seen_classes=seen_classes,
                device=config.device,
                logit_bias=logit_bias,
            )
            model.train()
            stage_losses: list[float] = []
            stage_validation: list[float] = []
            replay_schedule = replay_schedules[stage]
            replay_memory = replay_memories[stage]
            si_path = {
                name: torch.zeros_like(parameter)
                for name, parameter in model.named_parameters()
            }
            fisher_sum = {
                name: torch.zeros_like(parameter)
                for name, parameter in model.named_parameters()
            }
            fisher_steps = 0

            if method == "replay_more_epochs":
                epoch_budget = config.replay_more_epochs
            elif method == "replay_early_stopping":
                epoch_budget = config.early_stopping_max_epochs
            else:
                epoch_budget = config.epochs_per_task
            steps_per_epoch = math.ceil(len(task.train_x) / config.batch_size)
            best_validation = -math.inf
            stale_epochs = 0
            best_model_state: dict[str, Any] | None = None
            best_optimizer_state: dict[str, Any] | None = None
            stage_examples = 0
            stop_stage = False

            for epoch_index in range(epoch_budget):
                epoch_start = epoch_index * steps_per_epoch
                epoch_end = epoch_start + steps_per_epoch
                for step_index in range(epoch_start, epoch_end):
                    indices = schedules[stage][step_index]
                    current_x = task.train_x[indices].to(config.device)
                    current_y = task.train_y[indices].to(config.device)

                    remaining = None
                    if config.max_train_examples_per_task is not None:
                        remaining = config.max_train_examples_per_task - stage_examples
                        if remaining <= 0:
                            stop_stage = True
                            break
                        if len(current_x) > remaining:
                            current_x = current_x[:remaining]
                            current_y = current_y[:remaining]

                    replay_x: Tensor | None = None
                    replay_y: Tensor | None = None
                    replay_targets: Tensor | None = None
                    if _uses_replay(method) and replay_memory is not None:
                        assert replay_schedule is not None
                        replay_indices = replay_schedule[step_index]
                        replay_x = replay_memory[0][replay_indices].to(config.device)
                        replay_y = replay_memory[1][replay_indices].to(config.device)
                        if remaining is not None:
                            replay_room = max(0, remaining - len(current_x))
                            replay_x = replay_x[:replay_room]
                            replay_y = replay_y[:replay_room]
                        if len(replay_x) == 0:
                            replay_x = None
                            replay_y = None
                        elif _uses_derpp(method):
                            replay_targets = torch.cat(der_logits_parts)[
                                replay_indices[: len(replay_x)]
                            ].to(config.device)

                    current_count = len(current_x)
                    replay_count = 0 if replay_x is None else len(replay_x)
                    stage_examples += current_count + replay_count
                    cost["current_examples"] += current_count
                    cost["replay_examples"] += replay_count
                    cost["learner_forward_examples"] += current_count + replay_count

                    optimizer.zero_grad(set_to_none=True)
                    current_logits = model(current_x)
                    current_loss = F.cross_entropy(
                        _mask_unseen_logits(current_logits, seen_classes), current_y
                    )
                    replay_logits: Tensor | None = None
                    replay_loss: Tensor | None = None
                    if replay_x is not None and replay_y is not None:
                        replay_logits = model(replay_x)
                        replay_loss = F.cross_entropy(
                            _mask_unseen_logits(replay_logits, seen_classes), replay_y
                        )

                    loss = current_loss
                    if replay_loss is not None:
                        if method == "replay_balanced":
                            loss = 0.5 * current_loss + 0.5 * replay_loss
                        elif method == "er_ace":
                            ace_logits = _mask_unseen_logits(
                                current_logits, tuple(task.classes)
                            )
                            loss = 0.5 * F.cross_entropy(ace_logits, current_y) + 0.5 * replay_loss
                        elif _uses_derpp(method):
                            assert replay_logits is not None and replay_targets is not None
                            loss = (
                                current_loss
                                + config.derpp_alpha
                                * F.mse_loss(replay_logits, replay_targets)
                                + config.derpp_beta * replay_loss
                            )
                        elif method != "agem":
                            total = current_count + replay_count
                            loss = (
                                current_count * current_loss + replay_count * replay_loss
                            ) / total

                    if _uses_distillation(method) and teacher is not None:
                        distillation = _distillation_loss(
                            current_logits,
                            teacher,
                            current_x,
                            old_classes=old_classes,
                            temperature=config.distillation_temperature,
                        )
                        cost["teacher_forward_examples"] += current_count
                        if method == "lwf_calibrated":
                            old_fraction = len(old_classes) / len(seen_classes)
                            new_fraction = 1.0 - old_fraction
                            loss = (
                                new_fraction * current_loss
                                + config.lwf_old_class_weight
                                * old_fraction
                                * distillation
                            )
                        else:
                            loss = loss + config.distillation_strength * distillation

                    if method == "ewc" and ewc_importance:
                        loss = loss + 0.5 * config.ewc_lambda * _parameter_penalty(
                            model, ewc_importance, ewc_anchors
                        )
                        cost["regularizer_flops"] += 4 * int(cost["model_parameters"])
                    if method == "si" and si_importance:
                        loss = loss + config.si_lambda * _parameter_penalty(
                            model, si_importance, si_anchors
                        )
                        cost["regularizer_flops"] += 4 * int(cost["model_parameters"])

                    parameter_before = None
                    if method == "si":
                        parameter_before = {
                            name: parameter.detach().clone()
                            for name, parameter in model.named_parameters()
                        }

                    if method == "agem" and replay_loss is not None:
                        current_loss.backward()
                        current_gradient = _gradient_vector(model)
                        optimizer.zero_grad(set_to_none=True)
                        replay_loss.backward()
                        reference_gradient = _gradient_vector(model)
                        dot = torch.dot(current_gradient, reference_gradient)
                        if dot < 0.0:
                            current_gradient = current_gradient - (
                                dot
                                / reference_gradient.square().sum().clamp_min(1e-12)
                            ) * reference_gradient
                        _set_gradient_vector(model, current_gradient)
                    else:
                        loss.backward()
                    cost["learner_backward_examples"] += current_count + replay_count

                    gradients_before_step = None
                    if method == "si":
                        gradients_before_step = {
                            name: (
                                torch.zeros_like(parameter)
                                if parameter.grad is None
                                else parameter.grad.detach().clone()
                            )
                            for name, parameter in model.named_parameters()
                        }
                    if method == "ewc":
                        fisher_steps += 1
                        for name, parameter in model.named_parameters():
                            if parameter.grad is not None:
                                fisher_sum[name].add_(parameter.grad.detach().square())

                    step_started = time.perf_counter()
                    optimizer.step()
                    cost["optimizer_step_seconds"] += time.perf_counter() - step_started
                    cost["optimizer_steps"] += 1
                    if isinstance(model, SlowHeatMLP):
                        slow_units = sum(
                            layer.slow_heat.numel() for layer in model.get_slow_layers()
                        )
                        cost["slowheat_hook_flops"] += 4 * slow_units * (
                            current_count + replay_count
                        )
                        masked_parameters = sum(
                            layer.weight.numel()
                            + (0 if layer.bias is None else layer.bias.numel())
                            for layer in model.get_slow_layers()
                        )
                        cost["mask_application_flops"] += masked_parameters

                    if method == "si":
                        assert parameter_before is not None
                        assert gradients_before_step is not None
                        for name, parameter in model.named_parameters():
                            delta = parameter.detach() - parameter_before[name]
                            si_path[name].add_(-gradients_before_step[name] * delta)
                        cost["regularizer_flops"] += 3 * int(cost["model_parameters"])
                    stage_losses.append(float(loss.detach().item()))

                score = _validation_average(
                    model,
                    tasks,
                    through_stage=stage,
                    seen_classes=seen_classes,
                    device=config.device,
                    logit_bias=logit_bias,
                )
                stage_validation.append(score)
                if method == "replay_early_stopping":
                    if score > best_validation + config.early_stopping_min_delta:
                        best_validation = score
                        stale_epochs = 0
                        best_model_state = deepcopy(model.state_dict())
                        best_optimizer_state = deepcopy(optimizer.state_dict())
                    else:
                        stale_epochs += 1
                    if stale_epochs >= config.early_stopping_patience:
                        stop_stage = True
                if stop_stage:
                    break

            if method == "replay_early_stopping" and best_model_state is not None:
                model.load_state_dict(best_model_state)
                assert best_optimizer_state is not None
                optimizer.load_state_dict(best_optimizer_state)
            completed_epochs.append(len(stage_validation))
            validation_history.append(stage_validation)
            training_losses.append(stage_losses)

            acquisition = _accuracy(
                model,
                task.validation_x,
                task.validation_y,
                seen_classes=seen_classes,
                device=config.device,
                logit_bias=logit_bias,
            )
            validation_acquisition.append(acquisition)

            if method == "ewc":
                for name, parameter in model.named_parameters():
                    estimate = fisher_sum[name] / max(1, fisher_steps)
                    if name in ewc_importance:
                        estimate = config.ewc_decay * ewc_importance[name] + estimate
                    ewc_importance[name] = estimate.detach().clone()
                    ewc_anchors[name] = parameter.detach().clone()
                cost["consolidation_flops"] += int(cost["model_parameters"])
            if method == "si":
                for name, parameter in model.named_parameters():
                    displacement = parameter.detach() - si_anchors[name]
                    increment = torch.relu(si_path[name]) / (
                        displacement.square() + config.si_epsilon
                    )
                    si_importance[name] = si_importance.get(
                        name, torch.zeros_like(increment)
                    ) + increment
                    si_anchors[name] = parameter.detach().clone()
                cost["consolidation_flops"] += 4 * int(cost["model_parameters"])

            if isinstance(model, SlowHeatMLP) and method != "slowheat_none":
                consolidation_started = time.perf_counter()
                if method in {
                    "slowheat_adaptive",
                    "slowheat_replay_hidden_adaptive_beta_30_budget_0.25",
                }:
                    model.adapt_capacity(
                        acquisition_score=acquisition,
                        target_score=config.adaptive_target_accuracy,
                        adaptation_rate=config.adaptive_rate,
                        minimum=config.adaptive_minimum,
                        maximum=config.adaptive_maximum,
                    )
                model.consolidate(strategy="max")
                cost["consolidation_seconds"] += (
                    time.perf_counter() - consolidation_started
                )
                cost["consolidation_flops"] += sum(
                    layer.slow_heat.numel()
                    * max(1, math.ceil(math.log2(layer.slow_heat.numel())))
                    for layer in model.get_slow_layers()
                )
                capacity_history.append(
                    [layer.capacity_metrics() for layer in model.get_slow_layers()]
                )

            if method == "slowheat_replay_hidden_beta_30_budget_0.25_calibrated":
                calibration_started = time.perf_counter()
                logit_bias = _calibrate_old_class_bias(
                    model, tasks, stage=stage, config=config
                )
                cost["head_calibration_seconds"] += (
                    time.perf_counter() - calibration_started
                )

            for task_index in range(stage + 1):
                matrix[stage, task_index] = _accuracy(
                    model,
                    tasks[task_index].test_x,
                    tasks[task_index].test_y,
                    seen_classes=seen_classes,
                    device=config.device,
                    logit_bias=logit_bias,
                )
                task_aware_matrix[stage, task_index] = _task_aware_accuracy(
                    model,
                    tasks[task_index],
                    device=config.device,
                    logit_bias=logit_bias,
                )

            if _uses_derpp(method):
                memory_inputs: list[Tensor] = []
                for label in task.classes:
                    class_indices = torch.nonzero(
                        task.train_y == label, as_tuple=False
                    ).flatten()[: config.replay_per_class]
                    memory_inputs.append(task.train_x[class_indices])
                remembered_x = torch.cat(memory_inputs).to(config.device)
                with torch.no_grad():
                    der_logits_parts.append(model(remembered_x).detach().cpu())
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
            "validation_history": validation_history,
            "completed_epochs": completed_epochs,
            "training_losses": training_losses,
            "baseline_scores": baseline_scores.tolist(),
            "pretrain_scores": pretrain_scores.tolist(),
            "capacity_history": capacity_history,
            "final_logit_bias": logit_bias.detach().cpu().tolist(),
            "elapsed_seconds": elapsed,
            "cost": _finalize_cost(model, cost),
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
                "learner_examples_processed",
                "total_model_examples_processed",
                "estimated_total_flops",
                "estimated_overhead_flops",
                "replay_memory_bytes",
                "stored_logits_bytes",
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
                        "learner_examples_processed": result["cost"][
                            "learner_examples_processed"
                        ],
                        "total_model_examples_processed": result["cost"][
                            "total_model_examples_processed"
                        ],
                        "estimated_total_flops": result["cost"][
                            "estimated_total_flops"
                        ],
                        "estimated_overhead_flops": result["cost"][
                            "estimated_overhead_flops"
                        ],
                        "replay_memory_bytes": result["cost"][
                            "replay_memory_bytes"
                        ],
                        "stored_logits_bytes": result["cost"][
                            "stored_logits_bytes"
                        ],
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
    "learner_examples_processed": ("cost", "learner_examples_processed"),
    "total_model_examples_processed": ("cost", "total_model_examples_processed"),
    "estimated_total_flops": ("cost", "estimated_total_flops"),
    "estimated_overhead_flops": ("cost", "estimated_overhead_flops"),
    "replay_memory_bytes": ("cost", "replay_memory_bytes"),
    "stored_logits_bytes": ("cost", "stored_logits_bytes"),
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
    task_loader: Callable[..., list[MNISTTask]] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run paired Split-MNIST experiments and aggregate repeated seeds."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds deve ser não vazio e conter valores únicos")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw: dict[int, dict[str, dict[str, Any]]] = {}
    for index, seed in enumerate(seeds):
        config = replace(base_config, seed=seed)
        seed_path = output_path / f"seed_{seed}"
        saved_config_path = seed_path / "config.json"
        saved_results_path = seed_path / "results.json"
        if resume and saved_config_path.is_file():
            with saved_config_path.open(encoding="utf-8") as handle:
                saved_config = json.load(handle)
            expected_config = json.loads(json.dumps(asdict(config)))
            if saved_config != expected_config:
                raise RuntimeError(
                    f"configuração salva da seed {seed} difere do pré-registro"
                )
        if resume and saved_results_path.is_file():
            if not saved_config_path.is_file():
                raise RuntimeError(
                    f"seed {seed} possui results.json sem config.json; "
                    "não é seguro reutilizá-la"
                )
            with saved_results_path.open(encoding="utf-8") as handle:
                raw[seed] = json.load(handle)
            if verbose:
                print(
                    f"[Split-MNIST] seed {index + 1}/{len(seeds)}: "
                    f"{seed} (reutilizada)",
                    flush=True,
                )
            continue
        if verbose:
            print(f"[Split-MNIST] seed {index + 1}/{len(seeds)}: {seed}", flush=True)
        loader = load_split_mnist if task_loader is None else task_loader
        tasks = loader(
            config,
            data_dir=data_dir,
            download=download if index == 0 else False,
        )
        raw[seed] = run_split_mnist(
            config,
            tasks,
            output_dir=seed_path,
        )

    aggregate: dict[str, Any] = {
        "seeds": seeds,
        "methods": {},
        "primary_endpoint": PRIMARY_ENDPOINT,
        "paired_analysis_note": (
            "Diferenças são sempre método menos referência. Student-t e "
            "bootstrap reamostram pares; contagem de sinais preserva empates."
        ),
    }
    csv_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
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
                aggregate[comparison_key][method][metric]["paired_values"] = [
                    {"seed": seed, "difference": difference}
                    for seed, difference in zip(seeds, differences, strict=True)
                ]
                aggregate[comparison_key][method][metric]["confirmatory"] = (
                    paired_confirmatory_summary(
                        differences,
                        bootstrap_resamples=base_config.bootstrap_resamples,
                        bootstrap_seed=base_config.bootstrap_seed,
                    )
                )
                paired_rows.extend(
                    {
                        "reference": reference,
                        "method": method,
                        "metric": metric,
                        "seed": seed,
                        "difference": difference,
                    }
                    for seed, difference in zip(seeds, differences, strict=True)
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
    if paired_rows:
        with (output_path / "paired_differences.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)
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
