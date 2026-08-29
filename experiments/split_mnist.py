"""Paired Split-MNIST class-incremental benchmark for Functional SlowHeat."""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dual_heater import SlowHeatAdamW, compute_cl_metrics
from experiments.artifacts import (
    atomic_text_writer,
    build_run_identity,
    ensure_run_identity,
    read_json_object,
    read_torch_checkpoint,
    task_data_fingerprint,
    write_json_atomic,
    write_torch_atomic,
)
from experiments.confirmatory_statistics import (
    PRIMARY_ENDPOINT,
    normal_summary,
    paired_confirmatory_summary,
)
from experiments.contracts import ContinualTask
from experiments.lpr import LPRPreconditioner
from experiments.method_specs import (
    MethodSpec,
    method_epoch_budget,
    structured_method_match,
)
from experiments.model_factory import build_paired_models as _build_paired_models
from experiments.provenance import write_environment_manifest
from experiments.replay_memory import (
    REPLAY_SELECTION_STRATEGIES,
    ReplayBuffer,
    ReplaySelectionStrategy,
    select_task_exemplars,
)
from experiments.synthetic_cl import make_batch_schedule
from experiments.validation import (
    require_finite_values,
    require_nonnegative_values,
    require_positive_integers,
)

EVALUATION_BATCH_SIZE = 1_024
TRAIN_SPLIT_SEED_MULTIPLIER = 1_003
TEST_SPLIT_SEED_MULTIPLIER = 2_003
REPLAY_SCHEDULE_SEED_OFFSET = 10_000
CALIBRATION_OFFSETS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
STAGE_CHECKPOINT_SCHEMA_VERSION = 1
STAGE_RESUMABLE_METHODS = {
    "replay",
    "slowheat_replay_hidden_beta_30_budget_0.25",
    "derpp",
    "slowheat_derpp_hidden_beta_30_budget_0.25",
}

_METHOD_SPECS = {
    "vanilla": MethodSpec(),
    "slowheat": MethodSpec(slowheat=True),
    "slowheat_adaptive": MethodSpec(slowheat=True),
    "slowheat_native_state": MethodSpec(slowheat=True),
    "slowheat_unidirectional": MethodSpec(slowheat=True),
    "slowheat_unbudgeted": MethodSpec(
        slowheat=True,
        disable_capacity_budget=True,
    ),
    "slowheat_none": MethodSpec(slowheat=True),
    "hard_freeze": MethodSpec(slowheat=True),
    "replay": MethodSpec(replay=True),
    "distillation": MethodSpec(distillation=True),
    "slowheat_replay": MethodSpec(slowheat=True, replay=True),
    "slowheat_distillation": MethodSpec(slowheat=True, distillation=True),
    "derpp": MethodSpec(replay=True, derpp=True),
    "slowheat_derpp_hidden_beta_30_budget_0.25": MethodSpec(
        slowheat=True,
        replay=True,
        derpp=True,
        strength=30.0,
        budget=0.25,
        protect_output=False,
    ),
    "er_ace": MethodSpec(replay=True, er_ace=True),
    "slowheat_er_ace_hidden_beta_30_budget_0.25": MethodSpec(
        slowheat=True,
        replay=True,
        er_ace=True,
        strength=30.0,
        budget=0.25,
        protect_output=False,
    ),
    "lpr": MethodSpec(replay=True, lpr=True),
    "slowheat_lpr": MethodSpec(
        slowheat=True,
        replay=True,
        lpr=True,
        strength=30.0,
        budget=0.25,
        protect_output=False,
    ),
    "classifier_expander": MethodSpec(
        replay=True,
        classifier_expander=True,
    ),
    "slowheat_classifier_expander": MethodSpec(
        slowheat=True,
        replay=True,
        classifier_expander=True,
        strength=30.0,
        budget=0.25,
        protect_output=False,
    ),
    "scroll": MethodSpec(
        replay=True,
        scroll=True,
        epoch_budget_policy="scroll",
    ),
    "slowheat_scroll": MethodSpec(
        slowheat=True,
        replay=True,
        scroll=True,
        strength=30.0,
        budget=0.25,
        protect_output=False,
        epoch_budget_policy="scroll",
    ),
    "agem": MethodSpec(replay=True),
    "ewc": MethodSpec(),
    "si": MethodSpec(),
    "lwf_calibrated": MethodSpec(distillation=True),
    "replay_balanced": MethodSpec(replay=True),
    "replay_more_epochs": MethodSpec(
        replay=True,
        epoch_budget_policy="replay_more",
    ),
    "replay_early_stopping": MethodSpec(
        replay=True,
        epoch_budget_policy="early_stopping",
    ),
    "replay_global_lr_reduction": MethodSpec(replay=True),
    "slowheat_replay_hidden_adaptive_beta_30_budget_0.25": MethodSpec(
        slowheat=True,
        replay=True,
        strength=30.0,
        budget=0.25,
        protect_output=False,
    ),
    "slowheat_replay_partial_output_beta_30_budget_0.25": MethodSpec(
        slowheat=True,
        replay=True,
        strength=30.0,
        budget=0.25,
        partial_output_protection=True,
    ),
    "slowheat_replay_hidden_beta_30_budget_0.25_calibrated": MethodSpec(
        slowheat=True,
        replay=True,
        strength=30.0,
        budget=0.25,
        protect_output=False,
    ),
}

SUPPORTED_METHODS = set(_METHOD_SPECS)


_structured_match = structured_method_match


@cache
def _method_spec(method: str) -> MethodSpec | None:
    if spec := _METHOD_SPECS.get(method):
        return spec
    match = _structured_match(method)
    if match is None:
        return None
    auxiliary = match.group("auxiliary")
    return MethodSpec(
        slowheat=True,
        replay=auxiliary == "replay",
        distillation=auxiliary == "distillation",
        strength=float(match.group("beta")),
        budget=(
            None
            if match.group("budget") is None
            else float(match.group("budget"))
        ),
        protect_output=match.group("scope") != "hidden",
    )


def _is_slowheat(method: str) -> bool:
    spec = _method_spec(method)
    return spec is not None and spec.slowheat


def _uses_replay(method: str) -> bool:
    spec = _method_spec(method)
    return spec is not None and spec.replay


def _uses_derpp(method: str) -> bool:
    spec = _method_spec(method)
    return spec is not None and spec.derpp


def _uses_er_ace(method: str) -> bool:
    spec = _method_spec(method)
    return spec is not None and spec.er_ace


def _uses_lpr(method: str) -> bool:
    spec = _method_spec(method)
    return spec is not None and spec.lpr


def _uses_classifier_expander(method: str) -> bool:
    spec = _method_spec(method)
    return spec is not None and spec.classifier_expander


def _uses_scroll(method: str) -> bool:
    spec = _method_spec(method)
    return spec is not None and spec.scroll


def _uses_distillation(method: str) -> bool:
    spec = _method_spec(method)
    return spec is not None and spec.distillation


def _is_unidirectional(method: str) -> bool:
    if method == "slowheat_unidirectional":
        return True
    match = _structured_match(method)
    return match is not None and match.group("auxiliary") == "unidirectional"


def _method_strength(method: str, default: float) -> float:
    spec = _method_spec(method)
    return default if spec is None or spec.strength is None else spec.strength


def _method_budget(method: str, default: float) -> float:
    spec = _method_spec(method)
    return default if spec is None or spec.budget is None else spec.budget


def _protects_output(method: str) -> bool:
    spec = _method_spec(method)
    return True if spec is None else spec.protect_output


@dataclass(frozen=True)
class SplitMNISTConfig:
    seed: int = 42
    class_order: tuple[int, ...] = tuple(range(10))
    classes_per_task: int = 2
    task_class_counts: tuple[int, ...] | None = None
    input_dim: int = 784
    scenario: str = "class_incremental"
    domain_task_count: int | None = None
    hidden_dims: tuple[int, ...] = (256, 128)
    backbone: str = "mlp"
    image_shape: tuple[int, int, int] | None = None
    cnn_architecture: str = "small"
    cnn_channels: tuple[int, int] = (32, 64)
    cnn_pooled_size: tuple[int, int] = (2, 2)
    vgg_channels: tuple[int, ...] = (64, 128, 256, 256, 512, 512, 512, 512)
    resnet_stage_channels: tuple[int, int, int, int] = (64, 128, 256, 512)
    resnet_blocks_per_stage: tuple[int, int, int, int] = (2, 2, 2, 2)
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
    replay_selection: ReplaySelectionStrategy = "first"
    distillation_strength: float = 1.0
    distillation_temperature: float = 2.0
    derpp_alpha: float = 0.5
    derpp_beta: float = 0.5
    lpr_omega: float = 4.0
    lpr_spatial_beta: float = 2.0
    lpr_update_frequency: int = 30
    classifier_expander_replay_weight: float = 2.5
    classifier_expander_distillation_weight: float = 0.4
    classifier_expander_classifier_weight: float = 0.1
    classifier_expander_classifier_epochs: int = 1
    classifier_expander_classifier_lr: float = 1e-3
    scroll_ridge: float = 0.1
    scroll_replay_epochs: int = 1
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
        if self.task_class_counts is not None:
            return len(self.task_class_counts)
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
            if self.task_class_counts is not None and (
                not self.task_class_counts
                or any(count < 1 for count in self.task_class_counts)
                or sum(self.task_class_counts) != len(self.class_order)
            ):
                raise ValueError(
                    "task_class_counts deve conter contagens positivas que "
                    "somem o número de classes"
                )
            if (
                self.task_class_counts is None
                and len(self.class_order) % self.classes_per_task != 0
            ):
                raise ValueError("classes_per_task deve dividir o número de classes")
            if self.domain_task_count is not None:
                raise ValueError("domain_task_count só é válido em domain_incremental")
        elif (
            self.domain_task_count is None
            or self.domain_task_count < 1
            or self.classes_per_task != len(self.class_order)
            or self.task_class_counts is not None
        ):
            raise ValueError(
                "domain_incremental requer domain_task_count >= 1, todas as "
                "classes e task_class_counts=None"
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
            "lpr_update_frequency": self.lpr_update_frequency,
            "classifier_expander_classifier_epochs": (
                self.classifier_expander_classifier_epochs
            ),
            "scroll_replay_epochs": self.scroll_replay_epochs,
        }
        require_positive_integers(integers)
        for name, value in {
            "train_per_class": self.train_per_class,
            "test_per_class": self.test_per_class,
        }.items():
            if value is not None and value < 1:
                raise ValueError(f"{name} deve ser >= 1 ou None")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims deve conter dimensões positivas")
        if self.backbone not in {"mlp", "cnn"}:
            raise ValueError("backbone deve ser 'mlp' ou 'cnn'")
        if self.backbone == "mlp":
            if self.image_shape is not None:
                raise ValueError("image_shape só é válido para backbone CNN")
        else:
            if (
                self.image_shape is None
                or len(self.image_shape) != 3
                or any(size < 1 for size in self.image_shape)
                or math.prod(self.image_shape) != self.input_dim
            ):
                raise ValueError(
                    "backbone CNN requer image_shape [C, H, W] compatível com input_dim"
                )
            if self.cnn_architecture not in {"small", "vgg11", "resnet18"}:
                raise ValueError(
                    "cnn_architecture deve ser 'small', 'vgg11' ou 'resnet18'"
                )
            if self.cnn_architecture == "small":
                if len(self.cnn_channels) != 2 or any(
                    width < 1 for width in self.cnn_channels
                ):
                    raise ValueError("cnn_channels deve conter dois canais positivos")
            elif self.cnn_architecture == "vgg11":
                if len(self.vgg_channels) != 8 or any(
                    width < 1 for width in self.vgg_channels
                ):
                    raise ValueError("vgg_channels deve conter oito canais positivos")
                if min(self.image_shape[1:]) < 32:
                    raise ValueError(
                        "VGG11 requer dimensões espaciais de pelo menos 32"
                    )
            else:
                if len(self.resnet_stage_channels) != 4 or any(
                    width < 1 for width in self.resnet_stage_channels
                ):
                    raise ValueError(
                        "resnet_stage_channels deve conter quatro canais positivos"
                    )
                if len(self.resnet_blocks_per_stage) != 4 or any(
                    count < 1 for count in self.resnet_blocks_per_stage
                ):
                    raise ValueError(
                        "resnet_blocks_per_stage deve conter quatro contagens positivas"
                    )
            if len(self.cnn_pooled_size) != 2 or any(
                size < 1 for size in self.cnn_pooled_size
            ):
                raise ValueError("cnn_pooled_size deve conter dimensões positivas")
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
            "lpr_omega": self.lpr_omega,
            "lpr_spatial_beta": self.lpr_spatial_beta,
            "classifier_expander_replay_weight": (
                self.classifier_expander_replay_weight
            ),
            "classifier_expander_distillation_weight": (
                self.classifier_expander_distillation_weight
            ),
            "classifier_expander_classifier_weight": (
                self.classifier_expander_classifier_weight
            ),
            "classifier_expander_classifier_lr": (
                self.classifier_expander_classifier_lr
            ),
            "scroll_ridge": self.scroll_ridge,
            "ewc_lambda": self.ewc_lambda,
            "ewc_decay": self.ewc_decay,
            "si_lambda": self.si_lambda,
            "si_epsilon": self.si_epsilon,
            "lwf_old_class_weight": self.lwf_old_class_weight,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "global_lr_reduction": self.global_lr_reduction,
            "partial_output_slow_strength": self.partial_output_slow_strength,
        }
        require_finite_values(finite_values)
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate deve ser > 0 e weight_decay >= 0")
        if self.slow_strength < 0.0:
            raise ValueError("slow_strength deve ser >= 0")
        if not 0.0 <= self.plasticity_budget <= 1.0:
            raise ValueError("plasticity_budget deve estar em [0, 1]")
        if self.optimizer_state_policy not in {"native", "follow_update"}:
            raise ValueError("optimizer_state_policy inválido")
        if self.replay_selection not in REPLAY_SELECTION_STRATEGIES:
            raise ValueError(
                "replay_selection deve ser first, loss, representative ou hybrid"
            )
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
            "lpr_omega": self.lpr_omega,
            "lpr_spatial_beta": self.lpr_spatial_beta,
            "classifier_expander_replay_weight": (
                self.classifier_expander_replay_weight
            ),
            "classifier_expander_distillation_weight": (
                self.classifier_expander_distillation_weight
            ),
            "classifier_expander_classifier_weight": (
                self.classifier_expander_classifier_weight
            ),
            "scroll_ridge": self.scroll_ridge,
        }
        require_nonnegative_values(nonnegative)
        if self.classifier_expander_classifier_lr <= 0.0:
            raise ValueError("classifier_expander_classifier_lr deve ser > 0")
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
        cnn_only = {
            method
            for method in self.methods
            if (
                _uses_lpr(method)
                or _uses_classifier_expander(method)
                or _uses_scroll(method)
            )
            and self.backbone != "cnn"
        }
        if cnn_only:
            raise ValueError(
                "LPR, Classifier Expander e SCROLL requerem backbone CNN neste "
                f"runner: {sorted(cnn_only)}"
            )
        invalid_budgets = {
            method
            for method in self.methods
            if not 0.0 <= _method_budget(method, self.plasticity_budget) <= 1.0
        }
        if invalid_budgets:
            raise ValueError(
                f"budgets embutidos fora de [0, 1]: {sorted(invalid_budgets)}"
            )


# General name for new dataset adapters; the legacy name remains the stable API.
ContinualExperimentConfig = SplitMNISTConfig

# Compatibility alias retained for notebooks and historical imports.
MNISTTask = ContinualTask


def config_payload(config: SplitMNISTConfig) -> dict[str, Any]:
    """Serialize configs without perturbing legacy uniform-task protocols."""

    payload = asdict(config)
    if config.replay_selection == "first":
        payload.pop("replay_selection")
    if config.task_class_counts is None:
        payload.pop("task_class_counts")
    if config.backbone == "mlp":
        for field in (
            "backbone",
            "image_shape",
            "cnn_architecture",
            "cnn_channels",
            "cnn_pooled_size",
            "vgg_channels",
            "resnet_stage_channels",
            "resnet_blocks_per_stage",
        ):
            payload.pop(field)
    elif config.cnn_architecture == "small":
        payload.pop("cnn_architecture")
        payload.pop("vgg_channels")
        payload.pop("resnet_stage_channels")
        payload.pop("resnet_blocks_per_stage")
    elif config.cnn_architecture == "vgg11":
        payload.pop("cnn_channels")
        payload.pop("resnet_stage_channels")
        payload.pop("resnet_blocks_per_stage")
    else:
        payload.pop("cnn_channels")
        payload.pop("cnn_pooled_size")
        payload.pop("vgg_channels")
    if not any(
        _uses_lpr(method)
        or _uses_classifier_expander(method)
        or _uses_scroll(method)
        for method in config.methods
    ):
        for field in (
            "lpr_omega",
            "lpr_spatial_beta",
            "lpr_update_frequency",
            "classifier_expander_replay_weight",
            "classifier_expander_distillation_weight",
            "classifier_expander_classifier_weight",
            "classifier_expander_classifier_epochs",
            "classifier_expander_classifier_lr",
            "scroll_ridge",
            "scroll_replay_epochs",
        ):
            payload.pop(field)
    return payload


def _classes_for_task(
    config: SplitMNISTConfig,
    task_index: int,
) -> tuple[int, ...]:
    """Return global labels assigned to one possibly non-uniform task."""

    if not 0 <= task_index < config.task_count:
        raise IndexError("task_index fora da sequência configurada")
    if config.scenario == "domain_incremental":
        return config.class_order
    counts = config.task_class_counts
    if counts is None:
        start = task_index * config.classes_per_task
        end = start + config.classes_per_task
    else:
        start = sum(counts[:task_index])
        end = start + counts[task_index]
    return config.class_order[start:end]


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
    if count is not None and len(indices) < count:
        raise ValueError(
            f"classe {label} contém {len(indices)} exemplos, mas {count} foram solicitados"
        )
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
    selected = indices[order]
    return selected if count is None else selected[:count]


def _split_train_validation_indices(
    targets: Tensor,
    label: int,
    *,
    train_count: int | None,
    validation_count: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Create a disjoint split and fail if the declared sample counts do not fit."""

    all_indices = _select_class_indices(targets, label, count=None, seed=seed)
    required = validation_count + (0 if train_count is None else train_count)
    if len(all_indices) < required:
        requested_train = "todos os restantes" if train_count is None else train_count
        raise ValueError(
            f"classe {label} contém {len(all_indices)} exemplos, insuficientes para "
            f"validação={validation_count} e treino={requested_train}"
        )
    validation_indices = all_indices[:validation_count]
    remaining = all_indices[validation_count:]
    train_indices = remaining if train_count is None else remaining[:train_count]
    if len(train_indices) == 0:
        raise ValueError(f"classe {label} ficou sem exemplos de treino")
    return train_indices, validation_indices


# Public adapter helpers; underscored aliases remain available for old notebooks.
classes_for_task = _classes_for_task
normalized_images = _normalized_images
select_class_indices = _select_class_indices
split_train_validation_indices = _split_train_validation_indices


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
        classes = _classes_for_task(config, task_index)
        train_parts: list[Tensor] = []
        train_label_parts: list[Tensor] = []
        validation_parts: list[Tensor] = []
        validation_label_parts: list[Tensor] = []
        test_parts: list[Tensor] = []
        test_label_parts: list[Tensor] = []

        for label in classes:
            train_indices, validation_indices = _split_train_validation_indices(
                train_targets,
                label,
                train_count=config.train_per_class,
                validation_count=config.validation_per_class,
                seed=config.seed * TRAIN_SPLIT_SEED_MULTIPLIER + label,
            )
            test_indices = _select_class_indices(
                test_targets,
                label,
                count=config.test_per_class,
                seed=config.seed * TEST_SPLIT_SEED_MULTIPLIER + label,
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


def build_paired_models(config: SplitMNISTConfig) -> dict[str, nn.Module]:
    """Build all methods with byte-identical trainable initialization."""

    config.validate()
    specs: dict[str, MethodSpec] = {}
    for method in config.methods:
        spec = _method_spec(method)
        assert spec is not None
        specs[method] = spec
    return _build_paired_models(config, specs)


def _slow_layers(model: nn.Module) -> list[nn.Module]:
    getter = getattr(model, "get_slow_layers", None)
    return list(getter()) if callable(getter) else []


def _slow_states(model: nn.Module) -> list[nn.Module]:
    getter = getattr(model, "get_slow_states", None)
    return list(getter()) if callable(getter) else _slow_layers(model)


def _cnn_classifier(model: nn.Module) -> nn.Module:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Module):
        raise TypeError("método requer CNN com classificador separado")
    return classifier


def _cnn_features(model: nn.Module, inputs: Tensor) -> Tensor:
    extractor = getattr(model, "forward_features", None)
    if not callable(extractor):
        raise TypeError("método requer CNN com forward_features")
    return extractor(inputs)


@torch.no_grad()
def _fit_scroll_ridge(
    model: nn.Module,
    reference: nn.Module,
    task: MNISTTask,
    *,
    covariance: Tensor | None,
    class_sums: Tensor | None,
    ridge: float,
    class_count: int,
    device: str,
) -> tuple[Tensor, Tensor, int]:
    """Accumulate SCROLL sufficient statistics and solve ridge regression."""

    reference.eval()
    feature_parts: list[Tensor] = []
    for start in range(0, len(task.train_x), EVALUATION_BATCH_SIZE):
        inputs = task.train_x[start : start + EVALUATION_BATCH_SIZE].to(device)
        feature_parts.append(_cnn_features(reference, inputs))
    features = torch.cat(feature_parts)
    features = torch.cat((features, torch.ones_like(features[:, :1])), dim=1)
    targets = F.one_hot(task.train_y.to(device), num_classes=class_count).to(
        features.dtype
    )
    stage_covariance = features.T @ features
    stage_class_sums = features.T @ targets
    covariance = stage_covariance if covariance is None else covariance + stage_covariance
    class_sums = stage_class_sums if class_sums is None else class_sums + stage_class_sums
    identity = torch.eye(
        covariance.shape[0], device=covariance.device, dtype=covariance.dtype
    )
    solution = torch.linalg.solve(covariance + ridge * identity, class_sums)
    classifier = _cnn_classifier(model)
    classifier.weight.copy_(solution[:-1].T)
    if classifier.bias is not None:
        classifier.bias.copy_(solution[-1])
    dimension = covariance.shape[0]
    operations = 2 * len(features) * dimension**2 + dimension**3
    return covariance, class_sums, operations


def _train_classifier_expander_head(
    model: nn.Module,
    memory: tuple[Tensor, Tensor],
    *,
    seen_classes: tuple[int, ...],
    config: SplitMNISTConfig,
    seed: int,
) -> tuple[list[float], int]:
    """Classifier Expander stage two: classifier-only cross-task training."""

    classifier = _cnn_classifier(model)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.classifier_expander_classifier_lr,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(memory[0]) / config.replay_batch_size)
    schedule = make_batch_schedule(
        sample_count=len(memory[0]),
        batch_size=config.replay_batch_size,
        steps=steps_per_epoch * config.classifier_expander_classifier_epochs,
        seed=seed,
    )
    losses: list[float] = []
    model.train()
    examples = 0
    for indices in schedule:
        inputs = memory[0][indices].to(config.device)
        targets = memory[1][indices].to(config.device)
        with torch.no_grad():
            features = _cnn_features(model, inputs)
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(features)
        loss = config.classifier_expander_classifier_weight * F.cross_entropy(
            _mask_unseen_logits(logits, seen_classes), targets
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        examples += len(inputs)
    return losses, examples


def _train_scroll_replay(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    memory: tuple[Tensor, Tensor],
    *,
    seen_classes: tuple[int, ...],
    config: SplitMNISTConfig,
    seed: int,
) -> tuple[list[float], int]:
    """SCROLL representation adaptation using updated replay memory only."""

    steps_per_epoch = math.ceil(len(memory[0]) / config.replay_batch_size)
    schedule = make_batch_schedule(
        sample_count=len(memory[0]),
        batch_size=config.replay_batch_size,
        steps=steps_per_epoch * config.scroll_replay_epochs,
        seed=seed,
    )
    losses: list[float] = []
    examples = 0
    model.train()
    for indices in schedule:
        inputs = memory[0][indices].to(config.device)
        targets = memory[1][indices].to(config.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(_mask_unseen_logits(logits, seen_classes), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        examples += len(inputs)
    return losses, examples


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
    if not _slow_layers(model):
        raise TypeError("método SlowHeat requer um modelo instrumentado")
    if _is_unidirectional(method):
        for layer in _slow_layers(model):
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
    for start in range(0, len(inputs), EVALUATION_BATCH_SIZE):
        batch_x = inputs[start : start + EVALUATION_BATCH_SIZE].to(device)
        batch_y = targets[start : start + EVALUATION_BATCH_SIZE].to(device)
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
    for start in range(0, len(task.test_x), EVALUATION_BATCH_SIZE):
        batch_x = task.test_x[start : start + EVALUATION_BATCH_SIZE].to(device)
        batch_y = task.test_y[start : start + EVALUATION_BATCH_SIZE].to(device)
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
    if config.task_class_counts is None:
        end = (stage + 1) * config.classes_per_task
    else:
        end = sum(config.task_class_counts[: stage + 1])
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
def _model_forward_flops(model: nn.Module, sample_input: Tensor) -> int:
    """Estimate one-example Linear/Conv2d forward FLOPs from actual shapes."""

    total = 0
    handles: list[Any] = []

    def count(module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal total
        examples = max(1, output.shape[0])
        outputs_per_example = output.numel() // examples
        weight = getattr(module, "weight", None)
        if not isinstance(weight, Tensor):
            return
        if weight.ndim == 4:
            operations_per_output = weight.shape[1] * weight.shape[2] * weight.shape[3]
        elif weight.ndim == 2:
            operations_per_output = weight.shape[1]
        else:
            return
        total += 2 * outputs_per_example * operations_per_output

    for module in model.modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, Tensor) and weight.ndim in {2, 4}:
            handles.append(module.register_forward_hook(count))
    was_training = model.training
    try:
        model.eval()
        device = next(model.parameters()).device
        model(sample_input[:1].to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return total


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
    for offset in CALIBRATION_OFFSETS:
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


def _new_cost_record(
    model: nn.Module,
    sample_input: Tensor | None = None,
) -> dict[str, float | int]:
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
        "selection_seconds": 0.0,
        "selector_forward_examples": 0,
        "selector_distance_flops": 0,
        "slowheat_hook_flops": 0,
        "mask_application_flops": 0,
        "regularizer_flops": 0,
        "consolidation_flops": 0,
        "replay_memory_bytes": 0,
        "stored_logits_bytes": 0,
        "replay_metadata_bytes": 0,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "forward_flops_per_example": (
            _linear_forward_flops(model)
            if sample_input is None
            else _model_forward_flops(model, sample_input)
        ),
    }


def _finalize_cost(model: nn.Module, cost: dict[str, float | int]) -> dict[str, Any]:
    forward_flops = int(
        cost.get("forward_flops_per_example", _linear_forward_flops(model))
    )
    learner = int(cost["learner_forward_examples"])
    teacher = int(cost["teacher_forward_examples"])
    selector = int(cost["selector_forward_examples"])
    backward = int(cost["learner_backward_examples"])
    core = forward_flops * (learner + teacher + selector + 2 * backward)
    overhead = sum(
        int(cost[key])
        for key in (
            "slowheat_hook_flops",
            "mask_application_flops",
            "regularizer_flops",
            "consolidation_flops",
            "selector_distance_flops",
        )
    )
    record = dict(cost)
    record.update(
        {
            "learner_examples_processed": int(cost["current_examples"])
            + int(cost["replay_examples"]),
            "total_model_examples_processed": learner + teacher + selector,
            "estimated_core_flops": core,
            "estimated_overhead_flops": overhead,
            "estimated_total_flops": core + overhead,
            "flop_convention": (
                "Linear/Conv2d forward=2 FLOPs/MAC/example from observed tensor "
                "shapes; backward=2x forward. "
                "Hooks, masks, regularizers and consolidation are reported "
                "separately as operation-count approximations."
            ),
        }
    )
    return record


def _backfill_cost_metadata(
    result: dict[str, Any],
    *,
    method: str,
    config: SplitMNISTConfig,
) -> None:
    """Add deterministic cost fields missing from older saved artifacts."""

    cost = result.setdefault("cost", {})
    history = result.get("selection_history")
    remembered_samples = (
        sum(int(stage.get("selected_count", 0)) for stage in history)
        if isinstance(history, list) and history
        else config.replay_per_class * len(config.class_order)
    )
    replay_bytes = (
        remembered_samples * (config.input_dim * 4 + 8)
        if _uses_replay(method)
        else 0
    )
    logit_bytes = (
        remembered_samples * len(config.class_order) * 4
        if _uses_derpp(method)
        else 0
    )
    cost.setdefault("replay_memory_bytes", replay_bytes)
    cost.setdefault("stored_logits_bytes", logit_bytes)
    cost.setdefault("replay_metadata_bytes", 0)
    cost.setdefault("selection_seconds", 0.0)
    cost.setdefault("selector_forward_examples", 0)
    cost.setdefault("selector_distance_flops", 0)


def run_split_mnist(
    config: SplitMNISTConfig,
    tasks: list[MNISTTask],
    *,
    output_dir: str | Path | None = None,
    resume: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run paired methods, including replay and regularization baselines.

    The function deliberately remains serial. Every method receives identical
    initialization, current-data schedules and replay indices for a given seed.
    """

    config.validate()
    if len(tasks) != config.task_count:
        raise ValueError("quantidade de tasks incompatível com a configuração")
    data_sha256 = (
        task_data_fingerprint(tasks)
        if output_dir is not None
        and any(method in STAGE_RESUMABLE_METHODS for method in config.methods)
        else None
    )

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
    results: dict[str, dict[str, Any]] = {}

    for method in config.methods:
        method_spec = _method_spec(method)
        assert method_spec is not None
        method_config = replace(config, methods=(method,))
        model = build_paired_models(method_config)[method]
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
        pretrain_scores = np.full(config.task_count, np.nan, dtype=np.float64)
        training_losses: list[list[float]] = []
        validation_acquisition: list[float] = []
        validation_history: list[list[float]] = []
        completed_epochs: list[int] = []
        capacity_history: list[list[dict[str, float]]] = []
        teacher: nn.Module | None = None
        replay_buffer = ReplayBuffer()
        ewc_importance: dict[str, Tensor] = {}
        ewc_anchors: dict[str, Tensor] = {}
        si_importance: dict[str, Tensor] = {}
        si_anchors: dict[str, Tensor] = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        logit_bias = torch.zeros(len(config.class_order), device=config.device)
        lpr = (
            LPRPreconditioner(
                omega=config.lpr_omega,
                spatial_beta=config.lpr_spatial_beta,
                update_frequency=config.lpr_update_frequency,
                batch_size=config.replay_batch_size,
            )
            if _uses_lpr(method)
            else None
        )
        scroll_reference: nn.Module | None = None
        scroll_covariance: Tensor | None = None
        scroll_class_sums: Tensor | None = None
        cost = _new_cost_record(model, tasks[0].train_x[:1])
        best_model_state: dict[str, Any] | None = None
        best_optimizer_state: dict[str, Any] | None = None
        start_stage = 0
        elapsed_before = 0.0
        checkpoint_path = (
            Path(output_dir) / "checkpoints" / f"{method}.pt"
            if output_dir is not None and method in STAGE_RESUMABLE_METHODS
            else None
        )
        if checkpoint_path is not None and checkpoint_path.is_file():
            if not resume:
                raise FileExistsError(
                    f"checkpoint existente requer resume=True: {checkpoint_path}"
                )
            checkpoint = read_torch_checkpoint(checkpoint_path)
            expected_config = config_payload(method_config)
            if (
                checkpoint.get("schema_version") != STAGE_CHECKPOINT_SCHEMA_VERSION
                or checkpoint.get("method") != method
                or checkpoint.get("replay_selection") != config.replay_selection
                or checkpoint.get("config") != expected_config
                or checkpoint.get("data_sha256") != data_sha256
            ):
                raise RuntimeError(
                    f"checkpoint incompatível com método, configuração ou dados: {method}"
                )
            start_stage = int(checkpoint["next_stage"])
            if not 0 <= start_stage <= config.task_count:
                raise RuntimeError("checkpoint contém next_stage inválido")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            replay_buffer.load_state_dict(checkpoint["replay_buffer"])
            matrix = checkpoint["accuracy_matrix"].numpy().copy()
            task_aware_matrix = checkpoint["task_aware_accuracy_matrix"].numpy().copy()
            pretrain_scores = checkpoint["pretrain_scores"].numpy().copy()
            training_losses = list(checkpoint["training_losses"])
            validation_acquisition = list(checkpoint["validation_acquisition"])
            validation_history = list(checkpoint["validation_history"])
            completed_epochs = list(checkpoint["completed_epochs"])
            capacity_history = list(checkpoint["capacity_history"])
            logit_bias = checkpoint["logit_bias"].to(config.device)
            cost = dict(checkpoint["cost"])
            elapsed_before = float(checkpoint["elapsed_seconds"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            cuda_rng_states = checkpoint.get("cuda_rng_states")
            if cuda_rng_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_rng_states)
        started = time.perf_counter()

        for stage in range(start_stage, config.task_count):
            task = tasks[stage]
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
            replay_memory = replay_buffer.as_memory()
            replay_schedule = (
                make_batch_schedule(
                    sample_count=len(replay_memory[0]),
                    batch_size=config.replay_batch_size,
                    steps=len(schedules[stage]),
                    seed=config.seed + REPLAY_SCHEDULE_SEED_OFFSET + stage,
                )
                if replay_memory is not None
                else None
            )
            si_path = {
                name: torch.zeros_like(parameter)
                for name, parameter in model.named_parameters()
            }
            fisher_sum = {
                name: torch.zeros_like(parameter)
                for name, parameter in model.named_parameters()
            }
            fisher_steps = 0

            epoch_budget = method_epoch_budget(
                method_spec,
                stage=stage,
                default=config.epochs_per_task,
                replay_more=config.replay_more_epochs,
                early_stopping=config.early_stopping_max_epochs,
            )
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
                            if replay_buffer.logits is None:
                                raise RuntimeError("DER++ requer logits no replay buffer")
                            replay_targets = replay_buffer.logits[
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
                        _mask_unseen_logits(
                            current_logits,
                            tuple(task.classes)
                            if _uses_classifier_expander(method)
                            else seen_classes,
                        ),
                        current_y,
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
                        if _uses_classifier_expander(method):
                            loss = (
                                current_loss
                                + config.classifier_expander_replay_weight
                                * replay_loss
                            )
                            if teacher is not None and old_classes:
                                with torch.no_grad():
                                    teacher_logits = teacher(replay_x)
                                seen_index = torch.tensor(
                                    seen_classes,
                                    device=config.device,
                                    dtype=torch.long,
                                )
                                loss = loss + (
                                    config.classifier_expander_distillation_weight
                                    * F.mse_loss(
                                        replay_logits.index_select(1, seen_index),
                                        teacher_logits.index_select(1, seen_index),
                                    )
                                )
                                cost["teacher_forward_examples"] += replay_count
                        elif method == "replay_balanced":
                            loss = 0.5 * current_loss + 0.5 * replay_loss
                        elif _uses_er_ace(method):
                            ace_logits = _mask_unseen_logits(
                                current_logits, tuple(task.classes)
                            )
                            loss = (
                                0.5 * F.cross_entropy(ace_logits, current_y)
                                + 0.5 * replay_loss
                            )
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

                    fisher_gradients: tuple[Tensor | None, ...] | None = None
                    fisher_parameters: tuple[tuple[str, nn.Parameter], ...] = ()
                    if method == "ewc":
                        # The empirical Fisher must come from the current-task
                        # likelihood, not from the EWC-regularized total loss.
                        fisher_parameters = tuple(model.named_parameters())
                        fisher_gradients = torch.autograd.grad(
                            current_loss,
                            tuple(parameter for _, parameter in fisher_parameters),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        fisher_steps += 1
                        for (name, _), gradient in zip(
                            fisher_parameters, fisher_gradients, strict=True
                        ):
                            if gradient is not None:
                                fisher_sum[name].add_(gradient.detach().square())
                        cost["learner_backward_examples"] += current_count

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

                    if lpr is not None and replay_memory is not None:
                        if lpr.should_update():
                            cost["regularizer_flops"] += lpr.update(
                                model,
                                replay_memory[0],
                                device=config.device,
                            )
                            cost["learner_forward_examples"] += len(replay_memory[0])
                        cost["regularizer_flops"] += lpr.precondition(model)

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
                    step_started = time.perf_counter()
                    optimizer.step()
                    if lpr is not None:
                        lpr.advance()
                    cost["optimizer_step_seconds"] += time.perf_counter() - step_started
                    cost["optimizer_steps"] += 1
                    slow_layers = _slow_layers(model)
                    slow_states = _slow_states(model)
                    if slow_states:
                        slow_units = sum(
                            state.slow_heat.numel() for state in slow_states
                        )
                        cost["slowheat_hook_flops"] += 4 * slow_units * (
                            current_count + replay_count
                        )
                        masked_parameters = sum(
                            layer.weight.numel()
                            + (0 if layer.bias is None else layer.bias.numel())
                            for layer in slow_layers
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

            if _uses_replay(method):
                selection_started = time.perf_counter()
                selection = select_task_exemplars(
                    model,
                    task,
                    task_index=stage,
                    seen_classes=seen_classes,
                    samples_per_class=config.replay_per_class,
                    strategy=config.replay_selection,
                    device=config.device,
                    batch_size=EVALUATION_BATCH_SIZE,
                    store_logits=_uses_derpp(method),
                )
                replay_buffer.append(selection)
                cost["selection_seconds"] += time.perf_counter() - selection_started
                cost["selector_forward_examples"] += (
                    selection.selector_forward_examples
                )
                cost["selector_distance_flops"] += (
                    selection.selector_distance_flops
                )
                cost["replay_memory_bytes"] = replay_buffer.replay_memory_bytes
                cost["stored_logits_bytes"] = replay_buffer.stored_logits_bytes
                cost["replay_metadata_bytes"] = replay_buffer.metadata_bytes

            if _uses_scroll(method):
                if scroll_reference is None:
                    # In the no-checkpoint benchmark, task 0 is the documented
                    # representation bootstrap shared by both SCROLL variants.
                    scroll_reference = deepcopy(model).eval()
                    for parameter in scroll_reference.parameters():
                        parameter.requires_grad_(False)
                (
                    scroll_covariance,
                    scroll_class_sums,
                    ridge_operations,
                ) = _fit_scroll_ridge(
                    model,
                    scroll_reference,
                    task,
                    covariance=scroll_covariance,
                    class_sums=scroll_class_sums,
                    ridge=config.scroll_ridge,
                    class_count=len(config.class_order),
                    device=config.device,
                )
                cost["regularizer_flops"] += ridge_operations
                cost["teacher_forward_examples"] += len(task.train_x)
                updated_memory = replay_buffer.as_memory()
                assert updated_memory is not None
                replay_stage_losses, replay_stage_examples = _train_scroll_replay(
                    model,
                    optimizer,
                    updated_memory,
                    seen_classes=seen_classes,
                    config=config,
                    seed=config.seed + REPLAY_SCHEDULE_SEED_OFFSET * 2 + stage,
                )
                stage_losses.extend(replay_stage_losses)
                cost["replay_examples"] += replay_stage_examples
                cost["learner_forward_examples"] += replay_stage_examples
                cost["learner_backward_examples"] += replay_stage_examples
                cost["optimizer_steps"] += len(replay_stage_losses)
                stage_validation.append(
                    _validation_average(
                        model,
                        tasks,
                        through_stage=stage,
                        seen_classes=seen_classes,
                        device=config.device,
                        logit_bias=logit_bias,
                    )
                )

            if _uses_classifier_expander(method) and stage > 0:
                updated_memory = replay_buffer.as_memory()
                assert updated_memory is not None
                head_losses, head_examples = _train_classifier_expander_head(
                    model,
                    updated_memory,
                    seen_classes=seen_classes,
                    config=config,
                    seed=config.seed + REPLAY_SCHEDULE_SEED_OFFSET * 3 + stage,
                )
                stage_losses.extend(head_losses)
                cost["replay_examples"] += head_examples
                cost["learner_forward_examples"] += head_examples
                cost["learner_backward_examples"] += head_examples
                cost["optimizer_steps"] += len(head_losses)
                stage_validation.append(
                    _validation_average(
                        model,
                        tasks,
                        through_stage=stage,
                        seen_classes=seen_classes,
                        device=config.device,
                        logit_bias=logit_bias,
                    )
                )

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

            slow_states = _slow_states(model)
            if slow_states and method != "slowheat_none":
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
                    for layer in slow_states
                )
                capacity_history.append(
                    [state.capacity_metrics() for state in slow_states]
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

            if _uses_distillation(method) or _uses_classifier_expander(method):
                teacher = deepcopy(model).eval()
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)

            if checkpoint_path is not None:
                assert data_sha256 is not None
                write_torch_atomic(
                    checkpoint_path,
                    {
                        "schema_version": STAGE_CHECKPOINT_SCHEMA_VERSION,
                        "method": method,
                        "replay_selection": config.replay_selection,
                        "config": config_payload(method_config),
                        "data_sha256": data_sha256,
                        "next_stage": stage + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "replay_buffer": replay_buffer.state_dict(),
                        "accuracy_matrix": torch.from_numpy(matrix.copy()),
                        "task_aware_accuracy_matrix": torch.from_numpy(
                            task_aware_matrix.copy()
                        ),
                        "pretrain_scores": torch.from_numpy(pretrain_scores.copy()),
                        "training_losses": training_losses,
                        "validation_acquisition": validation_acquisition,
                        "validation_history": validation_history,
                        "completed_epochs": completed_epochs,
                        "capacity_history": capacity_history,
                        "logit_bias": logit_bias.detach().cpu(),
                        "cost": cost,
                        "elapsed_seconds": elapsed_before
                        + time.perf_counter()
                        - started,
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_states": (
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available()
                            else None
                        ),
                    },
                )

        elapsed = elapsed_before + time.perf_counter() - started
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
            "selection_history": replay_buffer.selection_history,
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
        del (
            model,
            optimizer,
            teacher,
            scroll_reference,
            lpr,
            replay_buffer,
            ewc_importance,
            ewc_anchors,
            si_importance,
            si_anchors,
            best_model_state,
            best_optimizer_state,
        )

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_environment_manifest(
            output_path,
            project_root=Path(__file__).resolve().parents[1],
        )
        write_json_atomic(output_path / "config.json", config_payload(config))
        write_json_atomic(output_path / "results.json", results)
        with atomic_text_writer(output_path / "summary.csv", newline="") as handle:
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
                "replay_metadata_bytes",
                "selector_forward_examples",
                "selection_seconds",
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
                        "replay_metadata_bytes": result["cost"][
                            "replay_metadata_bytes"
                        ],
                        "selector_forward_examples": result["cost"][
                            "selector_forward_examples"
                        ],
                        "selection_seconds": result["cost"][
                            "selection_seconds"
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
    "replay_metadata_bytes": ("cost", "replay_metadata_bytes"),
    "selector_forward_examples": ("cost", "selector_forward_examples"),
    "selection_seconds": ("cost", "selection_seconds"),
}


def _aggregate_summary(values: list[float]) -> dict[str, float]:
    return normal_summary(values)


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
    loader = load_split_mnist if task_loader is None else task_loader
    loader_identity = (
        "experiments.split_mnist.load_split_mnist"
        if task_loader is None
        else f"{loader.__module__}.{loader.__qualname__}"
    )
    project_root = Path(__file__).resolve().parents[1]
    identity = build_run_identity(
        {
            "base_config": config_payload(base_config),
            "seeds": seeds,
            "paired_references": paired_references,
        },
        project_root=project_root,
        task_loader=loader_identity,
    )
    ensure_run_identity(output_path, identity, resume=resume)
    write_environment_manifest(
        output_path,
        project_root=project_root,
    )
    raw: dict[int, dict[str, dict[str, Any]]] = {}
    for index, seed in enumerate(seeds):
        config = replace(base_config, seed=seed)
        seed_path = output_path / f"seed_{seed}"
        saved_config_path = seed_path / "config.json"
        saved_results_path = seed_path / "results.json"
        saved_data_identity_path = seed_path / "data_identity.json"
        if resume and saved_config_path.is_file():
            with saved_config_path.open(encoding="utf-8") as handle:
                saved_config = json.load(handle)
            expected_config = json.loads(json.dumps(config_payload(config)))
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
            if not saved_data_identity_path.is_file():
                raise RuntimeError(
                    f"seed {seed} possui results.json sem data_identity.json; "
                    "não é seguro reutilizá-la"
                )
            data_identity = read_json_object(saved_data_identity_path)
            data_digest = data_identity.get("tasks_sha256")
            if (
                data_identity.get("schema_version") != 1
                or data_identity.get("loader") != identity["task_loader"]
                or not isinstance(data_digest, str)
                or len(data_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in data_digest
                )
            ):
                raise RuntimeError(
                    f"data_identity.json inválido ou incompatível na seed {seed}"
                )
            with saved_results_path.open(encoding="utf-8") as handle:
                raw[seed] = json.load(handle)
            for method, result in raw[seed].items():
                _backfill_cost_metadata(result, method=method, config=config)
            if verbose:
                print(
                    f"[Split-MNIST] seed {index + 1}/{len(seeds)}: "
                    f"{seed} (reutilizada)",
                    flush=True,
                )
            continue
        if verbose:
            print(f"[Split-MNIST] seed {index + 1}/{len(seeds)}: {seed}", flush=True)
        tasks = loader(
            config,
            data_dir=data_dir,
            download=download if index == 0 else False,
        )
        write_json_atomic(
            saved_data_identity_path,
            {
                "schema_version": 1,
                "loader": identity["task_loader"],
                "tasks_sha256": task_data_fingerprint(tasks),
            },
        )
        raw[seed] = run_split_mnist(
            config,
            tasks,
            output_dir=seed_path,
            resume=resume,
        )
        for method, result in raw[seed].items():
            _backfill_cost_metadata(result, method=method, config=config)

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
                if len(differences) >= 2:
                    confirmatory: dict[str, Any] = paired_confirmatory_summary(
                        differences,
                        bootstrap_resamples=base_config.bootstrap_resamples,
                        bootstrap_seed=base_config.bootstrap_seed,
                    )
                else:
                    confirmatory = {
                        "available": False,
                        "n_pairs": len(differences),
                        "reason": "requires_at_least_two_paired_seeds",
                    }
                aggregate[comparison_key][method][metric]["confirmatory"] = (
                    confirmatory
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

    write_json_atomic(output_path / "aggregate.json", aggregate)
    write_json_atomic(
        output_path / "multi_seed_config.json",
        {"base_config": config_payload(base_config), "seeds": seeds},
    )
    with atomic_text_writer(output_path / "aggregate.csv", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    if paired_rows:
        with atomic_text_writer(
            output_path / "paired_differences.csv", newline=""
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

    write_json_atomic(output_path / "epoch_sweep.json", sweep)
    with atomic_text_writer(output_path / "epoch_sweep.csv", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sweep
