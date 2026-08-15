"""Deterministic, CPU-first synthetic class-incremental benchmark."""

from __future__ import annotations

import argparse
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

from dual_heater import (
    SlowHeatAdamW,
    SlowHeatMLP,
    SlowHeatSGD,
    compute_cl_metrics,
)


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    n_features: int = 16
    classes_per_task: int = 2
    task_count: int = 2
    train_per_class: int = 24
    test_per_class: int = 12
    hidden_dims: tuple[int, ...] = (32, 16)
    batch_size: int = 16
    steps_per_task: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    slow_strength: float = 3.0
    plasticity_budget: float = 0.25
    optimizer_state_policy: str = "follow_update"
    reduced_lr_factor: float = 0.1
    methods: tuple[str, ...] = ("vanilla", "reduced_lr", "slowheat_max")

    @property
    def class_count(self) -> int:
        return self.classes_per_task * self.task_count

    def validate(self) -> None:
        positive_integers = {
            "n_features": self.n_features,
            "classes_per_task": self.classes_per_task,
            "task_count": self.task_count,
            "train_per_class": self.train_per_class,
            "test_per_class": self.test_per_class,
            "batch_size": self.batch_size,
            "steps_per_task": self.steps_per_task,
        }
        for name, value in positive_integers.items():
            if value < 1:
                raise ValueError(f"{name} deve ser >= 1")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims deve conter dimensões positivas")
        float_values = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "slow_strength": self.slow_strength,
            "plasticity_budget": self.plasticity_budget,
            "reduced_lr_factor": self.reduced_lr_factor,
        }
        for name, value in float_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} deve ser finito")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate deve ser > 0")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay deve ser >= 0")
        if self.slow_strength < 0.0:
            raise ValueError("slow_strength deve ser >= 0")
        if not 0.0 <= self.plasticity_budget <= 1.0:
            raise ValueError("plasticity_budget deve estar em [0, 1]")
        if self.optimizer_state_policy not in {"native", "follow_update"}:
            raise ValueError(
                "optimizer_state_policy deve ser 'native' ou 'follow_update'"
            )
        if not 0.0 < self.reduced_lr_factor <= 1.0:
            raise ValueError("reduced_lr_factor deve estar em (0, 1]")
        if not self.methods:
            raise ValueError("methods não pode ser vazio")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods não pode conter duplicatas")
        supported = {
            "vanilla",
            "reduced_lr",
            "slowheat_max",
            "slowheat_mean",
            "slowheat_sum",
            "slowheat_none",
            "slowheat_max_sgd",
            "slowheat_max_legacy_adamw",
            "slowheat_max_native_state",
            "slowheat_max_unidirectional",
            "slowheat_max_unbudgeted",
        }
        unknown = set(self.methods) - supported
        if unknown:
            raise ValueError(f"métodos desconhecidos: {sorted(unknown)}")


def _vanilla_mlp(dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for input_dim, output_dim in pairwise(dims[:-1]):
        layers.extend((nn.Linear(input_dim, output_dim), nn.GELU()))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def build_paired_models(config: SyntheticConfig) -> dict[str, nn.Module]:
    """Build methods with byte-identical trainable parameter initialization."""

    config.validate()
    dims = (config.n_features, *config.hidden_dims, config.class_count)
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
            if method in {"vanilla", "reduced_lr"}:
                model: nn.Module = _vanilla_mlp(dims)
            elif method.startswith("slowheat_"):
                budget = (
                    0.0
                    if method == "slowheat_max_unbudgeted"
                    else config.plasticity_budget
                )
                model = SlowHeatMLP(
                    *dims,
                    act="gelu",
                    slow_strength=config.slow_strength,
                    plasticity_budget=budget,
                    protect_output=True,
                )
            else:  # guarded by validate
                raise AssertionError(method)

            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(reference_parameters[name])
            models[method] = model.cpu()
    return models


def make_batch_schedule(
    *,
    sample_count: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> list[Tensor]:
    """Precompute a deterministic sequence of minibatch indices."""

    if sample_count < 1 or batch_size < 1 or steps < 1:
        raise ValueError("sample_count, batch_size e steps devem ser >= 1")
    generator = torch.Generator().manual_seed(seed)
    stream = torch.empty(0, dtype=torch.long)
    schedule: list[Tensor] = []
    for _ in range(steps):
        while stream.numel() < batch_size:
            stream = torch.cat((stream, torch.randperm(sample_count, generator=generator)))
        schedule.append(stream[:batch_size].clone())
        stream = stream[batch_size:]
    return schedule


def _make_tasks(config: SyntheticConfig) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(config.seed)
    centers = torch.randn(
        config.class_count,
        config.n_features,
        generator=generator,
    ) * 1.5
    tasks: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for task in range(config.task_count):
        train_inputs: list[Tensor] = []
        train_targets: list[Tensor] = []
        test_inputs: list[Tensor] = []
        test_targets: list[Tensor] = []
        start = task * config.classes_per_task
        stop = start + config.classes_per_task
        for label in range(start, stop):
            train_inputs.append(
                centers[label]
                + 0.8
                * torch.randn(
                    config.train_per_class,
                    config.n_features,
                    generator=generator,
                )
            )
            train_targets.append(
                torch.full((config.train_per_class,), label, dtype=torch.long)
            )
            test_inputs.append(
                centers[label]
                + 0.8
                * torch.randn(
                    config.test_per_class,
                    config.n_features,
                    generator=generator,
                )
            )
            test_targets.append(
                torch.full((config.test_per_class,), label, dtype=torch.long)
            )
        tasks.append(
            (
                torch.cat(train_inputs),
                torch.cat(train_targets),
                torch.cat(test_inputs),
                torch.cat(test_targets),
            )
        )
    return tasks


@torch.no_grad()
def _accuracy(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    seen_class_count: int,
) -> float:
    model.eval()
    predictions = model(inputs)[..., :seen_class_count].argmax(dim=-1)
    return float((predictions == targets).float().mean().item())


def _build_optimizer(
    method: str,
    model: nn.Module,
    config: SyntheticConfig,
) -> torch.optim.Optimizer:
    learning_rate = config.learning_rate
    if method == "reduced_lr":
        learning_rate *= config.reduced_lr_factor
    if method.startswith("slowheat_") and method != "slowheat_max_legacy_adamw":
        if method == "slowheat_max_sgd":
            optimizer = SlowHeatSGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=config.weight_decay,
                state_policy=config.optimizer_state_policy,
            )
        else:
            state_policy = (
                "native"
                if method == "slowheat_max_native_state"
                else config.optimizer_state_policy
            )
            optimizer = SlowHeatAdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=config.weight_decay,
                state_policy=state_policy,
            )
        assert isinstance(model, SlowHeatMLP)
        if method == "slowheat_max_unidirectional":
            for layer in model.get_slow_layers():
                optimizer.register_slow_heat_module(layer)
        else:
            optimizer.register_slow_heat_model(model)
        return optimizer
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )


def _json_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in matrix
    ]


def run_experiment(
    config: SyntheticConfig,
    *,
    output_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Run all methods on identical data, initialization and minibatches."""

    config.validate()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tasks = _make_tasks(config)
    schedules = [
        make_batch_schedule(
            sample_count=len(task[0]),
            batch_size=config.batch_size,
            steps=config.steps_per_task,
            seed=config.seed + task_index,
        )
        for task_index, task in enumerate(tasks)
    ]
    models = build_paired_models(config)
    results: dict[str, dict[str, Any]] = {}

    for method, model in models.items():
        optimizer = _build_optimizer(method, model, config)
        matrix = np.full((config.task_count, config.task_count), np.nan)
        baseline_scores = np.asarray(
            [
                _accuracy(
                    model,
                    task[2],
                    task[3],
                    seen_class_count=(task_index + 1) * config.classes_per_task,
                )
                for task_index, task in enumerate(tasks)
            ],
            dtype=np.float64,
        )
        pretrain_scores = np.empty(config.task_count, dtype=np.float64)
        losses: list[list[float]] = []
        started = time.perf_counter()

        for stage, (train_x, train_y, _, _) in enumerate(tasks):
            pretrain_scores[stage] = _accuracy(
                model,
                tasks[stage][2],
                tasks[stage][3],
                seen_class_count=(stage + 1) * config.classes_per_task,
            )
            model.train()
            task_losses: list[float] = []
            for indices in schedules[stage]:
                optimizer.zero_grad(set_to_none=True)
                seen_class_count = (stage + 1) * config.classes_per_task
                logits = model(train_x[indices])[..., :seen_class_count]
                loss = F.cross_entropy(logits, train_y[indices])
                loss.backward()
                optimizer.step()
                task_losses.append(float(loss.detach().item()))
            losses.append(task_losses)

            if method.startswith("slowheat_") and method != "slowheat_none":
                assert isinstance(model, SlowHeatMLP)
                if "mean" in method:
                    strategy = "mean"
                elif "sum" in method:
                    strategy = "sum"
                else:
                    strategy = "max"
                model.consolidate(strategy=strategy)

            for task_index in range(stage + 1):
                matrix[stage, task_index] = _accuracy(
                    model,
                    tasks[task_index][2],
                    tasks[task_index][3],
                    seen_class_count=(stage + 1) * config.classes_per_task,
                )

        elapsed = time.perf_counter() - started
        metrics = compute_cl_metrics(
            matrix,
            pretrain_scores=pretrain_scores,
            baseline_scores=baseline_scores,
        )
        results[method] = {
            "accuracy_matrix": _json_matrix(matrix),
            "baseline_scores": baseline_scores.tolist(),
            "pretrain_scores": pretrain_scores.tolist(),
            "training_losses": losses,
            "elapsed_seconds": elapsed,
            "metrics": asdict(metrics),
            "capacity": (
                [
                    layer.capacity_metrics()
                    for layer in model.get_slow_layers()
                ]
                if isinstance(model, SlowHeatMLP)
                else None
            ),
        }

    with (output_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True, allow_nan=False)
    with (output_path / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, allow_nan=False)
    with (output_path / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "final_average_accuracy",
                "average_forgetting",
                "backward_transfer",
                "forward_transfer",
                "elapsed_seconds",
            ),
        )
        writer.writeheader()
        for method, result in results.items():
            writer.writerow(
                {
                    "method": method,
                    **{
                        key: result["metrics"][key]
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


def load_config(path: str | Path) -> SyntheticConfig:
    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    values["hidden_dims"] = tuple(values["hidden_dims"])
    values["methods"] = tuple(values["methods"])
    return SyntheticConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_experiment(load_config(args.config), output_dir=args.output_dir)


if __name__ == "__main__":
    main()
