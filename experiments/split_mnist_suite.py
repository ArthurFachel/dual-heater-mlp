"""Notebook-friendly orchestration for baselines, fairness and ablations."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.split_mnist import (
    SplitMNISTConfig,
    config_payload,
    run_split_mnist_multi_seed,
)

CANDIDATE = "slowheat_replay_hidden_beta_30_budget_0.25"
SLOWHEAT_DERPP = "slowheat_derpp_hidden_beta_30_budget_0.25"

ALL_BASELINES = (
    "vanilla",
    "replay",
    "derpp",
    "er_ace",
    "agem",
    "ewc",
    "si",
    "lwf_calibrated",
    "replay_balanced",
    "replay_more_epochs",
    "replay_early_stopping",
    CANDIDATE,
)

ABLATION_METHODS = (
    "replay",
    CANDIDATE,
    "slowheat_hidden_beta_30_budget_0.25",
    "slowheat_replay_hidden_adaptive_beta_30_budget_0.25",
    "slowheat_replay_partial_output_beta_30_budget_0.25",
    "slowheat_replay_hidden_beta_30_budget_0.25_calibrated",
    "replay_global_lr_reduction",
)

SLOWHEAT_DERPP_METHODS = (
    "replay",
    "derpp",
    CANDIDATE,
    SLOWHEAT_DERPP,
)

# Complete set of methods implemented by the visual benchmark engine. The
# three explicit beta variants are structured configurations accepted by the
# same engine and used by the project's original Split-MNIST comparison.
ALL_VISUAL_METHODS = (
    "vanilla",
    "slowheat",
    "slowheat_beta_10",
    "slowheat_beta_30",
    "slowheat_beta_100",
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
    SLOWHEAT_DERPP,
    "er_ace",
    "agem",
    "ewc",
    "si",
    "lwf_calibrated",
    "replay_balanced",
    "replay_more_epochs",
    "replay_early_stopping",
    "replay_global_lr_reduction",
    CANDIDATE,
    "slowheat_hidden_beta_30_budget_0.25",
    "slowheat_replay_hidden_adaptive_beta_30_budget_0.25",
    "slowheat_replay_partial_output_beta_30_budget_0.25",
    "slowheat_replay_hidden_beta_30_budget_0.25_calibrated",
)

CLASS_ORDERS = (
    tuple(range(10)),
    (8, 9, 6, 7, 4, 5, 2, 3, 0, 1),
    (2, 7, 1, 6, 4, 9, 0, 5, 3, 8),
    (5, 0, 8, 3, 6, 1, 9, 4, 7, 2),
    (1, 4, 7, 0, 3, 6, 9, 2, 5, 8),
)


def baseline_config(*, device: str = "cpu") -> SplitMNISTConfig:
    """Return one configuration that can execute every requested baseline."""

    return SplitMNISTConfig(
        hidden_dims=(256, 128),
        batch_size=128,
        epochs_per_task=10,
        train_per_class=1_000,
        validation_per_class=200,
        test_per_class=500,
        learning_rate=1e-3,
        weight_decay=1e-4,
        slow_strength=30.0,
        plasticity_budget=0.25,
        optimizer_state_policy="follow_update",
        replay_per_class=20,
        replay_batch_size=64,
        # Fixed before baseline execution. These are baseline defaults, not
        # hyperparameters selected on the confirmatory seeds.
        derpp_alpha=0.5,
        derpp_beta=0.5,
        ewc_lambda=100.0,
        ewc_decay=1.0,
        si_lambda=1.0,
        si_epsilon=0.1,
        distillation_temperature=2.0,
        lwf_old_class_weight=1.0,
        replay_more_epochs=20,
        early_stopping_max_epochs=30,
        early_stopping_patience=3,
        methods=ALL_BASELINES,
        device=device,
    )


def _run(
    config: SplitMNISTConfig,
    *,
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    download: bool,
    verbose: bool,
    paired_references: tuple[str, ...] = ("replay",),
    resume: bool = False,
) -> dict[str, Any]:
    return run_split_mnist_multi_seed(
        config,
        seeds=seeds,
        data_dir=data_dir,
        output_dir=output_dir,
        download=download,
        verbose=verbose,
        paired_references=paired_references,
        resume=resume,
    )


def run_all_baselines(
    *,
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute all baselines serially inside one notebook call."""

    return _run(
        baseline_config(device=device),
        seeds=seeds,
        data_dir=data_dir,
        output_dir=output_dir,
        download=download,
        verbose=verbose,
        paired_references=("replay", "derpp"),
        resume=resume,
    )


def run_all_visual_methods(
    *,
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute every method supported by the visual benchmark engine."""

    config = replace(baseline_config(device=device), methods=ALL_VISUAL_METHODS)
    return _run(
        config,
        seeds=seeds,
        data_dir=data_dir,
        output_dir=output_dir,
        download=download,
        verbose=verbose,
        paired_references=("replay", "derpp"),
        resume=resume,
    )


def run_equal_example_budget(
    *,
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Cap current+replay learner examples at the vanilla ten-epoch budget."""

    config = replace(
        baseline_config(device=device),
        methods=tuple(
            method
            for method in ALL_BASELINES
            if method not in {"replay_more_epochs", "replay_early_stopping"}
        ),
        max_train_examples_per_task=20_000,
    )
    return _run(
        config,
        seeds=seeds,
        data_dir=data_dir,
        output_dir=output_dir,
        download=download,
        verbose=verbose,
        paired_references=("replay", "derpp"),
        resume=resume,
    )


def run_ablation_matrix(
    *,
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Run remaining method ablations and replay-memory sizes."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = replace(baseline_config(device=device), methods=ABLATION_METHODS)
    results: dict[str, Any] = {
        "method_ablations": _run(
            base,
            seeds=seeds,
            data_dir=data_dir,
            output_dir=root / "methods",
            download=download,
            verbose=verbose,
            resume=resume,
        ),
        "memory_sizes": {},
    }
    for index, memory_size in enumerate((5, 10, 20, 50, 100)):
        memory_config = replace(
            base,
            replay_per_class=memory_size,
            methods=("replay", CANDIDATE),
        )
        results["memory_sizes"][str(memory_size)] = _run(
            memory_config,
            seeds=seeds,
            data_dir=data_dir,
            output_dir=root / f"memory_{memory_size}",
            download=download if index == 0 else False,
            verbose=verbose,
            resume=resume,
        )
    with (root / "ablation_index.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seeds": seeds,
                "method_config": config_payload(base),
                "memory_sizes": [5, 10, 20, 50, 100],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return results


def run_slowheat_derpp_test(
    *,
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Compare DER++ and SlowHeat+DER++ as a separate exploratory test."""

    config = replace(
        baseline_config(device=device),
        methods=SLOWHEAT_DERPP_METHODS,
    )
    return _run(
        config,
        seeds=seeds,
        data_dir=data_dir,
        output_dir=output_dir,
        download=download,
        verbose=verbose,
        paired_references=("replay", "derpp"),
        resume=resume,
    )


def run_order_and_capacity_generalization(
    *,
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Run fixed class orders, larger MLPs and memory budgets."""

    root = Path(output_dir)
    base = replace(
        baseline_config(device=device), methods=SLOWHEAT_DERPP_METHODS
    )
    results: dict[str, Any] = {"class_orders": {}, "architectures": {}}
    for index, order in enumerate(CLASS_ORDERS):
        results["class_orders"][str(index)] = _run(
            replace(base, class_order=order),
            seeds=seeds,
            data_dir=data_dir,
            output_dir=root / f"order_{index}",
            download=download if index == 0 else False,
            verbose=verbose,
            paired_references=("replay", "derpp"),
            resume=resume,
        )
    for name, hidden_dims in {
        "mlp_256_128": (256, 128),
        "mlp_512_256": (512, 256),
        "mlp_512_512_256": (512, 512, 256),
    }.items():
        results["architectures"][name] = _run(
            replace(base, hidden_dims=hidden_dims),
            seeds=seeds,
            data_dir=data_dir,
            output_dir=root / name,
            download=False,
            verbose=verbose,
            paired_references=("replay", "derpp"),
            resume=resume,
        )
    return results
