"""Calibration and paired benchmarks for Functional DualHeat on CIFAR-10."""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.artifacts import atomic_text_writer, write_json_atomic
from experiments.confirmatory_statistics import PRIMARY_ENDPOINT
from experiments.split_mnist import SplitMNISTConfig, run_split_mnist_multi_seed
from experiments.visual_generalization import (
    generalization_configs,
    load_split_cifar10,
)

PILOT_SEEDS = (698_971_273, 1_288_660_088, 1_181_804_493)
MAIN_BENCHMARK_SEEDS = (
    2_101_606_466,
    1_872_839_281,
    637_029_796,
    1_357_204_345,
    1_758_462_037,
    748_737_880,
    1_611_510_062,
    333_205_446,
    130_501_536,
    492_612_040,
)
PILOT_ARCHITECTURES = ("vgg11", "resnet18")
PILOT_GRID = tuple(
    {
        "fast_decay": decay,
        "fast_strength": strength,
        "fast_threshold": threshold,
        "fast_eps": 1e-8,
    }
    for decay, strength, threshold in itertools.product(
        (0.90, 0.97),
        (0.5, 2.0),
        (0.0, 0.5),
    )
)
FUNCTIONAL_DUALHEAT_BENCHMARK_METHODS = (
    "vanilla",
    "fastheat",
    "slowheat",
    "dualheat",
    "lpr",
    "slowheat_lpr",
    "dualheat_lpr",
    "classifier_expander",
    "slowheat_classifier_expander",
    "dualheat_classifier_expander",
    "scroll",
    "slowheat_scroll",
    "dualheat_scroll",
)
PRIMARY_PAIRS = (
    ("slowheat", "dualheat"),
    ("slowheat_lpr", "dualheat_lpr"),
    ("slowheat_classifier_expander", "dualheat_classifier_expander"),
    ("slowheat_scroll", "dualheat_scroll"),
)
SECONDARY_PAIRS = (
    ("vanilla", "fastheat"),
    ("vanilla", "dualheat"),
    ("lpr", "dualheat_lpr"),
    ("classifier_expander", "dualheat_classifier_expander"),
    ("scroll", "dualheat_scroll"),
)
FASTHEAT_MANIFEST_SCHEMA_VERSION = 1


def _architecture_config(
    architecture: str,
    *,
    device: str,
) -> SplitMNISTConfig:
    configs = generalization_configs(device)
    names = {
        "vgg11": "split_cifar10_vgg11_all_methods",
        "resnet18": "split_cifar10_resnet18_all_methods",
    }
    if architecture not in names:
        raise ValueError("architecture deve ser 'vgg11' ou 'resnet18'")
    return configs[names[architecture]]


def _candidate_name(candidate: dict[str, float]) -> str:
    return (
        f"alpha_{candidate['fast_decay']:.2f}"
        f"_gamma_{candidate['fast_strength']:.1f}"
        f"_delta_{candidate['fast_threshold']:.1f}"
    )


def _final_validation_accuracy(result: dict[str, Any]) -> float:
    history = result.get("validation_history")
    if not isinstance(history, list) or not history or not history[-1]:
        raise RuntimeError("resultado não contém validação final completa")
    value = float(history[-1][-1])
    if not math.isfinite(value):
        raise RuntimeError("acurácia final de validação não é finita")
    return value


def select_fastheat_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered validation objective and exact tie-break."""

    if not candidates:
        raise ValueError("candidates não pode ser vazio")
    return min(
        candidates,
        key=lambda item: (
            -float(item["mean_paired_validation_difference"]),
            float(item["fast_strength"]),
            float(item["fast_decay"]),
            -float(item["fast_threshold"]),
        ),
    )


def run_functional_dualheat_pilot(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the complete 2x2x2 grid and freeze a validation-only selection."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    run_index = 0
    for candidate in PILOT_GRID:
        paired_differences: list[dict[str, Any]] = []
        for architecture in PILOT_ARCHITECTURES:
            base = _architecture_config(architecture, device=device)
            config = replace(
                base,
                methods=("slowheat", "dualheat"),
                epochs_per_task=5,
                slow_strength=30.0,
                plasticity_budget=0.25,
                optimizer_state_policy="follow_update",
                **candidate,
            )
            run_dir = destination / "runs" / _candidate_name(candidate) / architecture
            run_split_mnist_multi_seed(
                config,
                seeds=list(PILOT_SEEDS),
                data_dir=data_dir,
                output_dir=run_dir,
                download=download if run_index == 0 else False,
                verbose=verbose,
                paired_references=("slowheat",),
                task_loader=load_split_cifar10,
                resume=resume,
            )
            run_index += 1
            for seed in PILOT_SEEDS:
                with (run_dir / f"seed_{seed}" / "results.json").open(
                    encoding="utf-8"
                ) as handle:
                    result = json.load(handle)
                slowheat = _final_validation_accuracy(result["slowheat"])
                dualheat = _final_validation_accuracy(result["dualheat"])
                paired_differences.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "slowheat_validation_accuracy": slowheat,
                        "dualheat_validation_accuracy": dualheat,
                        "difference": dualheat - slowheat,
                    }
                )
        candidates.append(
            {
                **candidate,
                "mean_paired_validation_difference": sum(
                    item["difference"] for item in paired_differences
                )
                / len(paired_differences),
                "paired_differences": paired_differences,
            }
        )

    # Exact numerical ties prefer weaker intervention: gamma, alpha, then -delta.
    selected = select_fastheat_candidate(candidates)
    manifest = {
        "schema_version": FASTHEAT_MANIFEST_SCHEMA_VERSION,
        "status": "frozen_before_main_benchmark",
        "method": "Functional DualHeat = activation FastHeat + Functional SlowHeat",
        "selection_data": "validation_only",
        "selection_endpoint": (
            "mean paired DualHeat - SlowHeat final Class-IL validation accuracy, "
            "equal weight per architecture and seed"
        ),
        "tie_break": [
            "lower_fast_strength",
            "lower_fast_decay",
            "higher_fast_threshold",
        ],
        "architectures": list(PILOT_ARCHITECTURES),
        "seeds": list(PILOT_SEEDS),
        "epochs_per_task": 5,
        "slowheat": {
            "slow_strength": 30.0,
            "plasticity_budget": 0.25,
            "consolidation": "max",
            "optimizer_state_policy": "follow_update",
        },
        "grid": list(PILOT_GRID),
        "selected": {
            key: selected[key]
            for key in (
                "fast_decay",
                "fast_strength",
                "fast_threshold",
                "fast_eps",
                "mean_paired_validation_difference",
            )
        },
    }
    payload = {"manifest": manifest, "candidates": candidates}
    write_json_atomic(destination / "pilot_results.json", payload)
    write_json_atomic(destination / "selected_fastheat_config.json", manifest)
    return payload


def load_frozen_fastheat_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"manifesto FastHeat congelado não encontrado: {source}; execute o piloto"
        )
    with source.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    selected = manifest.get("selected", {})
    required = {"fast_decay", "fast_strength", "fast_threshold", "fast_eps"}
    if (
        manifest.get("schema_version") != FASTHEAT_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "frozen_before_main_benchmark"
        or not required <= set(selected)
    ):
        raise RuntimeError("manifesto FastHeat inválido ou não congelado")
    candidate = {key: float(selected[key]) for key in required}
    if candidate not in PILOT_GRID:
        raise RuntimeError("configuração congelada não pertence à grade pré-registrada")
    return manifest


def functional_dualheat_benchmark_config(
    architecture: str,
    *,
    manifest: dict[str, Any],
    device: str = "cpu",
) -> SplitMNISTConfig:
    selected = manifest["selected"]
    return replace(
        _architecture_config(architecture, device=device),
        methods=FUNCTIONAL_DUALHEAT_BENCHMARK_METHODS,
        epochs_per_task=5,
        slow_strength=30.0,
        plasticity_budget=0.25,
        optimizer_state_policy="follow_update",
        fast_decay=float(selected["fast_decay"]),
        fast_strength=float(selected["fast_strength"]),
        fast_threshold=float(selected["fast_threshold"]),
        fast_eps=float(selected["fast_eps"]),
    )


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(
        sorted(range(len(p_values)), key=p_values.__getitem__)
    ):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted


def _comparison(
    aggregate: dict[str, Any], reference: str, candidate: str
) -> dict[str, Any]:
    return aggregate[f"paired_differences_vs_{reference}"][candidate]


def _validate_pair_fairness(
    output_dir: Path,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for reference, candidate in PRIMARY_PAIRS:
        for seed in seeds:
            with (output_dir / f"seed_{seed}" / "results.json").open(
                encoding="utf-8"
            ) as handle:
                result = json.load(handle)
            left = result[reference]
            right = result[candidate]
            checks.append(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "seed": seed,
                    "completed_epochs_equal": (
                        left["completed_epochs"] == right["completed_epochs"]
                    ),
                    "current_examples_equal": (
                        left["cost"]["current_examples"]
                        == right["cost"]["current_examples"]
                    ),
                    "replay_examples_equal": (
                        left["cost"]["replay_examples"]
                        == right["cost"]["replay_examples"]
                    ),
                    "selection_history_equal": (
                        left["selection_history"] == right["selection_history"]
                    ),
                }
            )
    if any(
        not all(value for key, value in check.items() if key.endswith("_equal"))
        for check in checks
    ):
        raise RuntimeError("falha nas invariantes pareadas do benchmark DualHeat")
    return checks


def _write_analysis(
    *,
    architecture: str,
    aggregate: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
    fairness: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = []
    p_values = []
    for reference, candidate in PRIMARY_PAIRS:
        metrics = _comparison(aggregate, reference, candidate)
        endpoint = metrics[PRIMARY_ENDPOINT]
        p_value = endpoint["confirmatory"]["student_t"]["two_sided_p"]
        p_values.append(float(p_value))
        primary.append(
            {
                "reference": reference,
                "candidate": candidate,
                "metrics": metrics,
            }
        )
    for item, adjusted in zip(primary, _holm_adjust(p_values), strict=True):
        item["metrics"][PRIMARY_ENDPOINT]["holm_adjusted_p"] = adjusted
    report = {
        "status": "exploratory_paired_benchmark",
        "architecture": architecture,
        "methods": list(FUNCTIONAL_DUALHEAT_BENCHMARK_METHODS),
        "seeds": list(MAIN_BENCHMARK_SEEDS),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "multiplicity": "Holm over four accuracy contrasts within architecture",
        "fastheat_manifest": manifest,
        "primary_contrasts": primary,
        "secondary_contrasts": [
            {
                "reference": reference,
                "candidate": candidate,
                "metrics": _comparison(aggregate, reference, candidate),
            }
            for reference, candidate in SECONDARY_PAIRS
        ],
        "fairness_checks": fairness,
        "interpretation": (
            "Exploratory only. Bootstrap intervals, seed signs, forgetting, "
            "task-aware accuracy, classifier gap, time, FLOPs and memory are "
            "reported in each paired metric block."
        ),
    }
    write_json_atomic(output_dir / "functional_dualheat_analysis.json", report)
    rows = []
    for item in primary:
        accuracy = item["metrics"][PRIMARY_ENDPOINT]
        confirmatory = accuracy["confirmatory"]
        rows.append(
            {
                "architecture": architecture,
                "reference": item["reference"],
                "candidate": item["candidate"],
                "mean_difference": confirmatory["mean_difference"],
                "bootstrap_ci95_low": confirmatory["paired_bootstrap"][
                    "ci95_percentile"
                ][0],
                "bootstrap_ci95_high": confirmatory["paired_bootstrap"][
                    "ci95_percentile"
                ][1],
                "positive_seeds": confirmatory["signs"]["positive"],
                "negative_seeds": confirmatory["signs"]["negative"],
                "ties": confirmatory["signs"]["ties"],
                "student_t_p": confirmatory["student_t"]["two_sided_p"],
                "holm_adjusted_p": accuracy["holm_adjusted_p"],
            }
        )
    with atomic_text_writer(
        output_dir / "functional_dualheat_primary.csv", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def run_functional_dualheat_benchmark(
    architecture: str,
    *,
    manifest_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Run a fresh 13-method benchmark using the frozen pilot selection."""

    manifest = load_frozen_fastheat_manifest(manifest_path)
    config = functional_dualheat_benchmark_config(
        architecture, manifest=manifest, device=device
    )
    destination = Path(output_dir)
    aggregate = run_split_mnist_multi_seed(
        config,
        seeds=list(MAIN_BENCHMARK_SEEDS),
        data_dir=data_dir,
        output_dir=destination,
        download=download,
        verbose=verbose,
        paired_references=(
            "vanilla",
            "slowheat",
            "lpr",
            "slowheat_lpr",
            "classifier_expander",
            "slowheat_classifier_expander",
            "scroll",
            "slowheat_scroll",
        ),
        task_loader=load_split_cifar10,
        resume=resume,
    )
    write_json_atomic(destination / "fastheat_manifest.json", manifest)
    fairness = _validate_pair_fairness(destination, MAIN_BENCHMARK_SEEDS)
    report = _write_analysis(
        architecture=architecture,
        aggregate=aggregate,
        manifest=manifest,
        output_dir=destination,
        fairness=fairness,
    )
    return {"aggregate": aggregate, "analysis": report}
