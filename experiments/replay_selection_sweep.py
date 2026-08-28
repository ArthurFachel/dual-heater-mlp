"""Exploratory sweep for model-ranked replay memories across visual streams."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.artifacts import (
    atomic_text_writer,
    read_json_object,
    write_json_atomic,
)
from experiments.confirmatory_split_mnist import CONFIRMATORY_SEEDS
from experiments.confirmatory_statistics import (
    PRIMARY_ENDPOINT,
    normal_summary,
    paired_confirmatory_summary,
)
from experiments.replay_memory import REPLAY_SELECTION_STRATEGIES
from experiments.split_mnist import (
    AGGREGATE_METRICS,
    SplitMNISTConfig,
    _result_metric,
    config_payload,
    run_split_mnist_multi_seed,
)
from experiments.split_mnist_suite import CANDIDATE, SLOWHEAT_DERPP, baseline_config
from experiments.visual_generalization import (
    generalization_configs,
    load_permuted_mnist,
    load_split_cifar10,
    load_split_cifar100,
)

SWEEP_SCHEMA_VERSION = 2
NO_MEMORY_CANDIDATE = "slowheat_hidden_beta_30_budget_0.25"
SWEEP_MEMORY_METHODS = ("replay", CANDIDATE, "derpp", SLOWHEAT_DERPP)
SWEEP_NO_MEMORY_METHODS = ("vanilla", NO_MEMORY_CANDIDATE)
SWEEP_METHODS = (*SWEEP_NO_MEMORY_METHODS, *SWEEP_MEMORY_METHODS)
SLOWHEAT_MEMORY_PAIRS = (
    ("slowheat_vs_replay", "replay", CANDIDATE),
    ("slowheat_vs_derpp", "derpp", SLOWHEAT_DERPP),
)
NO_MEMORY_REFERENCES = {
    "replay": "vanilla",
    CANDIDATE: NO_MEMORY_CANDIDATE,
    "derpp": "vanilla",
    SLOWHEAT_DERPP: NO_MEMORY_CANDIDATE,
}
SWEEP_DATASETS = (
    "split_mnist",
    "permuted_mnist",
    "split_cifar10",
    "split_cifar100",
    "split_cifar10_cnn",
)


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted


def replay_selection_configs(device: str = "cpu") -> dict[str, SplitMNISTConfig]:
    visual = generalization_configs(device)
    return {
        "split_mnist": replace(
            baseline_config(device=device), methods=SWEEP_METHODS
        ),
        "permuted_mnist": replace(
            visual["permuted_mnist"], methods=SWEEP_METHODS
        ),
        "split_cifar10": replace(
            visual["split_cifar10"], methods=SWEEP_METHODS
        ),
        "split_cifar100": replace(
            visual["split_cifar100"], methods=SWEEP_METHODS
        ),
        "split_cifar10_cnn": replace(
            visual["split_cifar10_cnn"], methods=SWEEP_METHODS
        ),
    }


def _loaders() -> dict[str, Any]:
    return {
        "split_mnist": None,
        "permuted_mnist": load_permuted_mnist,
        "split_cifar10": load_split_cifar10,
        "split_cifar100": load_split_cifar100,
        "split_cifar10_cnn": load_split_cifar10,
    }


def _raw_result(output: Path, seed: int) -> dict[str, dict[str, Any]]:
    return read_json_object(output / f"seed_{seed}" / "results.json")


def _build_report(
    root: Path,
    *,
    seeds: list[int],
    datasets: tuple[str, ...],
    selectors: tuple[str, ...],
    configs: dict[str, SplitMNISTConfig],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw: dict[str, dict[str, dict[int, dict[str, dict[str, Any]]]]] = {}
    no_memory_raw: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for dataset in datasets:
        no_memory_raw[dataset] = {
            seed: _raw_result(root / dataset / "no_memory", seed) for seed in seeds
        }
        raw[dataset] = {}
        for selector in selectors:
            raw[dataset][selector] = {
                seed: _raw_result(root / dataset / selector, seed) for seed in seeds
            }

    report: dict[str, Any] = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "status": "exploratory_not_independent_confirmation",
        "primary_endpoint": PRIMARY_ENDPOINT,
        "seeds": seeds,
        "datasets": list(datasets),
        "selectors": list(selectors),
        "learners": list(SWEEP_METHODS),
        "learner_run_count": len(seeds)
        * len(datasets)
        * (
            len(SWEEP_NO_MEMORY_METHODS)
            + len(selectors) * len(SWEEP_MEMORY_METHODS)
        ),
        "ranking_ownership": "each_learner_ranks_its_own_training_images",
        "no_memory_references": dict(NO_MEMORY_REFERENCES),
        "no_memory_execution": (
            "Vanilla is the no-cache control for Replay and DER++; hidden-only "
            "SlowHeat is the no-cache control for SlowHeat+Replay and "
            "SlowHeat+DER++. Both controls are trained once per dataset/seed."
        ),
        "comparison_warning": (
            "Every ranked learner selects its own memory. Replay/SlowHeat+Replay and "
            "DER++/SlowHeat+DER++ contrasts are algorithm-level comparisons, not "
            "isolated SlowHeat effects."
        ),
        "configs": {name: config_payload(configs[name]) for name in datasets},
        "summaries": {},
        "paired_differences_vs_first": {},
        "slowheat_vs_replay": {},
        "slowheat_vs_derpp": {},
        "memory_vs_no_memory": {},
        "multiplicity": (
            "Holm over loss, representative and hybrid final-accuracy contrasts "
            "within each dataset/backbone/learner family."
        ),
    }
    summary_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        report["summaries"][dataset] = {}
        report["paired_differences_vs_first"][dataset] = {}
        report["slowheat_vs_replay"][dataset] = {}
        report["slowheat_vs_derpp"][dataset] = {}
        report["memory_vs_no_memory"][dataset] = {}
        for method in SWEEP_METHODS:
            report["summaries"][dataset][method] = {}
            report["paired_differences_vs_first"][dataset][method] = {}
            method_selectors = (
                ("none",) if method in SWEEP_NO_MEMORY_METHODS else selectors
            )
            for selector in method_selectors:
                selector_summary: dict[str, Any] = {}
                for metric in AGGREGATE_METRICS:
                    values = [
                        _result_metric(
                            (
                                no_memory_raw[dataset][seed]
                                if selector == "none"
                                else raw[dataset][selector][seed]
                            )[method],
                            metric,
                        )
                        for seed in seeds
                    ]
                    selector_summary[metric] = normal_summary(values)
                    selector_summary[metric]["values"] = [
                        {"seed": seed, "value": value}
                        for seed, value in zip(seeds, values, strict=True)
                    ]
                    summary_rows.append(
                        {
                            "dataset": dataset,
                            "backbone": configs[dataset].backbone,
                            "learner": method,
                            "selector": selector,
                            "metric": metric,
                            **normal_summary(values),
                        }
                    )
                report["summaries"][dataset][method][selector] = selector_summary

            comparisons: list[tuple[str, dict[str, Any]]] = []
            if method in SWEEP_NO_MEMORY_METHODS:
                continue
            for selector in selectors:
                if selector == "first":
                    continue
                metric_summaries: dict[str, Any] = {}
                for metric in AGGREGATE_METRICS:
                    differences = [
                        _result_metric(raw[dataset][selector][seed][method], metric)
                        - _result_metric(raw[dataset]["first"][seed][method], metric)
                        for seed in seeds
                    ]
                    summary = normal_summary(differences)
                    summary["paired_values"] = [
                        {"seed": seed, "difference": difference}
                        for seed, difference in zip(seeds, differences, strict=True)
                    ]
                    summary["inference"] = (
                        paired_confirmatory_summary(
                            differences,
                            bootstrap_resamples=configs[dataset].bootstrap_resamples,
                            bootstrap_seed=configs[dataset].bootstrap_seed,
                        )
                        if len(seeds) >= 2
                        else {
                            "available": False,
                            "reason": "requires_at_least_two_paired_seeds",
                        }
                    )
                    metric_summaries[metric] = summary
                    difference_rows.extend(
                        {
                            "comparison": "selector_minus_first",
                            "dataset": dataset,
                            "backbone": configs[dataset].backbone,
                            "learner": method,
                            "selector": selector,
                            "reference": "first",
                            "metric": metric,
                            "seed": seed,
                            "difference": difference,
                        }
                        for seed, difference in zip(seeds, differences, strict=True)
                    )
                report["paired_differences_vs_first"][dataset][method][selector] = (
                    metric_summaries
                )
                comparisons.append((selector, metric_summaries))
            if len(seeds) >= 2 and comparisons:
                p_values = [
                    comparison[PRIMARY_ENDPOINT]["inference"]["student_t"][
                        "two_sided_p"
                    ]
                    for _, comparison in comparisons
                ]
                for (selector, comparison), adjusted in zip(
                    comparisons, _holm_adjust(p_values), strict=True
                ):
                    comparison[PRIMARY_ENDPOINT]["holm_adjusted_p"] = adjusted

        for report_key, reference, candidate in SLOWHEAT_MEMORY_PAIRS:
            for selector in selectors:
                metric_summaries = {}
                for metric in AGGREGATE_METRICS:
                    differences = [
                        _result_metric(
                            raw[dataset][selector][seed][candidate], metric
                        )
                        - _result_metric(
                            raw[dataset][selector][seed][reference], metric
                        )
                        for seed in seeds
                    ]
                    metric_summaries[metric] = normal_summary(differences)
                    metric_summaries[metric]["paired_values"] = [
                        {"seed": seed, "difference": difference}
                        for seed, difference in zip(seeds, differences, strict=True)
                    ]
                    difference_rows.extend(
                        {
                            "comparison": f"{candidate}_minus_{reference}",
                            "dataset": dataset,
                            "backbone": configs[dataset].backbone,
                            "learner": candidate,
                            "selector": selector,
                            "reference": reference,
                            "metric": metric,
                            "seed": seed,
                            "difference": difference,
                        }
                        for seed, difference in zip(
                            seeds, differences, strict=True
                        )
                    )
                report[report_key][dataset][selector] = metric_summaries

        for memory_method, no_memory_method in NO_MEMORY_REFERENCES.items():
            report["memory_vs_no_memory"][dataset][memory_method] = {}
            for selector in selectors:
                metric_summaries = {}
                for metric in AGGREGATE_METRICS:
                    differences = [
                        _result_metric(
                            raw[dataset][selector][seed][memory_method], metric
                        )
                        - _result_metric(
                            no_memory_raw[dataset][seed][no_memory_method], metric
                        )
                        for seed in seeds
                    ]
                    metric_summaries[metric] = normal_summary(differences)
                    metric_summaries[metric]["paired_values"] = [
                        {"seed": seed, "difference": difference}
                        for seed, difference in zip(seeds, differences, strict=True)
                    ]
                    difference_rows.extend(
                        {
                            "comparison": "memory_minus_no_memory",
                            "dataset": dataset,
                            "backbone": configs[dataset].backbone,
                            "learner": memory_method,
                            "selector": selector,
                            "reference": no_memory_method,
                            "metric": metric,
                            "seed": seed,
                            "difference": difference,
                        }
                        for seed, difference in zip(seeds, differences, strict=True)
                    )
                report["memory_vs_no_memory"][dataset][memory_method][selector] = (
                    metric_summaries
                )
    return report, summary_rows, difference_rows


def run_replay_selection_sweep(
    *,
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = True,
    datasets: tuple[str, ...] = SWEEP_DATASETS,
    selectors: tuple[str, ...] = REPLAY_SELECTION_STRATEGIES,
) -> dict[str, Any]:
    """Run and aggregate the fixed replay-selection matrix."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds deve ser não vazio e sem duplicatas")
    if set(seeds).intersection(CONFIRMATORY_SEEDS):
        raise ValueError("seeds confirmatórias são reservadas e não podem entrar no sweep")
    if not datasets or any(name not in SWEEP_DATASETS for name in datasets):
        raise ValueError("dataset inválido no sweep de replay")
    if (
        not selectors
        or "first" not in selectors
        or any(selector not in REPLAY_SELECTION_STRATEGIES for selector in selectors)
    ):
        raise ValueError("selectors deve incluir first e apenas estratégias conhecidas")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "sweep_index.json"
    configs = replay_selection_configs(device)
    identity = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "status": "running",
        "seeds": seeds,
        "datasets": list(datasets),
        "selectors": list(selectors),
        "methods": list(SWEEP_METHODS),
        "learner_run_count": len(seeds)
        * len(datasets)
        * (
            len(SWEEP_NO_MEMORY_METHODS)
            + len(selectors) * len(SWEEP_MEMORY_METHODS)
        ),
        "configs": {name: config_payload(configs[name]) for name in datasets},
    }
    if index_path.is_file():
        saved = read_json_object(index_path)
        comparable = {key: saved.get(key) for key in identity if key != "status"}
        expected = {key: value for key, value in identity.items() if key != "status"}
        if not resume or comparable != expected:
            raise RuntimeError("índice existente é incompatível ou resume está desativado")
    write_json_atomic(index_path, identity)

    loaders = _loaders()
    for dataset in datasets:
        if verbose:
            print(f"[replay-selection] {dataset}/no_memory", flush=True)
        run_split_mnist_multi_seed(
            replace(
                configs[dataset],
                methods=SWEEP_NO_MEMORY_METHODS,
                replay_selection="first",
            ),
            seeds=seeds,
            data_dir=data_dir,
            output_dir=root / dataset / "no_memory",
            download=download,
            verbose=verbose,
            paired_references=("vanilla",),
            task_loader=loaders[dataset],
            resume=resume,
        )
        for selector in selectors:
            if verbose:
                print(f"[replay-selection] {dataset}/{selector}", flush=True)
            config = replace(
                configs[dataset],
                methods=SWEEP_MEMORY_METHODS,
                replay_selection=selector,
            )
            run_split_mnist_multi_seed(
                config,
                seeds=seeds,
                data_dir=data_dir,
                output_dir=root / dataset / selector,
                download=False,
                verbose=verbose,
                paired_references=("replay", "derpp"),
                task_loader=loaders[dataset],
                resume=resume,
            )

    report, summary_rows, difference_rows = _build_report(
        root,
        seeds=seeds,
        datasets=datasets,
        selectors=selectors,
        configs=configs,
    )
    write_json_atomic(root / "sweep_report.json", report)
    with atomic_text_writer(root / "sweep_summary.csv", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with atomic_text_writer(root / "sweep_differences.csv", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(difference_rows[0]))
        writer.writeheader()
        writer.writerows(difference_rows)
    identity["status"] = "completed"
    identity["report"] = "sweep_report.json"
    write_json_atomic(index_path, identity)
    return report
