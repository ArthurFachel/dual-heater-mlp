"""Evaluate each MLP learner against the same learner + Functional SlowHeat.

DualHeat is the proposal's public name here, not the legacy DualHeatMLP class.
This exploratory suite never changes the frozen Split-MNIST confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from experiments.artifacts import write_json_atomic
from experiments.confirmatory_split_mnist import CONFIRMATORY_SEEDS
from experiments.confirmatory_statistics import (
    PRIMARY_ENDPOINT,
    paired_confirmatory_summary,
)
from experiments.provenance import relative_path
from experiments.split_mnist import (
    AGGREGATE_METRICS,
    SplitMNISTConfig,
    _result_metric,
    config_payload,
    run_split_mnist_multi_seed,
)
from experiments.split_mnist_suite import (
    CANDIDATE,
    SLOWHEAT_DERPP,
    SLOWHEAT_ER_ACE,
    baseline_config,
)
from experiments.visual_generalization import (
    generalization_configs,
    load_permuted_mnist,
    load_split_cifar10,
    load_split_cifar100,
)


@dataclass(frozen=True)
class MethodPair:
    label: str
    reference: str
    candidate: str


METHOD_PAIRS = (
    MethodPair(
        "Convencional (AdamW)", "vanilla", "slowheat_hidden_beta_30_budget_0.25"
    ),
    MethodPair("Replay", "replay", CANDIDATE),
    MethodPair("DER++", "derpp", SLOWHEAT_DERPP),
    MethodPair("ER-ACE", "er_ace", SLOWHEAT_ER_ACE),
)
PAIRED_METHODS = tuple(
    method for pair in METHOD_PAIRS for method in (pair.reference, pair.candidate)
)
DATASETS = ("split_mnist", "permuted_mnist", "split_cifar10", "split_cifar100")
# These quantities must be identical WITHIN a pair, not across learner families.
MATCHED_COSTS = (
    "model_parameters",
    "optimizer_steps",
    "current_examples",
    "replay_examples",
    "learner_examples_processed",
    "replay_memory_bytes",
    "stored_logits_bytes",
)


def paired_config(name: str = "split_mnist", device: str = "cpu") -> SplitMNISTConfig:
    if name not in DATASETS:
        raise ValueError(f"dataset não suportado pela suíte MLP: {name}")
    base = (
        baseline_config(device=device)
        if name == "split_mnist"
        else generalization_configs(device)[name]
    )
    return replace(
        base,
        methods=PAIRED_METHODS,
        slow_strength=30.0,
        plasticity_budget=0.25,
        optimizer_state_policy="follow_update",
    )


def _validate_seeds(seeds: list[int], *, new_run: bool = False) -> None:
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds deve conter inteiros não negativos")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds não pode conter duplicatas")
    if new_run and set(seeds).intersection(CONFIRMATORY_SEEDS):
        raise ValueError("seeds reservadas à confirmação não podem entrar nesta suíte")


def pair_protocol(config: SplitMNISTConfig, seeds: list[int]) -> dict[str, Any]:
    config.validate()
    _validate_seeds(seeds, new_run=True)
    if config.backbone != "mlp" or config.methods != PAIRED_METHODS:
        raise ValueError("a suíte requer MLP e os quatro pares completos")
    if config.optimizer_state_policy != "follow_update":
        raise ValueError(
            "a suíte principal requer optimizer_state_policy=follow_update"
        )
    return {
        "schema_version": 1,
        "status": "exploratory_not_independent_confirmation",
        "component": "DualHeat (Functional SlowHeat; not legacy DualHeatMLP)",
        "pairs": [asdict(pair) for pair in METHOD_PAIRS],
        "seeds": seeds,
        "config": json.loads(json.dumps(config_payload(config))),
        "component_settings": {
            "importance": "normalized abs(z * dL/dz)",
            "strength": 30.0,
            "plasticity_budget": 0.25,
            "scope": "hidden_only",
            "protection": "factorized_incoming_rows_and_outgoing_columns",
            "consolidation": "max_at_known_task_boundaries",
            "optimizer_state_policy": "follow_update",
        },
        "primary_endpoint": PRIMARY_ENDPOINT,
        "difference_direction": "candidate_minus_reference",
        "multiplicity": "Holm over the four accuracy contrasts within this dataset",
        "fairness": "paired initialization, data, batches, replay indices and epochs",
        "limitations": [
            "Fixed defaults, not individually tuned baselines; no universal benefit claim.",
            "No task ID at inference; known task boundaries at consolidation.",
            "Elapsed time is descriptive; no isolated warm-up timing or peak memory measurement.",
            "Replay/logit bytes are not total or peak memory.",
        ],
    }


def _write_json(path: Path, payload: Any) -> None:
    write_json_atomic(path, payload)


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(
        sorted(range(len(p_values)), key=p_values.__getitem__)
    ):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted


def summarize_pair_results(
    source_dir: str | Path, *, output_dir: str | Path
) -> dict[str, Any]:
    """Recompute only the four matched contrasts from complete per-seed files.

    Historical all-method runs are allowed, but incomplete seeds, mismatched
    configurations or unequal within-pair training resources are rejected.
    Source artifacts are read-only; no marginal CI is used to infer a difference.
    """
    source = Path(source_dir)
    fingerprints: dict[str, str] = {}

    def read(relative: str) -> dict[str, Any]:
        content = (source / relative).read_bytes()
        fingerprints[relative] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    manifest = read("multi_seed_config.json")
    seeds = manifest["seeds"]
    _validate_seeds(seeds)
    base = manifest["base_config"]
    if base.get("backbone", "mlp") != "mlp":
        raise ValueError("este relatório avalia apenas MLPs")
    if not set(PAIRED_METHODS).issubset(base["methods"]):
        raise ValueError("os resultados não contêm os quatro pares completos")
    if base.get("optimizer_state_policy") != "follow_update":
        raise ValueError(
            "os quatro pares requerem optimizer_state_policy=follow_update"
        )
    raw = {}
    for seed in seeds:
        saved_config = read(f"seed_{seed}/config.json")
        if saved_config != {**base, "seed": seed}:
            raise ValueError(f"configuração incompatível na seed {seed}")
        raw[seed] = read(f"seed_{seed}/results.json")
        for pair in METHOD_PAIRS:
            if pair.reference not in raw[seed] or pair.candidate not in raw[seed]:
                raise ValueError(f"par incompleto: {pair.label}, seed {seed}")
            reference, candidate = raw[seed][pair.reference], raw[seed][pair.candidate]
            for key in MATCHED_COSTS:
                if reference["cost"][key] != candidate["cost"][key]:
                    raise ValueError(
                        f"recursos diferentes: {pair.label}, seed {seed}, {key}"
                    )
            if reference["completed_epochs"] != candidate["completed_epochs"]:
                raise ValueError(f"épocas diferentes: {pair.label}, seed {seed}")

    comparisons = []
    seed_rows = []
    for pair in METHOD_PAIRS:
        metrics = {}
        for metric in AGGREGATE_METRICS:
            reference = [
                _result_metric(raw[seed][pair.reference], metric) for seed in seeds
            ]
            candidate = [
                _result_metric(raw[seed][pair.candidate], metric) for seed in seeds
            ]
            differences = [c - r for r, c in zip(reference, candidate, strict=True)]
            summary = (
                paired_confirmatory_summary(
                    differences,
                    bootstrap_resamples=base.get("bootstrap_resamples", 10_000),
                    bootstrap_seed=base.get("bootstrap_seed", 20_260_815),
                )
                if len(seeds) >= 2
                else {
                    "n_pairs": 1,
                    "mean_difference": differences[0],
                    "inference_unavailable": "requires_at_least_two_paired_seeds",
                }
            )
            metrics[metric] = {
                "reference_mean": sum(reference) / len(seeds),
                "candidate_mean": sum(candidate) / len(seeds),
                **summary,
            }
            seed_rows.extend(
                {
                    "reference": pair.reference,
                    "candidate": pair.candidate,
                    "metric": metric,
                    "seed": seed,
                    "reference_value": r,
                    "candidate_value": c,
                    "difference": c - r,
                }
                for seed, r, c in zip(seeds, reference, candidate, strict=True)
            )
        comparisons.append({**asdict(pair), "metrics": metrics})

    if len(seeds) >= 2:
        adjusted = _holm_adjust(
            [
                comparison["metrics"][PRIMARY_ENDPOINT]["student_t"]["two_sided_p"]
                for comparison in comparisons
            ]
        )
        for comparison, p_value in zip(comparisons, adjusted, strict=True):
            comparison["metrics"][PRIMARY_ENDPOINT]["holm_adjusted_p"] = p_value

    report = {
        "status": "exploratory_reanalysis"
        if not (source / "pair_protocol.json").exists()
        else "exploratory_paired_suite",
        "component": "DualHeat (Functional SlowHeat; not legacy DualHeatMLP)",
        "source_dir": relative_path(source, base=output_dir),
        "source_dir_base": "report_directory",
        "source_sha256": fingerprints,
        "source_environment_available": (source / "environment.json").is_file(),
        "config": base,
        "seeds": seeds,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "difference_direction": "candidate_minus_reference",
        "inference_note": (
            "Student-t and paired bootstrap intervals are pointwise, not simultaneous. "
            "Accuracy p-values use Holm across four contrasts within this dataset only. "
            "Secondary metrics and cross-dataset conclusions remain exploratory."
        ),
        "pairs": comparisons,
    }
    _write_pair_report(Path(output_dir), report, seed_rows)
    return report


def _write_pair_report(
    destination: Path, report: dict[str, Any], seed_rows: list[dict[str, Any]]
) -> None:
    rows = []
    lines = [
        "# Método versus método + DualHeat\n",
        "Implementação: **Functional SlowHeat**, não a classe legada DualHeatMLP.\n",
        (
            f"Execução de origem: `{report['source_dir']}` "
            "(caminho relativo à pasta deste relatório).\n"
        ),
        (
            f"MLP {report['config']['hidden_dims']}; "
            f"cenário `{report['config']['scenario']}`; "
            f"{report['config']['epochs_per_task']} épocas por tarefa.\n"
        ),
        (
            f"Análise exploratória; {len(report['seeds'])} seeds pareadas. "
            "Não constitui confirmação independente.\n"
        ),
        (
            "Delta = com componente − sem componente. Acurácia e forgetting em pontos "
            "percentuais; esquecer menos só é útil se a aquisição for preservada.\n"
        ),
        "| Método | Sem (%) | Com (%) | Delta (pp) | IC95% t (pp) | Delta forgetting (pp) | Tempo com/sem |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for comparison in report["pairs"]:
        metrics = comparison["metrics"]
        accuracy = metrics[PRIMARY_ENDPOINT]
        elapsed = metrics["elapsed_seconds"]
        ratio = (
            elapsed["candidate_mean"] / elapsed["reference_mean"]
            if elapsed["reference_mean"] > 0
            else None
        )
        ci = accuracy.get("student_t", {}).get("ci95", [None, None])
        row = {
            "reference": comparison["reference"],
            "candidate": comparison["candidate"],
            "n_pairs": len(report["seeds"]),
            "accuracy_reference_percent": 100 * accuracy["reference_mean"],
            "accuracy_candidate_percent": 100 * accuracy["candidate_mean"],
            "accuracy_delta_pp": 100 * accuracy["mean_difference"],
            "accuracy_ci95_t_low_pp": None if ci[0] is None else 100 * ci[0],
            "accuracy_ci95_t_high_pp": None if ci[1] is None else 100 * ci[1],
            "accuracy_p_holm": accuracy.get("holm_adjusted_p"),
            "forgetting_delta_pp": 100
            * metrics["average_forgetting"]["mean_difference"],
            "elapsed_ratio_of_means": ratio,
            "estimated_flops_delta": metrics["estimated_total_flops"][
                "mean_difference"
            ],
        }
        rows.append(row)
        ci_text = (
            "indisponível"
            if ci[0] is None
            else f"[{100 * ci[0]:+.3f}, {100 * ci[1]:+.3f}]"
        )
        time_text = "indisponível" if ratio is None else f"{ratio:.2f}×"
        lines.append(
            f"| {comparison['label']} | {row['accuracy_reference_percent']:.3f} "
            f"| {row['accuracy_candidate_percent']:.3f} | {row['accuracy_delta_pp']:+.3f} "
            f"| {ci_text} | {row['forgetting_delta_pp']:+.3f} | {time_text} |"
        )
    lines.extend(
        [
            "\n## Interpretação e limites\n",
            (
                "- ICs são pontuais; o JSON/CSV inclui p de acurácia ajustado por Holm "
                "para os quatro pares deste dataset. Não há correção entre datasets."
            ),
            "- Ganhos negativos e empates permanecem no relatório. Não se assume benefício universal.",
            (
                "- Mesmos parâmetros treináveis, épocas, passos, exemplos e memória de replay/logits "
                "são verificados dentro de cada par. Isso não iguala o custo computacional."
            ),
            (
                "- Tempo observado é descritivo, sem medição isolada com aquecimento; "
                "a ordem dos métodos é fixa. FLOPs são estimativas."
            ),
            "- Bytes de replay/logits não são memória total. Pico de memória não foi medido.",
            (
                "- Fronteiras de tarefa conhecidas; inferência sem task ID. "
                "Baselines usam defaults, sem ajuste individual nesta suíte."
            ),
            (
                "- Consulte pair_report.json para bootstrap pareado, sinais, demais métricas, "
                "configuração e hashes dos arquivos de origem."
            ),
        ]
    )
    _write_json(destination / "pair_report.json", report)
    for name, records in (
        ("pair_summary.csv", rows),
        ("pair_differences.csv", seed_rows),
    ):
        with (destination / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (destination / "pair_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_dualheat_pairs(
    *,
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    name: str = "split_mnist",
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = True,
    config: SplitMNISTConfig | None = None,
) -> dict[str, Any]:
    selected = paired_config(name, device) if config is None else config
    loaders = {
        "split_mnist": None,
        "permuted_mnist": load_permuted_mnist,
        "split_cifar10": load_split_cifar10,
        "split_cifar100": load_split_cifar100,
    }
    if name not in loaders:
        raise ValueError(f"dataset desconhecido: {name}")
    protocol = {"dataset": name, **pair_protocol(selected, seeds)}
    output = Path(output_dir)
    lock = output / "pair_protocol.json"
    if lock.exists():
        if not resume or json.loads(lock.read_text(encoding="utf-8")) != protocol:
            raise ValueError(
                "protocolo diferente ou resume desativado; use outro output_dir"
            )
    elif output.exists() and any(output.iterdir()):
        raise ValueError("output_dir deve estar vazio para iniciar a suíte")
    else:
        _write_json(lock, protocol)
    run_split_mnist_multi_seed(
        selected,
        seeds=seeds,
        data_dir=data_dir,
        output_dir=output,
        download=download,
        verbose=verbose,
        paired_references=(),  # The dedicated report computes exactly four contrasts.
        task_loader=loaders[name],
        resume=resume,
    )
    return summarize_pair_results(output, output_dir=output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=["split_mnist"]
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/dualheat_pairs")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summarize-from", type=Path, help="reanalisar artefatos sem treinar"
    )
    args = parser.parse_args(argv)
    if args.summarize_from:
        if args.dry_run:
            parser.error("--dry-run não pode ser combinado com --summarize-from")
        summarize_pair_results(args.summarize_from, output_dir=args.output_dir)
        report_path = relative_path(args.output_dir / "pair_report.md", base=Path.cwd())
        print(f"Relatório exploratório: {report_path}")
        return 0
    if not 1 <= args.num_seeds <= 2**31:
        parser.error("--num-seeds deve estar entre 1 e 2**31")
    if args.seeds is not None and len(args.seeds) != args.num_seeds:
        parser.error("--seeds deve conter exatamente --num-seeds valores")
    seeds = args.seeds
    if seeds is None:
        seeds = random.Random(20_260_826).sample(range(2**31), args.num_seeds)
    protocols = {}
    for name in dict.fromkeys(args.datasets):
        protocols[name] = pair_protocol(paired_config(name, args.device), seeds)
    if args.dry_run:
        print(json.dumps(protocols, indent=2, sort_keys=True))
        return 0
    for name in protocols:
        output = args.output_dir / name
        run_dualheat_pairs(
            name=name,
            seeds=seeds,
            data_dir=args.data_dir,
            output_dir=output,
            device=args.device,
            download=not args.no_download,
        )
        report_path = relative_path(output / "pair_report.md", base=Path.cwd())
        print(f"Relatório exploratório: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
