"""Two-task mechanism diagnostic for BERT SlowHeat on CLINC150."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from experiments.artifacts import write_json_atomic
from experiments.bert_slowheat_diagnostic_common import (
    condition_endpoints,
    format_mean_std,
    format_mean_std_scientific,
    summarize_values,
)
from experiments.provenance import write_environment_manifest
from experiments.split_clinc150 import (
    PlasticityMaskMode,
    SplitCLINC150Config,
    load_clinc150_tasks,
    run_split_clinc150,
)


@dataclass(frozen=True)
class DiagnosticCondition:
    name: str
    method: str
    slow_strength: float
    mask_mode: PlasticityMaskMode


def diagnostic_conditions() -> tuple[DiagnosticCondition, ...]:
    return (
        DiagnosticCondition("vanilla", "vanilla", 3.0, "soft"),
        DiagnosticCondition(
            "slowheat_beta_3", "slowheat_ffn_attention", 3.0, "soft"
        ),
        DiagnosticCondition(
            "slowheat_beta_10", "slowheat_ffn_attention", 10.0, "soft"
        ),
        DiagnosticCondition(
            "slowheat_beta_30", "slowheat_ffn_attention", 30.0, "soft"
        ),
        DiagnosticCondition(
            "slowheat_hard", "slowheat_ffn_attention", 3.0, "hard"
        ),
        DiagnosticCondition(
            "slowheat_random_hard",
            "slowheat_ffn_attention",
            3.0,
            "random_hard",
        ),
    )


def summarize_diagnostic(
    raw: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not raw:
        raise ValueError("resultados diagnósticos não podem ser vazios")
    seeds = sorted(raw)
    names = [condition.name for condition in diagnostic_conditions()]
    for seed in seeds:
        if set(raw[seed]) != set(names):
            raise ValueError(f"seed {seed} não contém as seis condições diagnósticas")
    endpoints = {
        seed: {
            name: condition_endpoints(raw[seed][name])
            for name in names
        }
        for seed in seeds
    }
    metric_names = tuple(endpoints[seeds[0]][names[0]])
    conditions = {
        name: {
            metric: summarize_values(
                [endpoints[seed][name][metric] for seed in seeds]
            )
            for metric in metric_names
        }
        for name in names
    }
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for name in names:
        if name == "vanilla":
            continue
        paired[name] = {}
        for metric in metric_names:
            by_seed: dict[str, float | None] = {}
            for seed in seeds:
                candidate = endpoints[seed][name][metric]
                baseline = endpoints[seed]["vanilla"][metric]
                by_seed[str(seed)] = (
                    None
                    if candidate is None or baseline is None
                    else candidate - baseline
                )
            paired[name][metric] = {
                **summarize_values(list(by_seed.values())),
                "by_seed": by_seed,
            }
    contrast_pairs = {
        "learned_hard_minus_random_hard": (
            "slowheat_hard",
            "slowheat_random_hard",
        ),
        "hard_minus_beta_3": ("slowheat_hard", "slowheat_beta_3"),
    }
    paired_contrasts: dict[str, dict[str, dict[str, Any]]] = {}
    for contrast, (left, right) in contrast_pairs.items():
        paired_contrasts[contrast] = {}
        for metric in metric_names:
            by_seed = {}
            for seed in seeds:
                left_value = endpoints[seed][left][metric]
                right_value = endpoints[seed][right][metric]
                by_seed[str(seed)] = (
                    None
                    if left_value is None or right_value is None
                    else left_value - right_value
                )
            paired_contrasts[contrast][metric] = {
                **summarize_values(list(by_seed.values())),
                "by_seed": by_seed,
            }
    return {
        "schema_version": 1,
        "endpoint_source": "validation",
        "task_count": 2,
        "seeds": seeds,
        "conditions": conditions,
        "raw_by_seed": {str(seed): endpoints[seed] for seed in seeds},
        "paired_vs_vanilla": paired,
        "paired_contrasts": paired_contrasts,
    }


def diagnostic_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BERT SlowHeat two-task diagnostic",
        "",
        "Validation means ± sample standard deviation across paired seeds.",
        "",
        "| Condition | T1 acquisition (%) | T1 retention (%) | T1 forgetting (%) | "
        "T2 acquisition (%) | Final average (%) | Protected entries | "
        "Protected RMS drift | Plastic RMS drift | Time (s) | Tokens/s | "
        "Peak allocated (MiB) | Peak reserved (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in diagnostic_conditions():
        item = summary["conditions"][condition.name]
        lines.append(
            f"| {condition.name} | "
            f"{format_mean_std(item['task1_acquisition'], 100.0)} | "
            f"{format_mean_std(item['task1_retention'], 100.0)} | "
            f"{format_mean_std(item['task1_forgetting'], 100.0)} | "
            f"{format_mean_std(item['task2_acquisition'], 100.0)} | "
            f"{format_mean_std(item['final_average_accuracy'], 100.0)} | "
            f"{format_mean_std(item['protected_count'])} | "
            f"{format_mean_std_scientific(item['protected_rms_drift'])} | "
            f"{format_mean_std_scientific(item['plastic_rms_drift'])} | "
            f"{format_mean_std(item['elapsed_seconds'])} | "
            f"{format_mean_std(item['tokens_per_second'])} | "
            f"{format_mean_std(item['peak_memory_mib'])} | "
            f"{format_mean_std(item['peak_reserved_mib'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation gates:",
            "",
            "- Learned hard > random hard is a diagnostic signal that ranking helps.",
            "- Hard > β=3 is a diagnostic signal that soft protection is too weak.",
            "- Hard does not retain T1: FFN/attention unit protection is insufficient.",
            "- Retention rises while T2 acquisition falls: tune the plasticity budget.",
            "",
        ]
    )
    return "\n".join(lines)


def load_completed_diagnostic(
    output_dir: str | Path,
) -> dict[int, dict[str, dict[str, Any]]]:
    source = Path(output_dir)
    seed_dirs = sorted(source.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"nenhuma seed encontrada em {source}")

    raw: dict[int, dict[str, dict[str, Any]]] = {}
    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError as error:
            raise ValueError(f"diretório de seed inválido: {seed_dir.name}") from error
        raw[seed] = {}
        for condition in diagnostic_conditions():
            result_path = (
                seed_dir
                / condition.name
                / condition.method
                / "results.json"
            )
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            raw[seed][condition.name] = json.loads(
                result_path.read_text(encoding="utf-8")
            )
    return raw


def write_diagnostic_reports(
    output_dir: str | Path,
    raw: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    destination = Path(output_dir)
    summary = summarize_diagnostic(raw)
    write_json_atomic(destination / "diagnostic_summary.json", summary)
    (destination / "diagnostic_table.md").write_text(
        diagnostic_markdown(summary), encoding="utf-8"
    )
    return summary


def run_diagnostic(
    base_config: SplitCLINC150Config,
    tasks,
    *,
    metadata: dict[str, Any],
    seeds: list[int],
    output_dir: str | Path,
    resume: bool = False,
    telemetry: bool = False,
    telemetry_every: int = 10,
) -> dict[int, dict[str, dict[str, Any]]]:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds deve ser não vazio e sem duplicatas")
    if base_config.evaluate_test:
        raise ValueError("diagnóstico deve permanecer validation-only")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_environment_manifest(
        destination,
        project_root=Path(__file__).resolve().parents[1],
    )
    raw: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        raw[seed] = {}
        for condition in diagnostic_conditions():
            config = replace(
                base_config,
                seed=seed,
                methods=(condition.method,),
                slow_strength=condition.slow_strength,
                plasticity_mask_mode=condition.mask_mode,
                task_limit=2,
                evaluate_test=False,
            )
            result = run_split_clinc150(
                config,
                tasks,
                metadata=metadata,
                output_dir=destination / f"seed_{seed}" / condition.name,
                resume=resume,
                telemetry=telemetry,
                telemetry_every=telemetry_every,
            )[condition.method]
            raw[seed][condition.name] = result

    write_diagnostic_reports(destination, raw)
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/bert_slowheat_diagnostic",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs-per-task", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--task-limit", type=int, choices=[2], default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--telemetry-every", type=int, default=10)
    parser.add_argument(
        "--summarize-from",
        help="reanalisar results.json existentes sem carregar dados ou treinar",
    )
    parser.set_defaults(evaluate_test=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_from is not None:
        raw = load_completed_diagnostic(args.summarize_from)
        write_diagnostic_reports(args.summarize_from, raw)
        return
    config = SplitCLINC150Config(
        device=args.device,
        batch_size=args.batch_size,
        epochs_per_task=args.epochs_per_task,
        max_length=args.max_length,
        task_limit=args.task_limit,
        evaluate_test=False,
    )
    tasks, metadata = load_clinc150_tasks(config, include_test=False)
    run_diagnostic(
        config,
        tasks,
        metadata=metadata,
        seeds=args.seeds,
        output_dir=args.output_dir,
        resume=args.resume,
        telemetry=args.telemetry,
        telemetry_every=args.telemetry_every,
    )


if __name__ == "__main__":
    main()
