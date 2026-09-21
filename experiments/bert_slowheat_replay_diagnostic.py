"""Two-task replay interaction diagnostic for BERT SlowHeat on CLINC150."""

from __future__ import annotations

import argparse
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
class ReplayDiagnosticCondition:
    name: str
    method: str
    mask_mode: PlasticityMaskMode


def replay_diagnostic_conditions() -> tuple[ReplayDiagnosticCondition, ...]:
    return (
        ReplayDiagnosticCondition("vanilla", "vanilla", "soft"),
        ReplayDiagnosticCondition("replay", "replay", "soft"),
        ReplayDiagnosticCondition(
            "slowheat_hard", "slowheat_ffn_attention", "hard"
        ),
        ReplayDiagnosticCondition(
            "slowheat_hard_replay",
            "slowheat_ffn_attention_replay",
            "hard",
        ),
    )


def run_replay_diagnostic(
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
        for condition in replay_diagnostic_conditions():
            config = replace(
                base_config,
                seed=seed,
                methods=(condition.method,),
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

    summary = summarize_replay_diagnostic(raw)
    write_json_atomic(destination / "diagnostic_summary.json", summary)
    (destination / "diagnostic_table.md").write_text(
        replay_diagnostic_markdown(summary), encoding="utf-8"
    )
    return raw


def summarize_replay_diagnostic(
    raw: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not raw:
        raise ValueError("resultados diagnósticos não podem ser vazios")
    seeds = sorted(raw)
    names = [condition.name for condition in replay_diagnostic_conditions()]
    for seed in seeds:
        if set(raw[seed]) != set(names):
            raise ValueError(f"seed {seed} não contém as quatro condições")

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
    contrast_pairs = {
        "replay_minus_vanilla": ("replay", "vanilla"),
        "hard_minus_vanilla": ("slowheat_hard", "vanilla"),
        "hard_replay_minus_replay": (
            "slowheat_hard_replay",
            "replay",
        ),
        "hard_replay_minus_hard": (
            "slowheat_hard_replay",
            "slowheat_hard",
        ),
    }
    paired_contrasts: dict[str, dict[str, dict[str, Any]]] = {}
    for contrast, (left, right) in contrast_pairs.items():
        paired_contrasts[contrast] = {}
        for metric in metric_names:
            by_seed: dict[str, float | None] = {}
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
        "primary_endpoint": "final_average_accuracy",
        "task_count": 2,
        "seeds": seeds,
        "conditions": conditions,
        "raw_by_seed": {str(seed): endpoints[seed] for seed in seeds},
        "paired_contrasts": paired_contrasts,
    }


def replay_diagnostic_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BERT SlowHeat replay interaction diagnostic",
        "",
        "Validation means ± sample standard deviation across paired seeds.",
        "Primary endpoint: final average accuracy.",
        "",
        "| Condition | Final average (%) | T1 retention (%) | T1 forgetting (%) | "
        "T2 acquisition (%) | Protected RMS drift | Replay memory (MiB) | "
        "Tokens | Tokens/s | Time (s) | Peak allocated (MiB) | Peak reserved (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in replay_diagnostic_conditions():
        item = summary["conditions"][condition.name]
        lines.append(
            f"| {condition.name} | "
            f"{format_mean_std(item['final_average_accuracy'], 100.0)} | "
            f"{format_mean_std(item['task1_retention'], 100.0)} | "
            f"{format_mean_std(item['task1_forgetting'], 100.0)} | "
            f"{format_mean_std(item['task2_acquisition'], 100.0)} | "
            f"{format_mean_std_scientific(item['protected_rms_drift'])} | "
            f"{format_mean_std(item['replay_memory_mib'])} | "
            f"{format_mean_std(item['tokens_processed'])} | "
            f"{format_mean_std(item['tokens_per_second'])} | "
            f"{format_mean_std(item['elapsed_seconds'])} | "
            f"{format_mean_std(item['peak_memory_mib'])} | "
            f"{format_mean_std(item['peak_reserved_mib'])} |"
        )
    lines.extend(
        [
            "",
            "Decision gate:",
            "",
            "- Primary contrast: hard_replay_minus_replay on final average "
            "accuracy.",
            "- Report all paired seed differences; n=3 remains diagnostic.",
            "- Reject a mechanism benefit if any seed reverses sign or if T2 acquisition "
            "drops by more than 2 percentage points on average.",
            "- Do not start the ten-task sequence until this gate is reviewed.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/bert_slowheat_replay_diagnostic",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--replay-batch-size", type=int, default=2)
    parser.add_argument("--replay-per-class", type=int, default=20)
    parser.add_argument("--epochs-per-task", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--task-limit", type=int, choices=[2], default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--telemetry-every", type=int, default=10)
    parser.set_defaults(evaluate_test=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SplitCLINC150Config(
        device=args.device,
        batch_size=args.batch_size,
        replay_batch_size=args.replay_batch_size,
        replay_per_class=args.replay_per_class,
        epochs_per_task=args.epochs_per_task,
        max_length=args.max_length,
        task_limit=args.task_limit,
        evaluate_test=False,
    )
    tasks, metadata = load_clinc150_tasks(config, include_test=False)
    run_replay_diagnostic(
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
