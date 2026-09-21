"""Low-memory replay diagnostic for BERT SlowHeat on two CLINC150 tasks."""

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

REPLAY_BUDGETS = (1, 5, 10)
PRIMARY_BUDGET = 1


@dataclass(frozen=True)
class ReplayBudgetCondition:
    name: str
    method: str
    mask_mode: PlasticityMaskMode
    replay_per_class: int


def replay_budget_conditions() -> tuple[ReplayBudgetCondition, ...]:
    conditions: list[ReplayBudgetCondition] = []
    for replay_per_class in REPLAY_BUDGETS:
        conditions.extend(
            [
                ReplayBudgetCondition(
                    f"replay_memory_{replay_per_class}",
                    "replay",
                    "soft",
                    replay_per_class,
                ),
                ReplayBudgetCondition(
                    f"slowheat_hard_replay_memory_{replay_per_class}",
                    "slowheat_ffn_attention_replay",
                    "hard",
                    replay_per_class,
                ),
            ]
        )
    return tuple(conditions)


def run_replay_budget_diagnostic(
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
        for condition in replay_budget_conditions():
            config = replace(
                base_config,
                seed=seed,
                methods=(condition.method,),
                plasticity_mask_mode=condition.mask_mode,
                replay_per_class=condition.replay_per_class,
                replay_batch_size=2,
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
    summary = summarize_replay_budget_diagnostic(raw)
    write_json_atomic(destination / "diagnostic_summary.json", summary)
    (destination / "diagnostic_table.md").write_text(
        replay_budget_markdown(summary), encoding="utf-8"
    )
    return raw


def summarize_replay_budget_diagnostic(
    raw: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not raw:
        raise ValueError("resultados diagnósticos não podem ser vazios")
    seeds = sorted(raw)
    names = [condition.name for condition in replay_budget_conditions()]
    for seed in seeds:
        if set(raw[seed]) != set(names):
            raise ValueError(f"seed {seed} não contém as seis condições")
        for replay_per_class in REPLAY_BUDGETS:
            replay_name = f"replay_memory_{replay_per_class}"
            hard_name = f"slowheat_hard_replay_memory_{replay_per_class}"
            replay_bytes = int(raw[seed][replay_name]["replay_memory_bytes"])
            hard_bytes = int(raw[seed][hard_name]["replay_memory_bytes"])
            if replay_bytes != hard_bytes:
                raise ValueError(
                    "memória de replay desigual para "
                    f"seed={seed}, replay_per_class={replay_per_class}: "
                    f"{replay_bytes} != {hard_bytes}"
                )

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
    paired_contrasts: dict[str, dict[str, dict[str, Any]]] = {}
    for replay_per_class in REPLAY_BUDGETS:
        replay_name = f"replay_memory_{replay_per_class}"
        hard_name = f"slowheat_hard_replay_memory_{replay_per_class}"
        contrast_name = f"hard_replay_minus_replay_memory_{replay_per_class}"
        paired_contrasts[contrast_name] = {}
        for metric in metric_names:
            by_seed: dict[str, float | None] = {}
            for seed in seeds:
                left = endpoints[seed][hard_name][metric]
                right = endpoints[seed][replay_name][metric]
                by_seed[str(seed)] = (
                    None if left is None or right is None else left - right
                )
            paired_contrasts[contrast_name][metric] = {
                **summarize_values(list(by_seed.values())),
                "by_seed": by_seed,
            }
    return {
        "schema_version": 1,
        "endpoint_source": "validation",
        "primary_endpoint": "final_average_accuracy",
        "primary_budget_per_class": PRIMARY_BUDGET,
        "primary_contrast": "hard_replay_minus_replay_memory_1",
        "secondary_budgets_per_class": [5, 10],
        "task_count": 2,
        "seeds": seeds,
        "conditions": conditions,
        "raw_by_seed": {str(seed): endpoints[seed] for seed in seeds},
        "paired_contrasts": paired_contrasts,
    }


def replay_budget_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BERT SlowHeat low-memory replay diagnostic",
        "",
        "Validation means ± sample standard deviation across paired seeds.",
        "Primary budget: 1 replay example per class.",
        "Secondary budgets: 5 and 10 replay examples per class.",
        "Primary endpoint: final average accuracy.",
        "",
        (
            "| Condition | Examples/class | Final average (%) | T1 retention (%) | "
            "T1 forgetting (%) | T2 acquisition (%) | Actual replay memory (MiB) | "
            "Protected RMS drift | Tokens | Tokens/s | Time (s) | "
            "Peak allocated (MiB) | Peak reserved (MiB) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in replay_budget_conditions():
        item = summary["conditions"][condition.name]
        lines.append(
            f"| {condition.name} | {condition.replay_per_class} | "
            f"{format_mean_std(item['final_average_accuracy'], 100.0)} | "
            f"{format_mean_std(item['task1_retention'], 100.0)} | "
            f"{format_mean_std(item['task1_forgetting'], 100.0)} | "
            f"{format_mean_std(item['task2_acquisition'], 100.0)} | "
            f"{format_mean_std(item['replay_memory_mib'])} | "
            f"{format_mean_std_scientific(item['protected_rms_drift'])} | "
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
            "- Primary contrast: hard_replay_minus_replay_memory_1 on final average accuracy.",
            "- Require a positive paired difference in every seed.",
            "- Require mean T2 acquisition loss no worse than 2 percentage points.",
            "- Require exactly zero protected RMS and max drift for hard+replay.",
            "- Budgets 5 and 10 are secondary diagnostics only.",
            "- Do not select the best-looking secondary budget as a confirmatory result.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/bert_slowheat_replay_budget_diagnostic",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--replay-batch-size", type=int, choices=[2], default=2)
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
        epochs_per_task=args.epochs_per_task,
        max_length=args.max_length,
        task_limit=args.task_limit,
        evaluate_test=False,
    )
    tasks, metadata = load_clinc150_tasks(config, include_test=False)
    run_replay_budget_diagnostic(
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
