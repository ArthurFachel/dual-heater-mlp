"""Mechanism-only capacity diagnostic for SlowHeat on Qwen2.

Purpose
-------
Measure the quantities needed to FIX `slow_strength` and `ffn_plasticity_budget`
by a criterion declared in advance, without ever reading accuracy. The script
deliberately has no access to an evaluation loop: it trains on the first N
CLINC150 domains, consolidates after each, and reports only mechanism
statistics (signal density, participation ratio, effective plasticity, churn).

Why this is not post-hoc selection
----------------------------------
The selection rule is passed in on the command line and evaluated over an
analytic grid on a fixed importance vector, so it costs no extra runs and
cannot be tuned against an endpoint that is never computed here. The chosen
point is written to a manifest to be frozen into the real protocol.

Run (CPU, two tasks):
    CUDA_VISIBLE_DEVICES= HF_HOME=.hf-cache PYTHONPATH=. \\
        .venv/bin/python experiments/qwen_capacity_diagnostic.py \\
            --tasks 2 --min-effective-plasticity 0.6
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dual_heater.optim import SlowHeatAdamW
from dual_heater.qwen import (
    QwenSlowHeatConfig,
    SlowHeatQwen2ForSequenceClassification,
)
from experiments.capacity_calibration import (
    dominant_unit_overlap,
    free_fraction_scoped,
    heat_concentration,
    importance_profile,
    iso_plasticity_family_scoped,
    jaccard,
    participation_ratio,
    plasticity_bounds_scoped,
    positive_fraction,
    protected_heat_scoped,
    select_by_declared_criterion,
    sweep_capacity_scoped,
)
from experiments.split_clinc150 import CLINC150_DOMAINS, build_clinc150_tasks

DEFAULT_STRENGTHS = (1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_BUDGETS = (0.10, 0.25, 0.50, 0.75, 0.90)
ISO_BUDGETS = (0.05, 0.10, 0.25, 0.50, 0.75)


def _layer_importances(model) -> list[Tensor]:
    """Per-tracker importance memory, in layer order.

    Returned as a LIST, not concatenated. Capacity scope is part of the
    mechanism: under the default `local` scope each layer normalizes to its own
    maximum, so pooling the family into one vector and dividing by a single
    global maximum reports a different (and much more plastic) protection
    profile than the model applies. See `experiments/capacity_calibration.py`.
    """

    return [
        tracker.importance_memory.detach().clone()
        for tracker in model.get_ffn_trackers()
    ]


def _pooled_importance(model) -> Tensor:
    """Concatenated importance, for scope-free statistics only.

    Valid for `positive_fraction` and `participation_ratio` (which do not
    normalize), NOT for heat or plasticity. Use `_layer_importances` with the
    scoped helpers for anything involving protection.
    """

    return torch.cat(_layer_importances(model))


def _resolve_domains(domains: list[str] | None, tasks: int) -> list[str]:
    """Declared task sequence, in training order.

    ``--domains`` wins over ``--tasks`` because order is part of the protocol:
    the anomaly diagnostic has to be able to state ``credit_cards -> banking``
    explicitly rather than have it inferred from a count and a dict order.
    """

    if domains is None:
        if not 2 <= tasks <= len(CLINC150_DOMAINS):
            raise ValueError("--tasks deve estar entre 2 e 10")
        return list(CLINC150_DOMAINS)[:tasks]
    if len(domains) < 2:
        raise ValueError("--domains exige ao menos dois domínios")
    unknown = [name for name in domains if name not in CLINC150_DOMAINS]
    if unknown:
        raise ValueError(
            "domínio desconhecido: " + ", ".join(unknown) + "; válidos: "
            + ", ".join(CLINC150_DOMAINS)
        )
    if len(set(domains)) != len(domains):
        raise ValueError("--domains não pode repetir domínios")
    return list(domains)


def _layer_report(model, budget: float, scope: str) -> list[dict[str, float]]:
    """Per-layer signal stats plus the protection the chosen scope assigns."""

    layers = _layer_importances(model)
    heat = protected_heat_scoped(layers, budget, scope=scope)
    offset = 0
    report = []
    for index, memory in enumerate(layers):
        width = memory.numel()
        layer_heat = heat[offset : offset + width]
        offset += width
        report.append(
            {
                "layer": index,
                "units": int(width),
                "positive_fraction": positive_fraction(memory),
                "participation_ratio": participation_ratio(memory),
                "protected_units": int(
                    torch.count_nonzero(layer_heat > 0.0).item()
                ),
                "mean_heat": float(layer_heat.mean().item()),
                "max_heat": float(layer_heat.max().item()),
                **heat_concentration(layer_heat),
                # Raw shape, before the budget truncates and the maximum
                # normalizes: distinguishes "one unit carries the layer" from
                # "a small group carries the layer".
                "raw": importance_profile(memory),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset", default="clinc/clinc_oos")
    parser.add_argument("--dataset-config", default="plus")
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help=(
            "explicit domain sequence, in training order (e.g. "
            "--domains credit_cards banking). Overrides --tasks. Every name "
            "must exist in CLINC150_DOMAINS and appear at most once; task "
            "order is part of the protocol, so it is declared, never inferred."
        ),
    )
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps-per-task", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--budget",
        type=float,
        default=0.25,
        help="budget at which the criterion is evaluated",
    )
    parser.add_argument(
        "--min-effective-plasticity",
        type=float,
        default=None,
        help="pre-declared floor; the criterion picks the largest beta meeting it",
    )
    parser.add_argument(
        "--iso-plasticity",
        type=float,
        action="append",
        default=None,
        help=(
            "target effective plasticity E*; emits one iso-plasticity arm per "
            "reachable budget, all with matched E* and different spread. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--capacity-scope",
        default="local",
        choices=("local", "global", "hierarchical"),
        help=(
            "must match the scope used in the real run; all heat and plasticity "
            "numbers depend on it"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", default="results/qwen_capacity_diagnostic/manifest.json"
    )
    args = parser.parse_args()

    domains = _resolve_domains(args.domains, args.tasks)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    scope = args.capacity_scope

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset(args.dataset, args.dataset_config)
    by_domain = {
        task.domain: task
        for task in build_clinc150_tasks(
            dataset, tokenizer, max_length=args.max_length, include_test=False
        )
    }
    tasks = [by_domain[name] for name in domains]
    print(f"tasks: {[task.domain for task in tasks]}")

    slowheat = QwenSlowHeatConfig(
        # These two are what the diagnostic exists to choose; the values used
        # while COLLECTING importance only affect the optimizer, not the
        # analytic grid, so they are held at the historical defaults.
        slow_strength=3.0,
        ffn_plasticity_budget=args.budget,
        capacity_scope=scope,
        freeze_unbound_parameters=True,
    )
    model = SlowHeatQwen2ForSequenceClassification.from_pretrained(
        args.model,
        num_labels=150,
        slowheat_config=slowheat,
        dtype=torch.float32,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.train()

    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    model.register_plasticity_masks(optimizer)

    stages: list[dict[str, Any]] = []
    previous_mask: Tensor | None = None
    # Per-task, per-layer importance captured BEFORE consolidation merges it
    # into the persistent memory. Needed for the cross-task top-k overlap: once
    # `consolidate` runs, `task_ema` is zeroed and the per-task signal is gone.
    task_importances: list[list[Tensor]] = []

    for stage, task in enumerate(tasks):
        split = task.train
        count = len(split.labels)
        order = torch.randperm(count)
        losses = []
        for step in range(args.steps_per_task):
            start = (step * args.batch_size) % max(count - args.batch_size, 1)
            index = order[start : start + args.batch_size]
            if len(index) == 0:
                break
            optimizer.zero_grad()
            output = model(
                input_ids=split.input_ids[index].to(device),
                attention_mask=split.attention_mask[index].to(device),
                labels=split.labels[index].to(device),
            )
            output.loss.backward()
            optimizer.step()
            losses.append(float(output.loss.item()))

        pooled_before = _pooled_importance(model)
        task_importances.append(
            [
                tracker.task_ema.detach().clone()
                for tracker in model.get_ffn_trackers()
            ]
        )
        model.consolidate()
        layers = _layer_importances(model)
        pooled = torch.cat(layers)
        mask = protected_heat_scoped(layers, args.budget, scope=scope) > 0.0

        record: dict[str, Any] = {
            "stage": stage,
            "domain": task.domain,
            "steps": len(losses),
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "pooled_units": int(pooled.numel()),
            "positive_fraction": positive_fraction(pooled),
            "participation_ratio": participation_ratio(pooled),
            "importance_growth": float(
                (pooled.sum() - pooled_before.sum()).item()
            ),
            "protected_units": int(torch.count_nonzero(mask).item()),
            "layers": _layer_report(model, args.budget, scope),
        }
        if previous_mask is not None:
            record["protection_jaccard_vs_previous"] = jaccard(previous_mask, mask)
            record["free_pool_turnover"] = 1.0 - jaccard(~previous_mask, ~mask)
        previous_mask = mask
        stages.append(record)
        print(
            f"stage {stage} ({task.domain}): loss {record['first_loss']:.3f} -> "
            f"{record['last_loss']:.3f} | density {record['positive_fraction']:.3f} "
            f"| PR {record['participation_ratio']:.1f} | protected "
            f"{record['protected_units']}"
        )
        if "protection_jaccard_vs_previous" in record:
            print(
                f"           churn: jaccard {record['protection_jaccard_vs_previous']:.3f}"
                f" | free-pool turnover {record['free_pool_turnover']:.3f}"
            )

    layers = _layer_importances(model)
    pooled = torch.cat(layers)
    grid = sweep_capacity_scoped(
        layers,
        strengths=list(DEFAULT_STRENGTHS),
        budgets=list(DEFAULT_BUDGETS),
        scope=scope,
    )

    print("\neffective plasticity grid (rows = budget, cols = beta)")
    header = "  budget |" + "".join(f"{beta:>9.0f}" for beta in DEFAULT_STRENGTHS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for budget in DEFAULT_BUDGETS:
        row = [
            point
            for point in grid
            if point.budget == budget
        ]
        row.sort(key=lambda point: point.slow_strength)
        cells = "".join(f"{point.effective_plasticity:>9.3f}" for point in row)
        print(f"  {budget:>6.2f} |{cells}")

    print("\nreachable plasticity interval per budget (floor = free fraction)")
    print("  budget | floor    reachable E")
    print("  " + "-" * 38)
    for budget in DEFAULT_BUDGETS:
        lower, upper = plasticity_bounds_scoped(layers, budget, scope=scope)
        print(f"  {budget:>6.2f} | {lower:.3f}    [{lower:.3f}, {upper:.3f}]")
    print(
        "  note: beta cannot push E below the floor, so budget and beta are\n"
        "        not interchangeable knobs."
    )

    print("\nprotection concentration (is the protected set functional?)")
    print("  budget | protected  effective  ratio  >=0.5  >=0.1  q50")
    print("  " + "-" * 62)
    concentration: dict[str, dict[str, float]] = {}
    for budget in DEFAULT_BUDGETS:
        heat = protected_heat_scoped(layers, budget, scope=scope)
        report = heat_concentration(heat)
        concentration[f"{budget}"] = report
        print(
            f"  {budget:>6.2f} | {int(report['protected_units']):>9,} "
            f"{report['effective_protected_units']:>10.0f} "
            f"{report['concentration_ratio']:>6.3f} "
            f"{int(report['units_above_0p5']):>6,} "
            f"{int(report['units_above_0p1']):>6,} "
            f"{report.get('protected_heat_q50', 0.0):>5.3f}"
        )
    print(
        "  ratio = effective/nominal protected units. Low ratio means the\n"
        "  protected count is nominal: a few extreme units set the normalizer\n"
        "  and most 'protected' units keep nearly full plasticity."
    )

    # Cross-task dominant-unit overlap: are the units that carry a layer the
    # same ones across tasks, or does each task claim a different handful?
    # Compared against chance (k/N); overlap at chance means the ranking carries
    # no cross-task information and per-layer concentration is task-specific.
    overlap_ks = (10, 100)
    layer_overlap: list[dict[str, Any]] = []
    if len(task_importances) >= 2:
        first_task, last_task = task_importances[0], task_importances[-1]
        print(
            f"\ndominant-unit overlap between task 0 ({domains[0]}) and task "
            f"{len(task_importances) - 1} ({domains[len(task_importances) - 1]})"
        )
        print("  layer |   k=10 (chance)     k=100 (chance)")
        print("  " + "-" * 46)
        for index, (left, right) in enumerate(
            zip(first_task, last_task, strict=True)
        ):
            entry: dict[str, Any] = {"layer": index}
            for k in overlap_ks:
                entry[f"top_{k}"] = dominant_unit_overlap(left, right, k=k)
            layer_overlap.append(entry)
            first = entry[f"top_{overlap_ks[0]}"]
            second = entry[f"top_{overlap_ks[1]}"]
            print(
                f"  {index:>5} | {first['overlap']:>6.3f} "
                f"({first['chance']:.4f})   {second['overlap']:>6.3f} "
                f"({second['chance']:.4f})"
            )

    iso_families: dict[str, list[dict[str, Any]]] = {}
    if args.iso_plasticity:
        for target in args.iso_plasticity:
            family = iso_plasticity_family_scoped(
                layers,
                target_plasticity=target,
                budgets=list(ISO_BUDGETS),
                scope=scope,
            )
            iso_families[f"{target}"] = [asdict(point) for point in family]
            print(f"\niso-plasticity family at E* = {target}")
            if not family:
                print("  no reachable budget: every floor exceeds the target")
                continue
            print("  budget | protected        beta        achieved E")
            print("  " + "-" * 50)
            for point in family:
                print(
                    f"  {point.budget:>6.2f} | {point.protected_units:>9,} "
                    f"({point.protected_fraction * 100:>2.0f}%) "
                    f"{point.slow_strength:>11.2f}  {point.achieved_plasticity:.6f}"
                )
            counts = [point.protected_units for point in family]
            print(
                f"  spread: {max(counts) / max(min(counts), 1):.2f}x in protected "
                f"count at matched plasticity"
            )

    selection: dict[str, Any] | None = None
    if args.min_effective_plasticity is not None:
        try:
            chosen = select_by_declared_criterion(
                grid,
                budget=args.budget,
                minimum_effective_plasticity=args.min_effective_plasticity,
            )
            selection = asdict(chosen)
            print(
                f"\nselected by declared criterion (E >= "
                f"{args.min_effective_plasticity}): beta={chosen.slow_strength}, "
                f"budget={chosen.budget}, E={chosen.effective_plasticity:.3f}"
            )
        except ValueError as error:
            print(f"\ncriterion unmet at budget {args.budget}: {error}")

    pooled_pr = participation_ratio(pooled)
    pr_fraction = pooled_pr / pooled.numel()
    if pr_fraction < 0.05:
        print(
            f"\nWARNING: participation ratio is {pr_fraction * 100:.1f}% of units. "
            "Importance is heavy-tailed, so beta loses physical meaning and the "
            "iso-plasticity arms will need absurd strengths. Consider "
            "renormalizing heat by a high percentile instead of the maximum."
        )

    manifest = {
        "criterion": {
            "rule": "largest slow_strength with effective_plasticity >= floor",
            "budget": args.budget,
            "minimum_effective_plasticity": args.min_effective_plasticity,
            "declared_before_run": args.min_effective_plasticity is not None,
            "reads_accuracy": False,
        },
        "iso_plasticity": {
            "rule": "solve beta so effective_plasticity == target at each budget",
            "budgets": list(ISO_BUDGETS),
            "targets": args.iso_plasticity or [],
            "families": iso_families,
        },
        "capacity_scope": scope,
        "bounds": {
            f"{budget}": {
                "floor": free_fraction_scoped(layers, budget, scope=scope),
                "protected_units": int(
                    torch.count_nonzero(
                        protected_heat_scoped(layers, budget, scope=scope) > 0.0
                    ).item()
                ),
            }
            for budget in DEFAULT_BUDGETS
        },
        "concentration": concentration,
        "dominant_unit_overlap": {
            "rule": (
                "top-k intersection of per-task importance (task_ema captured "
                "before consolidation), first task vs last task"
            ),
            "k": list(overlap_ks),
            "layers": layer_overlap,
        },
        "signal": {
            "pooled_units": int(pooled.numel()),
            "positive_fraction": positive_fraction(pooled),
            "participation_ratio": pooled_pr,
            "participation_fraction": pr_fraction,
            "heavy_tailed_warning": pr_fraction < 0.05,
        },
        "protocol": {
            "model": args.model,
            "dataset": f"{args.dataset}:{args.dataset_config}",
            "tasks": [task.domain for task in tasks],
            "task_order_declared": args.domains is not None,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "steps_per_task": args.steps_per_task,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "stages": stages,
        "grid": [asdict(point) for point in grid],
        "selection": selection,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest written: {path}")


if __name__ == "__main__":
    main()
