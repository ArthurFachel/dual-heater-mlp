#!/usr/bin/env python3
"""EXPLORATORY (F5, 2026-11-05): does the BERT soft-30 selection gain generalize to
other CLINC150 domain pairs?

Same protocol as scripts/run_soft_2x2.py (budget 0.50, soft strength 30, SlowHeat |z|
vs uniform_soft, seed band 14.000.003+), only the two-task stream changes: the
requested pair is moved to the front of the task list and run_diagnostic keeps its
default task_limit=2. banking -> credit_cards is the original pair.

Pre-declared reading (written before any run, also in fila.md F5): on a pair,
retention T1 diff (SlowHeat - uniform_soft) > 0 in >= 8/10 seeds AND mean FAA diff
> 0 = generalizes on that pair. Overall: generalizes if that holds on >= 3 of 4 pairs.
NOT pre-registered.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from experiments.bert_slowheat_diagnostic import DiagnosticCondition, run_diagnostic
from experiments.split_clinc150 import SplitCLINC150Config, load_clinc150_tasks

SEEDS = (
    14_000_003, 14_025_011, 14_050_017, 14_075_037, 14_100_043,
    14_125_059, 14_150_061, 14_175_087, 14_200_089, 14_225_097,
)
CONDITIONS = (
    DiagnosticCondition("vanilla", "vanilla", 3.0, "soft"),  # summary needs it; reference only
    DiagnosticCondition("slowheat_magnitude_soft_b30", "slowheat_ffn_attention", 30.0, "soft",
                        importance_criterion="magnitude"),
    DiagnosticCondition("slowheat_uniform_soft_b30", "slowheat_ffn_attention", 30.0, "uniform_soft",
                        importance_criterion="magnitude"),
)


def reorder(tasks, metadata, first: str, second: str):
    by_name = {t.domain: t for t in tasks}
    if first not in by_name or second not in by_name or first == second:
        raise SystemExit(f"bad pair {first} -> {second}; domains: {sorted(by_name)}")
    rest = [t for t in tasks if t.domain not in (first, second)]
    ordered = [by_name[first], by_name[second], *rest]
    meta = dict(metadata)
    meta["domain_order"] = [t.domain for t in ordered]
    meta["f5_pair"] = [first, second]
    return ordered, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pair", nargs=2, required=True, metavar=("T1", "T2"))
    p.add_argument("--output-dir", default="results/domain_pairs")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--budget", type=float, default=0.5)
    p.add_argument("--epochs-per-task", type=int, default=None, help="smoke only")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    base = SplitCLINC150Config(device=a.device, evaluate_test=False)
    if a.epochs_per_task is not None:
        base = replace(base, epochs_per_task=a.epochs_per_task)
    tasks, metadata = load_clinc150_tasks(base, include_test=False)
    tasks, metadata = reorder(tasks, metadata, *a.pair)
    config = replace(base, ffn_plasticity_budget=a.budget, attention_plasticity_budget=a.budget,
                     domain_order=tuple(metadata["domain_order"]))
    out = Path(a.output_dir) / f"{a.pair[0]}__{a.pair[1]}" / f"budget_{a.budget:g}"
    t0 = time.time()
    run_diagnostic(config, tasks, metadata=metadata, seeds=a.seeds, output_dir=out,
                   conditions=CONDITIONS, resume=True)
    print(f"pair {a.pair[0]} -> {a.pair[1]} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
