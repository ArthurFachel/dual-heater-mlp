#!/usr/bin/env python3
"""EXPLORATORY: retention x acquisition curve across protection budgets.

At one budget (0.25) SlowHeat beat the plasticity-matched uniform control by +3.9 pp
of task-1 retention but acquired task 2 2.6 pp worse. One point cannot separate
"better selection" from "another point on the same trade-off". This sweeps the
plasticity budget for the three protected arms (random hard, magnitude hard,
uniform) so each arm traces a retention-vs-acquisition curve.

NOT pre-registered. Seeds from a fresh band (13.500.003+), disjoint from every band
in goals/, docs/ and from results/uniform_lr_control (13.000.003+).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from experiments.bert_slowheat_diagnostic import DiagnosticCondition, run_diagnostic
from experiments.split_clinc150 import SplitCLINC150Config, load_clinc150_tasks

SEEDS = (
    13_500_003, 13_525_011, 13_550_017, 13_575_037, 13_600_043,
    13_625_059, 13_650_061, 13_675_087, 13_700_089, 13_725_097,
)
BUDGETS = (0.1, 0.25, 0.5, 0.75)

PROTECTED = (
    DiagnosticCondition("slowheat_random_hard", "slowheat_ffn_attention", 3.0, "random_hard"),
    DiagnosticCondition("slowheat_magnitude_hard", "slowheat_ffn_attention", 3.0, "hard",
                        importance_criterion="magnitude"),
    DiagnosticCondition("slowheat_uniform_hard", "slowheat_ffn_attention", 3.0, "uniform_hard",
                        importance_criterion="magnitude"),
)
# vanilla does not depend on the budget; it is rerun per budget because the
# diagnostic summary computes every contrast against it (and it doubles as a
# determinism check across budgets).
VANILLA = (DiagnosticCondition("vanilla", "vanilla", 3.0, "soft"),)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results/budget_curve")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--budgets", nargs="+", type=float, default=list(BUDGETS))
    p.add_argument("--epochs-per-task", type=int, default=None, help="smoke only")
    a = p.parse_args()
    base = SplitCLINC150Config(device="cuda", evaluate_test=False)
    if a.epochs_per_task is not None:
        base = replace(base, epochs_per_task=a.epochs_per_task)
    tasks, metadata = load_clinc150_tasks(base, include_test=False)
    out = Path(a.output_dir)
    started = time.time()
    for budget in a.budgets:
        config = replace(base, ffn_plasticity_budget=budget, attention_plasticity_budget=budget)
        run_diagnostic(config, tasks, metadata=metadata, seeds=a.seeds,
                       output_dir=out / f"budget_{budget:g}", conditions=VANILLA + PROTECTED)
        print(f"budget {budget:g} done at {(time.time() - started) / 60:.1f} min", flush=True)
    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
