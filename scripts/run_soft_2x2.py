#!/usr/bin/env python3
"""EXPLORATORY: 2x2 cell 'concentrated importance + SOFT mask' on BERT/CLINC150.

Phase 0 of the concentration law found that every setting where selection beat a
same-plasticity control used a hard mask, and every setting where it did not used a
soft mask. This run fills the missing cell: BERT CLINC (concentrated FFN importance,
CV ~0.59) with a SOFT mask, SlowHeat |z| vs a uniform control with the same mean
soft factor (uniform_soft). Two strengths: slow_strength 3 (project default for
BERT) and 30 (the MLP/CNN soft default, 1/(1+30h)).

NOT pre-registered. Fresh seed band 14.000.003+, disjoint from 13.000.003+ and
13.500.003+ and from every band in goals/.
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
BUDGETS = (0.1, 0.25, 0.5, 0.75)

CONDITIONS = (
    DiagnosticCondition("vanilla", "vanilla", 3.0, "soft"),
    DiagnosticCondition("slowheat_magnitude_soft_b3", "slowheat_ffn_attention", 3.0, "soft",
                        importance_criterion="magnitude"),
    DiagnosticCondition("slowheat_uniform_soft_b3", "slowheat_ffn_attention", 3.0, "uniform_soft",
                        importance_criterion="magnitude"),
    DiagnosticCondition("slowheat_magnitude_soft_b30", "slowheat_ffn_attention", 30.0, "soft",
                        importance_criterion="magnitude"),
    DiagnosticCondition("slowheat_uniform_soft_b30", "slowheat_ffn_attention", 30.0, "uniform_soft",
                        importance_criterion="magnitude"),
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results/soft_2x2")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--budgets", nargs="+", type=float, default=list(BUDGETS))
    p.add_argument("--epochs-per-task", type=int, default=None, help="smoke only")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    base = SplitCLINC150Config(device=a.device, evaluate_test=False)
    if a.epochs_per_task is not None:
        base = replace(base, epochs_per_task=a.epochs_per_task)
    tasks, metadata = load_clinc150_tasks(base, include_test=False)
    out = Path(a.output_dir)
    started = time.time()
    for budget in a.budgets:
        config = replace(base, ffn_plasticity_budget=budget, attention_plasticity_budget=budget)
        run_diagnostic(config, tasks, metadata=metadata, seeds=a.seeds,
                       output_dir=out / f"budget_{budget:g}", conditions=CONDITIONS,
                       resume=True)
        print(f"budget {budget:g} done at {(time.time() - started) / 60:.1f} min", flush=True)
    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
