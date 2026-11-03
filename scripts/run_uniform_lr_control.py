#!/usr/bin/env python3
"""EXPLORATORY: does R-A survive a plasticity-matched uniform control?

R-A (goals/protocol_importance_criterion_ablation.md) showed magnitude-ranked hard
protection beats a random hard mask of the same size by +12.23 pp of task-1
retention. Random-vs-selected holds the protected COUNT fixed, but it does not
separate "protect the right units" from "remove plasticity in a way that happens
to suit this task". This run adds `slowheat_uniform_hard`: every covered entry is
updated with the same factor, equal to the mean of the hard mask over the covered
surface (experiments.split_clinc150.UniformPlasticity). Same surface plasticity,
no unit selection, nothing frozen.

NOT pre-registered. Seeds come from a fresh band (13.000.003+) disjoint from every
band in goals/ and docs/. Nothing here is confirmatory.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from experiments.bert_slowheat_diagnostic import DiagnosticCondition, run_diagnostic
from experiments.split_clinc150 import SplitCLINC150Config, load_clinc150_tasks

SEEDS = (
    13_000_003, 13_025_011, 13_050_017, 13_075_037, 13_100_043,
    13_125_059, 13_150_061, 13_175_087, 13_200_089, 13_225_097,
)

CONDITIONS = (
    DiagnosticCondition("vanilla", "vanilla", 3.0, "soft"),
    DiagnosticCondition("slowheat_random_hard", "slowheat_ffn_attention", 3.0, "random_hard"),
    DiagnosticCondition("slowheat_magnitude_hard", "slowheat_ffn_attention", 3.0, "hard",
                        importance_criterion="magnitude"),
    DiagnosticCondition("slowheat_uniform_hard", "slowheat_ffn_attention", 3.0, "uniform_hard",
                        importance_criterion="magnitude"),
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results/uniform_lr_control")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--epochs-per-task", type=int, default=None, help="smoke only")
    a = p.parse_args()
    config = SplitCLINC150Config(device="cuda", evaluate_test=False)
    if a.epochs_per_task is not None:
        config = replace(config, epochs_per_task=a.epochs_per_task)
    tasks, metadata = load_clinc150_tasks(config, include_test=False)
    started = time.time()
    run_diagnostic(config, tasks, metadata=metadata, seeds=a.seeds,
                   output_dir=Path(a.output_dir), conditions=CONDITIONS)
    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
