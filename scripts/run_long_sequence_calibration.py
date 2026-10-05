#!/usr/bin/env python3
"""Calibration seed for the long-sequence criterion extension (Section I).

Runs ONE seed of the four-arm matrix at T=5 to measure real wall-clock cost
before committing to the ten pre-registered seeds. Section I of
goals/protocol_long_sequence_criterion.md requires a measured cost, not an
estimate by analogy -- estimating by analogy already failed twice in this
project (CPU/GPU confusion, 4.7x error).

The seed used here is LONG_SEQUENCE_CALIBRATION_SEED, deliberately OUTSIDE the
confirmatory band: the calibration runs the full matrix and therefore writes
accuracy, and spending a band seed on it would mean reading one confirmatory
seed before the other nine exist. Its output lives in a separate directory so a
calibration run is never silently mistaken for confirmatory evidence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.bert_slowheat_diagnostic import (
    LONG_SEQUENCE_CALIBRATION_SEED,
    LONG_SEQUENCE_CRITERION_SEEDS,
    LONG_SEQUENCE_TASK_LIMIT,
    criterion_ablation_conditions,
    run_diagnostic,
)
from experiments.split_clinc150 import SplitCLINC150Config, load_clinc150_tasks

OUTPUT_DIR = Path("results/long_sequence_criterion_calibration")

#: T=2, four arms, ten seeds on a 1080 Ti took 10.9 min (66.4 s per seed).
#: Printed next to the new measurement so the scaling is visible, never used
#: as a substitute for measuring.
TWO_TASK_SECONDS_PER_SEED = 66.4


def main() -> None:
    conditions = criterion_ablation_conditions()
    seed = LONG_SEQUENCE_CALIBRATION_SEED
    if seed in LONG_SEQUENCE_CRITERION_SEEDS:
        raise RuntimeError(
            "a seed de calibração não pode pertencer à banda confirmatória"
        )

    config = SplitCLINC150Config(device="cuda", evaluate_test=False)
    tasks, metadata = load_clinc150_tasks(config, include_test=False)

    started = time.time()
    run_diagnostic(
        config,
        tasks,
        metadata=metadata,
        seeds=[seed],
        output_dir=OUTPUT_DIR,
        conditions=conditions,
        task_limit=LONG_SEQUENCE_TASK_LIMIT,
    )
    elapsed = time.time() - started

    total = len(LONG_SEQUENCE_CRITERION_SEEDS)
    calibration = {
        "seed": seed,
        "task_limit": LONG_SEQUENCE_TASK_LIMIT,
        "arms": [condition.name for condition in conditions],
        "elapsed_seconds_one_seed": round(elapsed, 1),
        "two_task_seconds_per_seed": TWO_TASK_SECONDS_PER_SEED,
        "cost_ratio_vs_two_tasks": round(elapsed / TWO_TASK_SECONDS_PER_SEED, 2),
        "projected_hours_remaining": round(elapsed * (total - 1) / 3600.0, 2),
        "projected_hours_all_seeds": round(elapsed * total / 3600.0, 2),
        "section_i_budget_hours": 4.0,
    }
    calibration["within_budget"] = (
        calibration["projected_hours_all_seeds"] <= calibration["section_i_budget_hours"]
    )
    (OUTPUT_DIR / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    print(json.dumps(calibration, indent=2))
    if not calibration["within_budget"]:
        print(
            "\nATENÇÃO: a projeção excede as 4 h do §I. Reavaliar o escopo "
            "antes de lançar as seeds confirmatórias."
        )


if __name__ == "__main__":
    main()
