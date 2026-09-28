#!/usr/bin/env python3
"""Calibration seed for the importance-criterion ablation (Section I).

Runs ONE seed of the four-arm ablation matrix to measure real wall-clock cost
before committing to the ten pre-registered seeds. Section I of
goals/protocol_importance_criterion_ablation.md requires a measured cost, not an
estimate by analogy -- estimating by analogy already failed twice in this
project (CPU/GPU confusion, 4.7x error).

The seed used here is IMPORTANCE_CRITERION_ABLATION_SEEDS[0]. Its output lives
in a separate directory so a calibration run is never silently mistaken for
confirmatory evidence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.bert_slowheat_diagnostic import (
    IMPORTANCE_CRITERION_ABLATION_SEEDS,
    criterion_ablation_conditions,
    run_diagnostic,
)
from experiments.split_clinc150 import SplitCLINC150Config, load_clinc150_tasks

OUTPUT_DIR = Path("results/criterion_ablation_calibration")


def main() -> None:
    conditions = criterion_ablation_conditions()
    seed = IMPORTANCE_CRITERION_ABLATION_SEEDS[0]

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
    )
    elapsed = time.time() - started

    remaining = len(IMPORTANCE_CRITERION_ABLATION_SEEDS) - 1
    calibration = {
        "seed": seed,
        "arms": [c.name for c in conditions],
        "elapsed_seconds_one_seed": round(elapsed, 1),
        "projected_hours_remaining_nine_seeds": round(elapsed * remaining / 3600.0, 2),
        "projected_hours_all_ten_seeds": round(
            elapsed * len(IMPORTANCE_CRITERION_ABLATION_SEEDS) / 3600.0, 2
        ),
    }
    (OUTPUT_DIR / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
