#!/usr/bin/env python3
"""Confirmatory run of the importance-criterion ablation.

Executes the ten pre-registered seeds of
goals/protocol_importance_criterion_ablation.md over the four-arm matrix
(vanilla, random, magnitude, functional).

Cost is known, not guessed: the calibration seed measured 66.4 s for the whole
matrix on a 1080 Ti, so ten seeds land near eleven minutes.
"""

from __future__ import annotations

import time
from pathlib import Path

from experiments.bert_slowheat_diagnostic import (
    IMPORTANCE_CRITERION_ABLATION_SEEDS,
    criterion_ablation_conditions,
    run_diagnostic,
)
from experiments.split_clinc150 import SplitCLINC150Config, load_clinc150_tasks

OUTPUT_DIR = Path("results/criterion_ablation")


def main() -> None:
    config = SplitCLINC150Config(device="cuda", evaluate_test=False)
    tasks, metadata = load_clinc150_tasks(config, include_test=False)

    started = time.time()
    run_diagnostic(
        config,
        tasks,
        metadata=metadata,
        seeds=list(IMPORTANCE_CRITERION_ABLATION_SEEDS),
        output_dir=OUTPUT_DIR,
        conditions=criterion_ablation_conditions(),
    )
    print(f"done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
