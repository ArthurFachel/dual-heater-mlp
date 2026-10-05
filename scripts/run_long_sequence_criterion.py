#!/usr/bin/env python3
"""Confirmatory run of the long-sequence criterion extension.

Executes the ten pre-registered seeds of
goals/protocol_long_sequence_criterion.md over the four-arm matrix
(vanilla, random, magnitude, functional) at T=5.

Cost must be known, not guessed: run scripts/run_long_sequence_calibration.py
first and check its projection against the 4 h budget of Section I.
"""

from __future__ import annotations

import time
from pathlib import Path

from experiments.bert_slowheat_diagnostic import (
    LONG_SEQUENCE_CRITERION_SEEDS,
    LONG_SEQUENCE_TASK_LIMIT,
    criterion_ablation_conditions,
    run_diagnostic,
)
from experiments.split_clinc150 import SplitCLINC150Config, load_clinc150_tasks

OUTPUT_DIR = Path("results/long_sequence_criterion")


def main() -> None:
    config = SplitCLINC150Config(device="cuda", evaluate_test=False)
    tasks, metadata = load_clinc150_tasks(config, include_test=False)

    started = time.time()
    run_diagnostic(
        config,
        tasks,
        metadata=metadata,
        seeds=list(LONG_SEQUENCE_CRITERION_SEEDS),
        output_dir=OUTPUT_DIR,
        conditions=criterion_ablation_conditions(),
        task_limit=LONG_SEQUENCE_TASK_LIMIT,
    )
    print(f"done in {(time.time() - started) / 60:.1f} min")
    print(
        "próximo passo: PYTHONPATH=src:. python "
        "scripts/analyze_criterion_degeneracy.py "
        f"--run-dir {OUTPUT_DIR} --seeds "
        + " ".join(str(seed) for seed in LONG_SEQUENCE_CRITERION_SEEDS)
    )


if __name__ == "__main__":
    main()
