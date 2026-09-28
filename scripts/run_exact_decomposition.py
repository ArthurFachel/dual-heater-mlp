#!/usr/bin/env python3
"""Confirmatory run of the `exact` decomposition.

Executes the ten pre-registered seeds of goals/protocol_exact_decomposition.md
over three arms: vanilla, frozen_a_control (LoRA-FA), exact (LoRA-FA+SlowHeat).

Hyperparameters are copied verbatim from the original confirmation manifest
(results/qwen_lora_confirmation/seed_700001/manifest.json) so the decomposition
is measured under the same conditions as the result it decomposes.

Cost is measured, not guessed: the calibration seed took 21.6 min for the three
arms on a 1080 Ti, so ten seeds land near 3.6 h.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from experiments.qwen_lora_slowheat import EXACT_DECOMPOSITION_SEEDS

OUTPUT_DIR = Path("results/exact_decomposition")
ARMS = ("vanilla", "frozen_a_control", "exact")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for index, seed in enumerate(EXACT_DECOMPOSITION_SEEDS, start=1):
        destination = OUTPUT_DIR / f"seed_{seed}"
        if (destination / "summary.md").is_file():
            print(f"[{index}/10] seed {seed}: already done, skipping", flush=True)
            continue

        print(f"[{index}/10] seed {seed}: starting", flush=True)
        command = [
            sys.executable,
            "experiments/qwen_lora_slowheat.py",
            "--output", str(destination),
            "--arms", *ARMS,
            "--seed", str(seed),
            "--tasks", "10",
            "--rank", "16",
            "--alpha", "16.0",
            "--train-per-class", "50",
            "--eval-per-class", "20",
            "--epochs-per-task", "3",
            "--batch-size", "8",
            "--max-length", "48",
            "--slow-strength", "3.0",
            "--plasticity-budget", "0.5",
            "--target-plasticity", "0.85",
            "--device", "cuda:0",
        ]
        outcome = subprocess.run(command, capture_output=True, text=True)
        if outcome.returncode != 0:
            print(f"seed {seed} FAILED:\n{outcome.stderr[-2000:]}", flush=True)
            raise SystemExit(1)

        elapsed = (time.time() - started) / 60
        print(f"[{index}/10] seed {seed}: done ({elapsed:.1f} min total)", flush=True)

    print(f"all seeds done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
