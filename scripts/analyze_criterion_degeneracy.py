#!/usr/bin/env python3
"""Recover the missing Section F metric from the finished criterion ablation.

goals/protocol_importance_criterion_ablation.md Section F declares TWO
mandatory diagnostics. The 2026-09-28 confirmatory run emitted only the ranking
variance: the overlap is a cross-arm quantity (it compares the protected sets
of two separate runs), and the runner's in-run hook had no access to a second
arm, so `top_k_overlap` was never computed. Without it the frozen
interpretation rule -- overlap(magnitude, random) > 0.8 means the arm
degenerated -- could not be evaluated, and the published tie rested on the
variance alone.

This script reads the arms' saved `slow_heat` buffers, which ARE the masks the
runs used, and writes the overlap report next to the existing summary. It
trains nothing and reads no accuracy.

    PYTHONPATH=src:. python scripts/analyze_criterion_degeneracy.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.artifacts import write_json_atomic
from experiments.bert_slowheat_diagnostic import (
    IMPORTANCE_CRITERION_ABLATION_SEEDS,
)
from experiments.criterion_degeneracy import (
    load_arm_states,
    summarize_degeneracy_overlap,
)

DEFAULT_RUN = Path("results/criterion_ablation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(IMPORTANCE_CRITERION_ABLATION_SEEDS),
    )
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    by_seed = {seed: load_arm_states(run_dir, seed) for seed in args.seeds}
    summary = summarize_degeneracy_overlap(by_seed)
    summary["run_dir"] = str(run_dir)

    destination = Path(args.output or run_dir / "degeneracy_overlap.json")
    write_json_atomic(destination, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nescrito em {destination}")


if __name__ == "__main__":
    main()
