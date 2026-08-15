"""Sequential multi-seed aggregation for the synthetic benchmark."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.synthetic_cl import SyntheticConfig, load_config, run_experiment

METRIC_NAMES = (
    "final_average_accuracy",
    "average_forgetting",
    "backward_transfer",
    "forward_transfer",
)


def exact_two_sided_sign_test(differences: list[float]) -> float | None:
    """Two-sided exact sign test, ignoring exact ties."""

    nonzero = [value for value in differences if value != 0.0]
    count = len(nonzero)
    if count == 0:
        return None
    positives = sum(value > 0.0 for value in nonzero)
    tail = min(positives, count - positives)
    probability = 2.0 * sum(
        math.comb(count, index) for index in range(tail + 1)
    ) / (2**count)
    return min(1.0, probability)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) == 1:
        return {"mean": mean, "std": 0.0, "ci95_normal_half_width": 0.0}
    std = float(np.std(array, ddof=1))
    return {
        "mean": mean,
        "std": std,
        "ci95_normal_half_width": float(1.96 * std / np.sqrt(len(array))),
    }


def run_multi_seed(
    base_config: SyntheticConfig,
    *,
    seeds: list[int],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run seeds serially to keep peak resource use bounded."""

    if not seeds:
        raise ValueError("seeds não pode ser vazio")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds deve conter valores únicos")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        config = replace(base_config, seed=seed)
        raw[seed] = run_experiment(
            config,
            output_dir=output_path / f"seed_{seed}",
        )

    aggregate: dict[str, Any] = {
        "seeds": seeds,
        "methods": {},
        "paired_differences_vs_vanilla": {},
        "interval_note": (
            "CI95 is a normal approximation; use a predeclared paired test "
            "and bootstrap or t interval for publication."
        ),
    }
    for method in base_config.methods:
        aggregate["methods"][method] = {}
        for metric in METRIC_NAMES:
            values = [raw[seed][method]["metrics"][metric] for seed in seeds]
            if any(value is None for value in values):
                aggregate["methods"][method][metric] = None
            else:
                aggregate["methods"][method][metric] = _summary(values)

    if "vanilla" in base_config.methods:
        for method in base_config.methods:
            if method == "vanilla":
                continue
            method_differences: dict[str, Any] = {}
            for metric in METRIC_NAMES:
                differences: list[float] = []
                for seed in seeds:
                    candidate = raw[seed][method]["metrics"][metric]
                    baseline = raw[seed]["vanilla"]["metrics"][metric]
                    if candidate is None or baseline is None:
                        differences = []
                        break
                    differences.append(float(candidate - baseline))
                method_differences[metric] = (
                    {
                        **_summary(differences),
                        "exact_sign_test_p": exact_two_sided_sign_test(differences),
                    }
                    if differences
                    else None
                )
            aggregate["paired_differences_vs_vanilla"][method] = method_differences

    with (output_path / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True, allow_nan=False)
    with (output_path / "multi_seed_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {"base_config": asdict(base_config), "seeds": seeds},
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_multi_seed(
        load_config(args.config),
        seeds=args.seeds,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
