#!/usr/bin/env python3
"""Pre-endpoint checks for the long-sequence criterion extension.

Section H of goals/protocol_long_sequence_criterion.md. Runs BEFORE anyone
looks at an endpoint: if any check fails, the contrast is not interpretable and
the run must be diagnosed, not analysed.

Item 3.3 of goals/roadmap_icml_ijcnn.md states the hard rule this script
enforces:

    "Repeat the variance falsifier of the normalized ranking. Without that
     number the |z| vs |z·dL/dz| tie is not reportable."

Usage:

    PYTHONPATH=src:. .venv/bin/python scripts/verify_long_sequence_criterion.py

Exit 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.bert_slowheat_diagnostic import (
    LONG_SEQUENCE_CRITERION_SEEDS,
    LONG_SEQUENCE_TASK_LIMIT,
    criterion_ablation_conditions,
)
from experiments.criterion_degeneracy import (
    ABLATION_ARM_PATHS,
    DEGENERACY_OVERLAP_THRESHOLD,
    load_arm_states,
    summarize_degeneracy_overlap,
)

DEFAULT_ROOT = Path("results/long_sequence_criterion")

#: Section G.1: chance on a 150-way head with T tasks seen, under the runner's
#: masking of unseen logits.
CLASSES_PER_TASK = 15
#: FAA below this multiple of chance means a degenerate regime, and the whole
#: confirmatory family is reported inconclusive regardless of p.
NEAR_CHANCE_FACTOR = 4.0

#: Section C: the T=2 run measured these. Reported side by side so a collapse
#: of the ranking with longer sequences is visible, never used as a pass/fail
#: threshold of its own.
TWO_TASK_RANKING_VARIANCE = {
    "slowheat_hard": 2.9124e-03,
    "slowheat_magnitude_hard": 2.9070e-03,
    "slowheat_random_hard": 3.1250e-03,
}
TWO_TASK_MAGNITUDE_VS_RANDOM = 0.5748018835536388


def _chance(task_limit: int) -> float:
    return 1.0 / (task_limit * CLASSES_PER_TASK)


def check_long_sequence_run(
    root: Path,
    *,
    seeds: tuple[int, ...] = LONG_SEQUENCE_CRITERION_SEEDS,
    task_limit: int = LONG_SEQUENCE_TASK_LIMIT,
) -> tuple[list[str], dict]:
    """Return (failures, report). An empty failure list means the run is sound."""

    failures: list[str] = []
    report: dict = {
        "run_dir": str(root),
        "task_limit": task_limit,
        "chance": _chance(task_limit),
    }

    summary_path = root / "diagnostic_summary.json"
    if not summary_path.is_file():
        return [f"V1: {summary_path} não existe"], report
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # V1: every pre-registered seed ran, and no other.
    if tuple(summary["seeds"]) != tuple(sorted(seeds)):
        failures.append(
            f"V1: seeds {summary['seeds']} não são a banda pré-registrada"
        )

    # V2: the sequence length is the frozen one. A run at another T is a
    # different protocol wearing this one's name.
    if summary["task_count"] != task_limit:
        failures.append(
            f"V2: task_count={summary['task_count']}, esperado {task_limit}"
        )

    # V3: exactly the four declared arms.
    expected_arms = {condition.name for condition in criterion_ablation_conditions()}
    if set(summary["conditions"]) != expected_arms:
        failures.append(f"V3: braços {sorted(summary['conditions'])}")

    # V4: capacity matched across the protected arms. Without it the contrast
    # mixes criterion with amount of protection -- the error lr_control exists
    # to prevent.
    protected_counts = {
        name: summary["conditions"][name]["protected_count"]["mean"]
        for name in summary["conditions"]
        if name != "vanilla"
    }
    report["protected_count"] = protected_counts
    distinct = {round(value, 6) for value in protected_counts.values() if value}
    if len(distinct) > 1:
        failures.append(f"V4: capacidade não pareada entre braços: {protected_counts}")

    # V5 -- THE VARIANCE FALSIFIER (item 3.3 of the roadmap, Section F of the
    # original pre-registration). A nearly flat ranking makes top-k protection
    # noise-driven, which would turn a criterion tie into an artefact.
    variances = {
        name: summary["conditions"][name]["ranking_variance"]["mean"]
        for name in summary["conditions"]
    }
    report["ranking_variance"] = variances
    report["ranking_variance_two_task_reference"] = TWO_TASK_RANKING_VARIANCE
    for name in ("slowheat_magnitude_hard", "slowheat_hard"):
        value = variances.get(name)
        if value is None:
            failures.append(f"V5: variância do ranking ausente em {name}")
        elif value <= 0.0:
            failures.append(f"V5: variância do ranking nula em {name} ({value})")

    magnitude_variance = variances.get("slowheat_magnitude_hard")
    functional_variance = variances.get("slowheat_hard")
    if magnitude_variance and functional_variance:
        ratio = magnitude_variance / functional_variance
        report["magnitude_over_functional_variance"] = ratio

    # V6 -- the cross-arm half of Section F, which the 2026-09-28 run never
    # emitted. The frozen rule lives here.
    try:
        by_seed = {
            seed: load_arm_states(root, seed, ABLATION_ARM_PATHS)
            for seed in summary["seeds"]
        }
    except FileNotFoundError as error:
        failures.append(f"V6: checkpoint ausente, sobreposição não calculável: {error}")
    else:
        overlap = summarize_degeneracy_overlap(by_seed)
        report["degeneracy_overlap"] = overlap
        report["magnitude_vs_random_two_task_reference"] = (
            TWO_TASK_MAGNITUDE_VS_RANDOM
        )
        if overlap["degenerate"]:
            failures.append(
                "V6: sobreposição magnitude-vs-aleatório "
                f"{overlap['pairs']['magnitude_vs_random']['mean']:.4f} > "
                f"{DEGENERACY_OVERLAP_THRESHOLD}; empate é INCONCLUSIVO, "
                "não evidência (§F)"
            )
        if not overlap["capacity_matched_every_seed"]:
            failures.append("V6: capacidade não pareada em alguma seed")

    # V7 -- Section G.1's near-chance gate. Declared before the seeds precisely
    # so it cannot be argued about afterwards.
    chance = _chance(task_limit)
    faa = {
        name: summary["conditions"][name]["final_average_accuracy"]["mean"]
        for name in summary["conditions"]
    }
    report["final_average_accuracy"] = faa
    best = max(value for value in faa.values() if value is not None)
    report["best_faa"] = best
    report["near_chance_threshold"] = NEAR_CHANCE_FACTOR * chance
    if best < NEAR_CHANCE_FACTOR * chance:
        failures.append(
            f"V7: melhor FAA {best:.4f} abaixo de {NEAR_CHANCE_FACTOR}x a chance "
            f"({chance:.4f}); família confirmatória é INCONCLUSIVA por regime "
            "degenerado (§G.1), independentemente de p"
        )

    return failures, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_ROOT))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.run_dir)
    failures, report = check_long_sequence_run(root)
    report["failures"] = failures
    report["passed"] = not failures

    destination = Path(args.output or root / "verification.json")
    if root.is_dir():
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        print("\nFALHAS:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\nOK: todas as verificações passaram; endpoints são interpretáveis.")


if __name__ == "__main__":
    main()
