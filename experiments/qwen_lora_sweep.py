"""Multi-seed, multi-GPU orchestrator for the LoRA SlowHeat benchmark.

Each worker process runs ONE seed across ALL arms on ONE GPU. Arms therefore
stay paired within a seed (identical data stream, identical batch order), which
is what the paired sign test requires. Seeds are independent, so distributing
them across heterogeneous cards changes wall-clock per seed but not any metric.

This is NOT DDP: there is no gradient communication, and the cards may differ.

Status: EXPLORATORY. Seeds come from the declared exploratory sequence
``11 * k``; the confirmatory seeds reserved in ``confirmatory_split_mnist`` are
rejected so an exploratory sweep cannot burn them.

Run:
    HF_HOME=.hf-cache .venv/bin/python -m experiments.qwen_lora_sweep \\
        --output results/qwen_lora_sweep --tasks 10 --gpus 0 1 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.confirmatory_split_mnist import CONFIRMATORY_SEEDS
from experiments.multi_seed import exact_two_sided_sign_test
from experiments.qwen_lora_slowheat import ARMS

#: Declared exploratory sequence, matching the repository convention.
DEFAULT_SEEDS: tuple[int, ...] = tuple(11 * index for index in range(1, 11))

#: Arm every other arm is compared against in the paired analysis.
REFERENCE_ARM = "vanilla"


@dataclass(frozen=True)
class Job:
    seed: int
    gpu: int
    output: Path


def _launch(job: Job, arguments: argparse.Namespace) -> subprocess.Popen:
    command = [
        ".venv/bin/python",
        "-m",
        "experiments.qwen_lora_slowheat",
        "--output",
        str(job.output),
        "--tasks",
        str(arguments.tasks),
        "--seed",
        str(job.seed),
        "--device",
        "cuda:0",  # remapped by CUDA_VISIBLE_DEVICES
        "--train-per-class",
        str(arguments.train_per_class),
        "--eval-per-class",
        str(arguments.eval_per_class),
        "--batch-size",
        str(arguments.batch_size),
        "--epochs-per-task",
        str(arguments.epochs_per_task),
        "--max-length",
        str(arguments.max_length),
        "--rank",
        str(arguments.rank),
        "--slow-strength",
        str(arguments.slow_strength),
        "--plasticity-budget",
        str(arguments.plasticity_budget),
        "--arms",
        *arguments.arms,
    ]
    if arguments.hard:
        command.append("--hard")
    if arguments.target_plasticity is not None:
        command += ["--target-plasticity", str(arguments.target_plasticity)]

    environment = {
        "HF_HOME": ".hf-cache",
        "PYTHONPATH": ".",
        "CUDA_VISIBLE_DEVICES": str(job.gpu),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "TOKENIZERS_PARALLELISM": "false",
    }
    job.output.mkdir(parents=True, exist_ok=True)
    log = (job.output / "run.log").open("w", encoding="utf-8")
    return subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
    )


def _aggregate(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed results into paired contrasts against the reference."""

    by_arm: dict[str, dict[str, list[float]]] = {}
    for manifest in manifests:
        for result in manifest["results"]:
            bucket = by_arm.setdefault(
                result["method"],
                {
                    "final_average_accuracy": [],
                    "average_forgetting": [],
                    "backward_transfer": [],
                    "wall_clock_seconds": [],
                    "peak_memory_mib": [],
                    "train_tokens": [],
                    "effective_plasticity": [],
                },
            )
            bucket["final_average_accuracy"].append(result["final_average_accuracy"])
            bucket["average_forgetting"].append(result["average_forgetting"])
            bucket["backward_transfer"].append(result["backward_transfer"])
            bucket["wall_clock_seconds"].append(result["wall_clock_seconds"])
            bucket["peak_memory_mib"].append(result["peak_memory_mib"])
            bucket["train_tokens"].append(float(result["train_tokens"]))
            bucket["effective_plasticity"].append(
                result["stages"][-1]["effective_plasticity"]
            )

    summary: dict[str, Any] = {}
    for arm, values in by_arm.items():
        summary[arm] = {
            key: {
                "mean": float(np.mean(series)),
                "std": float(np.std(series, ddof=1)) if len(series) > 1 else 0.0,
                "n": len(series),
            }
            for key, series in values.items()
        }

    contrasts: dict[str, Any] = {}
    if REFERENCE_ARM in by_arm:
        for arm in by_arm:
            if arm == REFERENCE_ARM:
                continue
            paired: dict[str, Any] = {}
            for metric in ("final_average_accuracy", "average_forgetting"):
                treatment = by_arm[arm][metric]
                reference = by_arm[REFERENCE_ARM][metric]
                if len(treatment) != len(reference):
                    continue
                differences = [
                    float(a - b) for a, b in zip(treatment, reference, strict=True)
                ]
                paired[metric] = {
                    "mean_difference": float(np.mean(differences)),
                    "std_difference": (
                        float(np.std(differences, ddof=1))
                        if len(differences) > 1
                        else 0.0
                    ),
                    "exact_sign_test_p": exact_two_sided_sign_test(differences),
                    "per_seed_differences": differences,
                }
            contrasts[arm] = paired
    return {"per_arm": summary, "paired_vs_" + REFERENCE_ARM: contrasts}


def _format_table(summary: dict[str, Any], arms: list[str]) -> str:
    header = (
        "| arm | FAA (mean+-sd) | forgetting | BWT | seconds | peak MiB | E_eff |"
    )
    separator = "|---|---|---|---|---|---|---|"
    rows = []
    for arm in arms:
        if arm not in summary:
            continue
        item = summary[arm]
        rows.append(
            "| {arm} | {faa:.4f}+-{faa_sd:.4f} | {f:+.4f}+-{f_sd:.4f} | "
            "{b:+.4f} | {s:.1f} | {m:.1f} | {e:.3f} |".format(
                arm=arm,
                faa=item["final_average_accuracy"]["mean"],
                faa_sd=item["final_average_accuracy"]["std"],
                f=item["average_forgetting"]["mean"],
                f_sd=item["average_forgetting"]["std"],
                b=item["backward_transfer"]["mean"],
                s=item["wall_clock_seconds"]["mean"],
                m=item["peak_memory_mib"]["mean"],
                e=item["effective_plasticity"]["mean"],
            )
        )
    return "\n".join([header, separator, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--train-per-class", type=int, default=50)
    parser.add_argument("--eval-per-class", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-per-task", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--slow-strength", type=float, default=3.0)
    parser.add_argument("--plasticity-budget", type=float, default=0.5)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--target-plasticity", type=float, default=None)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Re-aggregate existing per-seed manifests without running anything.",
    )
    arguments = parser.parse_args()

    reserved = set(arguments.seeds) & set(CONFIRMATORY_SEEDS)
    if reserved:
        raise SystemExit(
            f"seeds confirmatórias reservadas não podem ser usadas em exploração: "
            f"{sorted(reserved)}"
        )
    if "slice" in arguments.arms and arguments.rank < arguments.tasks:
        raise SystemExit(
            f"o braço 'slice' requer rank >= tasks ({arguments.rank} < {arguments.tasks})"
        )

    arguments.output.mkdir(parents=True, exist_ok=True)
    jobs = [
        Job(
            seed=seed,
            gpu=arguments.gpus[index % len(arguments.gpus)],
            output=arguments.output / f"seed_{seed}",
        )
        for index, seed in enumerate(arguments.seeds)
    ]

    started = time.perf_counter()
    if not arguments.summarize_only:
        pending = list(jobs)
        running: list[tuple[Job, subprocess.Popen]] = []
        completed = 0
        # One concurrent job per GPU: each run needs ~2.4 GiB, but serializing
        # per card keeps the wall-clock numbers comparable across arms.
        while pending or running:
            while pending and len(running) < len(arguments.gpus):
                busy = {job.gpu for job, _ in running}
                candidate = next(
                    (job for job in pending if job.gpu not in busy), None
                )
                if candidate is None:
                    break
                pending.remove(candidate)
                running.append((candidate, _launch(candidate, arguments)))
                print(f"[launch] seed {candidate.seed} -> gpu {candidate.gpu}")
            time.sleep(5)
            for job, process in list(running):
                if process.poll() is None:
                    continue
                running.remove((job, process))
                completed += 1
                elapsed = time.perf_counter() - started
                status = "ok" if process.returncode == 0 else f"FAIL {process.returncode}"
                print(
                    f"[done] seed {job.seed} gpu {job.gpu} {status} "
                    f"({completed}/{len(jobs)}, {elapsed / 60:.1f} min)"
                )

    manifests = []
    missing = []
    for job in jobs:
        path = job.output / "manifest.json"
        if path.exists():
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(job.seed)
    if not manifests:
        raise SystemExit("nenhuma seed produziu manifesto; veja os run.log")

    aggregate = _aggregate(manifests)
    elapsed = time.perf_counter() - started
    report = {
        "status": "exploratory_multi_seed",
        "claim_scope": (
            "paired exploratory sweep; arms are NOT matched on effective "
            "plasticity and no reduced-learning-rate control is present, so a "
            "difference is not attributable to the mechanism alone"
        ),
        "seeds": arguments.seeds,
        "completed_seeds": [
            job.seed for job in jobs if (job.output / "manifest.json").exists()
        ],
        "missing_seeds": missing,
        "tasks": arguments.tasks,
        "arms": arguments.arms,
        "gpus": arguments.gpus,
        "orchestrator_wall_clock_seconds": elapsed,
        **aggregate,
    }
    (arguments.output / "sweep.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    table = _format_table(aggregate["per_arm"], arguments.arms)
    (arguments.output / "summary.md").write_text(table + "\n", encoding="utf-8")
    print()
    print(table)
    if missing:
        print(f"\nseeds sem manifesto: {missing}")
    print(f"\nwall-clock do orquestrador: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
