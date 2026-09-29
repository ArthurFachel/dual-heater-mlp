#!/usr/bin/env python3
"""Confirmatory run of the plasticity-matched protocol.

Executes the ten pre-registered seeds of goals/protocol_plasticity_matched.md
over three arms:

    vanilla            lr 1e-4,        A trainable    E_surface = 1.000
    lr_control         lr 6.2246e-5,   A trainable    E_surface = 0.622462
    frozen_a_control   lr 1e-4,        A FROZEN       E_surface = 0.622462

The two treated arms are matched on surface plasticity by arithmetic, not by
bisection: 0.6224617... is simultaneously the fraction of trainable parameters
surviving the freeze (4_214_016 / 6_769_920) and the control's learning-rate
scale.

Why this run exists: the 28/09 confirmation reported `exact - lr_control` as
plasticity-matched at E = 0.850, but that average was scoped to the masked
parameters and could not see the 2_555_904 parameters the arm FROZE. Measured
over the reference arm's trainable surface the contrast was 0.532 against
0.850. See section A of the protocol.

`exact` is deliberately absent (P1): the decomposition established that the
SlowHeat mask contributes nothing (5+/5, p = 1.00), and it costs 1.8x the time.

Hyperparameters are copied verbatim from the original confirmation manifest
(results/qwen_lora_confirmation/seed_700001/manifest.json).

Cost, measured from the decomposition manifests on the same scenario:
vanilla 348 s + lr_control ~356 s + frozen_a_control 335 s = ~1040 s/seed.
One whole seed per GPU over three GPUs is four waves, about 70 minutes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from dual_heater.lora_slowheat import FROZEN_A_LR_SCALE
from experiments.qwen_lora_slowheat import PLASTICITY_MATCHED_SEEDS

OUTPUT_DIR = Path("results/plasticity_matched")

#: P1 of the protocol. Order is the order arms run in.
ARMS: tuple[str, ...] = ("vanilla", "lr_control", "frozen_a_control")

#: P8. Not tuned: it is the freezing fraction, derived from two parameter
#: counts with no accuracy consulted.
LEARNING_RATE_SCALE: float = FROZEN_A_LR_SCALE


def build_command(*, seed: int, destination: str | Path) -> list[str]:
    """Build the command for one seed.

    Extracted so a test can assert the frozen scenario and the learning-rate
    scale are in the command that actually runs. ``--target-plasticity`` is
    deliberately NOT passed: no arm here carries a mask, and in the runner it
    would take precedence over the explicit scale in older code paths.
    """

    return [
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
        "--learning-rate-scale", repr(LEARNING_RATE_SCALE),
        "--device", "cuda:0",
    ]


def shard_seeds(
    seeds: tuple[int, ...], *, shard: int, shards: int
) -> tuple[int, ...]:
    """Seeds this shard owns, round-robin so every shard gets whole seeds.

    Sharding by SEED (not by arm) keeps all three arms of a seed on one GPU, so
    they share the identical data stream, ordering and token count and the
    paired sign test stays valid.
    """

    if shards < 1:
        raise ValueError("shards deve ser >= 1")
    if not 0 <= shard < shards:
        raise ValueError("shard fora do intervalo [0, shards)")
    return tuple(seeds[shard::shards])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    arguments = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seeds = shard_seeds(
        PLASTICITY_MATCHED_SEEDS, shard=arguments.shard, shards=arguments.shards
    )
    total = len(seeds)
    label = f"shard {arguments.shard}/{arguments.shards}"
    print(f"{label}: {total} seeds -> {list(seeds)}", flush=True)

    for index, seed in enumerate(seeds, start=1):
        destination = OUTPUT_DIR / f"seed_{seed}"
        if (destination / "summary.md").is_file():
            print(f"[{label} {index}/{total}] seed {seed}: já concluída", flush=True)
            continue

        print(f"[{label} {index}/{total}] seed {seed}: iniciando", flush=True)
        outcome = subprocess.run(
            build_command(seed=seed, destination=destination),
            capture_output=True,
            text=True,
        )
        if outcome.returncode != 0:
            print(f"seed {seed} FALHOU:\n{outcome.stderr[-2000:]}", flush=True)
            raise SystemExit(1)

        elapsed = (time.time() - started) / 60
        print(
            f"[{label} {index}/{total}] seed {seed}: concluída "
            f"({elapsed:.1f} min neste shard)",
            flush=True,
        )

    print(f"{label} completo em {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
