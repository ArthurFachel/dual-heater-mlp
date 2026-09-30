#!/usr/bin/env python
"""Passada 1 do piloto de penalidade no host Split-CIFAR100.

`goals/protocol_penalty_reevaluation.md` §C.1 e §D. Mesma passada 1 do
Split-MNIST (`scripts/run_penalty_pass1.py`), outro host.

**Nenhum endpoint de acurácia é lido aqui.**

**Sharding por seed, não por arm.** Cada GPU executa TODAS as arms de um
subconjunto de seeds, nunca metade das arms de uma seed. Partir uma seed entre
devices quebraria o pareamento: as arms compartilham o fluxo de dados e a
inicialização dentro da seed, e é isso que torna a diferença pareada válida.

Uso (uma GPU por shard, em paralelo):

    for s in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$s PYTHONPATH=. .venv/bin/python \\
        scripts/run_penalty_pass1_cifar.py --shard $s --shards 3 &
    done
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.confirmatory_split_mnist import (  # noqa: E402
    PENALTY_REEVALUATION_SEEDS,
)
from experiments.penalty_reevaluation import (  # noqa: E402
    PUBLISHED_PENALTY_STRENGTHS,
)
from experiments.split_mnist import run_split_mnist  # noqa: E402
from experiments.visual_generalization import (  # noqa: E402
    generalization_configs,
    load_split_cifar100,
)

#: §E, emendado: medir TODO passo com âncora.
DECLARED_INTERVAL = 1

ARMS = ("vanilla", "ewc", "si", "mas")
PENALTY_ARMS = ("ewc", "si", "mas")

OUTPUT_DIR = Path("results/penalty_pass1_cifar")


def shard_seeds(seeds: list[int], *, shard: int, shards: int) -> list[int]:
    """Reparte as seeds entre devices. Cada seed inteira num só device."""

    if shards < 1:
        raise ValueError("shards deve ser >= 1")
    if not 0 <= shard < shards:
        raise ValueError(f"shard deve estar em [0, {shards})")
    return [seed for index, seed in enumerate(seeds) if index % shards == shard]


def build_config(seed: int, *, device: str):
    base = generalization_configs(device=device)["split_cifar100"]
    return replace(
        base,
        seed=seed,
        methods=ARMS,
        plasticity_sampling_interval=DECLARED_INTERVAL,
        device=device,
        **PUBLISHED_PENALTY_STRENGTHS,
    )


def should_reuse(record: object) -> bool:
    """Só reaproveita artefato gerado sob ESTA configuração."""

    if not isinstance(record, dict):
        return False
    if record.get("declared_interval") != DECLARED_INTERVAL:
        return False
    if record.get("penalty_strengths") != dict(PUBLISHED_PENALTY_STRENGTHS):
        return False
    if record.get("host") != "split_cifar100":
        return False
    return True


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = shard_seeds(
        list(PENALTY_REEVALUATION_SEEDS), shard=args.shard, shards=args.shards
    )
    if args.limit is not None:
        seeds = seeds[: args.limit]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[shard {args.shard}/{args.shards}] device={device}, "
        f"{len(seeds)} seeds: {seeds}",
        flush=True,
    )

    # Carrega o dataset UMA vez; as tarefas são determinísticas dado o config.
    reference = build_config(seeds[0], device=device)
    tasks = load_split_cifar100(reference, data_dir="data", download=False)

    started = time.perf_counter()
    for index, seed in enumerate(seeds, start=1):
        destination = OUTPUT_DIR / f"seed_{seed}.json"
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if should_reuse(existing):
                print(f"[{index}/{len(seeds)}] seed {seed}: já existe", flush=True)
                continue
            print(
                f"[{index}/{len(seeds)}] seed {seed}: outra configuração, regravando",
                flush=True,
            )

        config = build_config(seed, device=device)
        seed_started = time.perf_counter()
        results = run_split_mnist(config, tasks)
        seed_elapsed = time.perf_counter() - seed_started

        record: dict = {
            "seed": seed,
            "host": "split_cifar100",
            "device": device,
            "declared_interval": DECLARED_INTERVAL,
            "penalty_strengths": dict(PUBLISHED_PENALTY_STRENGTHS),
            "elapsed_seconds": seed_elapsed,
            "arms": {},
        }
        for arm in PENALTY_ARMS:
            samples = results[arm]["plasticity_samples"]
            record["arms"][arm] = {
                key: summarize([s[key] for s in samples if s[key] is not None])
                for key in ("norm_ratio", "plasticity_ratio", "direction_cosine")
            } if samples else {}
        destination.write_text(json.dumps(record, indent=2), encoding="utf-8")

        line = ", ".join(
            f"{arm} E={record['arms'][arm]['norm_ratio']['mean']:.4f}"
            for arm in PENALTY_ARMS
            if record["arms"][arm]
        )
        print(
            f"[{index}/{len(seeds)}] seed {seed}: {seed_elapsed / 60:.1f} min — {line}",
            flush=True,
        )

    print(
        f"\n[shard {args.shard}] concluído em "
        f"{(time.perf_counter() - started) / 60:.1f} min",
        flush=True,
    )
    print("NENHUMA acurácia foi lida nesta passada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
