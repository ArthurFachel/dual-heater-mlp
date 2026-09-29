#!/usr/bin/env python
"""Seed de calibração do piloto de re-avaliação de métodos de penalidade.

`goals/protocol_penalty_reevaluation.md` §I. Roda UMA seed, fora da banda
confirmatória, com a passada 1 (vanilla + os três métodos de penalidade) e o
passo-sombra ligado.

**O que este script mede:** wiring (o `plasticity_ratio` aparece no artefato?),
o `E` de cada método, e o custo de parede. **Nada mais.**

**O que este script NÃO é:** um resultado. Seus endpoints são `n = 1` e não têm
nenhum valor estatístico. Citar sua acurácia como resultado é proibido pelo §I
do protocolo. O script nem imprime acurácia, de propósito — a separação é
imposta pelo código e não pela disciplina de quem lê.

Uso:

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python \\
        scripts/run_penalty_calibration.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.confirmatory_split_mnist import (  # noqa: E402
    PENALTY_REEVALUATION_CALIBRATION_SEED,
    PENALTY_REEVALUATION_SEEDS,
)
from experiments.split_mnist import (  # noqa: E402
    SplitMNISTConfig,
    load_split_mnist,
    run_split_mnist,
)

#: §E do protocolo: 1 medição a cada 50 passos.
SAMPLING_INTERVAL = 50

#: Passada 1 do §C.1. Os `lr_control` NÃO entram aqui: o escalar de cada um
#: depende do `E` que esta run mede.
ARMS = ("vanilla", "ewc", "si", "mas")

OUTPUT_DIR = Path("results/penalty_calibration")


def build_config() -> SplitMNISTConfig:
    return SplitMNISTConfig(
        seed=PENALTY_REEVALUATION_CALIBRATION_SEED,
        methods=ARMS,
        plasticity_sampling_interval=SAMPLING_INTERVAL,
        device="cpu",
    )


def main() -> int:
    seed = PENALTY_REEVALUATION_CALIBRATION_SEED
    if seed in PENALTY_REEVALUATION_SEEDS:
        raise SystemExit("seed de calibração está dentro da banda confirmatória")

    config = build_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"seed de calibração: {seed} (fora da banda confirmatória)")
    print(f"arms: {', '.join(ARMS)}")
    print(f"tarefas: {config.task_count}, amostragem: 1/{SAMPLING_INTERVAL} passos")
    print("carregando Split-MNIST...", flush=True)

    tasks = load_split_mnist(config, data_dir="data", download=False)

    started = time.perf_counter()
    results = run_split_mnist(config, tasks)
    elapsed = time.perf_counter() - started

    print(f"\n=== custo medido: {elapsed / 60:.1f} min para 1 seed, {len(ARMS)} arms")

    summary: dict[str, object] = {
        "seed": seed,
        "sampling_interval": SAMPLING_INTERVAL,
        "arms": list(ARMS),
        "elapsed_seconds": elapsed,
        "per_arm": {},
    }

    print("\n=== plasticidade medida (passada 1)")
    header = f"{'arm':<10}{'E':>10}{'norm':>10}{'cos':>10}{'amostras':>10}"
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        samples = results[arm]["plasticity_samples"]
        ratios = [s["plasticity_ratio"] for s in samples if s["plasticity_ratio"] is not None]
        norms = [s["norm_ratio"] for s in samples if s["norm_ratio"] is not None]
        cosines = [s["direction_cosine"] for s in samples if s["direction_cosine"] is not None]

        record: dict[str, object] = {
            "samples": len(samples),
            "elapsed_seconds": results[arm].get("elapsed_seconds"),
        }
        if ratios:
            record["plasticity_ratio_mean"] = statistics.fmean(ratios)
            record["plasticity_ratio_min"] = min(ratios)
            record["plasticity_ratio_max"] = max(ratios)
            record["norm_ratio_mean"] = statistics.fmean(norms)
            record["direction_cosine_mean"] = statistics.fmean(cosines)
            print(
                f"{arm:<10}{record['plasticity_ratio_mean']:>10.4f}"
                f"{record['norm_ratio_mean']:>10.4f}"
                f"{record['direction_cosine_mean']:>10.4f}{len(samples):>10}"
            )
        else:
            print(f"{arm:<10}{'—':>10}{'—':>10}{'—':>10}{len(samples):>10}")
        summary["per_arm"][arm] = record  # type: ignore[index]

    destination = OUTPUT_DIR / f"calibration_seed_{seed}.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nresumo: {destination}")

    print(
        "\nAVISO: n=1. Estes números medem WIRING e CUSTO. Os endpoints de "
        "acurácia desta run não têm valor estatístico e não são reportados."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
