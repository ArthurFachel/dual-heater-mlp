#!/usr/bin/env python3
"""Protocolo L3 — existe λ onde o controle pareado tem poder?

Pré-registro: `goals/protocol_lambda_sweep.md`, congelado antes da primeira
seed da banda 9.000.011+. Item 2.12 de `goals/roadmap_icml_ijcnn.md`.

O L2 mostrou que sob SGD puro os três métodos reduzem a norma do update, mas
só por 0,7% a 4,5% nas forças publicadas — um `lr_control` a `lr x 0,993` é
indistinguível do vanilla. O L3 varre a força procurando um ponto onde o
controle seja um controle de verdade (§D.2: `E <= 0,90`) **e** o otimizador
continue são.

| predição | o que mede |
|---|---|
| T-P1 | `E` monótono decrescente em λ, por arm |
| T-P2 | `ewc` e `si` divergem em λ >= 10x |
| T-P3 | `mas` atinge `E <= 0,90` em 30x ou 100x sem divergir |
| T-P4 | nenhuma célula de `ewc`/`si` tem poder sem divergir |

**Nenhum endpoint de acurácia é lido** (T9), verificado por
`tests/test_lambda_sweep.py::test_runner_reads_no_accuracy_field`.

Uso:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
    PYTHONPATH=src:. python -u scripts/run_lambda_sweep.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.confirmatory_split_mnist import LAMBDA_SWEEP_SEEDS
from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
from experiments.split_mnist import (
    SplitMNISTConfig,
    load_split_mnist,
    run_split_mnist,
)

#: T4: os mesmos quatro arms do L2 e das passadas 1.
ARMS = ("vanilla", "ewc", "si", "mas")
PENALTY_ARMS = ("ewc", "si", "mas")

#: T3: multiplicador sobre as forças de Hsu et al. (2018), não valor absoluto —
#: multiplicar preserva a razão entre métodos que eles publicaram. O `300x` da
#: calibração foi EXCLUÍDO: os três arms divergiram lá, e um eixo cujo último
#: ponto não produz medida nenhuma só gasta CPU.
MULTIPLIER_AXIS = (1.0, 3.0, 10.0, 30.0, 100.0)

#: T2: o ponto são do eixo de lr do L2. Em `1e-2` o `si` diverge já nas forças
#: publicadas; em `1e-3` os três `E` colam em 1 e não há o que varrer.
LEARNING_RATE = 3e-3

#: T6: amostragem exaustiva.
SAMPLING_INTERVAL = 1

#: §D.2, declarado ANTES das seeds. Uma célula tem poder se `E <= 0,90` e o arm
#: não divergiu ali. Mover este número depois de ver os dados seria escolher o
#: resultado; ele está pinado em `tests/test_lambda_sweep.py`.
POWER_THRESHOLD = 0.90

OUTPUT_DIR = Path("results/lambda_sweep")


def strengths_for(multiplier: float) -> dict[str, Any]:
    """Escala os três λ. Os `decay` ficam: são estrutura, não força.

    `ewc_decay` controla como o Fisher acumula entre tarefas — escalá-lo junto
    mudaria o método, não a intensidade dele, e confundiria os dois eixos.
    """

    scaled = dict(PUBLISHED_PENALTY_STRENGTHS)
    for key in ("ewc_lambda", "si_lambda", "mas_lambda"):
        scaled[key] = PUBLISHED_PENALTY_STRENGTHS[key] * multiplier
    return scaled


def build_config(*, seed: int, multiplier: float) -> SplitMNISTConfig:
    """Config de um ponto do eixo. O `multiplier` TEM de chegar aqui.

    Se fosse ignorado, os cinco pontos rodariam na mesma força e o sweep mediria
    ruído entre seeds — com manifests de aparência perfeita. Este projeto já
    teve esse bug duas vezes (`importance_criterion` e `learning_rate_scale`).
    """

    return SplitMNISTConfig(
        seed=seed,
        methods=ARMS,
        optimizer="sgd",
        learning_rate=LEARNING_RATE,
        plasticity_sampling_interval=SAMPLING_INTERVAL,
        device="cpu",
        **strengths_for(multiplier),
    )


def destination_for(*, root: Path, multiplier: float, seed: int) -> Path:
    """Um diretório por ponto do eixo, senão um sobrescreve o outro em silêncio."""

    return root / f"mult_{multiplier:g}" / f"seed_{seed}.json"


def summarize_metric(values: Sequence[Any]) -> dict[str, Any]:
    """Resume os finitos, conta os que não são, e marca a célula (T8).

    `divergent` é verdadeiro com UMA amostra não-finita que seja. A calibração
    mediu `E = 2,3e9` com 29 de 128 amostras infinitas: a média dos 99 finitos
    restantes é um número, mas não é plasticidade — é o resíduo de um
    otimizador explodindo, e tratá-la como medida contaminaria o veredito.

    `None` não é divergência: é ausência de medida (gradiente nulo, por
    exemplo), e contá-la como explosão inventaria dado.
    """

    finite: list[float] = []
    non_finite = 0
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            finite.append(number)
        else:
            non_finite += 1

    divergent = non_finite > 0
    if not finite:
        return {
            "n": 0,
            "n_non_finite": non_finite,
            "divergent": divergent,
            "mean": None,
        }
    return {
        "n": len(finite),
        "n_non_finite": non_finite,
        "divergent": divergent,
        "mean": statistics.fmean(finite),
        "stdev": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def has_power(*, mean_e: float | None, divergent: bool) -> bool:
    """§D.2: as DUAS condições, e a conjunção é o ponto.

    Um `E` baixo num arm divergente é exatamente o caso que o §D.2 existe para
    excluir — e é o caso mais tentador de aceitar, porque o número parece bom.
    """

    if divergent or mean_e is None:
        return False
    return mean_e <= POWER_THRESHOLD


def build_record_header(
    *, seed: int, multiplier: float, elapsed: float
) -> dict[str, Any]:
    """Tudo que permite auditar o manifest sem o runner em mãos."""

    return {
        "seed": seed,
        "host": "split_mnist",
        "optimizer": "sgd",
        "learning_rate": LEARNING_RATE,
        "multiplier": multiplier,
        "penalty_strengths": strengths_for(multiplier),
        "declared_interval": SAMPLING_INTERVAL,
        "power_threshold": POWER_THRESHOLD,
        "elapsed_seconds": elapsed,
        "arms": {},
    }


def should_reuse(record: object, *, multiplier: float) -> bool:
    """Só reaproveita artefato gravado sob ESTE ponto do eixo.

    Chavear o resume em "o arquivo existe" leria um manifest de outra força, ou
    de uma run AdamW, e misturaria proveniências sem nada falhar.
    """

    if not isinstance(record, dict):
        return False
    return (
        record.get("optimizer") == "sgd"
        and record.get("multiplier") == multiplier
        and record.get("learning_rate") == LEARNING_RATE
        and record.get("declared_interval") == SAMPLING_INTERVAL
    )


METRIC_KEYS = (
    "norm_ratio",
    "direction_cosine",
    "penalty_scale",
    "penalty_alignment",
)


def run_one_seed(*, seed: int, multiplier: float, tasks: Any) -> dict[str, Any]:
    config = build_config(seed=seed, multiplier=multiplier)
    started = time.perf_counter()
    results = run_split_mnist(config, tasks)
    elapsed = time.perf_counter() - started

    record = build_record_header(seed=seed, multiplier=multiplier, elapsed=elapsed)
    for arm in PENALTY_ARMS:
        samples = results[arm]["plasticity_samples"]
        if not samples:
            record["arms"][arm] = {}
            continue
        summary: dict[str, Any] = {
            key: summarize_metric([s.get(key) for s in samples])
            for key in METRIC_KEYS
        }
        summary["n_samples"] = len(samples)
        summary["divergent"] = summary["norm_ratio"]["divergent"]
        summary["has_power"] = has_power(
            mean_e=summary["norm_ratio"]["mean"],
            divergent=summary["divergent"],
        )
        record["arms"][arm] = summary
    return record


def fmt(value: float | None, digits: int = 4) -> str:
    """Um arm divergente produz `|p|/|g|` na casa de 1e13; `{:.4f}` cola colunas."""

    if value is None:
        return "—"
    if abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def render(records: Sequence[dict[str, Any]], multiplier: float) -> str:
    lines = [f"\n=== multiplicador = {multiplier:g}x"]
    header = (
        f"{'arm':<6}{'E':>12}{'dp':>10}{'nao-fin':>9}"
        f"{'cos':>9}{'|p|/|g|':>11}{'poder':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for arm in PENALTY_ARMS:
        entries = [r["arms"][arm] for r in records if r["arms"].get(arm)]
        if not entries:
            lines.append(f"{arm:<6}{'—':>12}")
            continue

        per_seed = [
            e["norm_ratio"]["mean"]
            for e in entries
            if e["norm_ratio"]["mean"] is not None
        ]
        non_finite = sum(e["norm_ratio"]["n_non_finite"] for e in entries)
        # Uma célula só tem poder se TODAS as seeds a classificaram assim.
        powered = sum(1 for e in entries if e.get("has_power"))

        if not per_seed:
            lines.append(
                f"{arm:<6}{'TUDO DIVERGIU':>12}  ({non_finite} nao-finitas)"
            )
            continue

        def mean_of(key: str, rows: Sequence[dict[str, Any]] = entries) -> float | None:
            values = [
                e[key]["mean"] for e in rows if e[key]["mean"] is not None
            ]
            return statistics.fmean(values) if values else None

        spread = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        lines.append(
            f"{arm:<6}{fmt(statistics.fmean(per_seed)):>12}{fmt(spread):>10}"
            f"{non_finite:>9}{fmt(mean_of('direction_cosine')):>9}"
            f"{fmt(mean_of('penalty_scale')):>11}"
            f"{f'{powered}/{len(entries)}':>8}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--multiplier",
        type=float,
        default=None,
        help="roda só um ponto do eixo (default: os cinco declarados)",
    )
    args = parser.parse_args(argv)

    axis = MULTIPLIER_AXIS if args.multiplier is None else (args.multiplier,)
    seeds = list(LAMBDA_SWEEP_SEEDS)
    if args.limit is not None:
        seeds = seeds[: args.limit]

    print("L3 — sweep de lambda sob SGD puro")
    print(f"eixo: {', '.join(f'{m:g}x' for m in axis)}")
    print(f"{len(seeds)} seeds, lr = {LEARNING_RATE:g}, limiar de poder = {POWER_THRESHOLD}")
    print("NENHUMA acuracia e lida nesta passada.\n")

    reference = build_config(seed=seeds[0], multiplier=1.0)
    tasks = load_split_mnist(reference, data_dir="data", download=False)

    started = time.perf_counter()
    by_axis: dict[float, list[dict[str, Any]]] = {}

    for multiplier in axis:
        records: list[dict[str, Any]] = []
        for index, seed in enumerate(seeds, start=1):
            destination = destination_for(
                root=OUTPUT_DIR, multiplier=multiplier, seed=seed
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                existing = json.loads(destination.read_text(encoding="utf-8"))
                if should_reuse(existing, multiplier=multiplier):
                    records.append(existing)
                    print(f"[{multiplier:g}x {index}/{len(seeds)}] seed {seed}: ja existe")
                    continue
                print(f"[{multiplier:g}x {index}/{len(seeds)}] seed {seed}: outra config, regravando")

            record = run_one_seed(seed=seed, multiplier=multiplier, tasks=tasks)
            destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
            records.append(record)
            summary = ", ".join(
                f"{arm}="
                + (
                    "divergiu"
                    if record["arms"][arm].get("divergent")
                    else fmt(record["arms"][arm]["norm_ratio"]["mean"])
                )
                for arm in PENALTY_ARMS
                if record["arms"].get(arm)
            )
            print(
                f"[{multiplier:g}x {index}/{len(seeds)}] seed {seed}: "
                f"{record['elapsed_seconds']:.0f}s — {summary}"
            )
        by_axis[multiplier] = records

    print(f"\n=== custo total: {(time.perf_counter() - started) / 60:.1f} min")
    for multiplier, records in by_axis.items():
        if records:
            print(render(records, multiplier))

    print(
        "\nT-P1: E monotono em lambda?  T-P2: ewc/si divergem em >=10x?  "
        f"T-P3: mas chega a E <= {POWER_THRESHOLD} sem divergir?  "
        "T-P4: nenhuma celula de ewc/si com poder?"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
