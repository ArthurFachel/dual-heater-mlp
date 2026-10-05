#!/usr/bin/env python3
"""Calibração exploratória do L3 — onde fica a grade de lambda?

**Isto NÃO é a run do L3.** É a medição que informa a grade e o custo antes de
congelar `goals/protocol_lambda_sweep.md`, no mesmo papel que a seed 8.999.993
teve para o L2 (`protocol_sgd_plasticity.md` §I).

Usa uma seed FORA de qualquer banda confirmatória, roda poucas tarefas, e não
lê nenhum endpoint de acurácia. O resultado serve para escolher onde colocar os
pontos do eixo — escolher a grade depois de ver `E` é legítimo exatamente
porque nenhuma acurácia foi lida.

Uso:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
    PYTHONPATH=src:. python scripts/calibrate_lambda_sweep.py
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections.abc import Sequence
from typing import Any

from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
from experiments.split_mnist import (
    SplitMNISTConfig,
    load_split_mnist,
    run_split_mnist,
)

ARMS = ("vanilla", "ewc", "si", "mas")
PENALTY_ARMS = ("ewc", "si", "mas")

#: Fora de toda banda confirmatória. Esta seed não pode ser reusada na run.
CALIBRATION_SEED = 8_999_977

#: O ponto são do eixo de lr do L2: `1e-2` faz o `si` divergir, `1e-3` deixa
#: todos os `E` colados em 1. `3e-3` é onde os três arms são mensuráveis.
LEARNING_RATE = 3e-3

#: Multiplicadores sobre as forças de Hsu et al. Grade larga de propósito: a
#: calibração existe para descobrir onde o efeito vive, não para confirmá-lo.
MULTIPLIERS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def strengths_for(multiplier: float) -> dict[str, Any]:
    """Escala só os lambdas. Os `decay` são estrutura, não força."""

    scaled = dict(PUBLISHED_PENALTY_STRENGTHS)
    for key in ("ewc_lambda", "si_lambda", "mas_lambda"):
        scaled[key] = PUBLISHED_PENALTY_STRENGTHS[key] * multiplier
    return scaled


def build_config(*, seed: int, multiplier: float) -> SplitMNISTConfig:
    return SplitMNISTConfig(
        seed=seed,
        methods=ARMS,
        optimizer="sgd",
        learning_rate=LEARNING_RATE,
        plasticity_sampling_interval=1,
        device="cpu",
        **strengths_for(multiplier),
    )


def summarize(values: Sequence[Any]) -> tuple[float | None, int]:
    """Média dos finitos, e a contagem dos que não foram."""

    finite = [
        float(v) for v in values if v is not None and math.isfinite(float(v))
    ]
    non_finite = sum(
        1 for v in values if v is not None and not math.isfinite(float(v))
    )
    return (statistics.fmean(finite) if finite else None), non_finite


def fmt(value: float | None, digits: int = 4) -> str:
    """Formata um float que pode ser `None` ou ter ordem de grandeza absurda.

    Mesmo motivo de `run_sgd_plasticity._fmt`: um arm que diverge produz
    `|p|/|g|` na casa de 1e13 e o `{:.4f}` cru cola a coluna na vizinha,
    tornando a tabela inteira ilegível justamente na linha que mais importa.
    """

    if value is None:
        return "—"
    if abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEED)
    args = parser.parse_args(argv)

    print("L3 — CALIBRAÇÃO exploratória, nao e a run")
    print(f"seed {args.seed} (fora de toda banda), lr = {LEARNING_RATE:g}, SGD puro")
    print(f"forcas base: {PUBLISHED_PENALTY_STRENGTHS}")
    print("NENHUMA acuracia e lida.\n")

    reference = build_config(seed=args.seed, multiplier=1.0)
    tasks = load_split_mnist(reference, data_dir="data", download=False)

    header = f"{'mult':>7}{'arm':>6}{'E':>12}{'cos':>9}{'|p|/|g|':>11}{'nao-fin':>9}"
    print(header)
    print("-" * len(header))

    started = time.perf_counter()
    for multiplier in MULTIPLIERS:
        config = build_config(seed=args.seed, multiplier=multiplier)
        results = run_split_mnist(config, tasks)
        for arm in PENALTY_ARMS:
            samples = results[arm]["plasticity_samples"]
            if not samples:
                print(f"{multiplier:>7g}{arm:>6}{'sem amostra':>12}")
                continue
            mean_e, non_finite = summarize([s.get("norm_ratio") for s in samples])
            mean_cos, _ = summarize([s.get("direction_cosine") for s in samples])
            mean_scale, _ = summarize([s.get("penalty_scale") for s in samples])
            shown_e = "divergiu" if mean_e is None else fmt(mean_e)
            print(
                f"{multiplier:>7g}{arm:>6}{shown_e:>12}"
                f"{fmt(mean_cos):>9}"
                f"{fmt(mean_scale):>11}"
                f"{non_finite:>9}"
            )
        print()

    print(f"custo: {(time.perf_counter() - started) / 60:.1f} min "
          f"para {len(MULTIPLIERS)} pontos, 1 seed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
