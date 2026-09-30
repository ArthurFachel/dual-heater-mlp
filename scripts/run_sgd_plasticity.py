#!/usr/bin/env python3
"""Protocolo L2 — `E > 1` é aritmético ou é artefato de otimizador adaptativo?

Pré-registro: `goals/protocol_sgd_plasticity.md`, congelado em `f9f9bb0` e
emendado em `80bd322` (S4: learning rate vira EIXO declarado), ambos antes da
primeira seed da banda.

Roda os mesmos quatro arms das passadas 1, nas mesmas forças, com **SGD puro**
no lugar do AdamW, em **três learning rates**, e grava as métricas que testam
as predições C1–C5:

| campo | predição |
|---|---|
| `norm_ratio` | C1: `E > 1` persiste sob SGD? |
| `penalty_alignment` | C2: `E·cos = 1 + g·p/‖g‖²`? |
| `predicted_above_one` | C3: o sinal previsto casa com o medido? |
| `penalty_scale` | C4: `‖p‖/‖g‖ > 0,5` no `mas`? |
| (o eixo) | C5: `E` é monótono no learning rate? |

O eixo existe porque `lr = 1e-2` faz o `si` divergir numericamente (medido na
calibração: `norm_ratio` até 5,5e18, 26 de 128 amostras não-finitas) enquanto
`3e-3` e `1e-3` ficam sãos. A divergência é reportada como **dado**: amostras
não-finitas são contadas no manifest, nunca descartadas em silêncio.

**Nenhum endpoint de acurácia é lido** (S9), verificado por
`tests/test_run_sgd_plasticity.py::test_runner_reads_no_accuracy_field`.

Uso:
    # custo e wiring, seed FORA da banda
    PYTHONPATH=. .venv/bin/python scripts/run_sgd_plasticity.py --calibrate

    # a run declarada: 3 pontos do eixo x 12 seeds
    PYTHONPATH=. .venv/bin/python scripts/run_sgd_plasticity.py
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

from experiments.confirmatory_split_mnist import SGD_PLASTICITY_SEEDS
from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
from experiments.split_mnist import (
    SplitMNISTConfig,
    load_split_mnist,
    run_split_mnist,
)

#: Os mesmos quatro arms das passadas 1 (S2). `vanilla` é âncora descritiva;
#: só os três de penalidade produzem amostras do passo-sombra.
ARMS = ("vanilla", "ewc", "si", "mas")
PENALTY_ARMS = ("ewc", "si", "mas")

#: S4 emendado: o eixo declarado, nesta ordem. `1e-2` continua aqui apesar de
#: fazer o `si` divergir — a fronteira é o dado, e removê-la esconderia onde
#: ela está.
LEARNING_RATE_AXIS = (1e-2, 3e-3, 1e-3)

#: S7: amostragem exaustiva.
SAMPLING_INTERVAL = 1

OUTPUT_DIR = Path("results/sgd_plasticity")

#: FORA da banda declarada (S6), para medir custo e wiring sem gastar seed do
#: pré-registro. Verificado por teste.
CALIBRATION_SEED = 8_999_993

METRIC_KEYS = (
    "norm_ratio",
    "plasticity_ratio",
    "direction_cosine",
    "penalty_alignment",
    "penalty_scale",
)


def build_config(*, seed: int, learning_rate: float) -> SplitMNISTConfig:
    """Config de um ponto do eixo. O `learning_rate` TEM de chegar aqui.

    Se este argumento fosse ignorado, os três pontos rodariam no mesmo lr e o
    eixo mediria ruído entre seeds, com manifests de aparência normal.
    """

    return SplitMNISTConfig(
        seed=seed,
        methods=ARMS,
        optimizer="sgd",
        learning_rate=learning_rate,
        plasticity_sampling_interval=SAMPLING_INTERVAL,
        device="cpu",
        **PUBLISHED_PENALTY_STRENGTHS,
    )


def destination_for(*, root: Path, learning_rate: float, seed: int) -> Path:
    """Um diretório por ponto do eixo.

    Pontos diferentes no mesmo caminho fariam a run que termina por último
    sobrescrever a outra em silêncio, e o agregado misturaria learning rates.
    """

    return root / f"lr_{learning_rate:g}" / f"seed_{seed}.json"


def summarize_metric(values: Sequence[Any]) -> dict[str, Any]:
    """Resume ignorando não-finitos, mas CONTANDO quantos foram.

    Um `norm_ratio` de 5,5e18 não pode entrar numa média — ela deixaria de
    significar qualquer coisa. Mas omitir a contagem faria o relatório
    apresentar uma média sobre 102 de 128 passos como se fossem 128.
    """

    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    non_finite = sum(
        1
        for value in values
        if value is not None and not math.isfinite(float(value))
    )
    if not finite:
        return {"n": 0, "n_non_finite": non_finite, "mean": None}
    return {
        "n": len(finite),
        "n_non_finite": non_finite,
        "mean": statistics.fmean(finite),
        "stdev": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def prediction_agreement(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """C3: fração de passos em que a predição bateu com o `E` medido.

    Amostras não-finitas não contam como acerto nem como erro: são ausência de
    medida, e classificá-las de qualquer um dos dois lados inventaria dado.
    """

    agreements = [
        (sample["norm_ratio"] > 1.0) is sample["predicted_above_one"]
        for sample in samples
        if sample.get("norm_ratio") is not None
        and math.isfinite(float(sample["norm_ratio"]))
        and sample.get("predicted_above_one") is not None
    ]
    return {
        "n": len(agreements),
        "agreed": sum(agreements),
        "rate": (sum(agreements) / len(agreements)) if agreements else None,
    }


def identity_residual(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """C2: `|E·cos − (1 + alignment)|`, que sob SGD tem de ser ~0."""

    residuals = [
        abs(
            sample["norm_ratio"] * sample["direction_cosine"]
            - (1.0 + sample["penalty_alignment"])
        )
        for sample in samples
        if all(
            sample.get(key) is not None and math.isfinite(float(sample[key]))
            for key in ("norm_ratio", "direction_cosine", "penalty_alignment")
        )
    ]
    return summarize_metric(residuals)


def build_record_header(
    *, seed: int, learning_rate: float, elapsed: float
) -> dict[str, Any]:
    """Tudo que é preciso para auditar o manifest offline."""

    return {
        "seed": seed,
        "host": "split_mnist",
        "optimizer": "sgd",
        "learning_rate": learning_rate,
        "declared_interval": SAMPLING_INTERVAL,
        "penalty_strengths": dict(PUBLISHED_PENALTY_STRENGTHS),
        "elapsed_seconds": elapsed,
        "arms": {},
    }


def should_reuse(record: object, *, learning_rate: float) -> bool:
    """Só reaproveita artefato gravado sob ESTA configuração.

    Chavear o resume em "o arquivo existe" preservaria artefato de outro ponto
    do eixo ou de uma run AdamW, e misturaria proveniências no agregado sem
    nada falhar.
    """

    if not isinstance(record, dict):
        return False
    return (
        record.get("optimizer") == "sgd"
        and record.get("learning_rate") == learning_rate
        and record.get("declared_interval") == SAMPLING_INTERVAL
        and record.get("penalty_strengths") == dict(PUBLISHED_PENALTY_STRENGTHS)
    )


def run_one_seed(*, seed: int, learning_rate: float) -> dict[str, Any]:
    config = build_config(seed=seed, learning_rate=learning_rate)
    tasks = load_split_mnist(config, data_dir="data", download=False)
    started = time.perf_counter()
    results = run_split_mnist(config, tasks)
    elapsed = time.perf_counter() - started

    record = build_record_header(
        seed=seed, learning_rate=learning_rate, elapsed=elapsed
    )
    for arm in PENALTY_ARMS:
        samples = results[arm]["plasticity_samples"]
        if not samples:
            record["arms"][arm] = {}
            continue
        summary: dict[str, Any] = {
            key: summarize_metric([s.get(key) for s in samples])
            for key in METRIC_KEYS
        }
        summary["prediction_agreement"] = prediction_agreement(samples)
        summary["identity_residual"] = identity_residual(samples)
        summary["n_samples"] = len(samples)
        record["arms"][arm] = summary
    return record


def _fmt(value: float | None, digits: int = 4) -> str:
    """Formata um float que pode ser `None` ou ter ordem de grandeza absurda.

    A calibração produziu `cos = 0.6598868169855` e `|p|/|g| = 0.5905` na mesma
    linha de um `E` divergente, e o `{:9.4f}` cru concatenou os dois campos
    ilegivelmente. Colunas de diagnóstico precisam sobreviver a um arm que
    explodiu.
    """

    if value is None:
        return "—"
    if abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def render(records: Sequence[dict[str, Any]], learning_rate: float) -> str:
    lines: list[str] = []
    lines.append(f"\n=== lr = {learning_rate:g}")
    header = (
        f"{'arm':<6}{'E':>9}{'dp':>9}{'nao-fin':>9}{'cos':>9}"
        f"{'|p|/|g|':>10}{'g.p/|g|2':>11}{'C3':>7}{'C2 resid':>11}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for arm in PENALTY_ARMS:
        arms = [r["arms"][arm] for r in records if r["arms"].get(arm)]
        if not arms:
            lines.append(f"{arm:<6}{'—':>9}")
            continue

        def mean_of(key: str, field: str = "mean") -> float | None:
            values = [
                entry[key][field]
                for entry in arms
                if entry.get(key) and entry[key].get(field) is not None
            ]
            return statistics.fmean(values) if values else None

        per_seed = [
            entry["norm_ratio"]["mean"]
            for entry in arms
            if entry["norm_ratio"]["mean"] is not None
        ]
        non_finite = sum(entry["norm_ratio"]["n_non_finite"] for entry in arms)
        rates = [
            entry["prediction_agreement"]["rate"]
            for entry in arms
            if entry["prediction_agreement"]["rate"] is not None
        ]
        if not per_seed:
            lines.append(
                f"{arm:<6}{'TUDO DIVERGIU':>9}  ({non_finite} amostras nao-finitas)"
            )
            continue
        spread = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        residual = mean_of("identity_residual", "max")
        mean_e = statistics.fmean(per_seed)
        # Um `E` divergente não cabe em `{:9.4f}` — ele estoura a coluna e
        # desalinha a tabela inteira, que foi como a divergência do `si`
        # apareceu na calibração. Notação científica preserva a legibilidade e
        # deixa a ordem de grandeza visível, que é o que importa aqui.
        wide = abs(mean_e) >= 1e4
        lines.append(
            f"{arm:<6}{(f'{mean_e:.2e}' if wide else f'{mean_e:.4f}'):>9}"
            f"{(f'{spread:.1e}' if wide else f'{spread:.4f}'):>9}"
            f"{non_finite:>9}{_fmt(mean_of('direction_cosine')):>9}"
            f"{_fmt(mean_of('penalty_scale')):>10}"
            f"{_fmt(mean_of('penalty_alignment')):>11}"
            f"{(statistics.fmean(rates) if rates else 0.0):>6.0%}"
            f"{(residual if residual is not None else float('nan')):>11.2e}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="uma seed FORA da banda, para medir custo e wiring (§F)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="roda só um ponto do eixo (default: os três declarados)",
    )
    args = parser.parse_args(argv)

    axis = LEARNING_RATE_AXIS if args.learning_rate is None else (args.learning_rate,)
    if args.calibrate:
        seeds = [CALIBRATION_SEED]
        root = OUTPUT_DIR / "calibration"
    else:
        seeds = list(SGD_PLASTICITY_SEEDS)
        if args.limit is not None:
            seeds = seeds[: args.limit]
        root = OUTPUT_DIR

    print(f"L2 — SGD puro, eixo de lr: {', '.join(f'{lr:g}' for lr in axis)}")
    print(f"{len(seeds)} seed(s), arms: {', '.join(ARMS)}")
    print(f"forças: {PUBLISHED_PENALTY_STRENGTHS}")
    print("NENHUMA acurácia é lida nesta passada.\n")

    started = time.perf_counter()
    by_axis: dict[float, list[dict[str, Any]]] = {}

    for learning_rate in axis:
        records: list[dict[str, Any]] = []
        for index, seed in enumerate(seeds, start=1):
            destination = destination_for(
                root=root, learning_rate=learning_rate, seed=seed
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                existing = json.loads(destination.read_text(encoding="utf-8"))
                if should_reuse(existing, learning_rate=learning_rate):
                    records.append(existing)
                    print(
                        f"[lr={learning_rate:g} {index}/{len(seeds)}] "
                        f"seed {seed}: já existe, pulando"
                    )
                    continue
                print(
                    f"[lr={learning_rate:g} {index}/{len(seeds)}] "
                    f"seed {seed}: outra configuração, regravando"
                )

            record = run_one_seed(seed=seed, learning_rate=learning_rate)
            destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
            records.append(record)
            summary = ", ".join(
                f"{arm} E="
                + (
                    f"{record['arms'][arm]['norm_ratio']['mean']:.4f}"
                    if record["arms"][arm].get("norm_ratio", {}).get("mean")
                    is not None
                    else "divergiu"
                )
                for arm in PENALTY_ARMS
                if record["arms"].get(arm)
            )
            print(
                f"[lr={learning_rate:g} {index}/{len(seeds)}] seed {seed}: "
                f"{record['elapsed_seconds']:.0f}s — {summary}"
            )
        by_axis[learning_rate] = records

    elapsed = time.perf_counter() - started
    print(f"\n=== custo total: {elapsed / 60:.1f} min")

    for learning_rate, records in by_axis.items():
        if records:
            print(render(records, learning_rate))

    print(
        "\nC1: E > 1 persiste?  C2: resíduo ~0?  C3: 100%?  "
        "C4: |p|/|g| > 0,5 no mas?  C5: E monótono no lr?"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
