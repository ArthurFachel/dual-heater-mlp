#!/usr/bin/env python3
"""Passada 2 — o MAS sobrevive a um controle pareado em plasticidade?

Pré-registro: `goals/protocol_penalty_pass2.md`, congelado antes da primeira
seed da banda 9.500.011+. Item 2.13 de `goals/roadmap_icml_ijcnn.md`.

**ESTA PASSADA LÊ ACURÁCIA.** É a primeira da Fase 2 que o faz. Todas as
anteriores (passadas 1, L2, L3) foram mecanismo-only, e era isso que permitia
redesenhar sem queimar pré-registro. A partir daqui o resultado é o resultado.

O desenho encolheu de 6 comparações para 1 ao longo da Fase 2:

| etapa | sobrou |
|---|---|
| desenho original | 6 (3 métodos x 2 hosts) |
| passada 1 (AdamW, G8) | 1, com lr_scale 0,9958 — vazia |
| L2 (SGD) | 3 pareáveis, mas fracas |
| L3 (sweep de lambda) | **1 com poder: mas a 30x** |

Contraste primário: `mas − lr_control` em esquecimento médio. O `lr_control`
remove exatamente a mesma plasticidade que o MAS remove (E = 0,854), mas sem
nenhum mecanismo de consolidação.

Uso:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
    PYTHONPATH=src:. python -u scripts/run_penalty_pass2.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.confirmatory_split_mnist import PENALTY_PASS2_SEEDS
from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
from experiments.split_mnist import (
    SplitMNISTConfig,
    load_split_mnist,
    run_split_mnist,
)

#: P5: três arms, não sete. Os outros métodos não têm controle construível —
#: o L3 mostrou que para `ewc` e `si` não existe lambda onde o controle seja
#: simultaneamente útil e numericamente estável.
ARMS = ("vanilla", "mas", "lr_control")

#: §C.1 — CONGELADO. Média das 10 seeds do `mas` a 30x em
#: `results/lambda_sweep/mult_30/`, lida do agregado do L3 ANTES de qualquer
#: acurácia desta passada. Recalculá-lo depois de ver um endpoint invalidaria a
#: passada inteira. Pinado em `tests/test_penalty_pass2.py` com 16 dígitos.
FROZEN_LR_SCALE = 0.8541056558955461

#: P3: 30x o publicado por Hsu et al. O `E` congelado acima foi medido NESSA
#: força; rodar em outra descreveria um experimento diferente.
MAS_MULTIPLIER = 30.0

#: P2: o mesmo lr do L3. Mudá-lo moveria o `E` e invalidaria o escalar.
LEARNING_RATE = 3e-3

#: P7/P8/P9: declarados aqui para que não sejam escolhidos depois.
PRIMARY_ENDPOINT = "average_forgetting"
PRIMARY_CONTRAST = ("mas", "lr_control")
CONFIRMATORY_FAMILY = (PRIMARY_CONTRAST,)

#: §D.2: a acurácia final é obrigatória na tabela. Um método pode reduzir
#: esquecimento colapsando a aquisição, e sem este número ao lado um `mas` que
#: simplesmente aprende menos pareceria estar preservando conhecimento.
REPORTED_METRICS = ("average_forgetting", "final_average_accuracy")

OUTPUT_DIR = Path("results/penalty_pass2")


def strengths_for_pass_two() -> dict[str, Any]:
    """As forças de Hsu, com o `mas_lambda` inflado pelo fator do L3."""

    scaled = dict(PUBLISHED_PENALTY_STRENGTHS)
    scaled["mas_lambda"] = PUBLISHED_PENALTY_STRENGTHS["mas_lambda"] * MAS_MULTIPLIER
    return scaled


def build_config(*, seed: int) -> SplitMNISTConfig:
    """A config declarada. Todo valor aqui está pinado por teste."""

    return SplitMNISTConfig(
        seed=seed,
        methods=ARMS,
        optimizer="sgd",
        learning_rate=LEARNING_RATE,
        learning_rate_scale=FROZEN_LR_SCALE,
        # §F: o `E` já foi medido no L3; remedi-lo custaria 3 passos por passo
        # e não entra em nenhum julgamento desta passada.
        plasticity_sampling_interval=0,
        device="cpu",
        **strengths_for_pass_two(),
    )


def build_record_header(*, seed: int, elapsed: float) -> dict[str, Any]:
    return {
        "seed": seed,
        "host": "split_mnist",
        "optimizer": "sgd",
        "learning_rate": LEARNING_RATE,
        "learning_rate_scale": FROZEN_LR_SCALE,
        "mas_multiplier": MAS_MULTIPLIER,
        "penalty_strengths": strengths_for_pass_two(),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_contrast": list(PRIMARY_CONTRAST),
        "elapsed_seconds": elapsed,
        "arms": {},
    }


def should_reuse(record: object) -> bool:
    """Só reaproveita artefato gravado sob ESTA configuração.

    O escalar é o que define a passada; um manifest com outro `lr_scale`
    descreve outro controle e não pode entrar no mesmo agregado.
    """

    if not isinstance(record, dict):
        return False
    return (
        record.get("optimizer") == "sgd"
        and record.get("learning_rate_scale") == FROZEN_LR_SCALE
        and record.get("mas_multiplier") == MAS_MULTIPLIER
        and record.get("learning_rate") == LEARNING_RATE
    )


def extract_metrics(arm_result: dict[str, Any]) -> dict[str, float]:
    """Lê os endpoints de `result["metrics"]`, não da raiz.

    `run_split_mnist` aninha as métricas sob `"metrics"` (ver `AGGREGATE_METRICS`
    em `experiments/split_mnist.py`). Ler da raiz levanta `KeyError`, que é o
    modo de falha barulhento e bom; o modo silencioso seria um `.get(metric, 0.0)`
    gravando zeros plausíveis em todos os manifests.
    """

    return {
        metric: float(arm_result["metrics"][metric]) for metric in REPORTED_METRICS
    }


def run_one_seed(*, seed: int, tasks: Any) -> dict[str, Any]:
    config = build_config(seed=seed)
    started = time.perf_counter()
    results = run_split_mnist(config, tasks)
    elapsed = time.perf_counter() - started

    record = build_record_header(seed=seed, elapsed=elapsed)
    for arm in ARMS:
        record["arms"][arm] = extract_metrics(results[arm])
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)

    seeds = [
        seed
        for index, seed in enumerate(PENALTY_PASS2_SEEDS)
        if index % args.shards == args.shard
    ]
    if args.limit is not None:
        seeds = seeds[: args.limit]

    print("Passada 2 — mas vs lr_control pareado, sob SGD")
    print(f"lr_scale CONGELADO: {FROZEN_LR_SCALE} (do L3, antes de qualquer acuracia)")
    print(f"mas_lambda = {strengths_for_pass_two()['mas_lambda']} ({MAS_MULTIPLIER:g}x Hsu)")
    print(f"{len(seeds)} seeds, arms: {', '.join(ARMS)}")
    print(f"primario: {PRIMARY_ENDPOINT}, contraste {PRIMARY_CONTRAST[0]} - {PRIMARY_CONTRAST[1]}")
    print("ESTA PASSADA LE ACURACIA.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = build_config(seed=seeds[0])
    tasks = load_split_mnist(reference, data_dir="data", download=False)

    started = time.perf_counter()
    for index, seed in enumerate(seeds, start=1):
        destination = OUTPUT_DIR / f"seed_{seed}.json"
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if should_reuse(existing):
                print(f"[{index}/{len(seeds)}] seed {seed}: ja existe")
                continue
            print(f"[{index}/{len(seeds)}] seed {seed}: outra config, regravando")

        record = run_one_seed(seed=seed, tasks=tasks)
        destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
        summary = ", ".join(
            f"{arm} F={record['arms'][arm]['average_forgetting']:.4f}"
            f"/A={record['arms'][arm]['final_average_accuracy']:.4f}"
            for arm in ARMS
        )
        print(
            f"[{index}/{len(seeds)}] seed {seed}: "
            f"{record['elapsed_seconds']:.0f}s — {summary}"
        )

    print(f"\n=== custo: {(time.perf_counter() - started) / 60:.1f} min")
    print("Agregue com scripts/analyze_penalty_pass2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
