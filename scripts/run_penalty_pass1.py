#!/usr/bin/env python
"""Passada 1 do piloto de re-avaliação de métodos de penalidade.

`goals/protocol_penalty_reevaluation.md` §C.1. Roda as 12 seeds
pré-registradas com `vanilla`, `ewc`, `si` e `mas`, mede a plasticidade de cada
método pelo passo-sombra, e classifica cada um como pareável ou não (G8).

**Nenhum endpoint de acurácia é lido aqui.** É o que mantém válido o
pré-registro: o `lr_scale` dos controles da passada 2 sai destes números, e
tê-los escolhido depois de ver acurácia seria seleção pós-hoc.

**Amostragem exaustiva (1/1).** O §E foi emendado em 29/09: mede-se TODO passo
com âncora, e `E_M` é a média exata em vez de uma estimativa. A 1/50 o ruído
amostral era ~14× o efeito e chegava a inverter o veredito G8. A medição não
perturba o treino (`test_measuring_does_not_change_the_trained_model`) e as
amostras carregam o índice do passo
(`test_samples_carry_the_step_index_so_runs_can_be_aligned`).

Uso:

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python \\
        scripts/run_penalty_pass1.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.confirmatory_split_mnist import (  # noqa: E402
    PENALTY_REEVALUATION_SEEDS,
)
from experiments.penalty_reevaluation import (  # noqa: E402
    PUBLISHED_PENALTY_STRENGTHS,
)
from experiments.split_mnist import (  # noqa: E402
    SplitMNISTConfig,
    load_split_mnist,
    run_split_mnist,
)

#: Taxa declarada no §E. **Emendada em 2026-09-29 de 50 para 1.**
#:
#: A 1/50 o `E` de uma seed vinha de 3 amostras. A medição exaustiva de uma
#: seed (128 passos) mostrou dp entre passos de 0,0772 contra um efeito
#: `|1 − E|` de 0,012: o IC95% da média com n=3 tem largura 0,171, ou seja o
#: ruído da estimativa é ~14× o efeito, e a série declarada chegou a inverter
#: o sinal do veredito G8 em relação à medição densa.
#:
#: O §E justificava 1/50 pelo custo 3× por passo medido. Medido neste host, o
#: custo total de amostrar TODO passo é ~30 s/seed/arm — irrelevante. A emenda
#: troca uma estimativa ruidosa por um valor exato.
DECLARED_INTERVAL = 1

#: Passada 1 do §C.1: sem os `lr_control`, cujo escalar depende destes números.
ARMS = ("vanilla", "ewc", "si", "mas")

PENALTY_ARMS = ("ewc", "si", "mas")

OUTPUT_DIR = Path("results/penalty_pass1")


def validate_intervals(*, dense_interval: int) -> None:
    """A grade densa tem de conter a declarada, ou o filtro perde amostras."""

    if dense_interval < 1:
        raise ValueError("dense_interval deve ser >= 1")
    if DECLARED_INTERVAL % dense_interval != 0:
        raise ValueError(
            f"dense_interval deve dividir {DECLARED_INTERVAL}, senão a série "
            "declarada não é um subconjunto da medida"
        )


def should_reuse(record: object, *, dense_interval: int) -> bool:
    """Um artefato só é reaproveitável se foi gerado sob ESTA configuração.

    Chavear o resume em "o arquivo existe" preserva artefatos de uma
    configuração anterior e faz o agregado misturar proveniências sem nada
    falhar — exatamente o que aconteceria com os 12 arquivos gravados a 1/5
    antes da emenda do §E. O guarda compara o que o arquivo declara ter usado.
    """

    if not isinstance(record, dict):
        return False
    for field in ("dense_interval", "penalty_strengths"):
        if field not in record:
            return False
    if record["dense_interval"] != dense_interval:
        return False
    if record.get("declared_interval", DECLARED_INTERVAL) != DECLARED_INTERVAL:
        return False
    if record["penalty_strengths"] != dict(PUBLISHED_PENALTY_STRENGTHS):
        return False
    return True


def build_config(seed: int, dense_interval: int) -> SplitMNISTConfig:
    return SplitMNISTConfig(
        seed=seed,
        methods=ARMS,
        plasticity_sampling_interval=dense_interval,
        device="cpu",
        **PUBLISHED_PENALTY_STRENGTHS,
    )


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
    parser.add_argument("--dense-interval", type=int, default=DECLARED_INTERVAL)
    parser.add_argument("--limit", type=int, default=None, help="só as N primeiras seeds")
    args = parser.parse_args()

    validate_intervals(dense_interval=args.dense_interval)

    seeds = list(PENALTY_REEVALUATION_SEEDS)
    if args.limit is not None:
        seeds = seeds[: args.limit]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"passada 1 — {len(seeds)} seeds, arms: {', '.join(ARMS)}")
    print(f"amostragem densa 1/{args.dense_interval}, série declarada 1/{DECLARED_INTERVAL}")
    print(f"forças: {PUBLISHED_PENALTY_STRENGTHS}\n")

    per_seed: dict[int, dict] = {}
    started = time.perf_counter()

    for index, seed in enumerate(seeds, start=1):
        destination = OUTPUT_DIR / f"seed_{seed}.json"
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if should_reuse(existing, dense_interval=args.dense_interval):
                per_seed[seed] = existing
                print(f"[{index}/{len(seeds)}] seed {seed}: já existe, pulando")
                continue
            print(
                f"[{index}/{len(seeds)}] seed {seed}: artefato de outra "
                "configuração, regravando"
            )

        config = build_config(seed, args.dense_interval)
        tasks = load_split_mnist(config, data_dir="data", download=False)
        seed_started = time.perf_counter()
        results = run_split_mnist(config, tasks)
        seed_elapsed = time.perf_counter() - seed_started

        record: dict = {
            "seed": seed,
            "dense_interval": args.dense_interval,
            "declared_interval": DECLARED_INTERVAL,
            "penalty_strengths": dict(PUBLISHED_PENALTY_STRENGTHS),
            "elapsed_seconds": seed_elapsed,
            "arms": {},
        }
        for arm in PENALTY_ARMS:
            samples = results[arm]["plasticity_samples"]
            declared = [s for s in samples if s["step"] % DECLARED_INTERVAL == 0]
            record["arms"][arm] = {
                "declared": {
                    key: summarize([s[key] for s in declared if s[key] is not None])
                    for key in ("norm_ratio", "plasticity_ratio", "direction_cosine")
                }
                if declared
                else {},
                "dense": {
                    key: summarize([s[key] for s in samples if s[key] is not None])
                    for key in ("norm_ratio", "plasticity_ratio", "direction_cosine")
                }
                if samples
                else {},
            }
        destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
        per_seed[seed] = record
        print(
            f"[{index}/{len(seeds)}] seed {seed}: {seed_elapsed:.0f}s — "
            + ", ".join(
                f"{arm} E={record['arms'][arm]['declared']['norm_ratio']['mean']:.4f}"
                for arm in PENALTY_ARMS
                if record["arms"][arm].get("declared")
            )
        )

    elapsed = time.perf_counter() - started
    print(f"\n=== custo total: {elapsed / 60:.1f} min para {len(seeds)} seeds")

    print("\n=== E por método (média exata sobre TODO passo com âncora)")
    header = (
        f"{'arm':<8}{'E':>10}{'dp entre seeds':>16}{'min':>9}{'max':>9}"
        f"{'cos':>9}{'pareável (G8)':>15}"
    )
    print(header)
    print("-" * len(header))

    aggregate: dict[str, dict] = {}
    for arm in PENALTY_ARMS:
        per_seed_means = [
            rec["arms"][arm]["declared"]["norm_ratio"]["mean"]
            for rec in per_seed.values()
            if rec["arms"][arm].get("declared")
        ]
        cos_means = [
            rec["arms"][arm]["declared"]["direction_cosine"]["mean"]
            for rec in per_seed.values()
            if rec["arms"][arm].get("declared")
        ]
        if not per_seed_means:
            print(f"{arm:<8}{'—':>10}")
            continue
        mean = statistics.fmean(per_seed_means)
        spread = statistics.stdev(per_seed_means) if len(per_seed_means) > 1 else 0.0
        pairable = mean <= 1.0
        aggregate[arm] = {
            "E_mean": mean,
            "E_stdev_between_seeds": spread,
            "E_min": min(per_seed_means),
            "E_max": max(per_seed_means),
            "direction_cosine_mean": statistics.fmean(cos_means),
            "seeds": len(per_seed_means),
            "pairable": pairable,
            "lr_scale": mean if pairable else None,
        }
        print(
            f"{arm:<8}{mean:>10.4f}{spread:>16.4f}{min(per_seed_means):>9.4f}"
            f"{max(per_seed_means):>9.4f}{statistics.fmean(cos_means):>9.4f}"
            f"{('sim' if pairable else 'NÃO'):>15}"
        )

    unpairable = [arm for arm, rec in aggregate.items() if not rec["pairable"]]
    if unpairable:
        print(
            f"\nG8: {', '.join(unpairable)} com E > 1 — o lr_control desses "
            "métodos NÃO é construível. O protocolo manda declará-los "
            "não-pareáveis e tirar seus contrastes da família confirmatória."
        )
        print(
            f"Família confirmatória cai de 6 para "
            f"{2 * (len(PENALTY_ARMS) - len(unpairable))} comparações."
        )

    summary_path = OUTPUT_DIR / "pass1_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "dense_interval": args.dense_interval,
                "declared_interval": DECLARED_INTERVAL,
                "penalty_strengths": dict(PUBLISHED_PENALTY_STRENGTHS),
                "penalty_strengths_source": (
                    "Hsu et al. 2018 (arXiv:1810.12488), "
                    "split_MNIST_incremental_class.sh"
                ),
                "elapsed_seconds": elapsed,
                "per_arm": aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nresumo: {summary_path}")
    print(
        "\nNENHUMA acurácia foi lida nesta passada. Os lr_scale acima são o "
        "insumo da passada 2 e devem ser congelados num commit antes dela."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
