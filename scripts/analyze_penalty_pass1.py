#!/usr/bin/env python3
"""Agrega a passada 1 de plasticidade e aplica a regra G8.

Pré-registro: `goals/protocol_penalty_reevaluation.md` (§C, §E, §F, G8) e
`goals/protocol_plasticity_generalization.md` (primária G1').

Lê os manifests do passo-sombra, agrega por arm sobre as seeds, e classifica
cada método como pareável ou não-pareável segundo G8: `E_M > 1` significa que
`lr × E` seria um AUMENTO de learning rate, logo o `lr_control` não existe.

**Este script não lê nenhum endpoint de acurácia, e não deve passar a ler.**
Os manifests da passada 1 não contêm acurácia por construção; se um dia
contiverem, o `lr_scale` da passada 2 continuaria tendo de sair daqui sem
consultá-la.

Uso:
    PYTHONPATH=. .venv/bin/python scripts/analyze_penalty_pass1.py \
        --root results/penalty_pass1_cifar
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.confirmatory_statistics import exact_two_sided_sign_test

#: Métrica primária G1' (razão de normas). As outras duas são diagnósticas e
#: nunca decidem G8.
PRIMARY = "norm_ratio"
DIAGNOSTICS = ("plasticity_ratio", "direction_cosine")
PENALTY_ARMS = ("ewc", "si", "mas")
EXPECTED_SEEDS = 12


def arm_metrics(run: dict[str, Any], arm: str) -> dict[str, Any]:
    """Extrai as métricas de um arm, tolerando os dois schemas da passada 1.

    O runner do MNIST (`scripts/run_penalty_pass1.py`) aninha as métricas em
    ``declared``/``dense`` — duas séries que são idênticas quando os dois
    intervalos são 1, que é o caso após a emenda de amostragem exaustiva. O
    runner do CIFAR (`scripts/run_penalty_pass1_cifar.py`) grava as métricas
    diretamente. Ler a série ``declared`` é o correto nos dois casos: é a que
    o protocolo declara como fonte de `E_M`.
    """

    payload = run["arms"][arm]
    if "declared" in payload:
        return payload["declared"]
    return payload


def load_runs(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("seed_*.json"))
    if not paths:
        raise SystemExit(f"nenhum manifest em {root}")
    return [json.loads(path.read_text()) for path in paths]


def check_integrity(runs: list[dict[str, Any]]) -> list[str]:
    """Verificações que precedem qualquer leitura de E."""

    failures: list[str] = []
    if len(runs) != EXPECTED_SEEDS:
        failures.append(f"esperava {EXPECTED_SEEDS} seeds, achei {len(runs)}")

    seeds = [run["seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        failures.append(f"seeds duplicadas: {seeds}")

    hosts = {run.get("host", "split_mnist") for run in runs}
    if len(hosts) != 1:
        failures.append(f"manifests de hosts diferentes: {sorted(hosts)}")

    # A amostragem tem de ser exaustiva (§E emendado). Uma série amostrada
    # reintroduz o ruído que já inverteu uma classificação G8 uma vez.
    intervals = {run.get("declared_interval") for run in runs}
    if intervals != {1}:
        failures.append(f"amostragem não exaustiva: declared_interval={intervals}")

    # Forças pareadas entre seeds: um arm rodado a força diferente não agrega.
    strengths = {json.dumps(run["penalty_strengths"], sort_keys=True) for run in runs}
    if len(strengths) != 1:
        failures.append(f"forças de penalidade divergem entre seeds: {strengths}")

    for run in runs:
        missing = set(PENALTY_ARMS) - set(run["arms"])
        if missing:
            failures.append(f"seed {run['seed']} sem arms {sorted(missing)}")
    return failures


def summarize_arm(runs: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Agrega um arm sobre as seeds e aplica G8."""

    per_seed = [arm_metrics(run, arm)[PRIMARY]["mean"] for run in runs]
    deviations = [value - 1.0 for value in per_seed]
    mean = st.mean(per_seed)

    summary: dict[str, Any] = {
        "arm": arm,
        "n_seeds": len(per_seed),
        "E_mean": mean,
        "E_stdev": st.stdev(per_seed) if len(per_seed) > 1 else 0.0,
        "E_min": min(per_seed),
        "E_max": max(per_seed),
        "above_one": sum(1 for value in deviations if value > 0.0),
        "below_one": sum(1 for value in deviations if value < 0.0),
        "p_exact": exact_two_sided_sign_test(deviations),
        # G8: E > 1 torna `lr * E` um aumento de learning rate.
        "pairable": mean <= 1.0,
        "lr_scale": mean if mean <= 1.0 else None,
    }
    for name in DIAGNOSTICS:
        summary[f"{name}_mean"] = st.mean(
            arm_metrics(run, arm)[name]["mean"] for run in runs
        )
    # O mínimo por passo do cosseno é o que revela rotação; a média esconde.
    summary["direction_cosine_min"] = min(
        arm_metrics(run, arm)["direction_cosine"]["min"] for run in runs
    )
    return summary


def analyze(root: Path) -> dict[str, Any]:
    runs = load_runs(root)
    failures = check_integrity(runs)
    summaries = [summarize_arm(runs, arm) for arm in PENALTY_ARMS]
    return {
        "root": str(root),
        "host": runs[0].get("host", "split_mnist"),
        "seeds": sorted(run["seed"] for run in runs),
        "penalty_strengths": runs[0]["penalty_strengths"],
        "integrity_failures": failures,
        "arms": summaries,
        "pairable_arms": [s["arm"] for s in summaries if s["pairable"]],
    }


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"host: {report['host']}   seeds: {len(report['seeds'])}")
    lines.append(f"forças: {report['penalty_strengths']}")
    if report["integrity_failures"]:
        lines.append("")
        lines.append("INTEGRIDADE FALHOU:")
        for item in report["integrity_failures"]:
            lines.append(f"  - {item}")
    lines.append("")
    lines.append(
        f"{'arm':<6}{'E(norma)':>10}{'dp':>9}{'min':>9}{'max':>9}"
        f"{'cos':>9}{'cos_min':>9}{'sinais':>10}{'p':>10}  G8"
    )
    for row in report["arms"]:
        p_value = "n/d" if row["p_exact"] is None else f"{row['p_exact']:.5f}"
        verdict = "pareável" if row["pairable"] else "NÃO-PAREÁVEL"
        lines.append(
            f"{row['arm']:<6}{row['E_mean']:>10.4f}{row['E_stdev']:>9.4f}"
            f"{row['E_min']:>9.4f}{row['E_max']:>9.4f}"
            f"{row['direction_cosine_mean']:>9.4f}{row['direction_cosine_min']:>9.4f}"
            f"{f'{row['above_one']}+/{row['below_one']}-':>10}{p_value:>10}  {verdict}"
        )
    lines.append("")
    pairable = report["pairable_arms"]
    if pairable:
        lines.append("lr_scale congelável para a passada 2:")
        for row in report["arms"]:
            if row["pairable"]:
                lines.append(f"  {row['arm']}: {row['lr_scale']:.10f}")
    else:
        lines.append(
            "NENHUM arm é pareável (G8): a família confirmatória deste host "
            "fica vazia e a passada 2 não tem controle a construir."
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/penalty_pass1"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = analyze(args.root)
    print(render(report))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2))
    return 1 if report["integrity_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
