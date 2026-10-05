#!/usr/bin/env python3
"""Agregador do L3 — julga T-P1 a T-P4 a partir dos manifests.

Pré-registro: `goals/protocol_lambda_sweep.md` §E. Item 2.12 de
`goals/roadmap_icml_ijcnn.md`.

As três decisões que um sweep usa para mentir estão todas declaradas no
protocolo e todas testadas em `tests/test_analyze_lambda_sweep.py`:

- o que conta como **divergente** (T8: uma amostra não-finita basta);
- o que conta como **poder** (§D.2: `E <= 0,90` E não divergente);
- o que conta como **monótono** (T-P1: uma inversão falsifica).

**Nenhum endpoint de acurácia é lido.**

Uso:
    PYTHONPATH=src:. python scripts/analyze_lambda_sweep.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PENALTY_ARMS = ("ewc", "si", "mas")
MULTIPLIER_AXIS = (1.0, 3.0, 10.0, 30.0, 100.0)
POWER_THRESHOLD = 0.90
RESULTS_DIR = Path("results/lambda_sweep")


def judge_monotonicity(series: Sequence[float | None]) -> dict[str, Any]:
    """T-P1: `E` decresce em λ? Células divergentes são PULADAS, não ordenadas.

    Tratar `None` como "o menor valor" faria qualquer série que termina em
    divergência parecer decrescente — exatamente o caso do `ewc`, que diverge
    nos três λ mais altos. Isso transformaria a falta de medida em evidência a
    favor da predição.
    """

    measured = [value for value in series if value is not None]
    n_skipped = len(series) - len(measured)
    if len(measured) < 2:
        return {
            "monotonic": None,
            "inversions": 0,
            "n_compared": len(measured),
            "n_skipped": n_skipped,
        }

    inversions = sum(
        1
        for earlier, later in itertools.pairwise(measured)
        if later > earlier
    )
    return {
        "monotonic": inversions == 0,
        "inversions": inversions,
        "n_compared": len(measured),
        "n_skipped": n_skipped,
    }


def divergence_rate(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """T-P2: fração de seeds em que o arm divergiu naquele λ.

    "Pelo menos metade" inclui a metade exata — com 10 seeds, 5 já conta.
    """

    n_divergent = sum(1 for cell in cells if cell.get("divergent"))
    total = len(cells)
    rate = (n_divergent / total) if total else 0.0
    return {
        "n_divergent": n_divergent,
        "n": total,
        "rate": rate,
        "majority": bool(total) and rate >= 0.5,
    }


def judge_power(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """§D.2 por célula, exigindo UNANIMIDADE entre as seeds.

    O controle da passada 2 é construído com um único `lr_scale`. Se as seeds
    discordam sobre a célula ter poder, esse valor não existe — uma maioria não
    constrói um controle.
    """

    n_with_power = sum(1 for cell in cells if cell.get("has_power"))
    n_divergent = sum(1 for cell in cells if cell.get("divergent"))
    sane = [
        cell["norm_ratio"]["mean"]
        for cell in cells
        if not cell.get("divergent") and cell["norm_ratio"]["mean"] is not None
    ]
    return {
        "n": len(cells),
        "n_with_power": n_with_power,
        "n_divergent": n_divergent,
        "unanimous": bool(cells) and n_with_power == len(cells),
        "mean_e": statistics.fmean(sane) if sane else None,
    }


def authorizes_pass_two(powered: dict[str, dict[float, dict[str, Any]]]) -> dict[str, Any]:
    """§E.1: a passada 2 só existe se alguma célula tiver poder unânime.

    Entre várias, escolhe o **menor λ**: §E.2 lembra que mudar λ muda o método,
    então o ponto de operação menos distante das forças publicadas é o menos
    distorcido. Escolher o λ mais forte compraria um controle mais vistoso ao
    custo de descrever um método que ninguém publicou.
    """

    candidates = [
        (multiplier, arm, verdict)
        for arm, by_multiplier in powered.items()
        for multiplier, verdict in by_multiplier.items()
        if verdict.get("unanimous")
    ]
    if not candidates:
        return {"authorized": False, "arm": None, "multiplier": None, "mean_e": None}

    multiplier, arm, verdict = min(candidates, key=lambda item: item[0])
    return {
        "authorized": True,
        "arm": arm,
        "multiplier": multiplier,
        "mean_e": verdict.get("mean_e"),
    }


def load_cells(root: Path, multiplier: float, arm: str) -> list[dict[str, Any]]:
    directory = root / f"mult_{multiplier:g}"
    cells: list[dict[str, Any]] = []
    for path in sorted(directory.glob("seed_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        entry = record.get("arms", {}).get(arm)
        if entry:
            cells.append(entry)
    return cells


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    print("L3 — veredito de T-P1 a T-P4")
    print(f"limiar de poder (§D.2): E <= {POWER_THRESHOLD}, e nao divergente\n")

    header = f"{'arm':<5}{'lambda':>9}{'E':>11}{'dp':>10}{'div':>7}{'poder':>8}"
    print(header)
    print("-" * len(header))

    powered: dict[str, dict[float, dict[str, Any]]] = {}
    series_by_arm: dict[str, list[float | None]] = {}

    for arm in PENALTY_ARMS:
        powered[arm] = {}
        series: list[float | None] = []
        for multiplier in MULTIPLIER_AXIS:
            cells = load_cells(args.results, multiplier, arm)
            if not cells:
                series.append(None)
                print(f"{arm:<5}{multiplier:>8g}x{'sem dado':>11}")
                continue

            power = judge_power(cells)
            divergence = divergence_rate(cells)
            powered[arm][multiplier] = power
            # Um arm que divergiu na maioria das seeds não entrega ponto de série.
            series.append(None if divergence["majority"] else power["mean_e"])

            per_seed = [
                cell["norm_ratio"]["mean"]
                for cell in cells
                if not cell.get("divergent")
                and cell["norm_ratio"]["mean"] is not None
            ]
            spread = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
            divergent_label = f"{divergence['n_divergent']}/{divergence['n']}"
            power_label = f"{power['n_with_power']}/{power['n']}"
            print(
                f"{arm:<5}{multiplier:>8g}x{fmt(power['mean_e']):>11}{fmt(spread):>10}"
                f"{divergent_label:>7}{power_label:>8}"
            )
        series_by_arm[arm] = series
        print()

    print("=== T-P1: E monotono decrescente em lambda?")
    for arm, series in series_by_arm.items():
        verdict = judge_monotonicity(series)
        state = {True: "SIM", False: "NAO", None: "indecidivel"}[verdict["monotonic"]]
        print(
            f"  {arm:<5}{state:>12}  "
            f"({verdict['n_compared']} pontos, {verdict['inversions']} inversoes, "
            f"{verdict['n_skipped']} pulados por divergencia)"
        )

    print("\n=== T-P2: ewc e si divergem em lambda >= 10x?")
    for arm in ("ewc", "si"):
        for multiplier in (10.0, 30.0, 100.0):
            cells = load_cells(args.results, multiplier, arm)
            if not cells:
                continue
            divergence = divergence_rate(cells)
            mark = "SIM" if divergence["majority"] else "NAO"
            print(
                f"  {arm:<5}{multiplier:>5g}x  {mark:>4}  "
                f"({divergence['n_divergent']}/{divergence['n']} seeds)"
            )

    verdict = authorizes_pass_two(powered)
    print("\n=== T-P3/T-P4: existe celula com poder unanime?")
    if verdict["authorized"]:
        print(
            f"  SIM — arm `{verdict['arm']}` em {verdict['multiplier']:g}x, "
            f"E = {fmt(verdict['mean_e'])}"
        )
        print("  §E.1: a passada 2 fica AUTORIZADA A EXISTIR, com pre-registro proprio.")
        print("  §E.2: mudar lambda muda o metodo — a ressalva vai junto, obrigatoriamente.")
    else:
        print("  NAO — nenhuma celula atinge o limiar sem divergir.")
        print("  §E.1: a passada 2 MORRE e o item 2.13 fecha sem rodar.")
        print("  A inconstrutibilidade e estrutural, nao acidente das forcas publicadas.")

    aggregate = {
        "power_threshold": POWER_THRESHOLD,
        "monotonicity": {
            arm: judge_monotonicity(series) for arm, series in series_by_arm.items()
        },
        "power": {
            arm: {str(m): v for m, v in by_m.items()} for arm, by_m in powered.items()
        },
        "pass_two": verdict,
    }
    destination = args.results / "aggregate.json"
    destination.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\nagregado: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
