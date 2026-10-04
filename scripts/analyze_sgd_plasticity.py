#!/usr/bin/env python3
"""Agrega a run do L2 e julga as predições C1–C5.

Pré-registro: `goals/protocol_sgd_plasticity.md`, congelado em `f9f9bb0` e
emendado em `80bd322` (eixo de learning rate).

Lê os manifests por ponto do eixo e responde, por arm:

| predição | critério |
|---|---|
| C1 | `E > 1` persiste sob SGD? (sinal exato sobre `E − 1` por seed) |
| C2 | resíduo da identidade `E·cos = 1 + g·p/‖g‖²` ~ 0 |
| C3 | predição pelo sinal de `2 g·p + ‖p‖²` acerta 100% dos passos |
| C4 | `‖p‖/‖g‖ > 0,5` no `mas` |
| C5 | `E` é monótono no learning rate |

**Não lê nenhum endpoint de acurácia**, e os manifests da passada não contêm
nenhum — verificado por teste.

Uso:
    PYTHONPATH=. .venv/bin/python scripts/analyze_sgd_plasticity.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.confirmatory_statistics import exact_two_sided_sign_test
from scripts.run_sgd_plasticity import (
    LEARNING_RATE_AXIS,
    OUTPUT_DIR,
    PENALTY_ARMS,
)

EXPECTED_SEEDS = 12
#: C2: a identidade é exata sob SGD; só erro numérico é tolerado.
IDENTITY_TOLERANCE = 1e-4
#: C4: limiar declarado no §C do protocolo.
PENALTY_DOMINANCE_THRESHOLD = 0.5


def load_axis_point(root: Path, learning_rate: float) -> list[dict[str, Any]]:
    directory = root / f"lr_{learning_rate:g}"
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("seed_*.json"))
    ]


def summarize_arm(records: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Agrega um arm num ponto do eixo e julga C1–C4."""

    entries = [r["arms"][arm] for r in records if r["arms"].get(arm)]
    per_seed = [
        entry["norm_ratio"]["mean"]
        for entry in entries
        if entry["norm_ratio"]["mean"] is not None
    ]
    non_finite = sum(entry["norm_ratio"]["n_non_finite"] for entry in entries)
    total_samples = sum(entry.get("n_samples", 0) for entry in entries)

    summary: dict[str, Any] = {
        "arm": arm,
        "n_seeds": len(per_seed),
        "n_samples": total_samples,
        "n_non_finite": non_finite,
        "divergent": bool(non_finite),
    }
    if not per_seed:
        summary["E_mean"] = None
        summary["c1_above_one"] = None
        return summary

    deviations = [value - 1.0 for value in per_seed]
    summary["E_mean"] = statistics.fmean(per_seed)
    summary["E_stdev"] = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
    summary["E_min"] = min(per_seed)
    summary["E_max"] = max(per_seed)
    summary["above_one"] = sum(1 for value in deviations if value > 0.0)
    summary["below_one"] = sum(1 for value in deviations if value < 0.0)
    summary["p_exact"] = exact_two_sided_sign_test(deviations)
    # C1 vale para este arm se E > 1 de forma unânime.
    summary["c1_above_one"] = summary["above_one"] == len(per_seed)

    for key in ("direction_cosine", "penalty_scale", "penalty_alignment"):
        values = [
            entry[key]["mean"]
            for entry in entries
            if entry.get(key) and entry[key]["mean"] is not None
        ]
        summary[key] = statistics.fmean(values) if values else None

    residuals = [
        entry["identity_residual"]["max"]
        for entry in entries
        if entry.get("identity_residual")
        and entry["identity_residual"].get("max") is not None
    ]
    summary["identity_residual_max"] = max(residuals) if residuals else None
    # C2 só é julgável onde a run não divergiu: um arm que explodiu não testa
    # a identidade, ele testa o otimizador.
    summary["c2_identity_holds"] = (
        None
        if summary["divergent"] or summary["identity_residual_max"] is None
        else summary["identity_residual_max"] < IDENTITY_TOLERANCE
    )

    agreed = sum(entry["prediction_agreement"]["agreed"] for entry in entries)
    counted = sum(entry["prediction_agreement"]["n"] for entry in entries)
    summary["prediction_agreed"] = agreed
    summary["prediction_counted"] = counted
    summary["prediction_rate"] = (agreed / counted) if counted else None
    summary["c3_prediction_exact"] = (agreed == counted) if counted else None

    summary["c4_penalty_dominates"] = (
        None
        if summary["penalty_scale"] is None
        else summary["penalty_scale"] > PENALTY_DOMINANCE_THRESHOLD
    )
    return summary


def check_monotonic(axis_summaries: dict[float, list[dict[str, Any]]], arm: str) -> dict[str, Any]:
    """C5: `E` é monótono no learning rate dentro deste arm?

    Pontos divergentes ficam de fora: um `E` de 1e16 não participa de uma
    afirmação sobre monotonicidade, ele a destruiria por construção.
    """

    points: list[tuple[float, float]] = []
    for learning_rate in sorted(axis_summaries, reverse=True):
        row = next(
            (s for s in axis_summaries[learning_rate] if s["arm"] == arm), None
        )
        if row and row["E_mean"] is not None and not row["divergent"]:
            points.append((learning_rate, row["E_mean"]))

    if len(points) < 2:
        return {"arm": arm, "n_points": len(points), "monotonic": None, "points": points}

    values = [value for _, value in points]
    increasing = all(b >= a for a, b in zip(values, values[1:], strict=False))
    decreasing = all(b <= a for a, b in zip(values, values[1:], strict=False))
    return {
        "arm": arm,
        "n_points": len(points),
        "monotonic": increasing or decreasing,
        "direction": "cresce quando lr cai" if increasing else ("cai quando lr cai" if decreasing else "não monótono"),
        "points": points,
    }


def analyze(root: Path) -> dict[str, Any]:
    axis_summaries: dict[float, list[dict[str, Any]]] = {}
    seeds_by_point: dict[float, int] = {}
    for learning_rate in LEARNING_RATE_AXIS:
        records = load_axis_point(root, learning_rate)
        seeds_by_point[learning_rate] = len(records)
        if records:
            axis_summaries[learning_rate] = [
                summarize_arm(records, arm) for arm in PENALTY_ARMS
            ]

    failures: list[str] = []
    for learning_rate, count in seeds_by_point.items():
        if count != EXPECTED_SEEDS:
            failures.append(
                f"lr={learning_rate:g}: esperava {EXPECTED_SEEDS} seeds, achei {count}"
            )

    return {
        "root": str(root),
        "axis": list(LEARNING_RATE_AXIS),
        "seeds_by_point": seeds_by_point,
        "integrity_failures": failures,
        "by_learning_rate": axis_summaries,
        "monotonicity": [check_monotonic(axis_summaries, arm) for arm in PENALTY_ARMS],
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    if report["integrity_failures"]:
        lines.append("INTEGRIDADE FALHOU:")
        for item in report["integrity_failures"]:
            lines.append(f"  - {item}")
        lines.append("")

    for learning_rate, rows in report["by_learning_rate"].items():
        lines.append(f"=== lr = {learning_rate:g}")
        header = (
            f"{'arm':<6}{'E':>10}{'dp':>9}{'sinais':>9}{'p':>9}"
            f"{'nao-fin':>9}{'cos':>9}{'|p|/|g|':>10}{'C1':>5}{'C2':>5}{'C3':>5}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows:
            if row["E_mean"] is None:
                lines.append(f"{row['arm']:<6}{'TUDO DIVERGIU':>10}")
                continue
            verdict = lambda key: (  # noqa: E731
                "—" if row[key] is None else ("sim" if row[key] else "NAO")
            )
            p_value = "—" if row.get("p_exact") is None else f"{row['p_exact']:.5f}"
            signs = f"{row['above_one']}+/{row['below_one']}-"
            lines.append(
                f"{row['arm']:<6}{_fmt(row['E_mean']):>10}{_fmt(row.get('E_stdev')):>9}"
                f"{signs:>9}"
                f"{p_value:>9}{row['n_non_finite']:>9}"
                f"{_fmt(row.get('direction_cosine')):>9}{_fmt(row.get('penalty_scale')):>10}"
                f"{verdict('c1_above_one'):>5}{verdict('c2_identity_holds'):>5}"
                f"{verdict('c3_prediction_exact'):>5}"
            )
        lines.append("")

    lines.append("=== C5: monotonicidade de E no learning rate")
    for entry in report["monotonicity"]:
        points = ", ".join(f"lr={lr:g}: {value:.4f}" for lr, value in entry["points"])
        verdict = (
            "—"
            if entry["monotonic"] is None
            else ("sim" if entry["monotonic"] else "NAO")
        )
        lines.append(
            f"  {entry['arm']:<5} {verdict:>4} ({entry.get('direction', 'n/d')}) — {points}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = analyze(args.root)
    print(render(report))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if report["integrity_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
