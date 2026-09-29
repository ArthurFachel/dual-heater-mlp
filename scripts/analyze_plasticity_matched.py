#!/usr/bin/env python3
"""Agrega a run de plasticidade pareada no contraste pré-registrado.

Pré-registro: ``goals/protocol_plasticity_matched.md``.

- Contraste primário (P3): ``frozen_a_control − lr_control`` em forgetting médio.
- Secundário (P4): o mesmo contraste em final average accuracy.
- Teste (P5): sinal exato bicaudal sobre as diferenças pareadas por seed.
- Correção (P6): Holm sobre a família de 2. Nada mais entra na família.
- Critério de sucesso (P9): ``p_Holm < 0,025`` no primário **e** mediana negativa.
- ``vanilla`` (P10) é descritivo e fica FORA da família.

Reusa ``exact_two_sided_sign_test`` e ``holm`` em vez de reimplementar
estatística. Roda as verificações H1-H8 antes de qualquer endpoint e recusa
agregar uma run que não passa — um contraste com pareamento quebrado não é
interpretável.

Uso:
    PYTHONPATH=. .venv/bin/python scripts/analyze_plasticity_matched.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.analyze_confirmation import holm
from experiments.confirmatory_statistics import exact_two_sided_sign_test
from scripts.verify_plasticity_matched import DEFAULT_ROOT, check_plasticity_matched

#: Endpoints na ordem da família confirmatória declarada em P6.
FAMILY = (
    ("average_forgetting", "forgetting", "primário"),
    ("final_average_accuracy", "FAA", "secundário"),
)
TREATMENT = "frozen_a_control"
COMPARATOR = "lr_control"
DESCRIPTIVE = "vanilla"
ALL_ARMS = (DESCRIPTIVE, COMPARATOR, TREATMENT)
#: P9. Metade de 0,05 porque o critério é bicaudal com direção declarada.
ALPHA_PRIMARY = 0.025


def load_seed_results(root: Path) -> dict[int, dict[str, dict[str, Any]]]:
    """Mapa seed -> braço -> registro de resultado."""

    out: dict[int, dict[str, dict[str, Any]]] = {}
    for seed_dir in sorted(root.glob("seed_*")):
        manifest = json.loads((seed_dir / "manifest.json").read_text())
        seed = int(seed_dir.name.removeprefix("seed_"))
        out[seed] = {row["method"]: row for row in manifest["results"]}
    return out


def arm_table(results: dict[int, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Valores ABSOLUTOS por braço. Vão sempre ao lado dos contrastes (P.195)."""

    table = []
    for arm in ALL_ARMS:
        row: dict[str, Any] = {"arm": arm, "n_seeds": len(results)}
        for key, label, _ in FAMILY:
            values = [results[s][arm][key] for s in sorted(results)]
            row[f"{label}_mean"] = st.mean(values)
            row[f"{label}_stdev"] = st.stdev(values) if len(values) > 1 else 0.0
        row["surface_plasticity"] = results[next(iter(results))][arm][
            "surface_plasticity"
        ]
        row["trainable_parameters"] = results[next(iter(results))][arm][
            "trainable_parameters"
        ]
        # Aquisição da última tarefa: a diagonal final. Sem ela um ganho de
        # retenção pode estar escondendo colapso de aquisição.
        matrix = results[next(iter(results))][arm]["accuracy_matrix"]
        acquisitions = [
            results[s][arm]["accuracy_matrix"][i][i]
            for s in sorted(results)
            for i in (len(matrix) - 1,)
        ]
        row["acquisition_last_mean"] = st.mean(acquisitions)
        first_retentions = [
            results[s][arm]["accuracy_matrix"][-1][0] for s in sorted(results)
        ]
        row["retention_first_mean"] = st.mean(first_retentions)
        table.append(row)
    return table


def paired_contrast(
    results: dict[int, dict[str, dict[str, Any]]],
    key: str,
    *,
    treatment: str = TREATMENT,
    comparator: str = COMPARATOR,
) -> dict[str, Any]:
    """Diferenças pareadas por seed com sinal exato. Nunca despareia."""

    seeds = sorted(results)
    diffs = [results[s][treatment][key] - results[s][comparator][key] for s in seeds]
    return {
        "contrast": f"{treatment} − {comparator}",
        "endpoint": key,
        "n": len(diffs),
        "per_seed": dict(zip(seeds, diffs, strict=True)),
        "mean": st.mean(diffs),
        "median": st.median(diffs),
        "negative": sum(1 for d in diffs if d < 0.0),
        "positive": sum(1 for d in diffs if d > 0.0),
        "p_raw": exact_two_sided_sign_test(diffs),
    }


def analyze(root: Path) -> dict[str, Any]:
    results = load_seed_results(root)
    contrasts = [paired_contrast(results, key) for key, _, _ in FAMILY]
    # Holm sobre a família de 2, na ordem declarada. p=None (todas as
    # diferenças exatamente zero) é tratado como 1.0, o valor conservador.
    adjusted = holm([c["p_raw"] if c["p_raw"] is not None else 1.0 for c in contrasts])
    for contrast, (_, label, role), p_holm in zip(contrasts, FAMILY, adjusted, strict=True):
        contrast["label"] = label
        contrast["role"] = role
        contrast["p_holm"] = p_holm

    primary = contrasts[0]
    verdict = {
        # P9: as duas condições, conjuntivas.
        "p_holm_below_alpha": primary["p_holm"] < ALPHA_PRIMARY,
        "median_negative": primary["median"] < 0.0,
    }
    verdict["significant"] = verdict["p_holm_below_alpha"] and verdict["median_negative"]
    verdict["branch"] = "A" if verdict["significant"] else "B"

    return {
        "root": str(root),
        "seeds": sorted(results),
        "arms": arm_table(results),
        "family": contrasts,
        "verdict": verdict,
        "alpha_primary": ALPHA_PRIMARY,
        "holm_floor": len(FAMILY) * 2 / 2 ** len(results),
    }


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"seeds ({len(report['seeds'])}): {report['seeds']}")
    lines.append("")
    lines.append("BRAÇOS (valores absolutos)")
    header = (
        f"{'braço':<18}{'E_surf':>9}{'treináveis':>12}"
        f"{'forgetting':>12}{'FAA':>9}{'aquis.últ':>11}{'reten.t0':>10}"
    )
    lines.append(header)
    for row in report["arms"]:
        lines.append(
            f"{row['arm']:<18}{row['surface_plasticity']:>9.4f}"
            f"{row['trainable_parameters']:>12,}"
            f"{row['forgetting_mean']:>12.4f}{row['FAA_mean']:>9.4f}"
            f"{row['acquisition_last_mean']:>11.4f}{row['retention_first_mean']:>10.4f}"
        )
    lines.append("")
    lines.append(f"FAMÍLIA CONFIRMATÓRIA (Holm sobre {len(report['family'])})")
    for contrast in report["family"]:
        p_raw = contrast["p_raw"]
        lines.append(
            f"  [{contrast['role']}] {contrast['contrast']} em {contrast['label']}: "
            f"média={contrast['mean']:+.4f} mediana={contrast['median']:+.4f} "
            f"sinais {contrast['negative']}−/{contrast['positive']}+ "
            f"p={'n/d' if p_raw is None else f'{p_raw:.4f}'} "
            f"p_Holm={contrast['p_holm']:.4f}"
        )
    lines.append("")
    verdict = report["verdict"]
    lines.append(
        f"VEREDITO P9: p_Holm<{report['alpha_primary']} = {verdict['p_holm_below_alpha']}, "
        f"mediana negativa = {verdict['median_negative']} "
        f"-> {'SIGNIFICATIVO (galho A)' if verdict['significant'] else 'EMPATE (galho B)'}"
    )
    lines.append(f"piso de Holm nesta n: {report['holm_floor']:.4f}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", type=Path, default=None, help="grava o relatório bruto")
    parser.add_argument(
        "--skip-integrity",
        action="store_true",
        help="NÃO use para resultado. Só para inspecionar uma run parcial.",
    )
    args = parser.parse_args(argv)

    failures = check_plasticity_matched(args.root)
    if failures and not args.skip_integrity:
        print("INTEGRIDADE FALHOU — não interprete endpoint:")
        for item in failures:
            print(" -", item)
        return 1
    if failures:
        print("AVISO: integridade reprovada, saída NÃO é resultado:")
        for item in failures:
            print(" -", item)
        print()

    report = analyze(args.root)
    print(render(report))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
