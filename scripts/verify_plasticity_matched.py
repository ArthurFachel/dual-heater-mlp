#!/usr/bin/env python3
"""Verificações H1-H8 de goals/protocol_plasticity_matched.md.

Roda ANTES de olhar qualquer endpoint. Se alguma falhar, o contraste não é
interpretável e a run precisa ser diagnosticada, não analisada.

Uso:

    PYTHONPATH=. .venv/bin/python scripts/verify_plasticity_matched.py

Saída esperada com a run completa: ``OK: 10 seeds, 8/8 verificações passaram``.
Exit 0 quando tudo passa, 1 quando alguma checagem falha.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_ROOT = Path("results/plasticity_matched")
EXPECTED_SEEDS = 10
TARGET_E = 0.6224617129892229
ARMS = ("vanilla", "lr_control", "frozen_a_control")
#: Os dois braços do contraste primário, que precisam estar pareados em E.
MATCHED_ARMS = ("frozen_a_control", "lr_control")
#: Tolerância do pareamento. O protocolo exige igualdade exata na superfície;
#: 1e-9 é folga de serialização JSON, não de método.
E_TOLERANCE = 1e-9


def check_plasticity_matched(
    root: Path,
    *,
    expected_seeds: int = EXPECTED_SEEDS,
    target_e: float = TARGET_E,
) -> list[str]:
    """Retorna a lista de falhas H1-H8. Lista vazia significa run íntegra.

    Separada de ``main`` para ser testável sem tocar em ``sys.exit``.
    """

    seed_dirs = sorted(root.glob("seed_*"))
    failures: list[str] = []

    # H1: a run tem todas as seeds pré-registradas.
    if len(seed_dirs) != expected_seeds:
        failures.append(
            f"H1: esperava {expected_seeds} seeds, achei {len(seed_dirs)}"
        )

    for seed_dir in seed_dirs:
        manifest_path = seed_dir / "manifest.json"

        # H2: cada seed escreveu seu manifesto.
        if not manifest_path.exists():
            failures.append(f"H2: {seed_dir.name} sem manifest.json")
            continue
        manifest = json.loads(manifest_path.read_text())
        results = {row["method"]: row for row in manifest["results"]}

        # H3: os três braços declarados, nem mais nem menos. Um braço a mais é
        # condição não pré-registrada dentro de um experimento confirmatório.
        if set(results) != set(ARMS):
            failures.append(f"H3: {seed_dir.name} tem braços {sorted(results)}")
            continue

        # H4: os dois braços do contraste pareados na SUPERFÍCIE (não na média
        # da máscara — esse foi exatamente o erro do R3).
        e_frozen = results["frozen_a_control"]["surface_plasticity"]
        e_lr = results["lr_control"]["surface_plasticity"]
        if abs(e_frozen - e_lr) > E_TOLERANCE:
            failures.append(
                f"H4: {seed_dir.name} E não pareado: {e_frozen} vs {e_lr}"
            )

        # H5: ambos no alvo declarado no pré-registro.
        for arm in MATCHED_ARMS:
            got = results[arm]["surface_plasticity"]
            if abs(got - target_e) > E_TOLERANCE:
                failures.append(
                    f"H5: {seed_dir.name}/{arm} E={got}, alvo={target_e}"
                )

        # H6: vanilla é a referência não tratada, E = 1 exato.
        if results["vanilla"]["surface_plasticity"] != 1.0:
            failures.append(f"H6: {seed_dir.name} vanilla E != 1.0")

        # H7: tokens de treino pareados dentro da seed. Arms que compartilham o
        # stream de dados por construção têm de bater exatamente; divergência é
        # quebra de pareamento, não ruído.
        tokens = {arm: results[arm]["train_tokens"] for arm in ARMS}
        if len(set(tokens.values())) != 1:
            failures.append(f"H7: {seed_dir.name} tokens não pareados: {tokens}")

        # H8: o lr_control realmente recebeu o tratamento. O modo de falha que
        # já mordeu este projeto: o knob não chega no braço e o controle roda
        # como vanilla com outro nome.
        lr_scale = results["lr_control"]["learning_rate_scale"]
        if abs(lr_scale - target_e) > 1e-12:
            failures.append(f"H8: {seed_dir.name} lr_control scale={lr_scale}")

        # H8b: o frozen_a_control NÃO recebe o tratamento de LR — seu E vem de
        # congelar A. Se ele também tivesse scale != 1 o contraste somaria dois
        # tratamentos.
        frozen_scale = results["frozen_a_control"]["learning_rate_scale"]
        if abs(frozen_scale - 1.0) > 1e-12:
            failures.append(
                f"H8: {seed_dir.name} frozen_a_control scale={frozen_scale}, esperava 1.0"
            )

        # H8c: o pareamento vem de superfícies treináveis DIFERENTES. Igualdade
        # de trainable_parameters com E igual seria a fingerprint do bug R3.
        trainable = {arm: results[arm]["trainable_parameters"] for arm in MATCHED_ARMS}
        if trainable["frozen_a_control"] >= trainable["lr_control"]:
            failures.append(
                f"H8: {seed_dir.name} frozen_a_control deveria treinar menos "
                f"parâmetros que lr_control, got {trainable}"
            )

    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--expected-seeds", type=int, default=EXPECTED_SEEDS)
    args = parser.parse_args(argv)

    failures = check_plasticity_matched(
        args.root, expected_seeds=args.expected_seeds
    )
    if failures:
        print("FALHOU:")
        for item in failures:
            print(" -", item)
        return 1

    seeds = len(sorted(args.root.glob("seed_*")))
    print(f"OK: {seeds} seeds, 8/8 verificações passaram")
    return 0


if __name__ == "__main__":
    sys.exit(main())
