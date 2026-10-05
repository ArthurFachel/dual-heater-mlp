#!/usr/bin/env python3
"""Agregador da passada 2 — julga Q1 a Q3.

Pré-registro: `goals/protocol_penalty_pass2.md` §E. Item 2.13 de
`goals/roadmap_icml_ijcnn.md`.

| predição | o que mede |
|---|---|
| Q1 | `mas − lr_control` em esquecimento **não** é significativo |
| Q2 | `lr_control − vanilla` reduz esquecimento (o controle é um controle) |
| Q3 | `mas` a 30x tem acurácia final menor que `vanilla` |

Esquecimento é **menor é melhor**: uma diferença negativa significa que o
tratamento esquece MENOS. O relatório diz isso em texto porque um leitor que
assuma o contrário lê o resultado invertido.

Uso:
    PYTHONPATH=src:. python scripts/analyze_penalty_pass2.py
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

ARMS = ("vanilla", "mas", "lr_control")
PRIMARY_ENDPOINT = "average_forgetting"
RESULTS_DIR = Path("results/penalty_pass2")

#: P9: a família tem UMA comparação. Com m=1 a correção de Holm é a identidade,
#: e aplicá-la mesmo assim sugeriria no relatório uma família que não existe.
FAMILY_SIZE = 1

#: §D.2: quanto a acurácia final do tratamento pode ficar abaixo do `vanilla`
#: antes de a queda de esquecimento deixar de ser interpretável como
#: preservação. Cinco pontos percentuais é generoso de propósito.
COLLAPSE_MARGIN = 0.05

#: Quantas vezes o nível de chance um arm precisa atingir para que o regime
#: conte como "aprendeu alguma coisa". Com 10 classes a chance é 0,10, logo o
#: piso é 0,20. É uma barra baixa — e se nem ela for atingida, o contraste de
#: esquecimento não compara preservação de conhecimento, compara o quanto cada
#: arm deixou de aprender.
NEAR_CHANCE_FACTOR = 2.0


def paired_differences(
    records: Sequence[dict[str, Any]], *, treatment: str, control: str
) -> list[float]:
    """`treatment − control` por seed, nessa ordem.

    Inverter a ordem troca o sinal sem mudar nenhum valor absoluto, e o `p` do
    teste bilateral seria idêntico — nada no resultado denunciaria a troca.
    """

    return [
        record["arms"][treatment][PRIMARY_ENDPOINT]
        - record["arms"][control][PRIMARY_ENDPOINT]
        for record in records
    ]


def describe_direction(difference: float) -> str:
    """Esquecimento é menor-é-melhor; dizer isso evita a leitura invertida."""

    if difference < 0:
        return "o tratamento esquece MENOS que o controle"
    if difference > 0:
        return "o tratamento esquece MAIS que o controle"
    return "empate exato"


def _p_value_of(differences: Sequence[float]) -> float | None:
    """`p` do sinal exato, ou `None` quando não há nenhuma diferença não-nula.

    Com todas as diferenças exatamente zero o teste não tem o que testar e
    devolve `None`. Tratar isso como `p = 1` seria quase certo na prática, mas
    é uma afirmação sobre os dados que o teste não fez — e o caso aparece
    justamente quando dois arms são o MESMO arm, que é o modo de falha mais
    grave possível aqui.
    """

    result = exact_two_sided_sign_test(list(differences))
    p_value = result["p_value"] if isinstance(result, dict) else result
    return None if p_value is None else float(p_value)


def judge_primary(differences: Sequence[float]) -> dict[str, Any]:
    """P10/P11: sinal exato bilateral, sem correção (P9: m=1)."""

    p_value = _p_value_of(differences)
    return {
        "n": len(differences),
        "n_negative": sum(1 for d in differences if d < 0),
        "n_positive": sum(1 for d in differences if d > 0),
        # Um empate exato não é evidência a favor de nenhum lado.
        "n_zero": sum(1 for d in differences if d == 0),
        "mean": statistics.fmean(differences) if differences else None,
        "p_value": p_value,
        "significant": p_value is not None and p_value < 0.05,
        "family_size": FAMILY_SIZE,
    }


def summarize_arms(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """§D.2: esquecimento E acurácia, para os três arms, sempre."""

    summary: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        entry: dict[str, Any] = {}
        for metric in ("average_forgetting", "final_average_accuracy"):
            values = [record["arms"][arm][metric] for record in records]
            entry[metric] = {
                "mean": statistics.fmean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        summary[arm] = entry
    return summary


def flags_acquisition_collapse(
    *, mas_accuracy: float, vanilla_accuracy: float
) -> bool:
    """Q3/§D.2: o tratamento reduziu esquecimento por aprender menos?

    Se sim, a queda de esquecimento não é preservação de conhecimento, e o
    relatório precisa dizer isso sozinho em vez de depender de quem lê reparar
    numa coluna.
    """

    return (vanilla_accuracy - mas_accuracy) > COLLAPSE_MARGIN


def flags_near_chance_regime(
    *, accuracies: dict[str, float], n_classes: int
) -> bool:
    """Todos os arms ficaram perto do nível de chance?

    `flags_acquisition_collapse` compara arms ENTRE SI e por isso é cego para o
    caso em que todos colapsaram juntos: a ordenação pode ser perfeitamente
    consistente, com `p` de 0,00049, e ainda assim os três modelos mal terem
    aprendido. Um contraste de esquecimento nesse regime não mede preservação
    de conhecimento — mede quanto cada arm deixou de adquirir.

    Usa o MELHOR arm, não a média: se algum escapou do chão, o regime não é
    degenerado por completo, e um tratamento que funciona enquanto os controles
    colapsam seria um resultado em vez de um artefato.
    """

    chance = 1.0 / n_classes
    return max(accuracies.values()) < chance * NEAR_CHANCE_FACTOR


def control_is_distinguishable(records: Sequence[dict[str, Any]]) -> bool:
    """Q2: o `lr_control` precisa diferir do `vanilla`, senão não é controle.

    Este é o modo de falha que o L3 existiu para evitar — um controle a
    `lr × 0,996` seria vanilla com outro nome. A verificação fica aqui porque é
    no agregado que ela se torna visível.
    """

    differences = paired_differences(
        records, treatment="lr_control", control="vanilla"
    )
    p_value = _p_value_of(differences)
    return p_value is not None and p_value < 0.05


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    paths = sorted(args.results.glob("seed_*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not records:
        print("nenhum manifest encontrado")
        return 1

    print("Passada 2 — mas a 30x contra controle pareado em plasticidade")
    print(f"{len(records)} seeds, endpoint primario: {PRIMARY_ENDPOINT}")
    print(f"lr_scale congelado: {records[0]['learning_rate_scale']}")
    print("Esquecimento: MENOR e melhor.\n")

    summary = summarize_arms(records)
    header = f"{'arm':<12}{'esquecimento':>14}{'dp':>9}{'acuracia':>11}{'dp':>9}"
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        entry = summary[arm]
        print(
            f"{arm:<12}{fmt(entry['average_forgetting']['mean']):>14}"
            f"{fmt(entry['average_forgetting']['stdev']):>9}"
            f"{fmt(entry['final_average_accuracy']['mean']):>11}"
            f"{fmt(entry['final_average_accuracy']['stdev']):>9}"
        )

    print("\n=== Q1 (PRIMARIO): mas - lr_control em esquecimento")
    primary = judge_primary(
        paired_differences(records, treatment="mas", control="lr_control")
    )
    print(
        f"  media {fmt(primary['mean'])}  "
        f"sinais {primary['n_negative']}-/{primary['n_positive']}+"
        f"/{primary['n_zero']}=  "
        + (
            f"p = {primary['p_value']:.5f}"
            if primary["p_value"] is not None
            else "p indefinido (todas as diferencas sao zero)"
        )
    )
    print(f"  {describe_direction(primary['mean'])}")
    if primary["significant"]:
        print("  SIGNIFICATIVO — Q1 FALSIFICADA: o MAS faz algo alem de remover plasticidade.")
    else:
        print("  nao significativo — Q1 CONFIRMADA: o efeito do MAS nao sobrevive ao pareamento.")

    print("\n=== Q2: o lr_control e um controle de verdade?")
    distinguishable = control_is_distinguishable(records)
    secondary = judge_primary(
        paired_differences(records, treatment="lr_control", control="vanilla")
    )
    print(
        f"  lr_control - vanilla: media {fmt(secondary['mean'])}, "
        + (
            f"p = {secondary['p_value']:.5f}"
            if secondary["p_value"] is not None
            else "p indefinido"
        )
    )
    print(f"  {'SIM' if distinguishable else 'NAO — o desenho nao testa nada'}")

    print("\n=== Q3: a forca inflada custou aquisicao?")
    mas_accuracy = summary["mas"]["final_average_accuracy"]["mean"]
    vanilla_accuracy = summary["vanilla"]["final_average_accuracy"]["mean"]
    collapsed = flags_acquisition_collapse(
        mas_accuracy=mas_accuracy, vanilla_accuracy=vanilla_accuracy
    )
    print(f"  mas {fmt(mas_accuracy)} contra vanilla {fmt(vanilla_accuracy)}")
    if collapsed:
        print("  COLAPSO SINALIZADO: a queda de esquecimento NAO e preservacao de")
        print("  conhecimento — o arm aprendeu menos. Obrigatorio no relatorio.")
    else:
        print("  sem colapso: a acuracia final se manteve dentro da margem.")

    accuracies = {
        arm: summary[arm]["final_average_accuracy"]["mean"] for arm in ARMS
    }
    n_classes = len(records[0].get("class_order") or range(10))
    near_chance = flags_near_chance_regime(
        accuracies=accuracies, n_classes=n_classes
    )
    print(f"\n=== REGIME: algum arm aprendeu de fato? (chance = {1/n_classes:.2f})")
    print(
        "  melhor arm: "
        + fmt(max(accuracies.values()))
        + f"  piso exigido: {1/n_classes*NEAR_CHANCE_FACTOR:.2f}"
    )
    if near_chance:
        print("  *** QUASE-CHANCE: NENHUM arm escapou do chao. ***")
        print("  O contraste primario acima compara modelos que mal aprenderam.")
        print("  Ele NAO mede preservacao de conhecimento, e qualquer leitura do")
        print("  Q1 tem de carregar essa ressalva — inclusive uma significativa.")
    else:
        print("  ok: ao menos um arm esta claramente acima do nivel de chance.")

    aggregate = {
        "n_seeds": len(records),
        "learning_rate_scale": records[0]["learning_rate_scale"],
        "arms": summary,
        "primary": primary,
        "control_vs_vanilla": secondary,
        "control_is_distinguishable": distinguishable,
        "acquisition_collapse": collapsed,
        "near_chance_regime": near_chance,
        "best_arm_accuracy": max(accuracies.values()),
        "chance_level": 1.0 / n_classes,
    }
    destination = args.results / "aggregate.json"
    destination.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\nagregado: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
