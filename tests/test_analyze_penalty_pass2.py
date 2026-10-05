"""O agregador da passada 2 julga Q1-Q3.

Pré-registro: `goals/protocol_penalty_pass2.md` §E.

Aqui o risco é diferente do L3: esta passada lê acurácia, a família tem uma só
comparação, e o endpoint primário é esquecimento. As três armadilhas são
inverter o sinal do contraste, aplicar uma correção que o protocolo não pede,
e deixar a acurácia final fora da tabela — que é como um método que colapsou a
aquisição passa por método que preservou conhecimento.
"""

from __future__ import annotations

import pytest


def seed_record(*, vanilla_f, mas_f, control_f, vanilla_a=0.5, mas_a=0.5, control_a=0.5):
    """Um manifest de seed como o agregador o recebe."""

    return {
        "arms": {
            "vanilla": {
                "average_forgetting": vanilla_f,
                "final_average_accuracy": vanilla_a,
            },
            "mas": {
                "average_forgetting": mas_f,
                "final_average_accuracy": mas_a,
            },
            "lr_control": {
                "average_forgetting": control_f,
                "final_average_accuracy": control_a,
            },
        }
    }


# --- o sinal do contraste --------------------------------------------------


def test_contrast_is_treatment_minus_control_in_that_order() -> None:
    """`mas − lr_control`, não o contrário.

    Inverter a ordem troca o sinal e transformaria "o MAS esquece menos" em "o
    MAS esquece mais" sem mudar nenhum valor absoluto — e o `p` do teste de
    sinal bilateral seria idêntico, então nada denunciaria a troca.
    """

    from scripts.analyze_penalty_pass2 import paired_differences

    records = [seed_record(vanilla_f=0.9, mas_f=0.80, control_f=0.85)]
    diffs = paired_differences(records, treatment="mas", control="lr_control")
    assert diffs[0] == pytest.approx(-0.05)


def test_negative_difference_means_the_treatment_forgets_less() -> None:
    """Esquecimento é 'menor é melhor'. O relatório tem de dizer isso em texto,
    porque um leitor que assuma 'maior é melhor' lê o resultado invertido."""

    from scripts.analyze_penalty_pass2 import describe_direction

    assert "menos" in describe_direction(-0.05).lower()
    assert "mais" in describe_direction(+0.05).lower()


# --- P9: a família tem uma comparação -------------------------------------


def test_no_holm_correction_is_applied_to_a_single_comparison() -> None:
    """P9: com m=1 Holm é a identidade. Aplicá-lo 'por segurança' não muda o
    número, mas sugere no relatório uma família que não existe."""

    from scripts.analyze_penalty_pass2 import judge_primary

    verdict = judge_primary([-0.05] * 12)
    assert verdict["family_size"] == 1
    assert "p_holm" not in verdict


def test_primary_verdict_uses_the_exact_sign_test() -> None:
    from scripts.analyze_penalty_pass2 import judge_primary

    verdict = judge_primary([-0.05] * 12)
    assert verdict["p_value"] == pytest.approx(2.0 ** -11)
    assert verdict["significant"] is True
    assert verdict["n_negative"] == 12


def test_a_split_result_is_not_significant() -> None:
    """§G.4: 9 de 12 dá p = 0,146 e é reportado como nulo."""

    from scripts.analyze_penalty_pass2 import judge_primary

    verdict = judge_primary([-0.05] * 9 + [0.05] * 3)
    assert verdict["significant"] is False
    assert verdict["p_value"] > 0.05


def test_exact_ties_are_not_silently_counted_as_wins() -> None:
    """Uma diferença exatamente zero não é evidência a favor de ninguém."""

    from scripts.analyze_penalty_pass2 import judge_primary

    verdict = judge_primary([-0.05] * 6 + [0.0] * 6)
    assert verdict["n_negative"] == 6
    assert verdict["n_zero"] == 6


# --- Q3/§D.2: a acurácia é obrigatória ------------------------------------


def test_aggregate_reports_accuracy_for_every_arm() -> None:
    """§D.2, a armadilha central desta passada.

    Um `mas` a 30x pode reduzir esquecimento simplesmente aprendendo menos.
    Sem a acurácia final ao lado, isso se lê como preservação de conhecimento.
    """

    from scripts.analyze_penalty_pass2 import summarize_arms

    records = [
        seed_record(
            vanilla_f=0.9, mas_f=0.5, control_f=0.85,
            vanilla_a=0.60, mas_a=0.20, control_a=0.55,
        )
    ]
    summary = summarize_arms(records)
    for arm in ("vanilla", "mas", "lr_control"):
        assert "final_average_accuracy" in summary[arm]
        assert "average_forgetting" in summary[arm]


def test_collapse_of_acquisition_is_flagged() -> None:
    """Q3: se o `mas` tem acurácia final muito abaixo do `vanilla`, a queda de
    esquecimento não é preservação — e o relatório precisa dizer isso sozinho,
    não depender de quem lê reparar."""

    from scripts.analyze_penalty_pass2 import flags_acquisition_collapse

    assert flags_acquisition_collapse(mas_accuracy=0.20, vanilla_accuracy=0.60) is True
    assert flags_acquisition_collapse(mas_accuracy=0.58, vanilla_accuracy=0.60) is False


def test_near_chance_regime_is_flagged_for_every_arm() -> None:
    """O caso que a passada 2 encontrou de verdade, e que `flags_acquisition_
    collapse` NÃO pega porque só compara arms entre si.

    Em Split-MNIST class-IL com 10 classes o nível de chance é 0,10. Acurácias
    de 0,15 a 0,20 significam que todos os arms estão perto do chão, e um
    contraste de esquecimento entre modelos que mal aprenderam não mede
    preservação de conhecimento — mede o quanto cada um deixou de aprender.
    Comparar arms entre si não detecta isso: eles podem estar ordenados de
    forma perfeitamente consistente e ainda assim todos colapsados.
    """

    from scripts.analyze_penalty_pass2 import flags_near_chance_regime

    # 10 classes, chance = 0,10. Tudo entre 0,15 e 0,20 é quase-chance.
    assert flags_near_chance_regime(
        accuracies={"vanilla": 0.149, "mas": 0.198, "lr_control": 0.160},
        n_classes=10,
    ) is True
    # Um regime são: bem acima do chão.
    assert flags_near_chance_regime(
        accuracies={"vanilla": 0.55, "mas": 0.62, "lr_control": 0.58},
        n_classes=10,
    ) is False


def test_near_chance_flag_uses_the_best_arm_not_the_mean() -> None:
    """Se QUALQUER arm escapou do chão, o regime não é degenerado por completo.

    O caso tem de ser construído onde média e máximo DISCORDAM, senão as duas
    implementações passam e o teste não testa nada. Com chance 0,10 o piso é
    0,20: aqui o máximo (0,25) está acima e a média (0,157) abaixo.

    Usar a média esconderia um desenho em que o tratamento funciona e os outros
    dois colapsaram — que seria um resultado, não um artefato.
    """

    from scripts.analyze_penalty_pass2 import flags_near_chance_regime

    accuracies = {"vanilla": 0.11, "mas": 0.25, "lr_control": 0.11}
    assert max(accuracies.values()) >= 0.20  # o melhor escapou
    assert sum(accuracies.values()) / len(accuracies) < 0.20  # a média, não
    assert flags_near_chance_regime(accuracies=accuracies, n_classes=10) is False


# --- Q2: o controle precisa ser um controle -------------------------------


def test_control_that_does_not_differ_from_vanilla_invalidates_the_design() -> None:
    """Q2: se `lr_control ≈ vanilla`, o pareamento não removeu plasticidade
    suficiente para importar, e o contraste primário não testa nada.

    Este é o modo de falha que o L3 existiu para evitar; a verificação fica
    aqui porque é no agregado que ela se torna visível.
    """

    from scripts.analyze_penalty_pass2 import control_is_distinguishable

    identical = [seed_record(vanilla_f=0.90, mas_f=0.80, control_f=0.90)] * 12
    distinct = [seed_record(vanilla_f=0.90, mas_f=0.80, control_f=0.85)] * 12

    assert control_is_distinguishable(identical) is False
    assert control_is_distinguishable(distinct) is True
