"""O agregador do L3 julga T-P1 a T-P4 a partir dos manifests.

Pré-registro: `goals/protocol_lambda_sweep.md` §E.

O agregador é onde um sweep mente com mais facilidade: ele decide o que conta
como "monótono", o que conta como "divergente" e o que conta como "poder". Cada
uma dessas três decisões está no protocolo, e cada uma tem teste aqui.
"""

from __future__ import annotations

import pytest


def cell(mean_e, *, divergent=False, n_non_finite=0):
    """Uma célula (arm, λ) como o agregador a recebe."""

    return {
        "norm_ratio": {
            "mean": mean_e,
            "divergent": divergent,
            "n_non_finite": n_non_finite,
        },
        "divergent": divergent,
        "has_power": (not divergent) and mean_e is not None and mean_e <= 0.90,
    }


# --- T-P1: monotonicidade --------------------------------------------------


def test_monotonic_decreasing_series_is_recognized() -> None:
    from scripts.analyze_lambda_sweep import judge_monotonicity

    verdict = judge_monotonicity([0.96, 0.93, 0.94 - 0.01, 0.87, 0.75])
    assert verdict["monotonic"] is True


def test_a_single_inversion_breaks_monotonicity() -> None:
    """Um ponto fora de ordem falsifica T-P1. Não existe 'quase monótono'."""

    from scripts.analyze_lambda_sweep import judge_monotonicity

    verdict = judge_monotonicity([0.96, 0.93, 0.95, 0.87])
    assert verdict["monotonic"] is False
    assert verdict["inversions"] == 1


def test_monotonicity_skips_divergent_cells_without_claiming_them() -> None:
    """§D/T8: uma célula divergente não testa monotonicidade, em nenhuma direção.

    Tratar `None` como 'o menor valor' faria qualquer série com uma divergência
    no fim parecer decrescente — que é exatamente o caso do `ewc` na calibração.
    """

    from scripts.analyze_lambda_sweep import judge_monotonicity

    verdict = judge_monotonicity([0.96, 0.93, None, None, None])
    assert verdict["monotonic"] is True
    assert verdict["n_compared"] == 2
    assert verdict["n_skipped"] == 3


def test_monotonicity_with_fewer_than_two_points_is_undecided() -> None:
    """Com um ponto só não há o que ser monótono — e dizer 'sim' seria inventar."""

    from scripts.analyze_lambda_sweep import judge_monotonicity

    assert judge_monotonicity([0.9])["monotonic"] is None
    assert judge_monotonicity([None, None])["monotonic"] is None


# --- T-P2: a fronteira de divergência --------------------------------------


def test_arm_is_called_divergent_when_most_seeds_diverge() -> None:
    """T-P2 fala em 'pelo menos metade das seeds'."""

    from scripts.analyze_lambda_sweep import divergence_rate

    cells = [cell(None, divergent=True, n_non_finite=9)] * 6 + [cell(0.9)] * 4
    rate = divergence_rate(cells)
    assert rate["n_divergent"] == 6
    assert rate["rate"] == pytest.approx(0.6)
    assert rate["majority"] is True


def test_a_minority_of_divergent_seeds_is_not_a_majority() -> None:
    from scripts.analyze_lambda_sweep import divergence_rate

    cells = [cell(None, divergent=True)] * 3 + [cell(0.9)] * 7
    assert divergence_rate(cells)["majority"] is False


def test_exactly_half_counts_as_at_least_half() -> None:
    """'pelo menos metade' inclui a metade exata; com 10 seeds isso é 5."""

    from scripts.analyze_lambda_sweep import divergence_rate

    cells = [cell(None, divergent=True)] * 5 + [cell(0.9)] * 5
    assert divergence_rate(cells)["majority"] is True


# --- T-P3/T-P4: poder ------------------------------------------------------


def test_cell_has_power_only_when_every_seed_agrees() -> None:
    """Uma célula com poder em 7 de 10 seeds não é uma célula com poder.

    O controle da passada 2 é construído com UM valor de lr_scale; se as seeds
    discordam sobre a célula ter poder, esse valor não existe.
    """

    from scripts.analyze_lambda_sweep import judge_power

    assert judge_power([cell(0.75)] * 10)["unanimous"] is True
    assert judge_power([cell(0.75)] * 7 + [cell(0.95)] * 3)["unanimous"] is False


def test_power_requires_non_divergence_even_with_a_low_mean() -> None:
    """O caso que o §D.2 existe para excluir, e o mais tentador de aceitar."""

    from scripts.analyze_lambda_sweep import judge_power

    verdict = judge_power([cell(0.4, divergent=True, n_non_finite=30)] * 10)
    assert verdict["unanimous"] is False
    assert verdict["n_with_power"] == 0


def test_power_verdict_reports_the_mean_only_over_sane_seeds() -> None:
    """A média do veredito ignora seeds divergentes — e o caso real é o
    perigoso: um arm divergente tem média FINITA, só que absurda.

    A calibração mediu `ewc` a 10x com média 2,31e9 e 29 amostras não-finitas.
    Incluir essa seed na média do veredito produziria um número que não é
    plasticidade de ninguém, e que arrastaria a média para longe de qualquer
    leitura possível. Usar `None` aqui não testaria nada: filtrar ou não daria
    o mesmo resultado.
    """

    from scripts.analyze_lambda_sweep import judge_power

    cells = [
        cell(0.80),
        cell(0.82),
        cell(2.31e9, divergent=True, n_non_finite=29),
    ]
    verdict = judge_power(cells)
    assert verdict["mean_e"] == pytest.approx(0.81)
    assert verdict["n_divergent"] == 1


def test_power_verdict_mean_ignores_a_seed_with_no_measurement() -> None:
    """E `None` continua sendo ausência de medida, não zero."""

    from scripts.analyze_lambda_sweep import judge_power

    cells = [cell(0.80), cell(0.82), cell(None, divergent=True, n_non_finite=5)]
    assert judge_power(cells)["mean_e"] == pytest.approx(0.81)


# --- o veredito final ------------------------------------------------------


def test_pass2_is_authorized_only_by_a_unanimous_powered_cell() -> None:
    """§E.1: T-P3 confirmada autoriza a passada 2 a EXISTIR, com pré-registro."""

    from scripts.analyze_lambda_sweep import authorizes_pass_two

    powered = {"mas": {30.0: {"unanimous": True, "mean_e": 0.87}}}
    assert authorizes_pass_two(powered)["authorized"] is True
    assert authorizes_pass_two(powered)["arm"] == "mas"
    assert authorizes_pass_two(powered)["multiplier"] == 30.0


def test_no_powered_cell_closes_pass_two() -> None:
    """§E.1: T-P3 falsificada MATA a passada 2, e isso é a saída mais forte."""

    from scripts.analyze_lambda_sweep import authorizes_pass_two

    verdict = authorizes_pass_two({"mas": {30.0: {"unanimous": False, "mean_e": 0.93}}})
    assert verdict["authorized"] is False


def test_the_weakest_powered_lambda_is_chosen_not_the_strongest() -> None:
    """Entre duas células com poder, a de menor λ é a menos distorcida.

    §E.2: mudar λ muda o método. Se 30x já dá poder, usar 100x afastaria o
    ponto de operação das forças publicadas sem ganho de desenho.
    """

    from scripts.analyze_lambda_sweep import authorizes_pass_two

    powered = {
        "mas": {
            100.0: {"unanimous": True, "mean_e": 0.75},
            30.0: {"unanimous": True, "mean_e": 0.87},
        }
    }
    assert authorizes_pass_two(powered)["multiplier"] == 30.0


def test_a_divergent_arm_never_authorizes_even_at_many_lambdas() -> None:
    from scripts.analyze_lambda_sweep import authorizes_pass_two

    powered = {
        "ewc": {
            10.0: {"unanimous": False, "mean_e": None},
            30.0: {"unanimous": False, "mean_e": None},
        }
    }
    assert authorizes_pass_two(powered)["authorized"] is False
