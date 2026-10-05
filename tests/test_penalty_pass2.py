"""Passada 2 — o `lr_control` pareado em plasticidade, sob SGD.

Pré-registro: `goals/protocol_penalty_pass2.md`.

Esta é a primeira passada da Fase 2 que lê acurácia, e o arm novo (`lr_control`)
tem um modo de falha silencioso conhecido: o `lr_scale` não chegar no
otimizador. O arm aparece na tabela, produz números plausíveis, e é `vanilla`
com outro nome. Esse bug exato já aconteceu duas vezes neste repo
(`importance_criterion` não propagado; `learning_rate_scale` não chegando ao
`lr_control` do host LoRA).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# --- P6/D.1: a banda -------------------------------------------------------


def test_pass2_band_has_twelve_distinct_seeds() -> None:
    from experiments.confirmatory_split_mnist import PENALTY_PASS2_SEEDS

    assert len(PENALTY_PASS2_SEEDS) == 12
    assert len(set(PENALTY_PASS2_SEEDS)) == 12


def test_pass2_band_is_disjoint_from_every_band_already_spent() -> None:
    """§D.1, incluindo a banda do L3 — que é a mais recente e a mais fácil de
    colidir, porque é a que forneceu o escalar deste desenho."""

    from experiments.bert_slowheat_diagnostic import (
        IMPORTANCE_CRITERION_ABLATION_SEEDS,
    )
    from experiments.confirmatory_split_mnist import (
        CONFIRMATORY_SEEDS,
        DECLARED_EXPLORATORY_SEEDS,
        DERPP_CONFIRMATORY_SEEDS,
        LAMBDA_SWEEP_SEEDS,
        PENALTY_PASS2_SEEDS,
        PENALTY_REEVALUATION_SEEDS,
        SGD_PLASTICITY_SEEDS,
    )
    from experiments.qwen_lora_slowheat import (
        EXACT_DECOMPOSITION_SEEDS,
        PLASTICITY_MATCHED_SEEDS,
    )
    from experiments.replay_selection_sweep import (
        REPLAY_SELECTOR_CONFIRMATORY_SEEDS,
    )

    seeds = set(PENALTY_PASS2_SEEDS)
    for name, band in (
        ("confirmatory", CONFIRMATORY_SEEDS),
        ("exploratory", DECLARED_EXPLORATORY_SEEDS),
        ("derpp", DERPP_CONFIRMATORY_SEEDS),
        ("replay selector", REPLAY_SELECTOR_CONFIRMATORY_SEEDS),
        ("criterion ablation", IMPORTANCE_CRITERION_ABLATION_SEEDS),
        ("exact decomposition", EXACT_DECOMPOSITION_SEEDS),
        ("plasticity matched", PLASTICITY_MATCHED_SEEDS),
        ("penalty reevaluation", PENALTY_REEVALUATION_SEEDS),
        ("sgd plasticity (L2)", SGD_PLASTICITY_SEEDS),
        ("lambda sweep (L3)", LAMBDA_SWEEP_SEEDS),
    ):
        overlap = seeds & set(band)
        assert not overlap, f"banda da passada 2 colide com {name}: {sorted(overlap)}"


# --- §C.1: o escalar congelado --------------------------------------------


def test_frozen_lr_scale_matches_the_protocol_to_the_last_digit() -> None:
    """§C.1. Este número foi lido do L3 ANTES de qualquer acurácia.

    Recalculá-lo depois de ver um endpoint invalidaria a passada inteira, e a
    forma de tornar isso visível é pinar os 16 dígitos aqui.
    """

    from scripts.run_penalty_pass2 import FROZEN_LR_SCALE

    assert FROZEN_LR_SCALE == 0.8541056558955461


def test_frozen_lr_scale_is_the_mean_of_the_L3_manifests() -> None:
    """O escalar tem de ser reproduzível a partir dos artefatos do L3.

    Se alguém o editar à mão, este teste quebra — que é o ponto.
    """

    import glob
    import json
    import statistics

    paths = sorted(glob.glob("results/lambda_sweep/mult_30/seed_*.json"))
    if len(paths) != 10:
        pytest.skip("artefatos do L3 ausentes neste checkout")

    from scripts.run_penalty_pass2 import FROZEN_LR_SCALE

    measured = statistics.fmean(
        json.loads(Path(path).read_text(encoding="utf-8"))["arms"]["mas"][
            "norm_ratio"
        ]["mean"]
        for path in paths
    )
    assert measured == pytest.approx(FROZEN_LR_SCALE, abs=1e-15)


def test_frozen_lr_scale_is_below_the_L3_power_threshold() -> None:
    """Coerência com o §D.2 do L3: um escalar acima de 0,90 não teria poder."""

    from scripts.run_penalty_pass2 import FROZEN_LR_SCALE

    assert FROZEN_LR_SCALE <= 0.90


# --- o arm novo: o `lr_scale` TEM de chegar no otimizador ------------------


def test_lr_control_is_a_registered_method() -> None:
    from experiments.split_mnist import SplitMNISTConfig

    SplitMNISTConfig(methods=("lr_control",)).validate()


def test_lr_control_actually_scales_the_optimizer_learning_rate() -> None:
    """O bug silencioso, como teste direto.

    Um `lr_control` que não escala o lr é `vanilla` com outro nome: ele roda,
    produz acurácia plausível, entra na tabela e torna o contraste primário uma
    comparação do método contra si mesmo.
    """

    from torch import nn

    from experiments.split_mnist import SplitMNISTConfig, _build_optimizer

    config = SplitMNISTConfig(
        optimizer="sgd", learning_rate=3e-3, learning_rate_scale=0.5
    )
    optimizer = _build_optimizer("lr_control", nn.Linear(4, 2), config)

    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(1.5e-3)


def test_the_scale_applies_only_to_lr_control() -> None:
    """`vanilla` e `mas` rodam no lr base; escalar os três não seria controle."""

    from torch import nn

    from experiments.split_mnist import SplitMNISTConfig, _build_optimizer

    config = SplitMNISTConfig(
        optimizer="sgd", learning_rate=3e-3, learning_rate_scale=0.5
    )
    for method in ("vanilla", "mas"):
        optimizer = _build_optimizer(method, nn.Linear(4, 2), config)
        for group in optimizer.param_groups:
            assert group["lr"] == pytest.approx(3e-3), (
                f"o escalar vazou para o arm `{method}`"
            )


def test_lr_control_applies_no_penalty() -> None:
    """O controle remove plasticidade SEM mecanismo de consolidação.

    Se ele acumulasse importância, seria um segundo MAS com lr menor, e o
    contraste primário mediria a diferença entre duas forças do mesmo método.
    """

    from experiments.split_mnist import _method_spec

    spec = _method_spec("lr_control")
    assert spec is not None
    for attribute in ("ewc", "si", "mas", "slowheat", "replay", "distillation"):
        assert not getattr(spec, attribute, False), (
            f"`lr_control` nao pode ter {attribute}"
        )


def test_default_scale_is_one_so_frozen_protocols_do_not_move() -> None:
    """O campo novo nao pode alterar nenhuma run existente."""

    from experiments.split_mnist import SplitMNISTConfig

    assert SplitMNISTConfig().learning_rate_scale == 1.0


def test_scale_field_stays_out_of_frozen_payloads() -> None:
    """Mesma convenção de `mas_lambda` e `optimizer`: o campo só entra no
    payload quando sai do default, senão move o sha256 de pré-registros
    fechados e a tentação vira atualizar o hash esperado."""

    from experiments.split_mnist import SplitMNISTConfig, config_payload

    assert "learning_rate_scale" not in config_payload(SplitMNISTConfig())
    assert (
        config_payload(SplitMNISTConfig(learning_rate_scale=0.5))[
            "learning_rate_scale"
        ]
        == 0.5
    )


def test_scale_must_be_in_the_open_unit_interval() -> None:
    """Um escalar > 1 é aumento de learning rate, não controle de plasticidade —
    exatamente o que a regra G8 recusa."""

    from experiments.split_mnist import SplitMNISTConfig

    with pytest.raises(ValueError):
        SplitMNISTConfig(learning_rate_scale=1.5).validate()
    with pytest.raises(ValueError):
        SplitMNISTConfig(learning_rate_scale=0.0).validate()


# --- P1-P5: a configuração da run -----------------------------------------


def test_config_pins_the_three_declared_arms() -> None:
    from scripts.run_penalty_pass2 import ARMS

    assert ARMS == ("vanilla", "mas", "lr_control")


def test_config_pins_sgd_and_the_declared_learning_rate() -> None:
    """P1 e P2. Mudar o lr base moveria o `E` e invalidaria o escalar (§G.2)."""

    from scripts.run_penalty_pass2 import build_config

    config = build_config(seed=9500011)
    config.validate()
    assert config.optimizer == "sgd"
    assert config.learning_rate == pytest.approx(3e-3)


def test_config_pins_the_inflated_mas_strength() -> None:
    """P3: `mas_lambda = 30`, que é 30x o publicado por Hsu. O `E` congelado
    foi medido NESSA forca; rodar noutra descreveria outro experimento."""

    from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
    from scripts.run_penalty_pass2 import build_config

    config = build_config(seed=9500011)
    assert config.mas_lambda == pytest.approx(
        PUBLISHED_PENALTY_STRENGTHS["mas_lambda"] * 30
    )


def test_config_carries_the_frozen_scale() -> None:
    from scripts.run_penalty_pass2 import FROZEN_LR_SCALE, build_config

    assert build_config(seed=9500011).learning_rate_scale == FROZEN_LR_SCALE


def test_shadow_step_is_off_in_pass_two() -> None:
    """§F: o `E` já foi medido no L3. Remedi-lo custaria 3 passos por passo e
    nao entra em nenhum julgamento desta passada."""

    from scripts.run_penalty_pass2 import build_config

    assert build_config(seed=9500011).plasticity_sampling_interval == 0


# --- P7-P11: o teste estatístico ------------------------------------------


def test_primary_endpoint_is_forgetting_not_accuracy() -> None:
    """P7. A acurácia média final é secundária e vai na tabela, mas o primário
    é esquecimento — trocar os dois depois de ver os números seria escolher."""

    from scripts.run_penalty_pass2 import PRIMARY_ENDPOINT

    assert PRIMARY_ENDPOINT == "average_forgetting"


def test_primary_contrast_is_mas_minus_lr_control() -> None:
    """P8. `mas - vanilla` é secundário: é a comparação que a literatura faz, e
    o ponto do desenho é que ela nao é a comparação certa."""

    from scripts.run_penalty_pass2 import PRIMARY_CONTRAST

    assert PRIMARY_CONTRAST == ("mas", "lr_control")


def test_family_has_exactly_one_comparison() -> None:
    """P9: com m=1 a correção de Holm é a identidade. Declarado para que
    ninguém a aplique depois para afrouxar um limiar."""

    from scripts.run_penalty_pass2 import CONFIRMATORY_FAMILY

    assert len(CONFIRMATORY_FAMILY) == 1


def test_twelve_unanimous_seeds_clear_the_threshold() -> None:
    """P11: o piso com n=12 é 0,00049, duas ordens abaixo de 0,05."""

    from experiments.confirmatory_statistics import exact_two_sided_sign_test

    result = exact_two_sided_sign_test([-0.05] * 12)
    p = result["p_value"] if isinstance(result, dict) else result
    assert p == pytest.approx(2.0 ** -11, rel=1e-9)
    assert p < 0.05


def test_a_nine_of_twelve_split_is_reported_as_null() -> None:
    """§G.4, como teste: esta passada nao distingue 'sem efeito' de 'efeito
    pequeno', e o limite tem de ser visível em vez de descoberto depois."""

    from experiments.confirmatory_statistics import exact_two_sided_sign_test

    result = exact_two_sided_sign_test([-0.05] * 9 + [0.05] * 3)
    p = result["p_value"] if isinstance(result, dict) else result
    assert p > 0.05


# --- D.2: a acurácia final é obrigatória ----------------------------------


def test_accuracy_is_reported_for_every_arm() -> None:
    """§D.2: um método pode reduzir esquecimento colapsando a aquisição.

    Sem a acurácia final ao lado, um `mas` que simplesmente aprende menos
    pareceria estar preservando conhecimento.
    """

    from scripts.run_penalty_pass2 import REPORTED_METRICS

    assert "average_forgetting" in REPORTED_METRICS
    assert "final_average_accuracy" in REPORTED_METRICS


def test_metrics_are_read_from_the_nested_metrics_dict() -> None:
    """`run_split_mnist` aninha os endpoints sob `"metrics"`, não na raiz.

    O smoke de 1 seed pegou isto como `KeyError`, que é a falha barulhenta e
    desejável. O teste existe para que a correção não regrida silenciosamente
    para um `.get(metric, 0.0)`, que gravaria zeros plausíveis em 12 manifests
    e só seria notado na hora de interpretar o contraste.
    """

    from scripts.run_penalty_pass2 import extract_metrics

    arm_result = {
        "metrics": {
            "average_forgetting": 0.25,
            "final_average_accuracy": 0.61,
            "backward_transfer": -0.1,
        },
        # Chaves homônimas na raiz: se o runner ler daqui, os números saem
        # errados sem levantar exceção nenhuma.
        "average_forgetting": 999.0,
        "final_average_accuracy": 999.0,
    }
    extracted = extract_metrics(arm_result)
    assert extracted["average_forgetting"] == pytest.approx(0.25)
    assert extracted["final_average_accuracy"] == pytest.approx(0.61)


def test_a_missing_endpoint_raises_instead_of_defaulting() -> None:
    """Um endpoint ausente tem de explodir, não virar zero."""

    from scripts.run_penalty_pass2 import extract_metrics

    with pytest.raises(KeyError):
        extract_metrics({"metrics": {"final_average_accuracy": 0.61}})
