"""Banda de seeds e métricas novas do protocolo L2 (SGD).

Pré-registro: `goals/protocol_sgd_plasticity.md`.

Duas coisas precisam ser verdadeiras antes da primeira seed: a banda é disjunta
de tudo já gasto (S6), e as métricas que testam as predições C2–C4 medem o que
o §B derivou. A segunda parte é a que importa mais — a derivação é uma predição
pontual declarada antes da run, e um instrumento que não a implementa
corretamente não pode falsificá-la.
"""

from __future__ import annotations

import pytest
import torch


def test_sgd_seed_band_has_twelve_distinct_seeds() -> None:
    from experiments.confirmatory_split_mnist import SGD_PLASTICITY_SEEDS

    assert len(SGD_PLASTICITY_SEEDS) == 12
    assert len(set(SGD_PLASTICITY_SEEDS)) == 12


def test_sgd_seed_band_is_disjoint_from_every_band_already_spent() -> None:
    """§D.1. Toda banda já gasta, verificada por import — não por inspeção."""

    from experiments.bert_slowheat_diagnostic import (
        IMPORTANCE_CRITERION_ABLATION_SEEDS,
    )
    from experiments.confirmatory_split_mnist import (
        CONFIRMATORY_SEEDS,
        DECLARED_EXPLORATORY_SEEDS,
        DERPP_CONFIRMATORY_SEEDS,
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

    seeds = set(SGD_PLASTICITY_SEEDS)
    for name, band in (
        ("confirmatory", CONFIRMATORY_SEEDS),
        ("exploratory", DECLARED_EXPLORATORY_SEEDS),
        ("derpp", DERPP_CONFIRMATORY_SEEDS),
        ("replay selector", REPLAY_SELECTOR_CONFIRMATORY_SEEDS),
        ("criterion ablation", IMPORTANCE_CRITERION_ABLATION_SEEDS),
        ("exact decomposition", EXACT_DECOMPOSITION_SEEDS),
        ("plasticity matched", PLASTICITY_MATCHED_SEEDS),
        ("penalty reevaluation", PENALTY_REEVALUATION_SEEDS),
    ):
        overlap = seeds & set(band)
        assert not overlap, f"banda L2 colide com {name}: {sorted(overlap)}"


def test_sgd_seed_band_matches_the_frozen_protocol() -> None:
    """Os valores do §D.1, literais. Mudar a banda muda o pré-registro."""

    from experiments.confirmatory_split_mnist import SGD_PLASTICITY_SEEDS

    assert tuple(SGD_PLASTICITY_SEEDS) == (
        8000011, 8025013, 8050021, 8075027, 8100037, 8125043,
        8150053, 8175059, 8200063, 8225069, 8250077, 8275081,
    )


# --- as métricas que testam a derivação do §B -----------------------------


def test_penalty_alignment_is_the_projection_on_the_loss_gradient() -> None:
    """`g·p / ‖g‖²`, o termo que decide o sinal de `E − 1`."""

    from dual_heater.plasticity import penalty_alignment

    loss_grad = {"w": torch.tensor([3.0, 4.0])}      # ‖g‖² = 25
    penalty_grad = {"w": torch.tensor([3.0, 4.0])}   # g·p = 25
    assert penalty_alignment(
        loss_grad=loss_grad, penalty_grad=penalty_grad
    ) == pytest.approx(1.0)

    opposed = {"w": torch.tensor([-1.5, -2.0])}      # g·p = -12.5
    assert penalty_alignment(
        loss_grad=loss_grad, penalty_grad=opposed
    ) == pytest.approx(-0.5)


def test_penalty_scale_is_the_norm_ratio() -> None:
    from dual_heater.plasticity import penalty_scale

    assert penalty_scale(
        loss_grad={"w": torch.tensor([3.0, 4.0])},
        penalty_grad={"w": torch.tensor([6.0, 8.0])},
    ) == pytest.approx(2.0)


def test_predicts_above_one_matches_the_derivation() -> None:
    """C3: `E > 1` ⟺ `2 g·p + ‖p‖² > 0`.

    Fronteira exata: com `p = −g`, temos `2 g·p + ‖p‖² = −2‖g‖² + ‖g‖² < 0`,
    e de fato `E = ‖g − g‖/‖g‖ = 0 < 1`.
    """

    from dual_heater.plasticity import predicts_above_one

    loss_grad = {"w": torch.tensor([1.0, 0.0])}

    # Ortogonal: 2*0 + 1 > 0 -> prevê E > 1, e de fato ‖(1,1)‖/‖(1,0)‖ = 1.41
    assert predicts_above_one(
        loss_grad=loss_grad, penalty_grad={"w": torch.tensor([0.0, 1.0])}
    ) is True
    # Anti-alinhado forte: p = -g
    assert predicts_above_one(
        loss_grad=loss_grad, penalty_grad={"w": torch.tensor([-1.0, 0.0])}
    ) is False
    # Anti-alinhado fraco: p = -0.25 g -> 2(-0.25) + 0.0625 < 0 -> E < 1
    assert predicts_above_one(
        loss_grad=loss_grad, penalty_grad={"w": torch.tensor([-0.25, 0.0])}
    ) is False


def test_the_factor_of_two_in_the_criterion_is_load_bearing() -> None:
    """O coeficiente 2 de `2 g·p + ‖p‖²` não é decorativo.

    As duas fórmulas — com fator 2 e com fator 1 — discordam exatamente quando
    `−‖p‖² < g·p < −‖p‖²/2`. Sem um caso nessa janela, trocar 2 por 1 passa em
    todos os outros testes, e o critério fica errado numa faixa estreita mas
    real de anti-alinhamento moderado.

    Construção: `‖p‖ = 1`, `g·p = −0,75`. Então
      correto (fator 2): 2(−0,75) + 1 = −0,5 < 0  -> prevê E < 1
      errado  (fator 1):  1(−0,75) + 1 = +0,25 > 0 -> preveria E > 1
    E o `E` verdadeiro confirma o correto.
    """

    from dual_heater.plasticity import norm_ratio, predicts_above_one

    # g = (1, 0); p = (-0.75, sqrt(1 - 0.5625)) tem ‖p‖ = 1 e g·p = -0.75
    loss_grad = {"w": torch.tensor([1.0, 0.0])}
    penalty_grad = {"w": torch.tensor([-0.75, (1.0 - 0.5625) ** 0.5])}

    assert float(penalty_grad["w"].norm()) == pytest.approx(1.0)
    assert float(torch.dot(penalty_grad["w"], loss_grad["w"])) == pytest.approx(-0.75)

    native = {"w": -(loss_grad["w"] + penalty_grad["w"])}
    unpenalized = {"w": -loss_grad["w"]}
    measured = norm_ratio(native=native, unpenalized=unpenalized)

    # A verdade: E < 1 nesta janela.
    assert measured < 1.0
    assert predicts_above_one(loss_grad=loss_grad, penalty_grad=penalty_grad) is False


@pytest.mark.parametrize("gp", [-0.95, -0.85, -0.75, -0.65, -0.55])
def test_criterion_is_exact_across_the_disagreement_window(gp: float) -> None:
    """Varre a janela inteira onde as duas fórmulas discordam.

    Com `‖p‖ = 1`, a janela é `−1 < g·p < −0,5`. Em toda ela a resposta certa
    é `E < 1`, e só a fórmula com o fator 2 acerta.
    """

    from dual_heater.plasticity import norm_ratio, predicts_above_one

    loss_grad = {"w": torch.tensor([1.0, 0.0])}
    penalty_grad = {"w": torch.tensor([gp, (1.0 - gp * gp) ** 0.5])}

    native = {"w": -(loss_grad["w"] + penalty_grad["w"])}
    unpenalized = {"w": -loss_grad["w"]}

    measured = norm_ratio(native=native, unpenalized=unpenalized)
    predicted = predicts_above_one(loss_grad=loss_grad, penalty_grad=penalty_grad)
    assert (measured > 1.0) is predicted
    assert predicted is False  # toda a janela tem E < 1


@pytest.mark.parametrize("seed", range(8))
def test_prediction_agrees_with_the_measured_norm_ratio_under_sgd(seed: int) -> None:
    """C3 contra a definição: o sinal previsto casa com o `E` de fato calculado.

    Este é o teste que amarra a derivação ao instrumento. Sob SGD,
    `Δ = −lr (g + p)`, então `norm_ratio` é exatamente `‖g+p‖/‖g‖`.
    """

    from dual_heater.plasticity import norm_ratio, predicts_above_one

    torch.manual_seed(seed)
    loss_grad = {"w": torch.randn(200)}
    penalty_grad = {"w": torch.randn(200) * 0.3}

    learning_rate = 0.05
    native = {"w": -learning_rate * (loss_grad["w"] + penalty_grad["w"])}
    unpenalized = {"w": -learning_rate * loss_grad["w"]}

    measured = norm_ratio(native=native, unpenalized=unpenalized)
    predicted = predicts_above_one(loss_grad=loss_grad, penalty_grad=penalty_grad)
    assert (measured > 1.0) is predicted


@pytest.mark.parametrize("seed", range(8))
def test_identity_E_times_cos_equals_one_plus_alignment(seed: int) -> None:
    """C2: `E · cos = 1 + g·p/‖g‖²`, exato sob SGD.

    Se esta identidade falhar, ou o instrumento está errado ou o update não é
    `−lr(g+p)` — e as duas possibilidades precisam ser distinguidas antes de
    interpretar qualquer número da run.
    """

    from dual_heater.plasticity import direction_cosine, norm_ratio, penalty_alignment

    torch.manual_seed(100 + seed)
    loss_grad = {"w": torch.randn(300)}
    penalty_grad = {"w": torch.randn(300) * 0.4}

    learning_rate = 0.01
    native = {"w": -learning_rate * (loss_grad["w"] + penalty_grad["w"])}
    unpenalized = {"w": -learning_rate * loss_grad["w"]}

    left = norm_ratio(native=native, unpenalized=unpenalized) * direction_cosine(
        native=native, unpenalized=unpenalized
    )
    right = 1.0 + penalty_alignment(loss_grad=loss_grad, penalty_grad=penalty_grad)
    assert left == pytest.approx(right, abs=1e-4)


def test_a_random_penalty_usually_increases_the_norm() -> None:
    """A afirmação do §B.1, que é o coração da predição C1.

    Uma perturbação independente do gradiente produz `E > 1` na maioria das
    vezes. Se este teste falhar, a leitura de que `E > 1` é o caso genérico
    está errada e o protocolo precisa ser relido antes da run.
    """

    from dual_heater.plasticity import norm_ratio

    torch.manual_seed(0)
    above = 0
    trials = 400
    for _ in range(trials):
        loss_grad = torch.randn(500)
        penalty_grad = torch.randn(500) * 0.1
        native = {"w": -(loss_grad + penalty_grad)}
        unpenalized = {"w": -loss_grad}
        if norm_ratio(native=native, unpenalized=unpenalized) > 1.0:
            above += 1
    assert above / trials > 0.7, f"apenas {above}/{trials} acima de 1"


def test_metrics_reject_a_zero_loss_gradient() -> None:
    """Denominador zero: sem evidência, e dizer isso é obrigatório."""

    from dual_heater.plasticity import penalty_alignment, penalty_scale

    zero = {"w": torch.zeros(4)}
    some = {"w": torch.ones(4)}
    assert penalty_alignment(loss_grad=zero, penalty_grad=some) is None
    assert penalty_scale(loss_grad=zero, penalty_grad=some) is None


def test_config_accepts_a_pure_sgd_optimizer() -> None:
    """S1: sem esse seletor o L2 não é executável — tudo roda AdamW."""

    from experiments.split_mnist import SplitMNISTConfig

    config = SplitMNISTConfig(optimizer="sgd")
    config.validate()
    assert config.optimizer == "sgd"


def test_config_defaults_to_adamw() -> None:
    """O default não pode mudar: protocolos congelados dependem dele."""

    from experiments.split_mnist import SplitMNISTConfig

    assert SplitMNISTConfig().optimizer == "adamw"


def test_config_rejects_an_unknown_optimizer() -> None:
    from experiments.split_mnist import SplitMNISTConfig

    with pytest.raises(ValueError):
        SplitMNISTConfig(optimizer="rmsprop").validate()


def test_optimizer_field_stays_out_of_frozen_payloads() -> None:
    """Adicionar o campo não pode mover o sha256 de um pré-registro fechado.

    Mesma convenção do `mas_lambda`: o campo só entra no payload quando sai do
    default. Sem isso, `tests/test_confirmatory_statistics.py` fica vermelho e
    a tentação é atualizar o hash esperado — o que reescreveria um
    pré-registro.
    """

    from experiments.split_mnist import SplitMNISTConfig, config_payload

    assert "optimizer" not in config_payload(SplitMNISTConfig())
    assert config_payload(SplitMNISTConfig(optimizer="sgd"))["optimizer"] == "sgd"


def test_sgd_optimizer_is_actually_sgd_and_has_no_momentum() -> None:
    """S1 exige SGD puro: momentum reintroduz o estado que o L2 quer remover."""

    import torch.nn as nn

    from experiments.split_mnist import SplitMNISTConfig, _build_optimizer

    model = nn.Linear(4, 2)
    optimizer = _build_optimizer("ewc", model, SplitMNISTConfig(optimizer="sgd"))

    assert isinstance(optimizer, torch.optim.SGD)
    for group in optimizer.param_groups:
        assert group["momentum"] == 0.0, "momentum reintroduz estado do otimizador"
        assert group["weight_decay"] == 0.0, (
            "weight decay soma outro termo ao update e confunde `p`"
        )


def test_adamw_remains_the_optimizer_when_not_requested() -> None:
    import torch.nn as nn

    from experiments.split_mnist import SplitMNISTConfig, _build_optimizer

    optimizer = _build_optimizer("ewc", nn.Linear(4, 2), SplitMNISTConfig())
    assert isinstance(optimizer, torch.optim.AdamW)


# --- o passo-sombra tem de EMITIR as métricas novas (S8) ------------------


def test_shadow_step_emits_the_gradient_metrics() -> None:
    """S8: sem estes campos no manifest, C2–C4 não são testáveis offline.

    O passo-sombra já computa os dois gradientes; não emiti-los obriga a
    re-rodar a experiência inteira para responder uma pergunta que os dados
    já continham.
    """

    import torch.nn as nn

    from dual_heater.plasticity import measure_shadow_step

    torch.manual_seed(3)
    model = nn.Linear(5, 3, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    inputs = torch.randn(8, 5)
    targets = torch.randint(0, 3, (8,))
    anchors = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    importance = {n: torch.ones_like(p) for n, p in model.named_parameters()}

    def unpenalized() -> torch.Tensor:
        return torch.nn.functional.cross_entropy(model(inputs), targets)

    def penalized() -> torch.Tensor:
        from dual_heater.ewc import ewc_penalty

        return unpenalized() + ewc_penalty(
            params=dict(model.named_parameters()),
            anchors=anchors,
            importance=importance,
            strength=5.0,
        )

    sample = measure_shadow_step(
        model=model,
        optimizer=optimizer,
        penalized_loss=penalized,
        unpenalized_loss=unpenalized,
    )

    for field in ("penalty_alignment", "penalty_scale", "predicted_above_one"):
        assert field in sample, f"campo {field} ausente do passo-sombra"
    assert isinstance(sample["predicted_above_one"], bool)
    assert sample["penalty_scale"] > 0.0


def test_shadow_step_prediction_agrees_with_its_own_measurement_under_sgd() -> None:
    """C3 medido de ponta a ponta pelo próprio instrumento, sob SGD.

    É o teste que fecha o laço: o passo-sombra prevê o sinal de `E − 1` a
    partir dos gradientes e mede `E` a partir dos updates, e os dois têm de
    concordar. Um desacordo sob SGD significa que o update não é `−lr(g+p)` —
    o que invalidaria a derivação do §B antes de a run começar.
    """

    import torch.nn as nn

    from dual_heater.ewc import ewc_penalty
    from dual_heater.plasticity import measure_shadow_step

    for seed in range(6):
        torch.manual_seed(seed)
        model = nn.Linear(6, 4, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        inputs = torch.randn(10, 6)
        targets = torch.randint(0, 4, (10,))
        anchors = {n: torch.randn_like(p) for n, p in model.named_parameters()}
        importance = {n: torch.rand_like(p) for n, p in model.named_parameters()}

        def unpenalized() -> torch.Tensor:
            return torch.nn.functional.cross_entropy(model(inputs), targets)

        def penalized() -> torch.Tensor:
            return unpenalized() + ewc_penalty(
                params=dict(model.named_parameters()),
                anchors=anchors,
                importance=importance,
                strength=20.0,
            )

        sample = measure_shadow_step(
            model=model,
            optimizer=optimizer,
            penalized_loss=penalized,
            unpenalized_loss=unpenalized,
        )
        measured_above = sample["norm_ratio"] > 1.0
        assert measured_above is sample["predicted_above_one"], (
            f"seed {seed}: E={sample['norm_ratio']:.6f} mas "
            f"predição={sample['predicted_above_one']}"
        )


def test_shadow_step_identity_holds_under_sgd() -> None:
    """C2 de ponta a ponta: `E · cos = 1 + alignment`, sob SGD."""

    import torch.nn as nn

    from dual_heater.ewc import ewc_penalty
    from dual_heater.plasticity import measure_shadow_step

    torch.manual_seed(42)
    model = nn.Linear(8, 5, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(12, 8)
    targets = torch.randint(0, 5, (12,))
    anchors = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    importance = {n: torch.rand_like(p) for n, p in model.named_parameters()}

    def unpenalized() -> torch.Tensor:
        return torch.nn.functional.cross_entropy(model(inputs), targets)

    def penalized() -> torch.Tensor:
        return unpenalized() + ewc_penalty(
            params=dict(model.named_parameters()),
            anchors=anchors,
            importance=importance,
            strength=15.0,
        )

    sample = measure_shadow_step(
        model=model,
        optimizer=optimizer,
        penalized_loss=penalized,
        unpenalized_loss=unpenalized,
    )
    left = sample["norm_ratio"] * sample["direction_cosine"]
    right = 1.0 + sample["penalty_alignment"]
    assert left == pytest.approx(right, abs=1e-4)


def test_shadow_step_identity_is_violated_under_adamw() -> None:
    """A ressalva do §D.2, como teste: sob AdamW a identidade NÃO vale.

    Isto não é um defeito — é o motivo de o protocolo declarar o contrafactual
    como local. Se este teste um dia passar a valer sob AdamW, a leitura do §B
    precisa ser revista, e é melhor descobrir por um teste vermelho do que por
    uma interpretação errada de uma run.
    """

    import torch.nn as nn

    from dual_heater.ewc import ewc_penalty
    from dual_heater.plasticity import measure_shadow_step

    torch.manual_seed(7)
    model = nn.Linear(8, 5, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.randn(12, 8)
    targets = torch.randint(0, 5, (12,))
    anchors = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    importance = {n: torch.rand_like(p) for n, p in model.named_parameters()}

    def unpenalized() -> torch.Tensor:
        return torch.nn.functional.cross_entropy(model(inputs), targets)

    def penalized() -> torch.Tensor:
        return unpenalized() + ewc_penalty(
            params=dict(model.named_parameters()),
            anchors=anchors,
            importance=importance,
            strength=15.0,
        )

    # Um passo real para o AdamW ter estado não trivial.
    optimizer.zero_grad()
    penalized().backward()
    optimizer.step()

    sample = measure_shadow_step(
        model=model,
        optimizer=optimizer,
        penalized_loss=penalized,
        unpenalized_loss=unpenalized,
    )
    left = sample["norm_ratio"] * sample["direction_cosine"]
    right = 1.0 + sample["penalty_alignment"]
    assert left != pytest.approx(right, abs=1e-4), (
        "a identidade do SGD passou a valer sob AdamW — reler o §B/§D.2"
    )
