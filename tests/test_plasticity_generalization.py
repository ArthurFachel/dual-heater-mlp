"""Plasticidade efetiva para métodos de penalidade.

Pré-registro: `goals/protocol_plasticity_generalization.md`.

O lema que torna LoRA e EWC comensuráveis: quando a modificação é um
escalonamento diagonal do update, `plasticity_ratio()` reduz ao `E` da máscara.
Se este teste quebra, os dois hosts não estão na mesma escala e a tabela
comparativa do artigo é inválida.
"""

from __future__ import annotations

import pytest
import torch


def test_ratio_is_one_without_penalty() -> None:
    from dual_heater.plasticity import plasticity_ratio

    delta = {"w": torch.tensor([1.0, -2.0, 3.0])}
    assert plasticity_ratio(native=delta, unpenalized=delta) == pytest.approx(1.0)


def test_ratio_reduces_to_the_mask_mean_for_diagonal_scaling() -> None:
    """O lema (G1/§B.1). Uma máscara [1, 0.5, 0] tem E = 0.5; o ratio dá 0.5."""

    from dual_heater.plasticity import plasticity_ratio

    unpenalized = {"w": torch.tensor([2.0, 2.0, 2.0])}
    mask = torch.tensor([1.0, 0.5, 0.0])
    native = {"w": unpenalized["w"] * mask}

    assert plasticity_ratio(native=native, unpenalized=unpenalized) == pytest.approx(0.5)


def test_ratio_ignores_parameters_with_zero_unpenalized_update() -> None:
    """G3: divisão por zero não é plasticidade zero, é ausência de evidência."""

    from dual_heater.plasticity import plasticity_ratio

    native = {"w": torch.tensor([1.0, 0.0])}
    unpenalized = {"w": torch.tensor([2.0, 0.0])}
    # Só o primeiro elemento é informativo: 1/2 = 0.5
    assert plasticity_ratio(native=native, unpenalized=unpenalized) == pytest.approx(0.5)


def test_frozen_parameters_count_as_zero_plasticity() -> None:
    """G5: o erro que originou este trabalho. Congelado é E=0, não 'fora da média'."""

    from dual_heater.plasticity import plasticity_ratio

    native = {"a": torch.zeros(4), "b": torch.ones(4)}
    unpenalized = {"a": torch.ones(4), "b": torch.ones(4)}
    assert plasticity_ratio(native=native, unpenalized=unpenalized) == pytest.approx(0.5)


def test_mean_is_per_element_not_per_tensor() -> None:
    """G4. Tensores de tamanhos diferentes: as duas médias divergem aqui.

    Por elemento: (6*1.0 + 2*0.0)/8 = 0.75
    Por tensor:   (1.0 + 0.0)/2      = 0.50
    """

    from dual_heater.plasticity import plasticity_ratio

    native = {"grande": torch.ones(6), "pequeno": torch.zeros(2)}
    unpenalized = {"grande": torch.ones(6), "pequeno": torch.ones(2)}
    assert plasticity_ratio(native=native, unpenalized=unpenalized) == pytest.approx(0.75)


def test_ratio_uses_magnitude_not_signed_value() -> None:
    """Um update que inverte de sinal teve sua magnitude preservada, não negada."""

    from dual_heater.plasticity import plasticity_ratio

    native = {"w": torch.tensor([-2.0])}
    unpenalized = {"w": torch.tensor([2.0])}
    assert plasticity_ratio(native=native, unpenalized=unpenalized) == pytest.approx(1.0)


def test_ratio_rejects_mismatched_keys() -> None:
    """Chaves diferentes significam superfícies diferentes: não são comparáveis."""

    from dual_heater.plasticity import plasticity_ratio

    with pytest.raises(ValueError):
        plasticity_ratio(native={"a": torch.ones(2)}, unpenalized={"b": torch.ones(2)})


def test_ratio_rejects_an_empty_surface() -> None:
    """Superfície vazia não tem plasticidade 1.0: não tem plasticidade."""

    from dual_heater.plasticity import plasticity_ratio

    with pytest.raises(ValueError):
        plasticity_ratio(native={}, unpenalized={})


def test_ratio_returns_none_when_no_element_is_informative() -> None:
    """Todos os denominadores zero: sem evidência, e dizer isso é obrigatório."""

    from dual_heater.plasticity import plasticity_ratio

    native = {"w": torch.zeros(3)}
    unpenalized = {"w": torch.zeros(3)}
    assert plasticity_ratio(native=native, unpenalized=unpenalized) is None


def test_norm_ratio_diagnostic() -> None:
    """D1 do protocolo."""

    from dual_heater.plasticity import norm_ratio

    native = {"w": torch.tensor([3.0, 4.0])}      # norma 5
    unpenalized = {"w": torch.tensor([6.0, 8.0])}  # norma 10
    assert norm_ratio(native=native, unpenalized=unpenalized) == pytest.approx(0.5)


def test_direction_cosine_is_one_under_diagonal_scaling() -> None:
    """D2: escalonamento diagonal POSITIVO uniforme não gira o update."""

    from dual_heater.plasticity import direction_cosine

    unpenalized = {"w": torch.tensor([1.0, 2.0, 3.0])}
    native = {"w": unpenalized["w"] * 0.4}
    assert direction_cosine(native=native, unpenalized=unpenalized) == pytest.approx(1.0)


def test_direction_cosine_detects_rotation() -> None:
    """O número que distingue 'removeu plasticidade' de 'girou o update'."""

    from dual_heater.plasticity import direction_cosine

    unpenalized = {"w": torch.tensor([1.0, 0.0])}
    native = {"w": torch.tensor([0.0, 1.0])}
    assert direction_cosine(native=native, unpenalized=unpenalized) == pytest.approx(0.0)


def test_shadow_step_measures_a_diagonal_mask_exactly() -> None:
    """Caso de resposta conhecida: sem penalidade -> ratio == 1.

    É o guarda-chuva do risco R5 do roadmap: antes de acreditar num varrido que
    diz que nenhum método sobrevive ao pareamento, o instrumento tem de acertar
    o caso em que a resposta é sabida.
    """

    import torch.nn as nn

    from dual_heater.plasticity import measure_shadow_step

    torch.manual_seed(0)
    model = nn.Linear(4, 3, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))

    def base_loss() -> torch.Tensor:
        return torch.nn.functional.cross_entropy(model(inputs), targets)

    report = measure_shadow_step(
        model=model,
        optimizer=optimizer,
        penalized_loss=base_loss,
        unpenalized_loss=base_loss,
    )
    # Sem penalidade os dois fluxos coincidem: ratio exatamente 1.
    assert report["plasticity_ratio"] == pytest.approx(1.0)
    assert report["direction_cosine"] == pytest.approx(1.0)
    assert report["norm_ratio"] == pytest.approx(1.0)


def test_shadow_step_detects_a_penalty_that_shrinks_the_update() -> None:
    """Uma penalidade quadrática ativa tem de baixar o ratio abaixo de 1."""

    import torch.nn as nn

    from dual_heater.ewc import ewc_penalty
    from dual_heater.plasticity import measure_shadow_step

    torch.manual_seed(1)
    model = nn.Linear(4, 3, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))

    anchors = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    importance = {n: torch.ones_like(p) for n, p in model.named_parameters()}

    def unpenalized() -> torch.Tensor:
        return torch.nn.functional.cross_entropy(model(inputs), targets)

    def penalized() -> torch.Tensor:
        return unpenalized() + ewc_penalty(
            params=dict(model.named_parameters()),
            anchors=anchors,
            importance=importance,
            strength=10.0,
        )

    report = measure_shadow_step(
        model=model,
        optimizer=optimizer,
        penalized_loss=penalized,
        unpenalized_loss=unpenalized,
    )
    assert report["plasticity_ratio"] is not None
    assert report["plasticity_ratio"] != pytest.approx(1.0)


def test_shadow_step_leaves_training_state_untouched() -> None:
    """A medição não pode alterar a trajetória que está sendo medida.

    Se o passo-sombra consumir o estado do otimizador ou deixar os parâmetros
    deslocados, o piloto mede uma run diferente da que reporta.
    """

    import copy

    import torch.nn as nn

    from dual_heater.plasticity import measure_shadow_step

    torch.manual_seed(2)
    model = nn.Linear(4, 3, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))

    def loss() -> torch.Tensor:
        return torch.nn.functional.cross_entropy(model(inputs), targets)

    # Um passo real para que o otimizador tenha estado não trivial.
    optimizer.zero_grad()
    loss().backward()
    optimizer.step()

    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    state_before = copy.deepcopy(optimizer.state_dict())

    measure_shadow_step(
        model=model, optimizer=optimizer, penalized_loss=loss, unpenalized_loss=loss
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), params_before[name]), (
            f"{name} foi deslocado pela medição"
        )
    state_after = optimizer.state_dict()
    for group_index, entries in state_before["state"].items():
        for key, value in entries.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, state_after["state"][group_index][key]), (
                    f"estado do otimizador ({key}) mudou durante a medição"
                )


def test_shadow_step_restores_pending_gradients() -> None:
    """A medição não pode consumir o `.grad` acumulado pelo chamador.

    Divergência deliberada do plano: `optimizer.zero_grad(set_to_none=True)`
    dentro do passo-sombra apaga uma acumulação em andamento, e o passo real
    seguinte aplicaria um gradiente diferente do que teria aplicado. Restaurar
    parâmetros e estado do otimizador não basta — o buffer de gradiente é um
    terceiro pedaço de estado de treino.

    O fixture usa batches DIFERENTES para o gradiente pendente e para a
    medição. Com o mesmo batch nos dois, o passo-sombra recomputaria por acaso
    o mesmo gradiente e o teste passaria mesmo sem restauração alguma — foi
    assim que a primeira versão deste teste sobreviveu à mutação M6.
    """

    import torch.nn as nn

    from dual_heater.plasticity import measure_shadow_step

    torch.manual_seed(3)
    model = nn.Linear(4, 3, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pending_inputs = torch.randn(8, 4)
    pending_targets = torch.randint(0, 3, (8,))
    shadow_inputs = torch.randn(8, 4) + 5.0
    shadow_targets = torch.randint(0, 3, (8,))

    def shadow_loss() -> torch.Tensor:
        return torch.nn.functional.cross_entropy(model(shadow_inputs), shadow_targets)

    # Gradiente pendente de OUTRO batch, como no meio de uma acumulação.
    optimizer.zero_grad()
    torch.nn.functional.cross_entropy(model(pending_inputs), pending_targets).backward()
    grads_before = {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    measure_shadow_step(
        model=model,
        optimizer=optimizer,
        penalized_loss=shadow_loss,
        unpenalized_loss=shadow_loss,
    )

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name}: gradiente pendente foi descartado"
        assert torch.equal(parameter.grad, grads_before[name]), (
            f"{name}: gradiente pendente foi alterado pela medição"
        )


def test_the_ratio_is_not_bounded_above_by_one_for_an_additive_penalty() -> None:
    """A propriedade que o §B do pré-registro NÃO garante, medida e fixada.

    O lema §B.1 dá `ratio = média(m) <= 1` quando a intervenção é um
    escalonamento diagonal. Uma penalidade aditiva não é: ela soma
    `λ Ω (θ − θ*)` ao gradiente, e num elemento onde o gradiente da loss e o
    da penalidade têm sinais OPOSTOS o update penalizado pode ter magnitude
    MAIOR que o não penalizado. O ratio daquele elemento passa de 1 e a média
    sobe junto.

    Isto não é bug nem do instrumento nem do wiring: é a diferença entre as
    duas famílias de intervenção, que é exatamente o que o protocolo do
    instrumento diz não ser único (§D.1). Fica fixado por teste para que
    ninguém "conserte" a métrica clampando em 1 — o clamp esconderia a
    evidência de que a intervenção não é diagonal, que é o que `D2` existe
    para reportar.
    """

    from dual_heater.plasticity import plasticity_ratio

    # Elemento 0: penalidade na mesma direção do update -> encolhe.
    # Elemento 1: penalidade na direção oposta -> AUMENTA a magnitude.
    unpenalized = {"w": torch.tensor([2.0, 2.0])}
    native = {"w": torch.tensor([1.0, 5.0])}

    ratio = plasticity_ratio(native=native, unpenalized=unpenalized)
    assert ratio == pytest.approx((0.5 + 2.5) / 2)
    assert ratio > 1.0


def test_the_cosine_stays_one_when_the_ratio_exceeds_one_by_scaling() -> None:
    """Separa "girou" de "cresceu": D2 é 1 mesmo com ratio > 1.

    Um update uniformemente AMPLIFICADO continua sendo escalonamento diagonal
    (com `m > 1`), e o cosseno reporta 1. É o par de números que o artigo
    precisa: `ratio` sozinho não distingue amplificação de rotação.
    """

    from dual_heater.plasticity import direction_cosine, plasticity_ratio

    unpenalized = {"w": torch.tensor([1.0, 2.0, 3.0])}
    native = {"w": unpenalized["w"] * 1.5}

    assert plasticity_ratio(native=native, unpenalized=unpenalized) == pytest.approx(1.5)
    assert direction_cosine(native=native, unpenalized=unpenalized) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# O que a primária G1' (norm_ratio) garante, e o que ela PERDE em relação à
# razão por elemento. Emenda de 2026-09-29 ao protocolo, §K linha 2.
# ---------------------------------------------------------------------------


def test_norm_ratio_reduces_to_the_mask_mean_for_a_UNIFORM_mask() -> None:
    """O lema §B.1 sobrevive à emenda, mas SÓ para máscara uniforme.

    Com `m_i = c` para todo i, `‖m ⊙ Δ‖ / ‖Δ‖ = c = média(m)`, qualquer que
    seja a distribuição de `Δ`. É o caso do `lr_control`, que é exatamente o
    braço contra o qual o pareamento é feito — por isso a emenda preserva a
    comensurabilidade que motivou o instrumento.
    """

    from dual_heater.plasticity import norm_ratio

    # Denominador deliberadamente heterogêneo: o lema não depende dele.
    unpenalized = {"w": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    for scale in (0.25, 0.5, 0.85, 1.0):
        native = {"w": unpenalized["w"] * scale}
        assert norm_ratio(native=native, unpenalized=unpenalized) == pytest.approx(
            scale
        ), f"máscara uniforme {scale}"


def test_norm_ratio_does_NOT_equal_the_mask_mean_for_a_heterogeneous_mask() -> None:
    """O custo da emenda, medido e fixado.

    `norm_ratio` é uma média QUADRÁTICA ponderada pela magnitude do update,
    não a média aritmética da máscara. Sob máscara heterogênea os dois números
    divergem, e a divergência não é pequena: máscara [1,1,0,0] sobre um
    denominador homogêneo dá `sqrt(0.5) = 0.707`, não `0.5`.

    Consequência declarada no protocolo (§B, emenda): a métrica primária deixa
    de ser comensurável com o `E` de `surface_plasticity()` para máscaras
    heterogêneas de LoRA. Ela continua comensurável com um `lr_control`, que é
    o pareamento que o piloto da Fase 2.4 realmente faz. Este teste existe
    para que ninguém afirme a equivalência forte por engano.
    """

    from dual_heater.plasticity import norm_ratio, plasticity_ratio

    unpenalized = {"w": torch.tensor([2.0, 2.0, 2.0, 2.0])}
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    native = {"w": unpenalized["w"] * mask}

    # A razão por elemento acerta a média da máscara...
    assert plasticity_ratio(native=native, unpenalized=unpenalized) == pytest.approx(0.5)
    # ...e a razão de normas dá a RMS, que é sqrt(1/2).
    assert norm_ratio(native=native, unpenalized=unpenalized) == pytest.approx(
        0.5**0.5, abs=1e-6
    )


def test_norm_ratio_is_bounded_by_one_when_the_penalty_only_shrinks() -> None:
    """A propriedade que motiva a emenda: um `lr_scale` utilizável.

    A razão por elemento é ilimitada acima (ver o teste do denominador
    pequeno), o que produziu `E = 1,78` na calibração e um `lr_scale > 1` —
    que não é um controle de plasticidade, é um aumento de learning rate.
    `surface_plasticity()` recusa `lr_scale` fora de [0, 1] por isso.

    A razão de normas não tem esse defeito quando o update encolhe: ela é
    exatamente ‖·‖ menor sobre ‖·‖ maior.
    """

    from dual_heater.plasticity import norm_ratio

    # Um único elemento com denominador minúsculo domina a média por elemento
    # e não move a norma.
    unpenalized = {"w": torch.tensor([1.0, 1.0, 1e-9])}
    native = {"w": torch.tensor([0.9, 0.9, 1e-3])}

    from dual_heater.plasticity import plasticity_ratio

    per_element = plasticity_ratio(native=native, unpenalized=unpenalized)
    assert per_element > 100.0, "o defeito que a emenda corrige"

    norms = norm_ratio(native=native, unpenalized=unpenalized)
    assert norms is not None
    assert 0.0 < norms <= 1.0
    assert norms == pytest.approx(0.9, abs=1e-3)


def test_norm_ratio_can_exceed_one_when_the_penalty_amplifies() -> None:
    """A emenda não promete um limite superior universal, e não deve prometer.

    Se a penalidade de fato AMPLIFICA o update — possível, porque ela é
    aditiva e não um escalonamento — a razão de normas passa de 1 e isso é a
    medição correta, não um artefato. O protocolo declara que, nesse caso, o
    `lr_control` daquele método não é construtível e o contraste é reportado
    como não-pareável em vez de pareado na marra.
    """

    from dual_heater.plasticity import norm_ratio

    unpenalized = {"w": torch.tensor([1.0, 1.0])}
    native = {"w": torch.tensor([2.0, 2.0])}

    assert norm_ratio(native=native, unpenalized=unpenalized) == pytest.approx(2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="precisa de CUDA")
def test_shadow_step_gives_the_same_answer_on_cuda_as_on_cpu() -> None:
    """Testes só-CPU não veem bug de colocação de device.

    O passo-sombra clona parâmetros, faz `deepcopy` do `state_dict` do
    otimizador e restaura os dois. Um tensor de estado que volte para o device
    errado passa despercebido na CPU e quebra (ou pior, silenciosamente muda o
    resultado) na primeira run de GPU. Este teste exige IGUALDADE numérica
    entre os dois devices, não apenas ausência de exceção.
    """

    import torch.nn as nn

    from dual_heater.ewc import ewc_penalty
    from dual_heater.plasticity import measure_shadow_step

    def run(device: str) -> dict[str, float | None]:
        torch.manual_seed(7)
        model = nn.Linear(16, 8, bias=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.randn(32, 16, generator=torch.Generator().manual_seed(1)).to(
            device
        )
        targets = torch.randint(
            0, 8, (32,), generator=torch.Generator().manual_seed(2)
        ).to(device)

        anchors = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        importance = {n: torch.ones_like(p) for n, p in model.named_parameters()}

        def unpenalized() -> torch.Tensor:
            return torch.nn.functional.cross_entropy(model(inputs), targets)

        def penalized() -> torch.Tensor:
            return unpenalized() + ewc_penalty(
                params=dict(model.named_parameters()),
                anchors=anchors,
                importance=importance,
                strength=100.0,
            )

        # Um passo real primeiro, para o otimizador ter estado não trivial.
        optimizer.zero_grad()
        penalized().backward()
        optimizer.step()

        return measure_shadow_step(
            model=model,
            optimizer=optimizer,
            penalized_loss=penalized,
            unpenalized_loss=unpenalized,
        )

    on_cpu = run("cpu")
    on_cuda = run("cuda")

    for key in ("plasticity_ratio", "norm_ratio", "direction_cosine"):
        assert on_cuda[key] == pytest.approx(on_cpu[key], rel=1e-4), (
            f"{key}: CUDA={on_cuda[key]} difere de CPU={on_cpu[key]}"
        )
