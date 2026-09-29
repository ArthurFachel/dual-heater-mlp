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
