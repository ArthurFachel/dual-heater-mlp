"""Baselines de penalidade quadrática: EWC e MAS como módulo próprio.

O termo `Σ Ω_i (θ_i − θ*_i)²` existia dentro de `experiments/split_mnist.py`,
acoplado àquele runner. Os hosts BERT e Qwen precisam do mesmo termo, e uma
segunda cópia seria o começo da divergência entre eles.

O teste que mais importa aqui não é o da fórmula: é o de que a penalidade
morde. O modo de falha silencioso é o método rodar com `strength` que nunca
chega na loss — aparece na tabela, produz números plausíveis, e é vanilla com
outro nome. Esse bug já mordeu este projeto duas vezes.
"""

from __future__ import annotations

import pytest
import torch


def test_ewc_penalty_is_zero_on_the_first_task() -> None:
    """Sem tarefa anterior não há âncora, logo não há o que penalizar."""

    from dual_heater.ewc import ewc_penalty

    assert ewc_penalty(params={}, anchors={}, importance={}, strength=1.0) == pytest.approx(0.0)


def test_ewc_penalty_grows_quadratically_with_drift() -> None:
    """Dobrar o drift quadruplica a penalidade."""

    from dual_heater.ewc import ewc_penalty

    anchors = {"w": torch.zeros(4)}
    importance = {"w": torch.ones(4)}

    near = ewc_penalty(
        params={"w": torch.full((4,), 0.1)},
        anchors=anchors,
        importance=importance,
        strength=1.0,
    )
    far = ewc_penalty(
        params={"w": torch.full((4,), 0.2)},
        anchors=anchors,
        importance=importance,
        strength=1.0,
    )
    assert far == pytest.approx(4.0 * near, rel=1e-5)


def test_ewc_penalty_respects_importance_weighting() -> None:
    """Um parâmetro com importância zero não contribui, por mais que drifte."""

    from dual_heater.ewc import ewc_penalty

    penalty = ewc_penalty(
        params={"w": torch.tensor([10.0, 10.0])},
        anchors={"w": torch.zeros(2)},
        importance={"w": torch.tensor([1.0, 0.0])},
        strength=1.0,
    )
    # Só o primeiro elemento conta: 0.5 * 1.0 * 10^2 = 50
    assert penalty == pytest.approx(50.0)


def test_ewc_penalty_scales_linearly_with_strength() -> None:
    """Se strength não multiplica, o knob não chega na loss."""

    from dual_heater.ewc import ewc_penalty

    kwargs = {
        "params": {"w": torch.ones(3)},
        "anchors": {"w": torch.zeros(3)},
        "importance": {"w": torch.ones(3)},
    }
    weak = ewc_penalty(**kwargs, strength=1.0)
    strong = ewc_penalty(**kwargs, strength=7.0)
    assert strong == pytest.approx(7.0 * weak)


def test_ewc_penalty_ignores_parameters_without_an_anchor() -> None:
    """Uma cabeça que cresceu entre tarefas não tem âncora e não é penalizada."""

    from dual_heater.ewc import ewc_penalty

    penalty = ewc_penalty(
        params={"old": torch.ones(2), "new_head": torch.full((5,), 100.0)},
        anchors={"old": torch.zeros(2)},
        importance={"old": torch.ones(2)},
        strength=1.0,
    )
    assert penalty == pytest.approx(1.0)  # 0.5 * 1 * (1^2 + 1^2)


def test_ewc_penalty_is_differentiable_with_respect_to_the_parameter() -> None:
    """Uma penalidade sem grad_fn não chega ao otimizador."""

    from dual_heater.ewc import ewc_penalty

    weight = torch.full((3,), 2.0, requires_grad=True)
    penalty = ewc_penalty(
        params={"w": weight},
        anchors={"w": torch.zeros(3)},
        importance={"w": torch.ones(3)},
        strength=1.0,
    )
    assert penalty.requires_grad
    penalty.backward()
    # d/dw de 0.5 * w^2 é w
    assert weight.grad == pytest.approx(torch.full((3,), 2.0))


def test_ewc_penalty_is_zero_at_the_anchor() -> None:
    """No ponto âncora a penalidade some, qualquer que seja a importância."""

    from dual_heater.ewc import ewc_penalty

    anchor = torch.randn(6)
    penalty = ewc_penalty(
        params={"w": anchor.clone()},
        anchors={"w": anchor},
        importance={"w": torch.rand(6) * 100.0},
        strength=9.0,
    )
    assert float(penalty) == pytest.approx(0.0, abs=1e-6)


def test_mas_omega_is_the_gradient_of_the_squared_output_norm() -> None:
    """MAS mede sensibilidade da saída, não do loss — não precisa de rótulo."""

    from dual_heater.ewc import accumulate_mas_omega

    weight = torch.eye(3, requires_grad=True)
    inputs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    omega: dict[str, torch.Tensor] = {"w": torch.zeros(3, 3)}

    count = accumulate_mas_omega(
        outputs=inputs @ weight,
        named_parameters=[("w", weight)],
        accumulator=omega,
    )

    assert count == 2
    # L = ||xW||^2, dL/dW = 2 x^T (xW). Para W=I e x=e_0: 2*e_0^T e_0.
    # MAS acumula |grad| por exemplo.
    assert omega["w"][0, 0] == pytest.approx(2.0)
    assert omega["w"][1, 1] == pytest.approx(8.0)  # x=2*e_1 -> 2*2*2
    assert omega["w"][2, 2] == pytest.approx(0.0)


def test_mas_omega_is_nonnegative() -> None:
    """Omega é magnitude; um valor negativo significa que faltou o abs."""

    from dual_heater.ewc import accumulate_mas_omega

    weight = torch.randn(4, 4, requires_grad=True)
    inputs = torch.randn(5, 4)
    omega = {"w": torch.zeros(4, 4)}
    accumulate_mas_omega(
        outputs=inputs @ weight,
        named_parameters=[("w", weight)],
        accumulator=omega,
    )
    assert bool((omega["w"] >= 0.0).all())
    assert float(omega["w"].sum()) > 0.0


def test_mas_reuses_the_same_penalty_as_ewc() -> None:
    """DRY: MAS é a mesma quadrática com Ω no lugar do Fisher."""

    from dual_heater.ewc import ewc_penalty, mas_penalty

    args = {
        "params": {"w": torch.full((3,), 0.5)},
        "anchors": {"w": torch.zeros(3)},
        "strength": 2.0,
    }
    omega = {"w": torch.tensor([1.0, 2.0, 3.0])}
    assert mas_penalty(**args, omega=omega) == pytest.approx(
        float(ewc_penalty(**args, importance=omega))
    )


def test_consolidate_averages_omega_over_examples() -> None:
    """Sem dividir pelo número de exemplos, Ω escala com o tamanho do dataset."""

    from dual_heater.ewc import consolidate_importance

    importance: dict[str, torch.Tensor] = {}
    anchors: dict[str, torch.Tensor] = {}
    consolidate_importance(
        named_parameters=[("w", torch.full((2,), 5.0))],
        accumulator={"w": torch.tensor([10.0, 20.0])},
        examples=10,
        importance=importance,
        anchors=anchors,
        decay=1.0,
    )
    assert importance["w"] == pytest.approx(torch.tensor([1.0, 2.0]))
    assert anchors["w"] == pytest.approx(torch.full((2,), 5.0))


def test_consolidate_accumulates_across_tasks_with_decay() -> None:
    """A importância da tarefa 2 soma à da tarefa 1, descontada por decay."""

    from dual_heater.ewc import consolidate_importance

    importance = {"w": torch.tensor([4.0])}
    anchors = {"w": torch.tensor([0.0])}
    consolidate_importance(
        named_parameters=[("w", torch.tensor([1.0]))],
        accumulator={"w": torch.tensor([2.0])},
        examples=1,
        importance=importance,
        anchors=anchors,
        decay=0.5,
    )
    assert importance["w"] == pytest.approx(torch.tensor([0.5 * 4.0 + 2.0]))
    # A âncora é reancorada no ponto atual, não mantida no antigo.
    assert anchors["w"] == pytest.approx(torch.tensor([1.0]))


def test_consolidate_rejects_a_zero_example_count() -> None:
    """Dividir por zero exemplos é bug de wiring, não um caso a tolerar."""

    from dual_heater.ewc import consolidate_importance

    with pytest.raises(ValueError):
        consolidate_importance(
            named_parameters=[("w", torch.zeros(1))],
            accumulator={"w": torch.zeros(1)},
            examples=0,
            importance={},
            anchors={},
            decay=1.0,
        )


def test_empirical_fisher_is_per_example_not_squared_batch_mean() -> None:
    """mean(g²) e (mean g)² diferem; a segunda esconde importância por
    cancelamento de gradiente. É a auditoria que o protocolo exige."""

    from dual_heater.ewc import accumulate_empirical_fisher

    # Dois exemplos com gradientes opostos: a média é zero, mas cada um
    # individualmente é informativo. Um Fisher (mean g)² daria zero.
    weight = torch.zeros(2, requires_grad=True)
    logits = torch.stack([weight, -weight])
    targets = torch.tensor([0, 1])

    accumulator = {"w": torch.zeros(2)}
    count = accumulate_empirical_fisher(
        logits=logits,
        targets=targets,
        named_parameters=[("w", weight)],
        accumulator=accumulator,
    )
    assert count == 2
    assert float(accumulator["w"].sum()) > 0.0
