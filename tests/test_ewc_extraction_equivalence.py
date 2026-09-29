"""A extração do EWC não pode ter mudado o número que o runner produz.

`experiments/split_mnist.py` tinha sua própria cópia da penalidade quadrática,
do Fisher empírico e da consolidação. Elas foram substituídas por delegação a
`dual_heater.ewc`. Estes testes reimplementam a aritmética da versão histórica
LITERALMENTE e exigem igualdade — se a extração tivesse mudado um fator, uma
ordem de soma ou uma convenção de meio, aqui é onde apareceria.

Sem isto, a extração é uma refatoração não verificada em cima de um método que
serve de baseline para dois artigos.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dual_heater.ewc import ewc_penalty
from experiments.split_mnist import (
    _accumulate_empirical_fisher,
    _consolidate_ewc_importance,
    _parameter_penalty,
)


def _historical_parameter_penalty(model, importance, anchors):
    """Cópia literal da implementação anterior à extração."""

    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, parameter in model.named_parameters():
        if name in importance:
            penalty = penalty + (
                importance[name] * (parameter - anchors[name]).square()
            ).sum()
    return penalty


def _historical_consolidate(model, *, fisher_sum, fisher_examples, importance, anchors, decay):
    """Cópia literal, incluindo o antigo `max(1, n)`."""

    divisor = max(1, fisher_examples)
    for name, parameter in model.named_parameters():
        estimate = fisher_sum[name] / divisor
        if name in importance:
            estimate = decay * importance[name] + estimate
        importance[name] = estimate.detach().clone()
        anchors[name] = parameter.detach().clone()


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 4))


def test_penalty_matches_the_pre_extraction_implementation() -> None:
    model = _model()
    torch.manual_seed(1)
    anchors = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    importance = {n: torch.rand_like(p) for n, p in model.named_parameters()}

    assert float(_parameter_penalty(model, importance, anchors)) == pytest.approx(
        float(_historical_parameter_penalty(model, importance, anchors)), rel=1e-9
    )


def test_penalty_keeps_the_no_half_factor_convention() -> None:
    """O runner soma o próprio 0.5; a delegação não pode aplicá-lo duas vezes.

    Este é o erro de migração que mais fácil passa despercebido: a força
    efetiva de EWC cairia pela metade e o baseline ficaria fraco sem erro
    nenhum.
    """

    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    anchors = {"weight": torch.zeros(1, 3)}
    importance = {"weight": torch.ones(1, 3)}

    # Σ Ω (θ − θ*)² = 3 * 4 = 12, SEM meio.
    assert float(_parameter_penalty(model, importance, anchors)) == pytest.approx(12.0)
    # Enquanto o módulo, com strength=1, aplica o meio: 6.
    assert float(
        ewc_penalty(
            params=dict(model.named_parameters()),
            anchors=anchors,
            importance=importance,
            strength=1.0,
        )
    ) == pytest.approx(6.0)


def test_penalty_is_zero_without_importance_as_before() -> None:
    model = _model()
    penalty = _parameter_penalty(model, {}, {})
    assert isinstance(penalty, torch.Tensor)
    assert float(penalty) == pytest.approx(0.0)


def test_fisher_accumulation_matches_the_pre_extraction_implementation() -> None:
    model = _model(seed=2)
    torch.manual_seed(3)
    inputs = torch.randn(7, 6)
    targets = torch.randint(0, 4, (7,))

    mine = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    count = _accumulate_empirical_fisher(model, model(inputs), targets, mine)

    # Versão histórica, reimplementada aqui.
    named = tuple(model.named_parameters())
    params = tuple(p for _, p in named)
    historical = {n: torch.zeros_like(p) for n, p in named}
    logits = model(inputs)
    for index in range(len(targets)):
        loss = torch.nn.functional.cross_entropy(
            logits[index : index + 1], targets[index : index + 1], reduction="sum"
        )
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        for (name, _), grad in zip(named, grads, strict=True):
            if grad is not None:
                historical[name].add_(grad.detach().square())

    assert count == 7
    for name in historical:
        assert mine[name] == pytest.approx(historical[name], abs=1e-6)


def test_consolidation_matches_the_pre_extraction_implementation() -> None:
    model = _model(seed=4)
    torch.manual_seed(5)
    fisher = {n: torch.rand_like(p) for n, p in model.named_parameters()}

    mine_importance = {n: torch.rand_like(p) for n, p in model.named_parameters()}
    historical_importance = {n: v.clone() for n, v in mine_importance.items()}
    mine_anchors: dict[str, torch.Tensor] = {}
    historical_anchors: dict[str, torch.Tensor] = {}

    _consolidate_ewc_importance(
        model,
        fisher_sum=fisher,
        fisher_examples=13,
        importance=mine_importance,
        anchors=mine_anchors,
        decay=0.7,
    )
    _historical_consolidate(
        model,
        fisher_sum=fisher,
        fisher_examples=13,
        importance=historical_importance,
        anchors=historical_anchors,
        decay=0.7,
    )

    for name in historical_importance:
        assert mine_importance[name] == pytest.approx(
            historical_importance[name], abs=1e-7
        )
        assert torch.equal(mine_anchors[name], historical_anchors[name])


def test_zero_examples_now_raises_instead_of_silently_dividing_by_one() -> None:
    """Divergência DELIBERADA da versão histórica, documentada na extração.

    O antigo `max(1, n)` transformava "a estimação nunca rodou" em importância
    numericamente válida e silenciosamente errada. O módulo recusa.
    """

    model = _model()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    with pytest.raises(ValueError):
        _consolidate_ewc_importance(
            model,
            fisher_sum=fisher,
            fisher_examples=0,
            importance={},
            anchors={},
            decay=1.0,
        )


def test_penalty_still_flows_gradient_to_the_model() -> None:
    """A penalidade tem de chegar ao otimizador pelo autograd, como antes."""

    model = nn.Linear(4, 2)
    anchors = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    importance = {n: torch.ones_like(p) for n, p in model.named_parameters()}

    penalty = _parameter_penalty(model, importance, anchors)
    assert penalty.requires_grad
    penalty.backward()
    assert model.weight.grad is not None
    assert float(model.weight.grad.abs().sum()) > 0.0
