"""Baselines de consolidação por penalidade quadrática: EWC, SI e MAS.

Os três compartilham a forma `(λ/2) Σ Ω_i (θ_i − θ*_i)²` e diferem apenas em
como `Ω` é estimado:

| método | Ω | referência |
|---|---|---|
| EWC | Fisher empírico diagonal, por exemplo | Kirkpatrick et al., PNAS 2017 (arXiv 1612.00796) |
| SI  | caminho do gradiente / deslocamento² | Zenke et al., ICML 2017 |
| MAS | |∂‖f(x)‖²/∂θ|, sem rótulo | Aljundi et al., ECCV 2018 |

Isto existia dentro de `experiments/split_mnist.py`, acoplado àquele runner.
Extraído porque os hosts BERT e Qwen precisam do mesmo termo, e a terceira
cópia seria o começo da divergência entre eles. Ver
`docs/audits/baseline_inventory.md` para o inventário que motivou a extração.

**Convenção de fator, que difere entre os métodos na literatura e no runner
do MLP:** `ewc_penalty` já inclui o `1/2`. O runner do Split-MNIST somava
`0.5 * ewc_lambda * _parameter_penalty(...)` para EWC e `si_lambda *
_parameter_penalty(...)` para SI — ou seja, SI ali não tem o meio. Ao migrar um
runner, passe `strength` de forma que o produto final seja o mesmo, ou a força
efetiva muda silenciosamente.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "accumulate_empirical_fisher",
    "accumulate_mas_omega",
    "consolidate_importance",
    "ewc_penalty",
    "mas_penalty",
]


def ewc_penalty(
    *,
    params: Mapping[str, Tensor],
    anchors: Mapping[str, Tensor],
    importance: Mapping[str, Tensor],
    strength: float,
) -> Tensor | float:
    """Penalidade quadrática ponderada: `(λ/2) Σ Ω_i (θ_i − θ*_i)²`.

    Retorna `0.0` (float, não tensor) quando não há âncora — o caso da
    primeira tarefa, onde não existe nada a preservar.

    Parâmetros presentes em `params` mas ausentes de `anchors` são ignorados:
    uma cabeça de classificação que cresceu entre tarefas não tem âncora e não
    deve ser penalizada por ter surgido.
    """

    total: Tensor | None = None
    for name, anchor in anchors.items():
        if name not in params or name not in importance:
            continue
        drift = params[name] - anchor
        term = (importance[name] * drift.pow(2)).sum()
        total = term if total is None else total + term
    if total is None:
        return 0.0
    return 0.5 * strength * total


def mas_penalty(
    *,
    params: Mapping[str, Tensor],
    anchors: Mapping[str, Tensor],
    omega: Mapping[str, Tensor],
    strength: float,
) -> Tensor | float:
    """MAS usa a MESMA quadrática, com Ω no lugar do Fisher.

    Existe como nome próprio só para que o chamador leia o método que está
    usando; não há uma segunda fórmula aqui de propósito.
    """

    return ewc_penalty(
        params=params, anchors=anchors, importance=omega, strength=strength
    )


def accumulate_empirical_fisher(
    *,
    logits: Tensor,
    targets: Tensor,
    named_parameters: Sequence[tuple[str, Tensor]],
    accumulator: MutableMapping[str, Tensor],
) -> int:
    """Acumula `Σ_i grad(log p(y_i|x_i))²` — POR EXEMPLO.

    O erro clássico é elevar ao quadrado o gradiente da loss média do
    minibatch, que computa `(mean g)²` e não `mean(g²)`. Os dois diferem
    exatamente quando os gradientes dos exemplos se cancelam, que é o caso em
    que a importância é alta e a versão errada reporta zero.

    Retorna o número de exemplos acumulados, para que o chamador possa dividir
    na consolidação.
    """

    parameters = tuple(parameter for _, parameter in named_parameters)
    for index in range(len(targets)):
        sample_loss = F.cross_entropy(
            logits[index : index + 1],
            targets[index : index + 1],
            reduction="sum",
        )
        gradients = torch.autograd.grad(
            sample_loss, parameters, retain_graph=True, allow_unused=True
        )
        for (name, _), gradient in zip(named_parameters, gradients, strict=True):
            if gradient is None:
                continue
            accumulator[name] = accumulator.get(
                name, torch.zeros_like(gradient)
            ) + gradient.detach().pow(2)
    return int(len(targets))


def accumulate_mas_omega(
    *,
    outputs: Tensor,
    named_parameters: Sequence[tuple[str, Tensor]],
    accumulator: MutableMapping[str, Tensor],
) -> int:
    """Acumula `Σ_i |∂‖f(x_i)‖²/∂θ|` — a sensibilidade MAS.

    Diferente do Fisher, não usa rótulo: mede quanto a SAÍDA da rede muda com
    o parâmetro, o que torna MAS aplicável a dados não rotulados. Como a
    quantidade é uma magnitude, o valor acumulado é sempre não negativo.
    """

    parameters = tuple(parameter for _, parameter in named_parameters)
    for index in range(len(outputs)):
        squared_norm = outputs[index].pow(2).sum()
        gradients = torch.autograd.grad(
            squared_norm, parameters, retain_graph=True, allow_unused=True
        )
        for (name, _), gradient in zip(named_parameters, gradients, strict=True):
            if gradient is None:
                continue
            accumulator[name] = accumulator.get(
                name, torch.zeros_like(gradient)
            ) + gradient.detach().abs()
    return int(len(outputs))


def consolidate_importance(
    *,
    named_parameters: Iterable[tuple[str, Tensor]],
    accumulator: Mapping[str, Tensor],
    examples: int,
    importance: MutableMapping[str, Tensor],
    anchors: MutableMapping[str, Tensor],
    decay: float = 1.0,
) -> None:
    """Fecha uma tarefa: normaliza o acumulador, soma à memória, reancora.

    `decay=1.0` é a soma acumulada clássica do EWC online. Valores menores
    esquecem tarefas antigas progressivamente.

    `examples` precisa ser positivo: dividir por zero aqui significa que a
    passada de estimação nunca rodou, que é bug de wiring e não um caso a
    tolerar silenciosamente com um `max(1, n)`.
    """

    if examples <= 0:
        raise ValueError(
            "examples deve ser > 0; zero significa que a estimação não rodou"
        )
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay deve estar em [0, 1]")

    for name, parameter in named_parameters:
        if name in accumulator:
            estimate = accumulator[name] / examples
            if name in importance:
                estimate = decay * importance[name] + estimate
            importance[name] = estimate.detach().clone()
        anchors[name] = parameter.detach().clone()
