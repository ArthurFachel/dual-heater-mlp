"""Plasticidade efetiva generalizada para métodos de penalidade.

Pré-registro: `goals/protocol_plasticity_generalization.md` (congelado antes
desta implementação).

`surface_plasticity()` em `lora_slowheat.py` mede a fração do update preservada
quando a intervenção é um escalonamento diagonal — máscara ou congelamento de
LoRA. Métodos de penalidade (EWC/SI/MAS) somam `λ Ω (θ − θ*)` ao gradiente e
podem GIRAR o update, não apenas encolhê-lo.

As funções aqui medem a razão entre o update aplicado e o update que teria sido
aplicado sem a penalidade, a partir do mesmo estado. O lema que torna os dois
hosts comensuráveis: sob escalonamento diagonal, `plasticity_ratio()` reduz
exatamente à média da máscara.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor

__all__ = [
    "direction_cosine",
    "measure_shadow_step",
    "norm_ratio",
    "plasticity_ratio",
]


def _validate(
    native: Mapping[str, Tensor], unpenalized: Mapping[str, Tensor]
) -> None:
    if set(native) != set(unpenalized):
        raise ValueError(
            "native e unpenalized descrevem superfícies diferentes: "
            f"{sorted(set(native) ^ set(unpenalized))}"
        )
    if not native:
        raise ValueError("superfície vazia não tem plasticidade definida")
    for name, tensor in native.items():
        if tensor.shape != unpenalized[name].shape:
            raise ValueError(f"{name}: formas incompatíveis")


def plasticity_ratio(
    *,
    native: Mapping[str, Tensor],
    unpenalized: Mapping[str, Tensor],
    eps: float = 0.0,
) -> float | None:
    """Razão média POR ELEMENTO entre o update aplicado e o não penalizado.

    Métrica primária G1. Retorna ``None`` quando nenhum elemento é informativo
    (todos os denominadores zero) — ausência de evidência precisa ser dizível,
    e 1.0 ou 0.0 seriam as duas leituras erradas.

    Elementos com ``|Δ_unpenalized| <= eps`` são excluídos (G3). Parâmetros
    congelados têm ``Δ_native = 0`` com denominador não nulo e contribuem com
    zero (G5) — o defeito R3 que esta função existe para não repetir.
    """

    _validate(native, unpenalized)

    numerators = torch.cat([native[name].detach().abs().flatten() for name in sorted(native)])
    denominators = torch.cat(
        [unpenalized[name].detach().abs().flatten() for name in sorted(native)]
    )

    informative = denominators > eps
    if not bool(informative.any()):
        return None

    ratios = numerators[informative] / denominators[informative]
    return float(ratios.mean())


def norm_ratio(
    *, native: Mapping[str, Tensor], unpenalized: Mapping[str, Tensor]
) -> float | None:
    """Diagnóstica D1: ``‖Δ_native‖ / ‖Δ_unpenalized‖``. Ignora direção."""

    _validate(native, unpenalized)

    native_norm = torch.cat(
        [native[name].detach().flatten() for name in sorted(native)]
    ).norm()
    unpenalized_norm = torch.cat(
        [unpenalized[name].detach().flatten() for name in sorted(native)]
    ).norm()
    if float(unpenalized_norm) == 0.0:
        return None
    return float(native_norm / unpenalized_norm)


def direction_cosine(
    *, native: Mapping[str, Tensor], unpenalized: Mapping[str, Tensor]
) -> float | None:
    """Diagnóstica D2: ``cos(Δ_native, Δ_unpenalized)``.

    Exatamente 1.0 sob escalonamento diagonal positivo. É o número que separa
    "removeu plasticidade" de "girou o update", e sem ele parear um método de
    penalidade contra um `lr_control` não é justificável.
    """

    _validate(native, unpenalized)

    flat_native = torch.cat([native[name].detach().flatten() for name in sorted(native)])
    flat_unpenalized = torch.cat(
        [unpenalized[name].detach().flatten() for name in sorted(native)]
    )
    if float(flat_native.norm()) == 0.0 or float(flat_unpenalized.norm()) == 0.0:
        return None
    return float(
        torch.dot(flat_native, flat_unpenalized)
        / (flat_native.norm() * flat_unpenalized.norm())
    )


def measure_shadow_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    penalized_loss: Callable[[], Tensor],
    unpenalized_loss: Callable[[], Tensor],
) -> dict[str, float | None]:
    """Mede a plasticidade de UM passo sem perturbar o treino.

    Executa o passo duas vezes a partir do mesmo estado (parâmetros e estado do
    otimizador), uma com a penalidade e outra sem, e restaura tudo ao final.
    Custa 3× um passo normal, por isso deve ser amostrado (G7).

    Sob AdamW o resultado é um contrafactual LOCAL: responde "dado este estado,
    qual seria este passo sem a penalidade", não "qual teria sido a trajetória".
    É a ressalva D.2 do pré-registro e vai no artigo.
    """

    import copy

    params_0 = {name: p.detach().clone() for name, p in model.named_parameters()}
    state_0 = copy.deepcopy(optimizer.state_dict())
    grads_0 = {
        name: (None if p.grad is None else p.grad.detach().clone())
        for name, p in model.named_parameters()
    }

    def run(loss_fn: Callable[[], Tensor]) -> dict[str, Tensor]:
        optimizer.zero_grad(set_to_none=True)
        loss_fn().backward()
        optimizer.step()
        delta = {
            name: (p.detach() - params_0[name]).clone()
            for name, p in model.named_parameters()
        }
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(params_0[name])
        optimizer.load_state_dict(copy.deepcopy(state_0))
        return delta

    try:
        native = run(penalized_loss)
        unpenalized = run(unpenalized_loss)
    finally:
        # Restaura também os gradientes: o chamador pode estar no meio de uma
        # acumulação, e consumi-la aqui mudaria a run que estamos medindo.
        for name, parameter in model.named_parameters():
            saved = grads_0[name]
            parameter.grad = None if saved is None else saved.clone()

    return {
        "plasticity_ratio": plasticity_ratio(native=native, unpenalized=unpenalized),
        "norm_ratio": norm_ratio(native=native, unpenalized=unpenalized),
        "direction_cosine": direction_cosine(native=native, unpenalized=unpenalized),
    }
