"""Optimizer-aware SlowHeat plasticity masking.

These optimizers scale the final parameter update after preconditioning,
momentum and decoupled weight decay. A raw gradient mask cannot provide this
semantics under AdamW because its moment normalization can cancel constant
gradient scaling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, TypeAlias, cast

import torch
from torch import Tensor
from torch.nn import Parameter

MaskSource: TypeAlias = Tensor | Callable[[], Tensor]
MaskSignature: TypeAlias = tuple[int, int, str]
MaskRegistration: TypeAlias = tuple[Parameter, MaskSource, str]


class _SlowHeatModule(Protocol):
    slow_heat: Tensor
    slow_strength: float
    weight: Parameter
    bias: Parameter | None
    gradient_masking: bool


class _PlasticityMaskMixin:
    """Shared registration and final-update masking operations."""

    _plasticity_masks: dict[int, MaskRegistration]
    _expected_mask_signatures: set[MaskSignature] | None
    param_groups: list[dict[str, Any]]

    def _initialize_plasticity_masks(self) -> None:
        self._plasticity_masks = {}
        self._expected_mask_signatures = None

    def _parameter_position(self, parameter: Parameter) -> tuple[int, int] | None:
        for group_index, group in enumerate(self.param_groups):
            for parameter_index, candidate in enumerate(group["params"]):
                if candidate is parameter:
                    return group_index, parameter_index
        return None

    def register_plasticity_mask(
        self,
        parameter: Parameter,
        mask: MaskSource,
        *,
        kind: str = "generic",
    ) -> None:
        """Register a static or dynamic ``[0, 1]`` mask for a parameter."""

        if self._parameter_position(parameter) is None:
            raise ValueError("parameter não pertence a este optimizer")
        self._plasticity_masks[id(parameter)] = (parameter, mask, kind)

    def register_slow_heat_module(self, module: torch.nn.Module) -> None:
        """Atomically register row masks derived from a SlowHeat layer.

        Legacy raw-gradient masking is disabled only after all trainable module
        parameters have been validated as members of this optimizer.
        """

        slow_heat = getattr(module, "slow_heat", None)
        slow_strength = getattr(module, "slow_strength", None)
        weight = getattr(module, "weight", None)
        if not isinstance(slow_heat, Tensor) or slow_strength is None:
            raise TypeError("module deve expor slow_heat e slow_strength")
        if not isinstance(weight, Parameter):
            raise TypeError("module deve expor weight treinável")

        typed_module = cast(_SlowHeatModule, module)
        parameters = [typed_module.weight]
        if isinstance(typed_module.bias, Parameter):
            parameters.append(typed_module.bias)
        if not all(self._parameter_position(item) is not None for item in parameters):
            raise ValueError("todos os parâmetros do módulo devem pertencer ao optimizer")

        def weight_mask() -> Tensor:
            factor = 1.0 / (
                1.0 + typed_module.slow_strength * typed_module.slow_heat
            )
            return factor.reshape(
                (-1,) + (1,) * (typed_module.weight.ndim - 1)
            )

        def bias_mask() -> Tensor:
            return 1.0 / (
                1.0 + typed_module.slow_strength * typed_module.slow_heat
            )

        typed_module.gradient_masking = False
        self.register_plasticity_mask(
            typed_module.weight,
            weight_mask,
            kind="slowheat_weight",
        )
        if isinstance(typed_module.bias, Parameter):
            self.register_plasticity_mask(
                typed_module.bias,
                bias_mask,
                kind="slowheat_bias",
            )

    def clear_plasticity_masks(self) -> None:
        """Remove optimizer masks; module gradient hooks are not re-enabled."""

        self._plasticity_masks.clear()

    def _current_mask_signatures(self) -> set[MaskSignature]:
        signatures: set[MaskSignature] = set()
        for parameter, _, kind in self._plasticity_masks.values():
            position = self._parameter_position(parameter)
            if position is None:
                raise RuntimeError("máscara registrada para parâmetro ausente")
            signatures.add((*position, kind))
        return signatures

    def _ensure_checkpoint_masks_registered(self) -> None:
        if (
            self._expected_mask_signatures is not None
            and self._current_mask_signatures() != self._expected_mask_signatures
        ):
            raise RuntimeError(
                "checkpoint protegido incompatível: register novamente as mesmas "
                "máscaras SlowHeat antes de chamar step()"
            )

    def _state_dict_with_mask_metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        current = self._current_mask_signatures()
        if self._expected_mask_signatures is not None:
            if current and current != self._expected_mask_signatures:
                raise RuntimeError(
                    "checkpoint protegido incompatível: não é seguro salvar "
                    "registros de máscaras diferentes dos esperados"
                )
            signatures = current or self._expected_mask_signatures
        else:
            signatures = current
        state["slowheat_masks"] = [
            {"group": group, "parameter": parameter, "kind": kind}
            for group, parameter, kind in sorted(signatures)
        ]
        return state

    def _load_mask_metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        state_copy = dict(state)
        metadata = state_copy.pop("slowheat_masks", None)
        if metadata is None:
            self._expected_mask_signatures = None
        else:
            self._expected_mask_signatures = {
                (int(item["group"]), int(item["parameter"]), str(item["kind"]))
                for item in metadata
            }
        return state_copy

    def _resolved_masks(self) -> list[tuple[Parameter, Tensor, Tensor]]:
        resolved: list[tuple[Parameter, Tensor, Tensor]] = []
        for parameter, source, _ in self._plasticity_masks.values():
            mask = source() if callable(source) else source
            if not isinstance(mask, Tensor):
                raise TypeError("a fonte da máscara deve retornar um Tensor")
            mask = mask.detach().to(device=parameter.device, dtype=parameter.dtype)
            try:
                expanded = torch.broadcast_to(mask, parameter.shape)
            except RuntimeError as error:
                raise ValueError(
                    "a máscara de plasticidade não é compatível com o parâmetro"
                ) from error
            if not torch.isfinite(expanded).all():
                raise ValueError("a máscara de plasticidade deve ser finita")
            if torch.any(expanded < 0.0) or torch.any(expanded > 1.0):
                raise ValueError("a máscara de plasticidade deve estar em [0, 1]")
            resolved.append((parameter, parameter.detach().clone(), expanded))
        return resolved

    @staticmethod
    def _apply_resolved_masks(
        snapshots: list[tuple[Parameter, Tensor, Tensor]],
    ) -> None:
        for parameter, previous, mask in snapshots:
            native_delta = parameter.detach() - previous
            parameter.copy_(previous + mask * native_delta)

    @staticmethod
    def _run_closure(closure: Callable[[], float] | None) -> float | None:
        if closure is None:
            return None
        with torch.enable_grad():
            return closure()


class SlowHeatAdamW(_PlasticityMaskMixin, torch.optim.AdamW):
    """AdamW with masks applied to the complete parameter update."""

    def __init__(self, params: Iterable[Parameter] | Iterable[dict[str, Any]], **kwargs):
        super().__init__(params, **kwargs)
        self._initialize_plasticity_masks()

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = self._run_closure(closure)
        self._ensure_checkpoint_masks_registered()
        snapshots = self._resolved_masks()
        torch.optim.AdamW.step(self)
        self._apply_resolved_masks(snapshots)
        return loss

    def state_dict(self) -> dict[str, Any]:
        return self._state_dict_with_mask_metadata(torch.optim.AdamW.state_dict(self))

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        torch.optim.AdamW.load_state_dict(self, self._load_mask_metadata(state_dict))


class SlowHeatSGD(_PlasticityMaskMixin, torch.optim.SGD):
    """SGD with masks applied to the complete parameter update."""

    def __init__(self, params: Iterable[Parameter] | Iterable[dict[str, Any]], **kwargs):
        super().__init__(params, **kwargs)
        self._initialize_plasticity_masks()

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = self._run_closure(closure)
        self._ensure_checkpoint_masks_registered()
        snapshots = self._resolved_masks()
        torch.optim.SGD.step(self)
        self._apply_resolved_masks(snapshots)
        return loss

    def state_dict(self) -> dict[str, Any]:
        return self._state_dict_with_mask_metadata(torch.optim.SGD.state_dict(self))

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        torch.optim.SGD.load_state_dict(self, self._load_mask_metadata(state_dict))
