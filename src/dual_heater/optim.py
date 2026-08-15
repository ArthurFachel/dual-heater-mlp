"""Optimizer-aware SlowHeat plasticity masking.

These optimizers scale the final parameter update after preconditioning,
momentum and decoupled weight decay. A raw gradient mask cannot provide this
semantics under AdamW because its moment normalization can cancel constant
gradient scaling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
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

    def get_lr_scales(self) -> Tensor: ...


@dataclass
class _ResolvedMask:
    parameter: Parameter
    parameter_before: Tensor
    mask: Tensor
    state_before: dict[str, Tensor]


class _PlasticityMaskMixin:
    """Shared registration and final-update masking operations."""

    _plasticity_masks: dict[int, MaskRegistration]
    _expected_mask_signatures: set[MaskSignature] | None
    param_groups: list[dict[str, Any]]
    state: dict[Parameter, dict[str, Any]]
    state_policy: str

    def _initialize_plasticity_masks(self, state_policy: str) -> None:
        if state_policy not in {"native", "follow_update"}:
            raise ValueError("state_policy deve ser 'native' ou 'follow_update'")
        self._plasticity_masks = {}
        self._expected_mask_signatures = None
        self.state_policy = state_policy

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

    def register_slow_heat_module(
        self,
        module: torch.nn.Module,
        *,
        input_module: torch.nn.Module | None = None,
    ) -> None:
        """Atomically register factorized masks derived from SlowHeat layers.

        Legacy raw-gradient masking is disabled only after all trainable module
        parameters have been validated as members of this optimizer. When an
        ``input_module`` is supplied, the weight mask protects both output rows
        and input columns using the strongest of the two protections.
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

        typed_input: _SlowHeatModule | None = None
        input_position: tuple[int, int] | None = None
        if input_module is not None:
            input_heat = getattr(input_module, "slow_heat", None)
            input_strength = getattr(input_module, "slow_strength", None)
            if not isinstance(input_heat, Tensor) or input_strength is None:
                raise TypeError("input_module deve expor slow_heat e slow_strength")
            typed_input = cast(_SlowHeatModule, input_module)
            input_position = self._parameter_position(typed_input.weight)
            if input_position is None:
                raise ValueError(
                    "o peso do input_module deve pertencer ao mesmo optimizer"
                )
            if typed_module.weight.ndim < 2:
                raise ValueError("proteção fatorada requer parâmetro com ao menos 2 dims")
            if typed_module.weight.shape[1] != typed_input.slow_heat.numel():
                raise ValueError(
                    "a importância de entrada não corresponde à dimensão do parâmetro"
                )

        def weight_mask() -> Tensor:
            output_factor = typed_module.get_lr_scales().reshape(
                (-1,) + (1,) * (typed_module.weight.ndim - 1)
            )
            if typed_input is None:
                return output_factor
            input_factor = typed_input.get_lr_scales().reshape(
                (1, -1) + (1,) * (typed_module.weight.ndim - 2)
            )
            return torch.minimum(output_factor, input_factor)

        def bias_mask() -> Tensor:
            return typed_module.get_lr_scales()

        typed_module.gradient_masking = False
        self.register_plasticity_mask(
            typed_module.weight,
            weight_mask,
            kind=(
                f"slowheat_weight_factorized_from_{input_position[0]}_{input_position[1]}"
                if input_position is not None
                else "slowheat_weight"
            ),
        )
        if isinstance(typed_module.bias, Parameter):
            self.register_plasticity_mask(
                typed_module.bias,
                bias_mask,
                kind="slowheat_bias",
            )

    def register_slow_heat_model(self, model: torch.nn.Module) -> None:
        """Register a sequential SlowHeat model with factorized connectivity."""

        getter = getattr(model, "get_slow_layers", None)
        if not callable(getter):
            raise TypeError("model deve expor get_slow_layers()")
        layers = list(getter())
        if not layers:
            raise ValueError("model não contém camadas SlowHeat")
        for index, layer in enumerate(layers):
            weight = getattr(layer, "weight", None)
            bias = getattr(layer, "bias", None)
            parameters = [weight] + ([bias] if isinstance(bias, Parameter) else [])
            if not all(
                isinstance(parameter, Parameter)
                and self._parameter_position(parameter) is not None
                for parameter in parameters
            ):
                raise ValueError(
                    "todos os parâmetros SlowHeat devem pertencer ao optimizer"
                )
            if index > 0 and weight.shape[1] != layers[index - 1].slow_heat.numel():
                raise ValueError("camadas SlowHeat não formam uma cadeia compatível")
        previous = None
        for layer in layers:
            self.register_slow_heat_module(layer, input_module=previous)
            previous = layer

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
        state["slowheat_state_policy"] = self.state_policy
        return state

    def _load_mask_metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        state_copy = dict(state)
        metadata = state_copy.pop("slowheat_masks", None)
        checkpoint_policy = state_copy.pop("slowheat_state_policy", None)
        if checkpoint_policy is not None and checkpoint_policy != self.state_policy:
            raise ValueError(
                "state_policy do checkpoint é incompatível com o optimizer atual"
            )
        if metadata is None:
            self._expected_mask_signatures = None
        else:
            self._expected_mask_signatures = {
                (int(item["group"]), int(item["parameter"]), str(item["kind"]))
                for item in metadata
            }
        return state_copy

    def _resolved_masks(self) -> list[_ResolvedMask]:
        resolved: list[_ResolvedMask] = []
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
            state_before = {
                key: value.detach().clone()
                for key, value in self.state.get(parameter, {}).items()
                if isinstance(value, Tensor) and value.shape == parameter.shape
            }
            resolved.append(
                _ResolvedMask(
                    parameter=parameter,
                    parameter_before=parameter.detach().clone(),
                    mask=expanded,
                    state_before=state_before,
                )
            )
        return resolved

    @staticmethod
    def _apply_resolved_masks(
        snapshots: list[_ResolvedMask],
    ) -> None:
        for snapshot in snapshots:
            native_delta = snapshot.parameter.detach() - snapshot.parameter_before
            snapshot.parameter.copy_(
                snapshot.parameter_before + snapshot.mask * native_delta
            )

    def _apply_state_policy(self, snapshots: list[_ResolvedMask]) -> None:
        """Make tensor-valued optimizer state follow the applied update mask."""

        if self.state_policy == "native":
            return
        for snapshot in snapshots:
            for key, current in self.state.get(snapshot.parameter, {}).items():
                if (
                    not isinstance(current, Tensor)
                    or current.shape != snapshot.parameter.shape
                ):
                    continue
                previous = snapshot.state_before.get(key)
                if previous is None:
                    previous = torch.zeros_like(current)
                current.copy_(previous + snapshot.mask * (current - previous))

    @staticmethod
    def _run_closure(closure: Callable[[], float] | None) -> float | None:
        if closure is None:
            return None
        with torch.enable_grad():
            return closure()


class SlowHeatAdamW(_PlasticityMaskMixin, torch.optim.AdamW):
    """AdamW with masks applied to the complete parameter update."""

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, Any]],
        *,
        state_policy: str = "follow_update",
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._initialize_plasticity_masks(state_policy)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = self._run_closure(closure)
        self._ensure_checkpoint_masks_registered()
        snapshots = self._resolved_masks()
        torch.optim.AdamW.step(self)
        self._apply_resolved_masks(snapshots)
        self._apply_state_policy(snapshots)
        return loss

    def state_dict(self) -> dict[str, Any]:
        return self._state_dict_with_mask_metadata(torch.optim.AdamW.state_dict(self))

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        torch.optim.AdamW.load_state_dict(self, self._load_mask_metadata(state_dict))


class SlowHeatSGD(_PlasticityMaskMixin, torch.optim.SGD):
    """SGD with masks applied to the complete parameter update."""

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, Any]],
        *,
        state_policy: str = "follow_update",
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._initialize_plasticity_masks(state_policy)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = self._run_closure(closure)
        self._ensure_checkpoint_masks_registered()
        snapshots = self._resolved_masks()
        torch.optim.SGD.step(self)
        self._apply_resolved_masks(snapshots)
        self._apply_state_policy(snapshots)
        return loss

    def state_dict(self) -> dict[str, Any]:
        return self._state_dict_with_mask_metadata(torch.optim.SGD.state_dict(self))

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        torch.optim.SGD.load_state_dict(self, self._load_mask_metadata(state_dict))
