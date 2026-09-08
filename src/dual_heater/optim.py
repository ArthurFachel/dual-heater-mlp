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


class _SlowHeatSource(Protocol):
    slow_heat: Tensor
    slow_strength: float

    def get_lr_scales(self) -> Tensor: ...


class _SlowHeatModule(_SlowHeatSource, Protocol):
    weight: Parameter
    bias: Parameter | None
    gradient_masking: bool


@dataclass
class _ResolvedMask:
    parameter: Parameter
    parameter_before: Tensor
    mask: Tensor
    state_before: dict[str, Tensor]


@dataclass(frozen=True)
class PlasticityMaskBinding:
    """Declarative optimizer-mask registration used by graph-based models."""

    parameter: Parameter
    mask: MaskSource
    kind: str


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

    def register_mask_bindings(
        self,
        bindings: Iterable[PlasticityMaskBinding],
    ) -> None:
        """Atomically register a complete set of graph-derived masks."""

        materialized = list(bindings)
        identifiers = [id(binding.parameter) for binding in materialized]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("bindings não pode registrar o mesmo parâmetro duas vezes")
        if any(not binding.kind for binding in materialized):
            raise ValueError("todo binding deve ter kind estável e não vazio")
        if any(
            self._parameter_position(binding.parameter) is None
            for binding in materialized
        ):
            raise ValueError("todos os parâmetros dos bindings devem pertencer ao optimizer")
        for binding in materialized:
            self.register_plasticity_mask(
                binding.parameter,
                binding.mask,
                kind=binding.kind,
            )

    def register_slow_heat_module(
        self,
        module: torch.nn.Module,
        *,
        input_module: torch.nn.Module | None = None,
        input_expansion: int = 1,
        input_key: str | None = None,
        hard: bool = False,
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
            raise ValueError(
                "todos os parâmetros do módulo devem pertencer ao optimizer"
            )

        if (
            not isinstance(input_expansion, int)
            or isinstance(input_expansion, bool)
            or input_expansion < 1
        ):
            raise ValueError("input_expansion deve ser um inteiro positivo")

        typed_input: _SlowHeatSource | None = None
        input_position: tuple[int, int] | None = None
        grouped_convolution = False
        groups = 1
        if input_module is not None:
            input_heat = getattr(input_module, "slow_heat", None)
            input_strength = getattr(input_module, "slow_strength", None)
            if not isinstance(input_heat, Tensor) or input_strength is None:
                raise TypeError("input_module deve expor slow_heat e slow_strength")
            typed_input = cast(_SlowHeatSource, input_module)
            input_weight = getattr(input_module, "weight", None)
            if isinstance(input_weight, Parameter):
                input_position = self._parameter_position(input_weight)
            if input_position is None and input_key is None:
                raise ValueError(
                    "fonte sem peso requer input_key estável para o checkpoint"
                )
            if typed_module.weight.ndim < 2:
                raise ValueError(
                    "proteção fatorada requer parâmetro com ao menos 2 dims"
                )
            source_units = typed_input.slow_heat.numel()
            in_channels = getattr(module, "in_channels", None)
            groups = getattr(module, "groups", 1)
            grouped_convolution = (
                typed_module.weight.ndim == 4
                and isinstance(in_channels, int)
                and isinstance(groups, int)
            )
            if grouped_convolution:
                compatible = input_expansion == 1 and in_channels == source_units
                compatible = compatible and groups >= 1
                compatible = compatible and in_channels % groups == 0
                compatible = compatible and typed_module.weight.shape[0] % groups == 0
                compatible = compatible and (
                    typed_module.weight.shape[1] == in_channels // groups
                )
            else:
                compatible = (
                    typed_module.weight.shape[1]
                    == source_units * input_expansion
                )
            if not compatible:
                raise ValueError(
                    "a importância de entrada não corresponde à dimensão do parâmetro"
                )

        def module_factor(source: _SlowHeatSource) -> Tensor:
            if hard:
                return (source.slow_heat <= 0.0).to(dtype=source.slow_heat.dtype)
            return source.get_lr_scales()

        def weight_mask() -> Tensor:
            output_factor = module_factor(typed_module).reshape(
                (-1,) + (1,) * (typed_module.weight.ndim - 1)
            )
            if typed_input is None:
                return output_factor
            source_factor = module_factor(typed_input)
            if grouped_convolution:
                outputs = typed_module.weight.shape[0]
                inputs_per_group = typed_module.weight.shape[1]
                outputs_per_group = outputs // groups
                output_group = (
                    torch.arange(outputs, device=source_factor.device)
                    // outputs_per_group
                )
                local_input = torch.arange(
                    inputs_per_group,
                    device=source_factor.device,
                )
                global_input = (
                    output_group[:, None] * inputs_per_group
                    + local_input[None, :]
                )
                input_factor = source_factor[global_input].reshape(
                    outputs,
                    inputs_per_group,
                    *([1] * (typed_module.weight.ndim - 2)),
                )
            else:
                input_factor = source_factor.repeat_interleave(
                    input_expansion
                ).reshape(
                    (1, -1) + (1,) * (typed_module.weight.ndim - 2)
                )
            return torch.minimum(output_factor, input_factor)

        def bias_mask() -> Tensor:
            return module_factor(typed_module)

        if typed_input is None:
            weight_kind = "slowheat_hard_weight" if hard else "slowheat_weight"
        else:
            prefix = "slowheat_hard" if hard else "slowheat"
            source_signature = (
                f"{input_position[0]}_{input_position[1]}"
                if input_position is not None
                else f"virtual_{input_key}"
            )
            weight_kind = f"{prefix}_weight_factorized_from_{source_signature}"
            if groups > 1:
                weight_kind += f"_groups_{groups}"
            if input_expansion > 1:
                weight_kind += f"_repeat_{input_expansion}"

        typed_module.gradient_masking = False
        self.register_plasticity_mask(
            typed_module.weight,
            weight_mask,
            kind=weight_kind,
        )
        if isinstance(typed_module.bias, Parameter):
            self.register_plasticity_mask(
                typed_module.bias,
                bias_mask,
                kind="slowheat_hard_bias" if hard else "slowheat_bias",
            )

    def register_slow_heat_channel_module(
        self,
        module: torch.nn.Module,
        *,
        source_module: torch.nn.Module,
        source_key: str,
        hard: bool = False,
    ) -> None:
        """Protect affine per-channel parameters such as GroupNorm weight/bias."""

        source_heat = getattr(source_module, "slow_heat", None)
        source_strength = getattr(source_module, "slow_strength", None)
        if not isinstance(source_heat, Tensor) or source_strength is None:
            raise TypeError("source_module deve expor slow_heat e slow_strength")
        source = cast(_SlowHeatSource, source_module)
        parameters = [
            parameter
            for parameter in (getattr(module, "weight", None), getattr(module, "bias", None))
            if isinstance(parameter, Parameter)
        ]
        if not parameters:
            raise ValueError("módulo de canal deve ter parâmetro affine")
        if any(parameter.ndim != 1 for parameter in parameters):
            raise ValueError("parâmetros affine de canal devem ser vetores")
        if any(parameter.numel() != source_heat.numel() for parameter in parameters):
            raise ValueError("parâmetros affine não correspondem à fonte de canais")
        if not all(self._parameter_position(parameter) is not None for parameter in parameters):
            raise ValueError("parâmetros affine devem pertencer ao optimizer")

        def channel_mask() -> Tensor:
            if hard:
                return (source.slow_heat <= 0.0).to(dtype=source.slow_heat.dtype)
            return source.get_lr_scales()

        prefix = "slowheat_hard" if hard else "slowheat"
        for parameter in parameters:
            self.register_plasticity_mask(
                parameter,
                channel_mask,
                kind=f"{prefix}_channel_affine_from_{source_key}",
            )

    def register_slow_heat_model(
        self,
        model: torch.nn.Module,
        *,
        hard: bool = False,
    ) -> None:
        """Register sequential or explicitly graph-connected SlowHeat models."""

        getter = getattr(model, "get_slow_layers", None)
        if not callable(getter):
            raise TypeError("model deve expor get_slow_layers()")
        layers = list(getter())
        if not layers:
            raise ValueError("model não contém camadas SlowHeat")
        registrations: list[
            tuple[torch.nn.Module, torch.nn.Module | None, int, str | None]
        ] = []
        graph_getter = getattr(model, "get_slow_connections", None)
        if callable(graph_getter):
            registrations = list(graph_getter())
            destinations = [id(layer) for layer, _, _, _ in registrations]
            if len(destinations) != len(set(destinations)):
                raise ValueError("grafo SlowHeat registra um destino mais de uma vez")
            if set(destinations) != {id(layer) for layer in layers}:
                raise ValueError(
                    "grafo SlowHeat deve registrar exatamente todas as camadas"
                )
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
            if callable(graph_getter):
                continue
            previous = layers[index - 1] if index > 0 else None
            expansion = 1
            if previous is not None:
                source_units = previous.slow_heat.numel()
                if (
                    weight.ndim == 4
                    and isinstance(getattr(layer, "in_channels", None), int)
                ):
                    compatible = layer.in_channels == source_units
                elif (
                    weight.ndim == 2
                    and isinstance(previous, torch.nn.Conv2d)
                    and weight.shape[1] % source_units == 0
                ):
                    expansion = weight.shape[1] // source_units
                    compatible = True
                else:
                    compatible = weight.shape[1] == source_units
                if not compatible:
                    raise ValueError(
                        "camadas SlowHeat não formam uma cadeia compatível"
                    )
            registrations.append((layer, previous, expansion, None))
        for layer, previous, expansion, input_key in registrations:
            self.register_slow_heat_module(
                layer,
                input_module=previous,
                input_expansion=expansion,
                input_key=input_key,
                hard=hard,
            )
        channel_getter = getattr(model, "get_slow_channel_modules", None)
        if callable(channel_getter):
            for module, source, source_key in channel_getter():
                self.register_slow_heat_channel_module(
                    module,
                    source_module=source,
                    source_key=source_key,
                    hard=hard,
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
            state_before = (
                {}
                if self.state_policy == "native"
                else {
                    key: value.detach().clone()
                    for key, value in self.state.get(parameter, {}).items()
                    if isinstance(value, Tensor) and value.shape == parameter.shape
                }
            )
            resolved.append(
                _ResolvedMask(
                    parameter=parameter,
                    parameter_before=parameter.detach().clone(),
                    mask=expanded,
                    state_before=state_before,
                )
            )
        return resolved

    def _resolved_mask_for_parameter(self, parameter: Parameter) -> Tensor | None:
        registration = self._plasticity_masks.get(id(parameter))
        if registration is None:
            return None
        registered_parameter, source, _ = registration
        if registered_parameter is not parameter:
            raise RuntimeError("registro de máscara aponta para parâmetro incorreto")
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
        return expanded

    @staticmethod
    def _apply_resolved_masks(
        snapshots: list[_ResolvedMask],
    ) -> None:
        for snapshot in snapshots:
            if torch.all(snapshot.mask == 1.0):
                continue
            native_delta = snapshot.parameter.detach() - snapshot.parameter_before
            snapshot.parameter.copy_(
                snapshot.parameter_before + snapshot.mask * native_delta
            )

    def _apply_state_policy(self, snapshots: list[_ResolvedMask]) -> None:
        """Make tensor-valued optimizer state follow the applied update mask."""

        if self.state_policy == "native":
            return
        for snapshot in snapshots:
            if torch.all(snapshot.mask == 1.0):
                continue
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

    def _step_with_masks(
        self,
        native_step: Callable[[], Any],
        closure: Callable[[], float] | None,
    ) -> float | None:
        loss = self._run_closure(closure)
        self._ensure_checkpoint_masks_registered()
        snapshots = self._resolved_masks()
        native_step()
        self._apply_resolved_masks(snapshots)
        self._apply_state_policy(snapshots)
        return loss


class SlowHeatAdamW(_PlasticityMaskMixin, torch.optim.AdamW):
    """AdamW with masks applied to the complete parameter and state updates.

    When masks are present, the pointwise implementation computes native AdamW
    moments in short-lived temporaries and applies the mask directly. It avoids
    the parameter-plus-state snapshots used by the generic optimizer path.
    """

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
        if not self._plasticity_masks:
            torch.optim.AdamW.step(self)
            return loss
        self._pointwise_masked_step()
        return loss

    def _pointwise_masked_step(self) -> None:
        """Apply AdamW without retaining full parameter/state snapshots."""

        if getattr(self, "grad_scale", None) is not None or getattr(
            self, "found_inf", None
        ) is not None:
            raise RuntimeError("SlowHeatAdamW mascarado não suporta GradScaler")
        for group in self.param_groups:
            if group.get("capturable", False):
                raise ValueError("SlowHeatAdamW mascarado não suporta capturable=True")
            if group.get("differentiable", False):
                raise ValueError(
                    "SlowHeatAdamW mascarado não suporta differentiable=True"
                )
            if group.get("fused", False):
                raise ValueError("SlowHeatAdamW mascarado não suporta fused=True")
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            if any(torch.is_tensor(value) for value in (lr, beta1, beta2)):
                raise ValueError(
                    "SlowHeatAdamW mascarado requer lr e betas escalares"
                )
            lr = float(lr)
            beta1 = float(beta1)
            beta2 = float(beta2)
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            amsgrad = bool(group["amsgrad"])
            maximize = bool(group["maximize"])

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdamW não suporta gradientes esparsos")
                if torch.is_complex(parameter):
                    raise ValueError(
                        "SlowHeatAdamW mascarado não suporta parâmetros complexos"
                    )
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format
                        )
                state["step"].add_(1)
                step = float(state["step"].item())
                grad = -parameter.grad if maximize else parameter.grad
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                mask = self._resolved_mask_for_parameter(parameter)

                if mask is None or bool(torch.all(mask == 1.0)):
                    if weight_decay != 0.0:
                        parameter.mul_(1.0 - lr * weight_decay)
                    exp_avg.lerp_(grad, 1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1.0 - beta2
                    )
                    denominator_source = exp_avg_sq
                    if amsgrad:
                        maximum = state["max_exp_avg_sq"]
                        torch.maximum(maximum, exp_avg_sq, out=maximum)
                        denominator_source = maximum
                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step
                    denominator = denominator_source.sqrt().div_(
                        bias_correction2**0.5
                    ).add_(eps)
                    parameter.addcdiv_(
                        exp_avg,
                        denominator,
                        value=-lr / bias_correction1,
                    )
                    continue

                # The parameter update uses the native moments, while the
                # persistent state optionally follows the same mask. This is
                # exactly the semantics of the previous snapshot path.
                native_exp_avg = torch.lerp(exp_avg, grad, 1.0 - beta1)
                native_exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(
                    grad, grad, value=1.0 - beta2
                )
                native_max = None
                if amsgrad:
                    native_max = torch.maximum(
                        state["max_exp_avg_sq"], native_exp_avg_sq
                    )

                if self.state_policy == "native":
                    exp_avg.copy_(native_exp_avg)
                    exp_avg_sq.copy_(native_exp_avg_sq)
                    if native_max is not None:
                        state["max_exp_avg_sq"].copy_(native_max)
                else:
                    exp_avg.lerp_(native_exp_avg, mask)
                    exp_avg_sq.lerp_(native_exp_avg_sq, mask)
                    if native_max is not None:
                        state["max_exp_avg_sq"].lerp_(native_max, mask)

                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay * mask)
                denominator_source = (
                    native_max if native_max is not None else native_exp_avg_sq
                )
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = denominator_source.sqrt().div_(
                    bias_correction2**0.5
                ).add_(eps)
                native_exp_avg.mul_(mask)
                parameter.addcdiv_(
                    native_exp_avg,
                    denominator,
                    value=-lr / bias_correction1,
                )

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
        return self._step_with_masks(
            lambda: torch.optim.SGD.step(self),
            closure,
        )

    def state_dict(self) -> dict[str, Any]:
        return self._state_dict_with_mask_metadata(torch.optim.SGD.state_dict(self))

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        torch.optim.SGD.load_state_dict(self, self._load_mask_metadata(state_dict))
