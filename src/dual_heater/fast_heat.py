"""Short-timescale activation competition for Functional DualHeat.

FastHeat is deliberately independent from the optimizer.  It stores one
non-trainable value per hidden unit/channel, applies a divisive lateral gate in
both training and evaluation, and updates its state only while the module is in
training mode.  Evaluation therefore uses the last online state without
learning from validation or test examples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FastHeatConfig:
    """Configuration shared by every FastHeat gate in one model."""

    fast_decay: float = 0.90
    fast_strength: float = 0.5
    fast_threshold: float = 0.5
    eps: float = 1e-8

    def __post_init__(self) -> None:
        values = (
            self.fast_decay,
            self.fast_strength,
            self.fast_threshold,
            self.eps,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("parâmetros FastHeat devem ser finitos")
        if not 0.0 <= self.fast_decay < 1.0:
            raise ValueError("fast_decay deve estar em [0, 1)")
        if self.fast_strength < 0.0:
            raise ValueError("fast_strength deve ser >= 0")
        if self.fast_threshold < 0.0:
            raise ValueError("fast_threshold deve ser >= 0")
        if self.eps <= 0.0:
            raise ValueError("eps deve ser > 0")


class FastHeatGate(nn.Module):
    """Apply normalized activation-based lateral inhibition.

    ``unit_dim`` identifies the unit/channel dimension.  Linear and sequence
    tensors use ``-1``; convolutional tensors use ``1``.  The current forward
    is gated with the previous state, then training forwards update the state:

    ``f <- relu(alpha*f + (1-alpha)*(normalized_magnitude-threshold))``.
    """

    def __init__(
        self,
        unit_count: int,
        *,
        unit_dim: int,
        config: FastHeatConfig | None = None,
    ) -> None:
        super().__init__()
        if (
            not isinstance(unit_count, int)
            or isinstance(unit_count, bool)
            or unit_count < 1
        ):
            raise ValueError("unit_count deve ser um inteiro positivo")
        if not isinstance(unit_dim, int) or isinstance(unit_dim, bool):
            raise TypeError("unit_dim deve ser um inteiro")
        self.unit_count = unit_count
        self.unit_dim = unit_dim
        self.config = FastHeatConfig() if config is None else config
        self.register_buffer("fast_heat", torch.zeros(unit_count))
        self._last_elements_per_example = 0

    def _resolved_unit_dim(self, ndim: int) -> int:
        resolved = self.unit_dim if self.unit_dim >= 0 else ndim + self.unit_dim
        if not 0 <= resolved < ndim:
            raise ValueError("unit_dim é incompatível com o tensor de entrada")
        return resolved

    def _view_shape(self, inputs: Tensor, unit_dim: int) -> tuple[int, ...]:
        shape = [1] * inputs.ndim
        shape[unit_dim] = self.unit_count
        return tuple(shape)

    def _lateral_scale(self) -> Tensor:
        if self.config.fast_strength == 0.0 or self.unit_count == 1:
            return torch.ones_like(self.fast_heat)
        mean_others = (self.fast_heat.sum() - self.fast_heat) / (self.unit_count - 1)
        return 1.0 / (1.0 + self.config.fast_strength * mean_others)

    def forward(self, inputs: Tensor) -> Tensor:
        if not torch.is_floating_point(inputs):
            raise TypeError("FastHeatGate requer tensor de ponto flutuante")
        unit_dim = self._resolved_unit_dim(inputs.ndim)
        if inputs.shape[unit_dim] != self.unit_count:
            raise ValueError(
                "dimensão de unidades do tensor não corresponde a unit_count"
            )
        batch_size = int(inputs.shape[0]) if inputs.ndim > 0 else 1
        self._last_elements_per_example = inputs.numel() // max(1, batch_size)

        scale = self._lateral_scale().to(dtype=inputs.dtype)
        output = inputs * scale.view(self._view_shape(inputs, unit_dim))

        if self.training:
            with torch.no_grad():
                reduce_dims = tuple(
                    dimension
                    for dimension in range(inputs.ndim)
                    if dimension != unit_dim
                )
                magnitude = inputs.detach().abs().mean(dim=reduce_dims)
                if magnitude.dtype in {torch.float16, torch.bfloat16}:
                    magnitude = magnitude.float()
                normalizer = magnitude.mean().clamp_min(self.config.eps)
                normalized = magnitude / normalizer
                self.fast_heat.mul_(self.config.fast_decay).add_(
                    normalized - self.config.fast_threshold,
                    alpha=1.0 - self.config.fast_decay,
                ).clamp_(min=0.0)
        return output

    @torch.no_grad()
    def reset_fast_heat(self) -> None:
        self.fast_heat.zero_()

    def estimated_flops_per_example(self) -> int:
        """Return a deterministic operation-count approximation.

        The estimate covers gating, magnitude reduction, normalization and the
        EMA update.  It intentionally follows the repository's approximate
        hook/mask accounting convention rather than hardware-specific kernels.
        """

        elements = self._last_elements_per_example
        if elements == 0:
            return 0
        return 3 * elements + 12 * self.unit_count

    def extra_repr(self) -> str:
        return (
            f"units={self.unit_count}, dim={self.unit_dim}, "
            f"alpha={self.config.fast_decay}, "
            f"gamma={self.config.fast_strength}, "
            f"delta={self.config.fast_threshold}"
        )


class FastHeatActivation(nn.Module):
    """Keep an activation and its FastHeat gate in one topology slot."""

    def __init__(
        self,
        activation_module: nn.Module,
        unit_count: int,
        *,
        unit_dim: int,
        config: FastHeatConfig,
    ) -> None:
        super().__init__()
        self.activation = activation_module
        self.gate = FastHeatGate(
            unit_count,
            unit_dim=unit_dim,
            config=config,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.gate(self.activation(inputs))


def fast_heat_states(module: nn.Module) -> list[FastHeatGate]:
    """Return every FastHeat gate without duplicating the root module."""

    return [child for child in module.modules() if isinstance(child, FastHeatGate)]


def get_fast_states(module: nn.Module) -> list[FastHeatGate]:
    """Public functional form of the models' ``get_fast_states`` method."""

    return fast_heat_states(module)


@torch.no_grad()
def reset_fast_heat(module: nn.Module) -> None:
    for gate in fast_heat_states(module):
        gate.reset_fast_heat()
