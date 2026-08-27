"""Paired model construction independent from the training engine."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Protocol

import torch
from torch import Tensor, nn

from dual_heater import SlowHeatCNN, SlowHeatMLP
from experiments.method_specs import MethodSpec


class ModelFactoryConfig(Protocol):
    seed: int
    input_dim: int
    hidden_dims: tuple[int, ...]
    class_order: tuple[int, ...]
    methods: tuple[str, ...]
    backbone: str
    image_shape: tuple[int, int, int] | None
    cnn_channels: tuple[int, int]
    cnn_pooled_size: tuple[int, int]
    slow_strength: float
    plasticity_budget: float
    partial_output_slow_strength: float
    device: str


class _VanillaMLP(nn.Sequential):
    """Native MLP exposing the same feature interface as visual backbones."""

    def forward_features(self, inputs: Tensor) -> Tensor:
        for module in list(self.children())[:-1]:
            inputs = module(inputs)
        return inputs


def _vanilla_mlp(dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for input_dim, output_dim in pairwise(dims[:-1]):
        layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return _VanillaMLP(*layers)


class _VanillaCNN(nn.Module):
    """Native control with the same trainable topology as ``SlowHeatCNN``."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        channels: tuple[int, int],
        pooled_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(pooled_size)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            channels[1] * pooled_size[0] * pooled_size[1],
            num_classes,
        )

    def forward_features(self, inputs: Tensor) -> Tensor:
        inputs = self.pool(self.activation1(self.conv1(inputs)))
        inputs = self.pool(self.activation2(self.conv2(inputs)))
        return self.flatten(self.adaptive_pool(inputs))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.forward_features(inputs))


def _vanilla_model(config: ModelFactoryConfig, dims: tuple[int, ...]) -> nn.Module:
    if config.backbone == "mlp":
        return _vanilla_mlp(dims)
    assert config.image_shape is not None
    return _VanillaCNN(
        config.image_shape[0],
        len(config.class_order),
        channels=config.cnn_channels,
        pooled_size=config.cnn_pooled_size,
    )


def build_paired_models(
    config: ModelFactoryConfig,
    method_specs: Mapping[str, MethodSpec],
) -> dict[str, nn.Module]:
    """Build methods from byte-identical native parameter initialization."""

    dims = (config.input_dim, *config.hidden_dims, len(config.class_order))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        reference = _vanilla_model(config, dims)
        reference_parameters = {
            name: parameter.detach().clone()
            for name, parameter in reference.named_parameters()
        }
        models: dict[str, nn.Module] = {}
        for method in config.methods:
            spec = method_specs[method]
            torch.manual_seed(config.seed)
            if not spec.slowheat:
                model = _vanilla_model(config, dims)
            else:
                budget = (
                    0.0 if spec.disable_capacity_budget else (
                        config.plasticity_budget
                        if spec.budget is None
                        else spec.budget
                    )
                )
                output_strength = (
                    config.partial_output_slow_strength
                    if spec.partial_output_protection
                    else None
                )
                strength = (
                    config.slow_strength if spec.strength is None else spec.strength
                )
                common = {
                    "act": "relu",
                    "slow_strength": strength,
                    "plasticity_budget": budget,
                    "protect_output": spec.protect_output,
                    "output_slow_strength": output_strength,
                }
                if config.backbone == "mlp":
                    model = SlowHeatMLP(*dims, **common)
                else:
                    assert config.image_shape is not None
                    model = SlowHeatCNN(
                        config.image_shape[0],
                        len(config.class_order),
                        channels=config.cnn_channels,
                        pooled_size=config.cnn_pooled_size,
                        **common,
                    )
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(reference_parameters[name])
            models[method] = model.to(config.device)
    return models
