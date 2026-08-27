"""Shared layer-building helpers."""

import math

from torch import nn


def validate_finite_hyperparameters(**values: float) -> None:
    invalid = {
        name: value for name, value in values.items() if not math.isfinite(value)
    }
    if invalid:
        raise ValueError(f"hiperparâmetros devem ser finitos: {invalid}")


def activation(name: str) -> nn.Module:
    normalized = name.lower()
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    if normalized == "leaky":
        return nn.LeakyReLU(0.1)
    if activation_type := activations.get(normalized):
        return activation_type()
    raise ValueError(f"Unknown activation: {name}")


def validate_mlp_dims(dims: tuple[int, ...]) -> None:
    if len(dims) < 2:
        raise ValueError("uma MLP requer ao menos as dimensões de entrada e saída")
    if any(not isinstance(width, int) or isinstance(width, bool) or width < 1 for width in dims):
        raise ValueError("todas as dimensões da MLP devem ser inteiros positivos")
