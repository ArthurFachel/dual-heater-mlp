"""Small validation primitives shared by experiment configurations."""

from __future__ import annotations

import math
from collections.abc import Mapping


def require_positive_integers(values: Mapping[str, int]) -> None:
    for name, value in values.items():
        if value < 1:
            raise ValueError(f"{name} deve ser >= 1")


def require_finite_values(values: Mapping[str, float]) -> None:
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} deve ser finito")


def require_nonnegative_values(values: Mapping[str, float]) -> None:
    invalid = {name: value for name, value in values.items() if value < 0.0}
    if invalid:
        raise ValueError(f"parâmetros devem ser >= 0: {invalid}")
