"""Shared typed contracts for continual-learning experiment adapters."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ContinualTask:
    """Materialized train/validation/test tensors for one continual stage."""

    classes: tuple[int, ...]
    train_x: Tensor
    train_y: Tensor
    validation_x: Tensor
    validation_y: Tensor
    test_x: Tensor
    test_y: Tensor
