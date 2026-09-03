"""Utilities for side-effect-free model evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn


@contextmanager
def evaluating(module: nn.Module) -> Iterator[nn.Module]:
    """Temporarily enter evaluation mode and restore the previous mode."""

    was_training = module.training
    module.eval()
    try:
        yield module
    finally:
        module.train(was_training)
