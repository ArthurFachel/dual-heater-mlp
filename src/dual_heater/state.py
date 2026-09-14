"""Precision contract for the mechanisms' scientific state.

Heats, importance estimates and consolidation counters decide which units get
protected.  They are part of the experiment's measurement apparatus, not part of
the computation graph, so a model-wide ``.half()``/``.bfloat16()`` must not
change them: a reduced-precision ranking can silently reorder units through
underflow or rounding, and a float counter stops incrementing exactly once it
exceeds the mantissa (2048 in fp16).

`FP32ScientificStateMixin` therefore lets every real transformation happen
(device moves, ``share_memory()``, pinning) and only undoes the dtype cast on
the named floating-point buffers a module declares as scientific state.
"""

from __future__ import annotations

import torch
from torch import nn


class FP32ScientificStateMixin(nn.Module):
    """Keep named scientific buffers FP32 while preserving device transforms."""

    _fp32_state_names: tuple[str, ...] = ()

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        for name in self._fp32_state_names:
            buffer = getattr(self, name, None)
            if buffer is not None and buffer.is_floating_point():
                setattr(self, name, buffer.float())
        return result


def register_counter(module: nn.Module, name: str, *, device=None) -> None:
    """Register an int64 scalar counter that cannot lose exactness."""

    module.register_buffer(
        name, torch.zeros((), dtype=torch.int64, device=device)
    )
