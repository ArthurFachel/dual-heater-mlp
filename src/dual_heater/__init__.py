"""Neuron-level plasticity mechanisms for continual learning research."""

from .dual_heat import DualHeatLinear, DualHeatMLP
from .fast_heat import FastHeatConfig, FastHeatGate, get_fast_states, reset_fast_heat
from .lora import DualHeatLoRALinear
from .metrics import CLMetrics, compute_cl_metrics
from .optim import SlowHeatAdamW, SlowHeatSGD
from .resnet import CIFARResNet18, FunctionalDualHeatResNet18, SlowHeatResNet18
from .slow_heat import (
    FunctionalDualHeatCNN,
    FunctionalDualHeatMLP,
    FunctionalDualHeatVGG11,
    SlowHeatChannelTracker,
    SlowHeatCNN,
    SlowHeatConv2d,
    SlowHeatLinear,
    SlowHeatMLP,
    SlowHeatVGG11,
)

__all__ = [
    "CIFARResNet18",
    "CLMetrics",
    "DualHeatLinear",
    "DualHeatLoRALinear",
    "DualHeatMLP",
    "FastHeatConfig",
    "FastHeatGate",
    "FunctionalDualHeatCNN",
    "FunctionalDualHeatMLP",
    "FunctionalDualHeatResNet18",
    "FunctionalDualHeatVGG11",
    "SlowHeatAdamW",
    "SlowHeatCNN",
    "SlowHeatChannelTracker",
    "SlowHeatConv2d",
    "SlowHeatLinear",
    "SlowHeatMLP",
    "SlowHeatResNet18",
    "SlowHeatSGD",
    "SlowHeatVGG11",
    "compute_cl_metrics",
    "get_fast_states",
    "reset_fast_heat",
]
