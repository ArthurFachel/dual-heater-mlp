"""Neuron-level plasticity mechanisms for continual learning research."""

from .dual_heat import DualHeatLinear, DualHeatMLP
from .lora import DualHeatLoRALinear
from .metrics import CLMetrics, compute_cl_metrics
from .optim import SlowHeatAdamW, SlowHeatSGD
from .resnet import CIFARResNet18, SlowHeatResNet18
from .slow_heat import (
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
]
