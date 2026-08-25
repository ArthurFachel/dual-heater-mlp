"""Neuron-level plasticity mechanisms for continual learning research."""

from .dual_heat import DualHeatLinear, DualHeatMLP
from .lora import DualHeatLoRALinear
from .metrics import CLMetrics, compute_cl_metrics
from .optim import SlowHeatAdamW, SlowHeatSGD
from .resnet import CIFARResNet18, SlowHeatResNet18
from .slow_heat import (
    SlowHeatCNN,
    SlowHeatChannelTracker,
    SlowHeatConv2d,
    SlowHeatLinear,
    SlowHeatMLP,
)

__all__ = [
    "CLMetrics",
    "DualHeatLinear",
    "DualHeatLoRALinear",
    "DualHeatMLP",
    "SlowHeatAdamW",
    "CIFARResNet18",
    "SlowHeatCNN",
    "SlowHeatChannelTracker",
    "SlowHeatConv2d",
    "SlowHeatLinear",
    "SlowHeatMLP",
    "SlowHeatResNet18",
    "SlowHeatSGD",
    "compute_cl_metrics",
]
