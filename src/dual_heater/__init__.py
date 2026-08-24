"""Neuron-level plasticity mechanisms for continual learning research."""

from .dual_heat import DualHeatLinear, DualHeatMLP
from .lora import DualHeatLoRALinear
from .metrics import CLMetrics, compute_cl_metrics
from .optim import SlowHeatAdamW, SlowHeatSGD
from .slow_heat import SlowHeatCNN, SlowHeatConv2d, SlowHeatLinear, SlowHeatMLP

__all__ = [
    "CLMetrics",
    "DualHeatLinear",
    "DualHeatLoRALinear",
    "DualHeatMLP",
    "SlowHeatAdamW",
    "SlowHeatCNN",
    "SlowHeatConv2d",
    "SlowHeatLinear",
    "SlowHeatMLP",
    "SlowHeatSGD",
    "compute_cl_metrics",
]
