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
from .transformer import SlowHeatAttentionTracker, SlowHeatFFNTracker

_BERT_EXPORTS = {
    "BertSlowHeatConfig",
    "ExactSlowHeatLoRAConfig",
    "SlowHeatBertForSequenceClassification",
    "build_exact_slowheat_lora",
    "exact_lora_mask_bindings",
    "register_exact_lora_masks",
}


def __getattr__(name: str):
    """Load optional BERT/PEFT integrations only when requested."""

    if name in _BERT_EXPORTS:
        from . import bert

        return getattr(bert, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    "SlowHeatAttentionTracker",
    "SlowHeatCNN",
    "SlowHeatChannelTracker",
    "SlowHeatConv2d",
    "SlowHeatFFNTracker",
    "SlowHeatLinear",
    "SlowHeatMLP",
    "SlowHeatResNet18",
    "SlowHeatSGD",
    "SlowHeatVGG11",
    "compute_cl_metrics",
    "get_fast_states",
    "reset_fast_heat",
]
