"""Neuron-level plasticity mechanisms for continual learning research."""

from .dual_heat import DualHeatLinear, DualHeatMLP
from .fast_heat import FastHeatConfig, FastHeatGate, get_fast_states, reset_fast_heat
from .lora import DualHeatLoRALinear
from .metrics import CLMetrics, compute_cl_metrics
from .optim import SlowHeatAdamW, SlowHeatSGD
from .slow_heat import (
    FunctionalDualHeatCNN,
    FunctionalDualHeatMLP,
    SlowHeatChannelTracker,
    SlowHeatCNN,
    SlowHeatConv2d,
    SlowHeatLinear,
    SlowHeatMLP,
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

_QWEN_EXPORTS = {
    "QwenSlowHeatConfig",
    "SlowHeatQwen2ForSequenceClassification",
}


def __getattr__(name: str):
    """Load optional BERT/PEFT integrations only when requested."""

    if name in _BERT_EXPORTS:
        from . import bert

        return getattr(bert, name)
    if name in _QWEN_EXPORTS:
        from . import qwen

        return getattr(qwen, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CLMetrics",
    "DualHeatLinear",
    "DualHeatLoRALinear",
    "DualHeatMLP",
    "FastHeatConfig",
    "FastHeatGate",
    "FunctionalDualHeatCNN",
    "FunctionalDualHeatMLP",
    "SlowHeatAdamW",
    "SlowHeatAttentionTracker",
    "SlowHeatCNN",
    "SlowHeatChannelTracker",
    "SlowHeatConv2d",
    "SlowHeatFFNTracker",
    "SlowHeatLinear",
    "SlowHeatMLP",
    "SlowHeatSGD",
    "compute_cl_metrics",
    "get_fast_states",
    "reset_fast_heat",
]
