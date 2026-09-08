"""Functional SlowHeat trackers for Transformer hidden units and attention heads."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from .slow_heat import _SlowHeatImportanceMixin

AttentionCombination = Literal["max", "mean", "sum"]


def _validated_token_mask(
    validity_mask: Tensor | None,
    tensor: Tensor,
) -> Tensor | None:
    """Return a detached ``[batch, tokens, 1]`` validity mask."""

    if validity_mask is None:
        return None
    if tensor.ndim != 3:
        raise ValueError("o tensor observado deve ter forma [B, T, U]")
    mask = validity_mask
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim != 2 or mask.shape != tensor.shape[:2]:
        raise ValueError("validity_mask deve ter forma [B, T]")
    if not torch.isfinite(mask).all():
        raise ValueError("validity_mask deve conter somente valores finitos")
    if torch.any(mask < 0) or torch.any(mask > 1):
        raise ValueError("validity_mask deve estar em [0, 1]")
    return mask.detach().to(device=tensor.device).unsqueeze(-1)


class SlowHeatFFNTracker(_SlowHeatImportanceMixin, nn.Module):
    """Track post-activation functional utility for Transformer FFN units."""

    def __init__(
        self,
        units: int,
        *,
        slow_strength: float = 3.0,
        plasticity_budget: float = 0.25,
        importance_decay: float = 0.99,
        importance_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not isinstance(units, int) or isinstance(units, bool) or units < 1:
            raise ValueError("units deve ser um inteiro positivo")
        self.units = units
        self._initialize_importance_state(
            unit_count=units,
            slow_strength=slow_strength,
            plasticity_budget=plasticity_budget,
            importance_decay=importance_decay,
            importance_eps=importance_eps,
            gradient_masking=False,
        )

    def observe(
        self,
        hidden: Tensor,
        validity_mask: Tensor | None = None,
    ) -> Tensor:
        if hidden.ndim < 2 or hidden.shape[-1] != self.units:
            raise ValueError(
                "SlowHeatFFNTracker requer tensor [..., units] compatível"
            )
        mask = (
            _validated_token_mask(validity_mask, hidden)
            if validity_mask is not None
            else None
        )
        if self.training and hidden.requires_grad:
            hidden.register_hook(
                self._functional_importance_hook(hidden.detach(), mask)
            )
        return hidden

    forward = observe

    def _reduce_contribution(self, contribution: Tensor) -> Tensor:
        return contribution.sum(dim=tuple(range(contribution.ndim - 1)))

    def extra_repr(self) -> str:
        return (
            f"units={self.units}, beta={self.slow_strength}, "
            f"plasticity={self.plasticity_budget:.3f}"
        )


class SlowHeatAttentionTracker(_SlowHeatImportanceMixin, nn.Module):
    """Track one functional-importance value per self-attention head.

    Q, K, V and the merged per-head output are observed without materializing
    attention probabilities. Each signal is normalized independently, then the
    four vectors are combined once per completed backward pass.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        slow_strength: float = 3.0,
        plasticity_budget: float = 0.25,
        importance_decay: float = 0.99,
        importance_eps: float = 1e-8,
        combination: AttentionCombination = "max",
    ) -> None:
        super().__init__()
        for name, value in {"num_heads": num_heads, "head_dim": head_dim}.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} deve ser um inteiro positivo")
        if combination not in {"max", "mean", "sum"}:
            raise ValueError("combination deve ser 'max', 'mean' ou 'sum'")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim
        self.combination: AttentionCombination = combination
        self._initialize_importance_state(
            unit_count=num_heads,
            slow_strength=slow_strength,
            plasticity_budget=plasticity_budget,
            importance_decay=importance_decay,
            importance_eps=importance_eps,
            gradient_masking=False,
        )

    def _reduce_contribution(self, contribution: Tensor) -> Tensor:
        shaped = contribution.reshape(
            contribution.shape[0],
            contribution.shape[1],
            self.num_heads,
            self.head_dim,
        )
        return shaped.sum(dim=(0, 1, 3))

    def _update_task_ema(self, signal: Tensor) -> None:
        with torch.no_grad():
            step = int(self.task_step.item())
            if step == 0:
                self.task_ema.copy_(signal)
            else:
                decay = min(
                    self.importance_decay,
                    1.0 - 1.0 / (1.0 + float(step)),
                )
                self.task_ema.mul_(decay).add_(signal, alpha=1.0 - decay)
            self.task_step.add_(1)

    def observe(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        head_output: Tensor,
        validity_mask: Tensor | None = None,
    ) -> Tensor:
        tensors = {
            "query": query,
            "key": key,
            "value": value,
            "output": head_output,
        }
        for name, tensor in tensors.items():
            if tensor.ndim != 3 or tensor.shape[-1] != self.hidden_size:
                raise ValueError(
                    f"{name} deve ter forma [B, T, {self.hidden_size}]"
                )
            if tensor.shape[:2] != head_output.shape[:2]:
                raise ValueError("todos os sinais de atenção devem alinhar B e T")
        mask = _validated_token_mask(validity_mask, head_output)
        if not self.training or not all(tensor.requires_grad for tensor in tensors.values()):
            return head_output

        normalized_signals: dict[str, Tensor] = {}

        def make_hook(name: str, activation: Tensor):
            def hook(grad: Tensor) -> Tensor:
                with torch.no_grad():
                    local_activation = activation
                    local_grad = grad.detach()
                    if local_activation.dtype in {torch.float16, torch.bfloat16}:
                        local_activation = local_activation.float()
                    if local_grad.dtype in {torch.float16, torch.bfloat16}:
                        local_grad = local_grad.float()
                    contribution = local_activation.abs() * local_grad.abs()
                    if mask is not None:
                        contribution.mul_(
                            mask.to(
                                device=contribution.device,
                                dtype=contribution.dtype,
                            )
                        )
                    signal = self._reduce_contribution(contribution)
                    normalized_signals[name] = signal / signal.mean().clamp_min(
                        self.importance_eps
                    )
                    if len(normalized_signals) == len(tensors):
                        stacked = torch.stack(
                            [normalized_signals[item] for item in tensors]
                        )
                        if self.combination == "max":
                            combined = stacked.amax(dim=0)
                        elif self.combination == "mean":
                            combined = stacked.mean(dim=0)
                        else:
                            combined = stacked.sum(dim=0)
                        self._update_task_ema(combined)
                return grad

            return hook

        for name, tensor in tensors.items():
            tensor.register_hook(make_hook(name, tensor.detach()))
        return head_output

    def expanded_head_scales(self, *, hard: bool = False) -> Tensor:
        factors = (
            (self.slow_heat <= 0.0).to(dtype=self.slow_heat.dtype)
            if hard
            else self.get_lr_scales()
        )
        return factors.repeat_interleave(self.head_dim)

    def extra_repr(self) -> str:
        return (
            f"heads={self.num_heads}, head_dim={self.head_dim}, "
            f"combine={self.combination}, beta={self.slow_strength}, "
            f"plasticity={self.plasticity_budget:.3f}"
        )
