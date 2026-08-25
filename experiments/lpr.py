"""Layerwise Proximal Replay gradient preconditioner.

This is a small, dependency-free adaptation of the algorithm published by
Yoo et al. (ICML 2024).  It intentionally targets only affine layers used by
the local benchmark (Conv2d, Linear and SlowHeatLinear).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _is_affine(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv2d, nn.Linear)) or (
        module.__class__.__name__ == "SlowHeatLinear"
        and hasattr(module, "weight")
    )


@dataclass
class _LayerInverse:
    matrix: Tensor
    has_bias: bool


class LPRPreconditioner:
    """Precondition gradients from replay activation covariances.

    For layer input ``Z``, LPR applies
    ``grad @ (I + omega * Z.T @ Z / N)^-1``.  Convolutional activations use
    unfolded patches and the spatial normalization from the reference code.
    """

    def __init__(
        self,
        *,
        omega: float = 4.0,
        spatial_beta: float = 2.0,
        update_frequency: int = 30,
        batch_size: int = 128,
    ) -> None:
        if omega < 0.0:
            raise ValueError("lpr_omega deve ser >= 0")
        if spatial_beta < 0.0:
            raise ValueError("lpr_spatial_beta deve ser >= 0")
        if update_frequency < 1 or batch_size < 1:
            raise ValueError("frequência e batch size do LPR devem ser positivos")
        self.omega = omega
        self.spatial_beta = spatial_beta
        self.update_frequency = update_frequency
        self.batch_size = batch_size
        self.steps = 0
        self.inverses: dict[str, _LayerInverse] = {}

    def should_update(self) -> bool:
        return not self.inverses or self.steps % self.update_frequency == 0

    @torch.no_grad()
    def update(self, model: nn.Module, replay_inputs: Tensor, *, device: str) -> int:
        """Recompute inverse preconditioners and return an operation estimate."""

        modules = {
            name: module
            for name, module in model.named_modules()
            if name and _is_affine(module)
        }
        sums: dict[str, Tensor] = {}
        sample_counts: dict[str, int] = {name: 0 for name in modules}
        spatial_counts: dict[str, int] = {}
        activations: dict[str, Tensor] = {}
        handles = []

        def capture(name: str):
            def hook(_module: nn.Module, inputs: tuple[Tensor, ...], _output: Tensor):
                activations[name] = inputs[0].detach()

            return hook

        for name, module in modules.items():
            handles.append(module.register_forward_hook(capture(name)))

        was_training = model.training
        model.eval()
        try:
            for start in range(0, len(replay_inputs), self.batch_size):
                batch = replay_inputs[start : start + self.batch_size].to(device)
                activations.clear()
                model(batch)
                for name, module in modules.items():
                    values = activations[name]
                    if isinstance(module, nn.Conv2d):
                        values = F.unfold(
                            values,
                            kernel_size=module.kernel_size,
                            dilation=module.dilation,
                            padding=module.padding,
                            stride=module.stride,
                        ).transpose(1, 2)
                        spatial = values.shape[1]
                        values = values.reshape(-1, values.shape[-1])
                    else:
                        spatial = 1
                        values = values.reshape(-1, values.shape[-1])
                    has_bias = getattr(module, "bias", None) is not None
                    if has_bias:
                        values = torch.cat(
                            (values, torch.ones_like(values[:, :1])), dim=1
                        )
                    covariance = values.T @ values
                    sums[name] = sums.get(name, torch.zeros_like(covariance)) + covariance
                    sample_counts[name] += len(batch)
                    spatial_counts[name] = spatial
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)

        operation_estimate = 0
        inverses: dict[str, _LayerInverse] = {}
        for name, module in modules.items():
            covariance = sums[name] / max(1, sample_counts[name])
            scale = self.omega / (spatial_counts[name] ** self.spatial_beta)
            dimension = covariance.shape[0]
            identity = torch.eye(
                dimension, device=covariance.device, dtype=covariance.dtype
            )
            inverses[name] = _LayerInverse(
                matrix=torch.linalg.inv(identity + scale * covariance),
                has_bias=getattr(module, "bias", None) is not None,
            )
            operation_estimate += dimension**3 + dimension**2 * sample_counts[name]
        self.inverses = inverses
        return operation_estimate

    @torch.no_grad()
    def precondition(self, model: nn.Module) -> int:
        """Apply cached LPR matrices in place and return an operation estimate."""

        operation_estimate = 0
        for name, module in model.named_modules():
            if name not in self.inverses:
                continue
            weight = getattr(module, "weight", None)
            bias = getattr(module, "bias", None)
            if weight is None or weight.grad is None:
                continue
            gradient = weight.grad.reshape(weight.shape[0], -1)
            if bias is not None:
                if bias.grad is None:
                    continue
                gradient = torch.cat((gradient, bias.grad.reshape(-1, 1)), dim=1)
            inverse = self.inverses[name].matrix
            transformed = gradient @ inverse
            if bias is not None:
                weight.grad.copy_(transformed[:, :-1].reshape_as(weight))
                bias.grad.copy_(transformed[:, -1].reshape_as(bias))
            else:
                weight.grad.copy_(transformed.reshape_as(weight))
            operation_estimate += 2 * transformed.shape[0] * inverse.numel()
        return operation_estimate

    def advance(self) -> None:
        self.steps += 1
