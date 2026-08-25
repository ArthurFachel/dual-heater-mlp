"""
SlowHeat: importância funcional por neurônio para continual learning.

Filosofia:
  O protótipo estima importância com |z * dL/dz|, normaliza a estatística dentro
  de cada camada e reserva uma fração explícita dos neurônios para plasticidade.
  A estatística é invariável à reescala recíproca de neurônios em redes
  positivamente homogêneas, ao contrário da magnitude de ativação isolada.

Protocolo de fronteira entre tasks:
  1. Treina normalmente na task k
  2. Chama model.consolidate() — funde evidência e recalcula o orçamento
  3. Reseta estatísticas internas para a próxima task

Parâmetros:
  slow_strength (float, default 3.0): β — modulação do gradiente = 1/(1+β·slow_heat)

Pipeline:
  1. z = Wx + b                              (pré-ativação)
  2. No backward: task_ema = EMA(normalize(|z * dL/dz|))
  3. consolidate(): funde a evidência e aplica o orçamento de capacidade
  4. Backward legado: grad /= (1 + β · slow_heat)

Arquitetura:
  SlowHeatLinear — camada linear com modulação de gradiente + max-consolidation
  SlowHeatMLP   — MLP sequencial usando SlowHeatLinear
"""


import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._layers import activation, validate_mlp_dims


def _adapt_budget(
    current: float,
    *,
    acquisition_score: float,
    target_score: float,
    adaptation_rate: float,
    minimum: float,
    maximum: float,
) -> float:
    values = (acquisition_score, target_score, adaptation_rate, minimum, maximum)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("parâmetros de adaptação devem ser finitos")
    if not 0.0 <= acquisition_score <= 1.0 or not 0.0 <= target_score <= 1.0:
        raise ValueError("scores de aquisição devem estar em [0, 1]")
    if adaptation_rate < 0.0:
        raise ValueError("adaptation_rate deve ser >= 0")
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("limites do orçamento devem satisfazer 0 <= min <= max <= 1")
    updated = current + adaptation_rate * (target_score - acquisition_score)
    return min(maximum, max(minimum, updated))


class _SlowHeatImportanceMixin:
    """Shared functional-importance lifecycle for linear and convolution layers."""

    slow_strength: float
    importance_decay: float
    importance_eps: float
    gradient_masking: bool
    importance_memory: Tensor
    slow_heat: Tensor
    task_ema: Tensor
    task_step: Tensor
    consolidated_tasks: Tensor
    plasticity_budget_state: Tensor

    def _initialize_importance_state(
        self,
        *,
        unit_count: int,
        slow_strength: float,
        plasticity_budget: float,
        importance_decay: float,
        importance_eps: float,
        gradient_masking: bool,
        state_device=None,
    ) -> None:
        if slow_strength < 0.0:
            raise ValueError("slow_strength deve ser >= 0")
        if not 0.0 <= plasticity_budget <= 1.0:
            raise ValueError("plasticity_budget deve estar em [0, 1]")
        if not 0.0 <= importance_decay < 1.0:
            raise ValueError("importance_decay deve estar em [0, 1)")
        if importance_eps <= 0.0:
            raise ValueError("importance_eps deve ser > 0")
        self.slow_strength = slow_strength
        self.importance_decay = importance_decay
        self.importance_eps = importance_eps
        self.gradient_masking = gradient_masking
        self.register_buffer(
            "plasticity_budget_state",
            torch.tensor(
                float(plasticity_budget),
                dtype=torch.float32,
                device=state_device,
            ),
        )
        self.register_buffer(
            "importance_memory",
            torch.zeros(unit_count, device=state_device),
        )
        self.register_buffer("slow_heat", torch.zeros(unit_count, device=state_device))
        self.register_buffer("task_ema", torch.zeros(unit_count, device=state_device))
        self.register_buffer(
            "task_step",
            torch.zeros(1, dtype=torch.long, device=state_device),
        )
        self.register_buffer(
            "consolidated_tasks",
            torch.zeros(1, dtype=torch.long, device=state_device),
        )

    def _reduce_contribution(self, contribution: Tensor) -> Tensor:
        raise NotImplementedError

    def _functional_importance_hook(
        self,
        preactivation: Tensor,
        validity_mask: Tensor | None = None,
    ):
        """Track normalized first-order contribution ``|z * dL/dz|``."""

        def hook(grad: Tensor) -> Tensor:
            with torch.no_grad():
                activation = preactivation
                detached_grad = grad.detach()
                if activation.dtype in {torch.float16, torch.bfloat16}:
                    activation = activation.float()
                if detached_grad.dtype in {torch.float16, torch.bfloat16}:
                    detached_grad = detached_grad.float()
                contribution = activation.abs() * detached_grad.abs()
                if validity_mask is not None:
                    contribution.mul_(
                        validity_mask.to(
                            device=contribution.device,
                            dtype=contribution.dtype,
                        )
                    )
                signal = self._reduce_contribution(contribution)
                normalizer = signal.mean().clamp_min(self.importance_eps)
                normalized = signal / normalizer
                step = int(self.task_step.item())
                if step == 0:
                    self.task_ema.copy_(normalized)
                else:
                    decay = min(
                        self.importance_decay,
                        1.0 - 1.0 / (1.0 + float(step)),
                    )
                    self.task_ema.mul_(decay).add_(normalized, alpha=1.0 - decay)
                self.task_step.add_(1)
            return grad

        return hook

    def _apply_capacity_budget(self) -> None:
        """Derive a scale-free protection vector with guaranteed free capacity."""

        self.slow_heat.zero_()
        max_protected = math.floor(
            (1.0 - self.plasticity_budget) * self.importance_memory.numel() + 1e-12
        )
        positive = int(torch.count_nonzero(self.importance_memory > 0.0).item())
        protected = min(max_protected, positive)
        if protected == 0:
            return
        order = torch.argsort(self.importance_memory, descending=True, stable=True)
        indices = order[:protected]
        selected = self.importance_memory[indices]
        self.slow_heat[indices] = selected / selected.max().clamp_min(
            self.importance_eps
        )

    def consolidate(self, strategy: str = "max") -> None:
        if strategy not in {"max", "mean", "sum"}:
            raise ValueError("strategy deve ser 'max', 'mean' ou 'sum'")
        if self.task_step.item() == 0:
            raise RuntimeError("não é possível consolidar uma task sem backward")
        with torch.no_grad():
            if strategy == "max":
                self.importance_memory.copy_(
                    torch.maximum(self.importance_memory, self.task_ema)
                )
            elif strategy == "mean":
                count = int(self.consolidated_tasks.item()) + 1
                self.importance_memory.add_(
                    (self.task_ema - self.importance_memory) / count
                )
            else:
                self.importance_memory.add_(self.task_ema)
            self._apply_capacity_budget()
            self.consolidated_tasks.add_(1)
            self.task_ema.zero_()
            self.task_step.zero_()

    def _gradient_mask_hook(self):
        def hook(grad: Tensor) -> Tensor:
            if not self.gradient_masking or self.slow_strength <= 0.0:
                return grad
            scale = 1.0 / (1.0 + self.slow_strength * self.slow_heat)
            return grad * scale.view(-1, *([1] * (grad.dim() - 1)))

        return hook

    def get_lr_scales(self) -> Tensor:
        return 1.0 / (1.0 + self.slow_strength * self.slow_heat)

    def capacity_metrics(self) -> dict[str, float]:
        protected = float((self.slow_heat > 0.0).float().mean().item())
        return {
            "protected_fraction": protected,
            "plastic_fraction": 1.0 - protected,
        }

    @property
    def plasticity_budget(self) -> float:
        return float(self.plasticity_budget_state.item())

    @plasticity_budget.setter
    def plasticity_budget(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("plasticity_budget deve estar em [0, 1]")
        self.plasticity_budget_state.fill_(value)

    def adapt_capacity(
        self,
        *,
        acquisition_score: float,
        target_score: float,
        adaptation_rate: float = 0.1,
        minimum: float = 0.05,
        maximum: float = 0.95,
    ) -> float:
        self.plasticity_budget = _adapt_budget(
            self.plasticity_budget,
            acquisition_score=acquisition_score,
            target_score=target_score,
            adaptation_rate=adaptation_rate,
            minimum=minimum,
            maximum=maximum,
        )
        with torch.no_grad():
            self._apply_capacity_budget()
        return self.plasticity_budget


# ─── SlowHeatLinear ─────────────────────────────

class SlowHeatLinear(_SlowHeatImportanceMixin, nn.Module):
    """
    Camada linear com utilidade funcional e proteção por capacidade.

    Rastreia `|z * dL/dz|` normalizado durante cada task. Ao final da task,
    `consolidate()` funde a evidência e deriva `slow_heat` sob um orçamento
    mínimo de plasticidade.

    Diferente do DualHeat:
      - Sem inibição lateral (sem fast heat, sem γ, sem δ)
      - Slow heat é MAX entre tasks, não média 1/n
      - Mais simples; eficácia contra forgetting ainda precisa ser validada
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        slow_strength: float = 3.0,
        plasticity_budget: float = 0.25,
        importance_decay: float = 0.99,
        importance_eps: float = 1e-8,
        gradient_masking: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._initialize_importance_state(
            unit_count=out_features,
            slow_strength=slow_strength,
            plasticity_budget=plasticity_budget,
            importance_decay=importance_decay,
            importance_eps=importance_eps,
            gradient_masking=gradient_masking,
        )

        # Parâmetros lineares padrão
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / in_features**0.5
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Hook legado de modulação do gradiente no backward
        self.weight.register_hook(self._gradient_mask_hook())
        if bias:
            self.bias.register_hook(self._gradient_mask_hook())

    def forward(self, x: Tensor) -> Tensor:
        z = F.linear(x, self.weight, self.bias)  # [B, out]

        if self.training and z.requires_grad:
            z.register_hook(self._functional_importance_hook(z.detach()))

        return z

    def _reduce_contribution(self, contribution: Tensor) -> Tensor:
        return contribution.sum(dim=tuple(range(contribution.dim() - 1)))

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"\u03b2={self.slow_strength}, plasticity={self.plasticity_budget:.3f}"
        )


# ─── SlowHeatConv2d ─────────────────────────────

class SlowHeatConv2d(_SlowHeatImportanceMixin, nn.Conv2d):
    """
    Conv2d com utilidade funcional e proteção por capacidade.

    Rastreia `|z * dL/dz|` normalizado por canal. `consolidate()` funde a
    evidência e deriva a proteção sob um orçamento mínimo de plasticidade.

    Gradient hook escala por canal: grad[cout] /= (1 + β·slow_heat[cout]).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        slow_strength: float = 3.0,
        plasticity_budget: float = 0.25,
        importance_decay: float = 0.99,
        importance_eps: float = 1e-8,
        gradient_masking: bool = True,
        stride: int | tuple[int, int] = 1,
        padding: str | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self._initialize_importance_state(
            unit_count=out_channels,
            slow_strength=slow_strength,
            plasticity_budget=plasticity_budget,
            importance_decay=importance_decay,
            importance_eps=importance_eps,
            gradient_masking=gradient_masking,
            state_device=self.weight.device,
        )
        # Hook legado de modulação do gradiente
        self.weight.register_hook(self._gradient_mask_hook())
        if self.bias is not None:
            self.bias.register_hook(self._gradient_mask_hook())

    def forward(
        self,
        x: Tensor,
        validity_mask: Tensor | None = None,
    ) -> Tensor:
        z = super().forward(x)

        broadcast_mask = None
        if validity_mask is not None:
            if validity_mask.ndim == 3:
                validity_mask = validity_mask.unsqueeze(1)
            try:
                broadcast_mask = torch.broadcast_to(validity_mask, z.shape).detach()
            except RuntimeError as error:
                raise ValueError(
                    "validity_mask deve ser compatível com [B, C_out, H, W]"
                ) from error
            if not torch.isfinite(broadcast_mask).all():
                raise ValueError("validity_mask deve conter somente valores finitos")
            if torch.any(broadcast_mask < 0) or torch.any(broadcast_mask > 1):
                raise ValueError("validity_mask deve estar em [0, 1]")

        if self.training and z.requires_grad:
            z.register_hook(
                self._functional_importance_hook(z.detach(), broadcast_mask)
            )

        return z

    def _reduce_contribution(self, contribution: Tensor) -> Tensor:
        return contribution.sum(dim=(0, 2, 3))

    def extra_repr(self) -> str:
        base = nn.Conv2d.extra_repr(self)
        return (
            f"{base}, \u03b2={self.slow_strength}, "
            f"plasticity={self.plasticity_budget:.3f}"
        )


# ─── SlowHeatCNN ─────────────────────────────────

class SlowHeatCNN(nn.Module):
    """Sequential CNN with functional, channel-wise SlowHeat protection.

    The adaptive pooling fixes the Conv→Linear mapping. Optimizer registration
    repeats each source-channel factor over every pooled spatial position.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        channels: tuple[int, int] = (32, 64),
        pooled_size: int | tuple[int, int] = (2, 2),
        act: str = "relu",
        slow_strength: float = 3.0,
        plasticity_budget: float = 0.25,
        importance_decay: float = 0.99,
        importance_eps: float = 1e-8,
        protect_output: bool = True,
        output_slow_strength: float | None = None,
    ) -> None:
        super().__init__()
        if (
            not isinstance(in_channels, int)
            or isinstance(in_channels, bool)
            or in_channels < 1
        ):
            raise ValueError("in_channels deve ser um inteiro positivo")
        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes < 1
        ):
            raise ValueError("num_classes deve ser um inteiro positivo")
        if len(channels) != 2 or any(
            not isinstance(width, int)
            or isinstance(width, bool)
            or width < 1
            for width in channels
        ):
            raise ValueError("channels deve conter dois inteiros positivos")
        if isinstance(pooled_size, int):
            pooled_shape = (pooled_size, pooled_size)
        else:
            pooled_shape = tuple(pooled_size)
        if len(pooled_shape) != 2 or any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            for size in pooled_shape
        ):
            raise ValueError("pooled_size deve conter dimensões positivas")

        common = {
            "slow_strength": slow_strength,
            "plasticity_budget": plasticity_budget,
            "importance_decay": importance_decay,
            "importance_eps": importance_eps,
        }
        self.conv1 = SlowHeatConv2d(
            in_channels,
            channels[0],
            kernel_size=3,
            padding=1,
            **common,
        )
        self.conv2 = SlowHeatConv2d(
            channels[0],
            channels[1],
            kernel_size=3,
            padding=1,
            **common,
        )
        self.activation1 = activation(act)
        self.activation2 = activation(act)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(pooled_shape)
        self.flatten = nn.Flatten()
        classifier_features = channels[1] * pooled_shape[0] * pooled_shape[1]
        if protect_output:
            classifier_common = dict(common)
            if output_slow_strength is not None:
                classifier_common["slow_strength"] = output_slow_strength
            self.classifier: nn.Module = SlowHeatLinear(
                classifier_features,
                num_classes,
                **classifier_common,
            )
        else:
            self.classifier = nn.Linear(classifier_features, num_classes)
        self.pooled_size = pooled_shape

    def forward_features(self, x: Tensor) -> Tensor:
        """Return penultimate features for ridge/classifier-only methods."""

        x = self.pool(self.activation1(self.conv1(x)))
        x = self.pool(self.activation2(self.conv2(x)))
        x = self.adaptive_pool(x)
        return self.flatten(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x))

    def get_slow_layers(
        self,
    ) -> list[SlowHeatConv2d | SlowHeatLinear]:
        return [
            module
            for module in (self.conv1, self.conv2, self.classifier)
            if isinstance(module, (SlowHeatConv2d, SlowHeatLinear))
        ]

    def consolidate(self, strategy: str = "max") -> None:
        for layer in self.get_slow_layers():
            layer.consolidate(strategy=strategy)

    def get_lr_scales(self) -> list[Tensor]:
        return [layer.get_lr_scales() for layer in self.get_slow_layers()]

    def adapt_capacity(
        self,
        *,
        acquisition_score: float,
        target_score: float,
        adaptation_rate: float = 0.1,
        minimum: float = 0.05,
        maximum: float = 0.95,
    ) -> list[float]:
        return [
            layer.adapt_capacity(
                acquisition_score=acquisition_score,
                target_score=target_score,
                adaptation_rate=adaptation_rate,
                minimum=minimum,
                maximum=maximum,
            )
            for layer in self.get_slow_layers()
        ]


# ─── SlowHeatMLP ────────────────────────────────

class SlowHeatMLP(nn.Sequential):
    """
    MLP com SlowHeatLinear nas ocultas e, por padrão, também na saída.

    Uso:
        net = SlowHeatMLP(784, 256, 128, 10, act="gelu", slow_strength=3.0)

    Entre tasks:
        net.consolidate()
    """
    def __init__(
        self,
        *dims: int,
        act: str = "relu",
        slow_strength: float = 3.0,
        plasticity_budget: float = 0.25,
        protect_output: bool = True,
        output_slow_strength: float | None = None,
    ):
        validate_mlp_dims(dims)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(
                SlowHeatLinear(
                    dims[i], dims[i + 1],
                    slow_strength=slow_strength,
                    plasticity_budget=plasticity_budget,
                )
            )
            layers.append(activation(act))
        if protect_output:
            layers.append(
                SlowHeatLinear(
                    dims[-2],
                    dims[-1],
                    slow_strength=(
                        slow_strength
                        if output_slow_strength is None
                        else output_slow_strength
                    ),
                    plasticity_budget=plasticity_budget,
                )
            )
        else:
            layers.append(nn.Linear(dims[-2], dims[-1]))
        super().__init__(*layers)

    def get_slow_layers(self) -> list[SlowHeatLinear]:
        return [m for m in self.modules() if isinstance(m, SlowHeatLinear)]

    def consolidate(self, strategy: str = "max"):
        """Consolida todos os SlowHeatLinear com a mesma estratégia."""
        for layer in self.get_slow_layers():
            layer.consolidate(strategy=strategy)

    def get_lr_scales(self) -> list[Tensor]:
        return [m.get_lr_scales() for m in self.get_slow_layers()]

    def adapt_capacity(
        self,
        *,
        acquisition_score: float,
        target_score: float,
        adaptation_rate: float = 0.1,
        minimum: float = 0.05,
        maximum: float = 0.95,
    ) -> list[float]:
        """Apply the same validation-driven capacity controller to every layer."""

        return [
            layer.adapt_capacity(
                acquisition_score=acquisition_score,
                target_score=target_score,
                adaptation_rate=adaptation_rate,
                minimum=minimum,
                maximum=maximum,
            )
            for layer in self.get_slow_layers()
        ]
