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

# ─── helpers ────────────────────────────────────

def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unknown activation: {name}")


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


# ─── SlowHeatLinear ─────────────────────────────

class SlowHeatLinear(nn.Module):
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
        if slow_strength < 0.0:
            raise ValueError("slow_strength deve ser >= 0")
        if not 0.0 <= plasticity_budget <= 1.0:
            raise ValueError("plasticity_budget deve estar em [0, 1]")
        if not 0.0 <= importance_decay < 1.0:
            raise ValueError("importance_decay deve estar em [0, 1)")
        if importance_eps <= 0.0:
            raise ValueError("importance_eps deve ser > 0")
        self.in_features = in_features
        self.out_features = out_features
        self.slow_strength = slow_strength
        self.importance_decay = importance_decay
        self.importance_eps = importance_eps
        self.gradient_masking = gradient_masking
        self.register_buffer(
            "plasticity_budget_state",
            torch.tensor(float(plasticity_budget), dtype=torch.float32),
        )

        # Parâmetros lineares padrão
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / in_features**0.5
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Evidência consolidada e intensidade de proteção derivada do orçamento.
        self.register_buffer("importance_memory", torch.zeros(out_features))
        self.register_buffer("slow_heat", torch.zeros(out_features))

        # Estatísticas intra-task: EMA suave da utilidade funcional normalizada.
        self.register_buffer("task_ema", torch.zeros(out_features))
        self.register_buffer("task_step", torch.zeros(1, dtype=torch.long))
        self.register_buffer("consolidated_tasks", torch.zeros(1, dtype=torch.long))

        # Hook legado de modulação do gradiente no backward
        self.weight.register_hook(self._gradient_mask_hook())
        if bias:
            self.bias.register_hook(self._gradient_mask_hook())

    def forward(self, x: Tensor) -> Tensor:
        z = F.linear(x, self.weight, self.bias)  # [B, out]

        if self.training and z.requires_grad:
            z.register_hook(self._functional_importance_hook(z.detach()))

        return z

    def _functional_importance_hook(self, preactivation: Tensor):
        """Track normalized first-order contribution ``|z * dL/dz|``."""

        def hook(grad: Tensor) -> Tensor:
            with torch.no_grad():
                reduce_dims = tuple(range(grad.dim() - 1))
                contribution = preactivation.abs() * grad.detach().abs()
                if contribution.dtype in {torch.float16, torch.bfloat16}:
                    contribution = contribution.float()
                signal = contribution.sum(dim=reduce_dims)
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
        max_protected = int(
            (1.0 - self.plasticity_budget) * self.importance_memory.numel()
        )
        positive = int(torch.count_nonzero(self.importance_memory > 0.0).item())
        protected = min(max_protected, positive)
        if protected == 0:
            return
        order = torch.argsort(self.importance_memory, descending=True, stable=True)
        indices = order[:protected]
        selected = self.importance_memory[indices]
        scale = selected.max().clamp_min(self.importance_eps)
        self.slow_heat[indices] = selected / scale

    def consolidate(self, strategy: str = "max"):
        """Consolida a importância da task e reseta só estatísticas intra-task."""
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
        """Escala gradiente: 1/(1 + β·slow_heat)."""
        def hook(grad: Tensor) -> Tensor:
            if not self.gradient_masking or self.slow_strength <= 0.0:
                return grad
            scale = 1.0 / (1.0 + self.slow_strength * self.slow_heat)
            return grad * scale.view(-1, *([1] * (grad.dim() - 1)))
        return hook

    def get_lr_scales(self) -> Tensor:
        """Fatores de plasticidade; só são LR efetivos sob update compatível."""
        return 1.0 / (1.0 + self.slow_strength * self.slow_heat)

    def capacity_metrics(self) -> dict[str, float]:
        """Report the realized protected/plastic fractions for diagnostics."""

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
        """Adjust free capacity from an external validation acquisition score."""

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

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"\u03b2={self.slow_strength}, plasticity={self.plasticity_budget:.3f}"
        )


# ─── SlowHeatConv2d ─────────────────────────────

class SlowHeatConv2d(nn.Module):
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
        kernel_size: int,
        slow_strength: float = 3.0,
        plasticity_budget: float = 0.25,
        importance_decay: float = 0.99,
        importance_eps: float = 1e-8,
        gradient_masking: bool = True,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
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
            torch.tensor(float(plasticity_budget), dtype=torch.float32),
        )
        self._stride = stride
        self._padding = padding

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
            / (in_channels * kernel_size * kernel_size)**0.5
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("importance_memory", torch.zeros(out_channels))
        self.register_buffer("slow_heat", torch.zeros(out_channels))
        # Within-task EMA
        self.register_buffer("task_ema", torch.zeros(out_channels))
        self.register_buffer("task_step", torch.zeros(1, dtype=torch.long))
        self.register_buffer("consolidated_tasks", torch.zeros(1, dtype=torch.long))

        # Hook legado de modulação do gradiente
        self.weight.register_hook(self._gradient_mask_hook())
        if bias:
            self.bias.register_hook(self._gradient_mask_hook())

    def forward(self, x) -> Tensor:
        z = F.conv2d(x, self.weight, self.bias, stride=self.stride,
                     padding=self.padding)

        if self.training and z.requires_grad:
            z.register_hook(self._functional_importance_hook(z.detach()))

        return z

    def _functional_importance_hook(self, preactivation: Tensor):
        def hook(grad: Tensor) -> Tensor:
            with torch.no_grad():
                contribution = preactivation.abs() * grad.detach().abs()
                if contribution.dtype in {torch.float16, torch.bfloat16}:
                    contribution = contribution.float()
                signal = contribution.sum(dim=[0, 2, 3])
                normalized = signal / signal.mean().clamp_min(self.importance_eps)
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
        self.slow_heat.zero_()
        max_protected = int(
            (1.0 - self.plasticity_budget) * self.importance_memory.numel()
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

    @property
    def stride(self):
        return getattr(self, '_stride', 1)
    @stride.setter
    def stride(self, v):
        self._stride = v
    @property
    def padding(self):
        return getattr(self, '_padding', 0)
    @padding.setter
    def padding(self, v):
        self._padding = v

    def consolidate(self, strategy: str = "max"):
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
            # grad shape: [out, in, kH, kW] or [out] for bias
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

    def extra_repr(self) -> str:
        return (
            f"in={self.weight.shape[1]}x{self.weight.shape[2]}, "
            f"out={self.weight.shape[0]}, "
            f"\u03b2={self.slow_strength}, plasticity={self.plasticity_budget:.3f}"
        )


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
    ):
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(
                SlowHeatLinear(
                    dims[i], dims[i + 1],
                    slow_strength=slow_strength,
                    plasticity_budget=plasticity_budget,
                )
            )
            layers.append(_act(act))
        if protect_output:
            layers.append(
                SlowHeatLinear(
                    dims[-2],
                    dims[-1],
                    slow_strength=slow_strength,
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
