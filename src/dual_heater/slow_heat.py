"""
SlowHeat: importância por neurônio consolidada por MAX para continual learning.

Filosofia:
  O protótipo investiga uma estimativa local de importância baseada em ativação.
  A consolidação MAX mantém importância monotônica entre tasks, mas isso não
  garante retenção funcional: outras camadas e parâmetros compartilhados ainda
  podem alterar o comportamento da rede.

Protocolo de fronteira entre tasks:
  1. Treina normalmente na task k
  2. Chama model.consolidate() — faz max do slow_heat anterior com a EMA da task atual
  3. Reseta estatísticas internas para a próxima task

Parâmetros:
  slow_strength (float, default 3.0): β — modulação do gradiente = 1/(1+β·slow_heat)

Pipeline:
  1. z = Wx + b                              (pré-ativação)
  2. Se training: task_ema = EMA(|z|)        (rastreio intra-task)
  3. consolidate(): slow_heat = max(slow_heat, task_ema)
  4. Backward legado: grad /= (1 + β · slow_heat)

Arquitetura:
  SlowHeatLinear — camada linear com modulação de gradiente + max-consolidation
  SlowHeatMLP   — MLP sequencial usando SlowHeatLinear
"""


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


# ─── SlowHeatLinear ─────────────────────────────

class SlowHeatLinear(nn.Module):
    """
    Camada linear com proteção local por neurônio via MAX-consolidation.

    Rastreia |z| (magnitude da pré-ativação) durante o treino de cada task.
    Ao final da task, consolidate() funde as estatísticas da task atual no
    slow_heat via MAX element-wise, garantindo proteção monotônica.

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
        gradient_masking: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if slow_strength < 0.0:
            raise ValueError("slow_strength deve ser >= 0")
        self.in_features = in_features
        self.out_features = out_features
        self.slow_strength = slow_strength
        self.gradient_masking = gradient_masking

        # Parâmetros lineares padrão
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / in_features**0.5
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Importância por neurônio consolidada entre estatísticas de task
        self.register_buffer("slow_heat", torch.zeros(out_features))

        # Estatísticas intra-task: EMA suave de |z|
        self.register_buffer("task_ema", torch.zeros(out_features))
        self.register_buffer("task_step", torch.zeros(1, dtype=torch.long))
        self.register_buffer("consolidated_tasks", torch.zeros(1, dtype=torch.long))

        # Hook legado de modulação do gradiente no backward
        self.weight.register_hook(self._gradient_mask_hook())
        if bias:
            self.bias.register_hook(self._gradient_mask_hook())

    def forward(self, x: Tensor) -> Tensor:
        z = F.linear(x, self.weight, self.bias)  # [B, out]

        if self.training:
            with torch.no_grad():
                reduce_dims = tuple(range(z.dim() - 1))
                mag = z.detach().abs().mean(dim=reduce_dims)  # [out]

                step = self.task_step.item()
                if step == 0:
                    # Primeiro passo da task: copia direto
                    self.task_ema.copy_(mag)
                else:
                    # Média online até 100 passos; depois EMA com decay 0.99
                    decay = min(0.99, 1.0 - 1.0 / (1.0 + float(step)))
                    self.task_ema.mul_(decay).add_(mag, alpha=1.0 - decay)
                self.task_step.add_(1)

        return z

    def consolidate(self, strategy: str = "max"):
        """Consolida a importância da task e reseta só estatísticas intra-task."""
        if strategy not in {"max", "mean", "sum"}:
            raise ValueError("strategy deve ser 'max', 'mean' ou 'sum'")
        with torch.no_grad():
            if strategy == "max":
                self.slow_heat.copy_(torch.maximum(self.slow_heat, self.task_ema))
            elif strategy == "mean":
                count = int(self.consolidated_tasks.item()) + 1
                self.slow_heat.add_((self.task_ema - self.slow_heat) / count)
            else:
                self.slow_heat.add_(self.task_ema)
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

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"\u03b2={self.slow_strength}"
        )


# ─── SlowHeatConv2d ─────────────────────────────

class SlowHeatConv2d(nn.Module):
    """
    Conv2d com proteção local por canal via MAX-consolidation.

    Rastreia |z| médio por canal (batch + espacial) dentro de cada task.
    consolidate() faz slow_heat = max(slow_heat, task_ema).

    Gradient hook escala por canal: grad[cout] /= (1 + β·slow_heat[cout]).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        slow_strength: float = 3.0,
        gradient_masking: bool = True,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        if slow_strength < 0.0:
            raise ValueError("slow_strength deve ser >= 0")
        self.slow_strength = slow_strength
        self.gradient_masking = gradient_masking
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

        # Per-channel importance (max across all task means)
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

        if self.training:
            with torch.no_grad():
                # |z| mean across [batch, H, W] — per channel
                mag = z.detach().abs().mean(dim=[0, 2, 3])  # [out_channels]
                step = self.task_step.item()
                if step == 0:
                    self.task_ema.copy_(mag)
                else:
                    decay = min(0.99, 1.0 - 1.0 / (1.0 + float(step)))
                    self.task_ema.mul_(decay).add_(mag, alpha=1.0 - decay)
                self.task_step.add_(1)

        return z

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
        with torch.no_grad():
            if strategy == "max":
                self.slow_heat.copy_(torch.maximum(self.slow_heat, self.task_ema))
            elif strategy == "mean":
                count = int(self.consolidated_tasks.item()) + 1
                self.slow_heat.add_((self.task_ema - self.slow_heat) / count)
            else:
                self.slow_heat.add_(self.task_ema)
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

    def extra_repr(self) -> str:
        return (
            f"in={self.weight.shape[1]}x{self.weight.shape[2]}, "
            f"out={self.weight.shape[0]}, "
            f"\u03b2={self.slow_strength}"
        )


# ─── SlowHeatMLP ────────────────────────────────

class SlowHeatMLP(nn.Sequential):
    """
    MLP com camadas SlowHeatLinear nas ocultas.

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
    ):
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(
                SlowHeatLinear(
                    dims[i], dims[i + 1],
                    slow_strength=slow_strength,
                )
            )
            layers.append(_act(act))
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
