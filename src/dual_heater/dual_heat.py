"""
DualHeat legado v3: inibição lateral + decay ativo + modulação local de plasticidade
                           + slow heat com memória limitada (forgetting)

Esta implementação histórica permanece inalterada para compatibilidade. Novos
experimentos devem usar ``FunctionalDualHeatMLP`` (ou as variantes CNN/VGG/
ResNet), que combinam FastHeat de ativação com Functional SlowHeat.

Novidade vs v2 (única mudança de comportamento):

  v2                                  →  v3
  ──                                     ──
  slow_heat = média amostral            slow_heat = média amostral com
  verdadeira (1/t), memória infinita.      janela de memória efetiva (opcional).
  Um neurônio que foi importante            Após slow_window passos, o
  em qualquer ponto do passado                estimador vira uma EMA de taxa
  continua protegido para sempre.            η = 1 - 1/slow_window, permitindo
                                               que neurônios "esfriem" quando
                                               deixam de ser usados.

Implementação: "capped incremental mean". Em vez de recalcular a fórmula
como uma EMA de taxa fixa (o que sofreria de cold-start bias, já que o
buffer começa em zero), simplesmente limitamos o denominador n:

    n_eff(t) = min(t, slow_window)
    heat_slow(t) = heat_slow(t-1) + (|output(t)| - heat_slow(t-1)) / n_eff(t)

Para t <= slow_window isso é EXATAMENTE a média amostral verdadeira (não
viesada, idêntica à v2 — ver seção 3.4 do documento teórico). Para
t > slow_window, a fórmula se torna uma EMA padrão com janela efetiva de
slow_window passos e half-life ≈ 0.693 * slow_window passos. A transição
entre os dois regimes é suave (mesma fórmula, só o teto do denominador
muda) e não introduz descontinuidade nem viés adicional.

slow_window=None preserva o comportamento v2 (memória infinita, sem
esquecimento) — retrocompatível por padrão.

Pipeline por passo (inalterado, exceto passo 4):
  1. z = Wx + b                          (pré-ativação)
  2. output = z / (1 + γ·mean_others)    (inibição lateral divisiva)
  3. fast_heat = max(0, α·|output| + (1-α)·fast_heat − δ)
  4. slow_heat += (|output| − slow_heat) / min(n, slow_window)
  5. grad /= (1 + β·slow_heat)           (hook legado de gradiente)
"""


import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._layers import activation, validate_finite_hyperparameters, validate_mlp_dims


class DualHeatLinear(nn.Module):
    """
    Linear layer com inibição lateral + importância por neurônio + slow heat com
    memória limitada (forgetting) opcional.

    Parâmetros
    ----------
    fast_decay : float (0.85-0.97)
        α — velocidade do EMA rápido. Meia-vida ≈ 1/(1-α) passos.

    fast_strength : float (1.0-5.0)
        γ — força da inibição lateral divisiva.

    fast_decay_rate : float (0.02-0.08)
        δ — decaimento ativo por passo. Cria threshold:
            neurônio com |output| médio < δ/(1-α) → heat = 0.

    slow_strength : float (0.0-5.0)
        β — força de modulação: grad /= (1 + β·slow_heat).
        Default 2.0 (CL-optimized).

    slow_window : int | None
        Janela de memória efetiva do slow_heat, em passos de treino.
        None (default) = memória infinita, comportamento v2 (nunca
        esquece — média amostral verdadeira desde o início do treino).
        Um inteiro (ex: 2000) limita a memória: depois de `slow_window`
        passos, o slow_heat passa a rastrear aproximadamente só os
        últimos `slow_window` passos, e a proteção de um neurônio
        decai naturalmente se ele parar de ser usado.
        Regra prática: escolha slow_window na ordem de grandeza do
        número de passos de treino por task (ou um pouco maior, se
        quiser reter importância por mais de uma task).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        fast_decay: float = 0.93,
        fast_strength: float = 2.0,
        fast_decay_rate: float = 0.04,
        slow_strength: float = 2.0,
        slow_window: int | None = None,
        importance: str = "activation",
        bias: bool = True,
    ):
        super().__init__()
        validate_finite_hyperparameters(
            fast_decay=fast_decay,
            fast_strength=fast_strength,
            fast_decay_rate=fast_decay_rate,
            slow_strength=slow_strength,
        )
        if not 0.0 <= fast_decay < 1.0:
            raise ValueError("fast_decay deve estar no intervalo [0, 1)")
        if fast_strength < 0.0:
            raise ValueError("fast_strength deve ser >= 0")
        if fast_decay_rate < 0.0:
            raise ValueError("fast_decay_rate deve ser >= 0")
        if slow_strength < 0.0:
            raise ValueError("slow_strength deve ser >= 0")
        if slow_window is not None and slow_window < 1:
            raise ValueError("slow_window deve ser >= 1 ou None")
        if importance not in {"activation", "sensitivity"}:
            raise ValueError("importance deve ser 'activation' ou 'sensitivity'")

        self.in_features = in_features
        self.out_features = out_features
        self.fast_decay = fast_decay
        self.fast_strength = fast_strength
        self.fast_decay_rate = fast_decay_rate
        self.slow_strength = slow_strength
        self.slow_window = slow_window
        self.importance = importance

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / in_features**0.5
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Fast heat (pós-inibição, com decay ativo)
        self.register_buffer("fast_heat", torch.zeros(out_features))
        # Slow heat (média amostral com janela limitada opcional)
        self.register_buffer("slow_heat", torch.zeros(out_features))
        self.register_buffer("slow_n", torch.ones(1))

        # Hook legado de modulação do gradiente
        self.weight.register_hook(self._gradient_mask_hook())
        if bias:
            self.bias.register_hook(self._gradient_mask_hook())

    def forward(self, x: Tensor) -> Tensor:
        # 1. Pré-ativação
        z = F.linear(x, self.weight, self.bias)  # [B, out]

        # 2. Inibição LATERAL divisiva — heat dos OUTROS neurônios
        if self.fast_strength > 0.0 and self.out_features > 1 and self.training:
            with torch.no_grad():
                sum_h = self.fast_heat.sum()
                mean_others = (sum_h - self.fast_heat) / (self.out_features - 1)
            view_shape = (1,) * (z.dim() - 1) + (self.out_features,)
            output = z / (1.0 + self.fast_strength * mean_others.view(view_shape))
        else:
            output = z

        # 3-4. Atualiza heats (só em treino, pós-inibição)
        if self.training:
            with torch.no_grad():
                reduce_dims = tuple(range(output.dim() - 1))
                post_mag = output.detach().abs().mean(dim=reduce_dims)  # [out]

                # Fast: EMA + decay ativo (inalterado)
                self.fast_heat.mul_(self.fast_decay).add_(
                    post_mag, alpha=1.0 - self.fast_decay
                ).sub_(self.fast_decay_rate).clamp_(min=0.0)

                # Slow: capped incremental mean (memória infinita ou limitada)
                n_true = self.slow_n.item()
                n_eff = min(n_true, self.slow_window) if self.slow_window else n_true
                if self.importance == "activation":
                    self.slow_heat.add_((post_mag - self.slow_heat) / n_eff)
                    self.slow_n += 1

            if self.importance == "sensitivity" and output.requires_grad:
                output.register_hook(self._sensitivity_hook(output.detach()))

        return output

    def _sensitivity_hook(self, activation: Tensor):
        """Cria hook que atualiza E[|activation| * |dL/dactivation|]."""
        def hook(grad: Tensor) -> Tensor:
            with torch.no_grad():
                reduce_dims = tuple(range(grad.dim() - 1))
                signal = (grad.detach().abs() * activation.abs()).sum(dim=reduce_dims)
                assert isinstance(self.slow_n, Tensor)
                n_true = float(self.slow_n.item())
                n_eff = min(n_true, self.slow_window) if self.slow_window else n_true
                self.slow_heat.add_((signal - self.slow_heat) / n_eff)
                self.slow_n += 1
            return grad
        return hook

    def _gradient_mask_hook(self):
        """Escala gradiente por 1/(1 + β·slow_heat)."""
        def hook(grad: Tensor) -> Tensor:
            if self.slow_strength <= 0.0:
                return grad
            scale = 1.0 / (1.0 + self.slow_strength * self.slow_heat)
            return grad * scale.view(-1, *([1] * (grad.dim() - 1)))
        return hook

    def get_lr_scales(self) -> Tensor:
        return 1.0 / (1.0 + self.slow_strength * self.slow_heat)

    def effective_memory_steps(self) -> float:
        """Quantos passos o slow_heat efetivamente pondera agora (min(t, W))."""
        n_true = self.slow_n.item()
        return min(n_true, self.slow_window) if self.slow_window else n_true

    def reset_slow_heat(self):
        """Zera slow_heat e o contador — útil se quiser reiniciar a memória
        manualmente em algum ponto de controle (não é chamado automaticamente:
        DualHeat não usa oracle de fronteira de tarefas por design)."""
        self.slow_heat.zero_()
        self.slow_n.fill_(1.0)

    def extra_repr(self) -> str:
        w = self.slow_window if self.slow_window else "∞"
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"α={self.fast_decay}, γ={self.fast_strength}, "
            f"δ={self.fast_decay_rate}, β={self.slow_strength}, "
            f"slow_window={w}"
        )


class DualHeatMLP(nn.Sequential):
    """MLP com DualHeatLinear v3 nas camadas ocultas."""
    def __init__(
        self,
        *dims: int,
        act: str = "relu",
        fast_decay: float = 0.93,
        fast_strength: float = 2.0,
        fast_decay_rate: float = 0.04,
        slow_strength: float = 2.0,
        slow_window: int | None = None,
        importance: str = "activation",
        protect_output: bool = False,
    ):
        validate_mlp_dims(dims)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(
                DualHeatLinear(
                    dims[i], dims[i + 1],
                    fast_decay=fast_decay,
                    fast_strength=fast_strength,
                    fast_decay_rate=fast_decay_rate,
                    slow_strength=slow_strength,
                    slow_window=slow_window,
                    importance=importance,
                )
            )
            layers.append(activation(act))
        if protect_output:
            layers.append(
                DualHeatLinear(
                    dims[-2], dims[-1],
                    fast_decay=fast_decay,
                    fast_strength=fast_strength,
                    fast_decay_rate=fast_decay_rate,
                    slow_strength=slow_strength,
                    slow_window=slow_window,
                    importance=importance,
                )
            )
        else:
            layers.append(nn.Linear(dims[-2], dims[-1]))
        super().__init__(*layers)

    def get_dual_layers(self) -> list[DualHeatLinear]:
        return [m for m in self.modules() if isinstance(m, DualHeatLinear)]

    def get_lr_scales(self) -> list[Tensor]:
        return [m.get_lr_scales() for m in self.get_dual_layers()]
