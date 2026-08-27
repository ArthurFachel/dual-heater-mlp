"""
DualHeatLoRALinear — extensão de DualHeat para adaptadores LoRA.

Ideia central
--------------
No DualHeat original, a proteção "lenta" atua diretamente sobre o gradiente de
`self.weight` (via register_hook no parâmetro). Em LoRA a base é congelada
e o único caminho treinável é o delta de baixo posto:

    delta = scaling * B @ (A @ x)          # shape (..., out_features)
    z     = base(x) + delta

Não existe um `.weight` treinável equivalente ao Linear cheio. Este protótipo
aplica o hook ao tensor `delta`, no espaço de saída, antes de somá-lo à base:

    delta.register_hook(scale_por_neuronio_de_saida)

Isso reduz as contribuições de gradiente associadas a saídas protegidas e
propaga a escala para A e B. Entretanto, `lora_A` é compartilhada entre todas
as saídas: updates induzidos por saídas pouco protegidas ainda podem alterar o
delta de saídas protegidas através de `lora_B`. Portanto, o hook não garante
proteção independente por saída e deve ser tratado como mecanismo experimental.

A inibição lateral rápida (fast heat) não precisa de nenhuma adaptação: ela
já opera sobre a saída (`z`/`output`), então funciona igual, seja a saída
vinda de um Linear cheio ou de base+LoRA.

Diferenças em relação ao DualHeatLinear original
--------------------------------------------------
1. base_weight/base_bias viram buffers (não Parameters) — ficam de fora do
   optimizer automaticamente, sem precisar filtrar por requires_grad.
2. O hook de plasticidade é registrado no tensor `delta` a cada forward (não pode ser
   registrado uma vez só no __init__ como no original, porque delta é um
   tensor novo a cada chamada — não é um parâmetro persistente).
3. `post_mag` agora reduz sobre todas as dimensões exceto a última
   (`dim=tuple(range(output.dim()-1))`), pra funcionar tanto com entradas
   2D (batch, features) quanto 3D (batch, seq, hidden), que é o formato
   usual em transformers/LLMs.
4. A escala de plasticidade (`scale`) é aplicada com broadcasting direto sobre a
   última dimensão do gradiente (out_features), em vez do `.view(-1, 1)`
   do original — porque lá o hook era sobre a matriz de pesos
   (out_features, in_features), aqui é sobre a ativação
   (..., out_features).

Uso pretendido: substituir as camadas onde você aplicaria LoRA normalmente
(q_proj, k_proj, v_proj, o_proj, gate/up/down_proj etc.) por esta classe,
inicializando base_weight/base_bias com os pesos pré-treinados do modelo
congelado.
"""


import torch
import torch.nn.functional as F
from torch import nn

from ._layers import validate_finite_hyperparameters


class DualHeatLoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        fast_decay: float = 0.93,
        fast_strength: float = 2.0,
        fast_decay_rate: float = 0.04,
        slow_strength: float = 2.0,
        slow_window: int | None = None,
        bias: bool = True,
        base_weight: torch.Tensor | None = None,
        base_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        validate_finite_hyperparameters(
            lora_alpha=lora_alpha,
            fast_decay=fast_decay,
            fast_strength=fast_strength,
            fast_decay_rate=fast_decay_rate,
            slow_strength=slow_strength,
        )
        if r < 1:
            raise ValueError("r deve ser >= 1")
        if lora_alpha <= 0.0:
            raise ValueError("lora_alpha deve ser > 0")
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

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r

        # Base pré-treinada, congelada. Buffer, não Parameter: nunca aparece
        # em model.parameters(), então o optimizer nunca tenta atualizá-la.
        self.register_buffer(
            "base_weight",
            base_weight.detach().clone() if base_weight is not None
            else torch.randn(out_features, in_features) / in_features**0.5,
        )
        if bias:
            self.register_buffer(
                "base_bias",
                base_bias.detach().clone() if base_bias is not None else torch.zeros(out_features),
            )
        else:
            self.base_bias = None

        # Único caminho treinável.
        self.lora_A = nn.Parameter(torch.randn(r, in_features) / in_features**0.5)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # init padrão LoRA: delta=0 no t=0

        self.fast_decay = fast_decay
        self.fast_strength = fast_strength
        self.fast_decay_rate = fast_decay_rate
        self.slow_strength = slow_strength
        self.slow_window = slow_window

        self.register_buffer("fast_heat", torch.zeros(out_features))
        self.register_buffer("slow_heat", torch.zeros(out_features))
        self.register_buffer("slow_n", torch.ones(1))

    def _plasticity_scale_output(self, grad: torch.Tensor) -> torch.Tensor:
        if self.slow_strength <= 0.0:
            return grad
        scale = 1.0 / (1.0 + self.slow_strength * self.slow_heat)  # (out_features,)
        return grad * scale  # broadcast sobre a última dim de (..., out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.base_weight, self.base_bias)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

        # Proteção lenta generalizada: atua na contribuição do adaptador por
        # neurônio de saída, e não pesos individuais (que nem existem aqui).
        if self.slow_strength > 0.0 and delta.requires_grad:
            delta.register_hook(self._plasticity_scale_output)

        z = base + delta

        # Inibição lateral rápida — inalterada, opera sobre a saída combinada.
        if self.fast_strength > 0.0 and self.out_features > 1 and self.training:
            with torch.no_grad():
                sum_h = self.fast_heat.sum()
                mean_others = (sum_h - self.fast_heat) / (self.out_features - 1)
            output = z / (1.0 + self.fast_strength * mean_others)
        else:
            output = z

        if self.training:
            with torch.no_grad():
                post_mag = output.detach().abs().mean(dim=tuple(range(output.dim() - 1)))

                self.fast_heat.mul_(self.fast_decay).add_(
                    post_mag, alpha=1.0 - self.fast_decay
                ).sub_(self.fast_decay_rate).clamp_(min=0.0)

                n_true = self.slow_n.item()
                n_eff = min(n_true, self.slow_window) if self.slow_window else n_true
                self.slow_heat.add_((post_mag - self.slow_heat) / n_eff)
                self.slow_n += 1

        return output

    def get_lr_scales(self) -> torch.Tensor:
        return 1.0 / (1.0 + self.slow_strength * self.slow_heat)

    def reset_slow_heat(self):
        self.slow_heat.zero_()
        self.slow_n.fill_(1.0)

    def extra_repr(self) -> str:
        w = self.slow_window if self.slow_window else "\u221e"
        return (
            f"in={self.in_features}, out={self.out_features}, r={self.r}, "
            f"\u03b1={self.fast_decay}, \u03b3={self.fast_strength}, "
            f"\u03b4={self.fast_decay_rate}, \u03b2={self.slow_strength}, "
            f"slow_window={w}"
        )


if __name__ == "__main__":
    # Sanity check: confirma que (1) gradiente chega em lora_A e lora_B,
    # (2) a base fica intocada, (3) funciona com input 3D estilo transformer,
    # (4) neurônios com slow_heat alto realmente recebem gradiente menor.
    torch.manual_seed(0)
    layer = DualHeatLoRALinear(in_features=16, out_features=8, r=4, slow_window=5)
    layer.train()
    with torch.no_grad():
        layer.lora_B.copy_(torch.randn_like(layer.lora_B) * 0.1)

    x = torch.randn(2, 5, 16, requires_grad=True)  # (batch, seq, hidden)
    out = layer(x)
    out.pow(2).mean().backward()
    assert layer.lora_A.grad.abs().sum() > 0
    assert layer.lora_B.grad.abs().sum() > 0
    assert not hasattr(layer.base_weight, "grad") or layer.base_weight.grad is None

    layer.zero_grad()
    with torch.no_grad():
        layer.slow_heat.copy_(torch.tensor([0.0] * 4 + [10.0] * 4))
    out2 = layer(torch.randn(2, 5, 16))
    out2.sum().backward()
    low = layer.lora_B.grad[:4].abs().mean().item()
    high = layer.lora_B.grad[4:].abs().mean().item()
    print(f"grad medio (heat baixo): {low:.4f}  |  grad medio (heat alto): {high:.4f}")
    assert high < low
    print("OK — protecao via hook no delta esta funcionando.")
