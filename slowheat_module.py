"""
slowheat_module.py — SlowHeat: Per-neuron MAX-importance EWC for Continual Learning.

Filosofia:
  O esquecimento em métodos EWC-style vem da diluição da importância — a média
  amostral 1/n de |output| cai quando tasks novas com ativações diferentes
  entram na conta. SlowHeat resolve isso consolidando a importância via
  element-wise MAX entre tasks.

  Uma vez que um neurônio se mostra importante, ele fica protegido para sempre.
  Isso dá a garantia mais forte possível de anti-esquecimento vinda de um
  método de regularização puro.

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
  4. Backward: grad /= (1 + β · slow_heat)   (proteção EWC)

Arquitetura:
  SlowHeatLinear — camada linear com hook EWC + max-consolidation
  SlowHeatMLP   — MLP sequencial usando SlowHeatLinear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional


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
    Camada linear com proteção EWC per-neuron via MAX-consolidation.

    Rastreia |z| (magnitude da pré-ativação) durante o treino de cada task.
    Ao final da task, consolidate() funde as estatísticas da task atual no
    slow_heat via MAX element-wise, garantindo proteção monotônica.

    Diferente do DualHeat:
      - Sem inibição lateral (sem fast heat, sem γ, sem δ)
      - Slow heat é MAX entre tasks, não média 1/n
      - Mais simples, mais forte contra forgetting
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        slow_strength: float = 3.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.slow_strength = slow_strength

        # Parâmetros lineares padrão
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / in_features**0.5
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Importância por neurônio: MAX de todas as médias de task
        self.register_buffer("slow_heat", torch.zeros(out_features))

        # Estatísticas intra-task: EMA suave de |z|
        self.register_buffer("task_ema", torch.zeros(out_features))
        self.register_buffer("task_step", torch.zeros(1, dtype=torch.long))

        # Hook EWC no backward
        self.weight.register_hook(self._ewc_hook())
        if bias:
            self.bias.register_hook(self._ewc_hook())

    def forward(self, x: Tensor) -> Tensor:
        z = F.linear(x, self.weight, self.bias)  # [B, out]

        if self.training:
            with torch.no_grad():
                mag = z.detach().abs().mean(0)  # [out], média do batch

                step = self.task_step.item()
                if step == 0:
                    # Primeiro passo da task: copia direto
                    self.task_ema.copy_(mag)
                else:
                    # EMA adaptativa: decay mais suave a cada passo
                    decay = min(0.99, 1.0 - 1.0 / (1.0 + float(step)))
                    self.task_ema.mul_(decay).add_(mag, alpha=1.0 - decay)
                self.task_step.add_(1)

        return z

    def consolidate(self):
        """
        Funde a task atual no slow_heat via MAX element-wise.
        Chamar entre tasks. Reseta estatísticas intra-task.
        """
        with torch.no_grad():
            self.slow_heat = torch.max(self.slow_heat, self.task_ema)
            self.task_ema.zero_()
            self.task_step.zero_()

    def _ewc_hook(self):
        """Escala gradiente: 1/(1 + β·slow_heat)."""
        def hook(grad: Tensor) -> Tensor:
            if self.slow_strength <= 0.0:
                return grad
            scale = 1.0 / (1.0 + self.slow_strength * self.slow_heat)
            return grad * scale.view(-1, *([1] * (grad.dim() - 1)))
        return hook

    def get_lr_scales(self) -> Tensor:
        """Learning rate efetivo por neurônio."""
        return 1.0 / (1.0 + self.slow_strength * self.slow_heat)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"\u03b2={self.slow_strength}"
        )


# ─── SlowHeatConv2d ─────────────────────────────

class SlowHeatConv2d(nn.Module):
    """
    Conv2d com proteção EWC per-channel via MAX-consolidation.

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
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.slow_strength = slow_strength
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

        # EWC gradient hook
        self.weight.register_hook(self._ewc_hook())
        if bias:
            self.bias.register_hook(self._ewc_hook())

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

    def consolidate(self):
        with torch.no_grad():
            self.slow_heat = torch.max(self.slow_heat, self.task_ema)
            self.task_ema.zero_()
            self.task_step.zero_()

    def _ewc_hook(self):
        def hook(grad: Tensor) -> Tensor:
            if self.slow_strength <= 0.0:
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
        layers: List[nn.Module] = []
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

    def get_slow_layers(self) -> List[SlowHeatLinear]:
        return [m for m in self.modules() if isinstance(m, SlowHeatLinear)]

    def consolidate(self):
        """Chamar entre tasks — consolida MAX em todos os SlowHeatLinear."""
        for layer in self.get_slow_layers():
            layer.consolidate()

    def get_lr_scales(self) -> List[Tensor]:
        return [m.get_lr_scales() for m in self.get_slow_layers()]


# ─── Benchmark ──────────────────────────────────

def _make_data(
    n_per_class: int = 500,
    n_features: int = 64,
    n_classes: int = 20,
    noise: float = 0.8,
    seed: int = 42,
):
    """Dataset sintético: centros Gaussianos com ruído."""
    torch.manual_seed(seed)
    centers = torch.randn(n_classes, n_features) * 1.5
    X, y = [], []
    for c in range(n_classes):
        Xc = centers[c] + torch.randn(n_per_class, n_features) * noise
        X.append(Xc)
        y.append(torch.full((n_per_class,), c, dtype=torch.long))
    X = torch.cat(X)
    y = torch.cat(y)
    return X, y


def _make_task_loaders(X, y, tasks, batch_size=64, train_frac=0.8):
    """Cria DataLoaders treino/teste por task."""
    from torch.utils.data import DataLoader, TensorDataset

    train_loaders, test_loaders = [], []
    for cs, ce in tasks:
        mask = (y >= cs) & (y < ce)
        Xt, yt = X[mask], y[mask]
        perm = torch.randperm(len(Xt))
        split = int(len(Xt) * train_frac)
        train_loaders.append(
            DataLoader(
                TensorDataset(Xt[perm[:split]], yt[perm[:split]]),
                batch_size=batch_size, shuffle=True,
            )
        )
        test_loaders.append(
            DataLoader(
                TensorDataset(Xt[perm[split:]], yt[perm[split:]]),
                batch_size=batch_size * 2,
            )
        )
    return train_loaders, test_loaders


def _eval_tasks(net, test_loaders, seen):
    """Accuracy por task."""
    net.eval()
    accs = []
    for t in seen:
        correct = total = 0
        for xb, yb in test_loaders[t]:
            logits = net(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        accs.append(correct / total)
    return accs


BENCHMARK_CONFIGS = {
    "Vanilla": {
        "fn": lambda D, C: nn.Sequential(
            nn.Linear(D, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, C),
        ),
        "needs_consolidate": False,
    },
    "DualHeat (defaults)": {
        "fn": lambda D, C: __import__("dual_heat_module", fromlist=["DualHeatMLP"]).DualHeatMLP(
            D, 128, 64, C, act="gelu"
        ),
        "needs_consolidate": False,
    },
    "DualHeat (EWC forte, b=5.0)": {
        "fn": lambda D, C: __import__("dual_heat_module", fromlist=["DualHeatMLP"]).DualHeatMLP(
            D, 128, 64, C, act="gelu", slow_strength=5.0
        ),
        "needs_consolidate": False,
    },
    "SlowHeat (b=3.0)": {
        "fn": lambda D, C: SlowHeatMLP(D, 128, 64, C, act="gelu", slow_strength=3.0),
        "needs_consolidate": True,
    },
    "SlowHeat (b=5.0)": {
        "fn": lambda D, C: SlowHeatMLP(D, 128, 64, C, act="gelu", slow_strength=5.0),
        "needs_consolidate": True,
    },
    "SlowHeat (b=10.0)": {
        "fn": lambda D, C: SlowHeatMLP(D, 128, 64, C, act="gelu", slow_strength=10.0),
        "needs_consolidate": True,
    },
}


def run_benchmark(
    tasks,
    train_loaders,
    test_loaders,
    steps_per_task: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
):
    """Roda benchmark de CL para todas as configs."""
    import time
    import numpy as np

    D = 64
    C = 20
    results = {}

    for name, cfg in BENCHMARK_CONFIGS.items():
        torch.manual_seed(seed)
        net = cfg["fn"](D, C)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

        t0 = time.time()
        history = []

        for ct in range(len(tasks)):
            net.train()
            for _ in range(steps_per_task):
                xb, yb = next(iter(train_loaders[ct]))
                opt.zero_grad()
                loss = F.cross_entropy(net(xb), yb)
                loss.backward()
                opt.step()

            # Consolidação pós-task (só SlowHeat)
            if cfg["needs_consolidate"]:
                net.consolidate()

            accs = _eval_tasks(net, test_loaders, range(ct + 1))
            history.append(accs)

        elapsed = time.time() - t0
        results[name] = (history, elapsed)
        print(f"{name:<30} ({elapsed:.1f}s)")
        for ct, accs in enumerate(history):
            s = " | ".join(f"T{t}={a:.3f}" for t, a in enumerate(accs))
            print(f"  Task {ct}: {s}")
        final = history[-1]
        forget = history[0][0] - history[-1][0]
        print(f"  Avg Acc={np.mean(final):.3f}  Forgetting={forget:.3f}")
        print()

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/fachel/Desktop/dual_heat")

    print("=" * 60)
    print("SlowHeat Benchmark — Continual Learning (4 tasks, 5 classes each)")
    print("=" * 60)
    print()

    X, y = _make_data()
    tasks = [(0, 5), (5, 10), (10, 15), (15, 20)]
    train_l, test_l = _make_task_loaders(X, y, tasks)

    run_benchmark(tasks, train_l, test_l)
