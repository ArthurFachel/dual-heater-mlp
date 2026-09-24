"""Three SlowHeat-in-LoRA mechanisms, plus the two reference arms.

Motivation
----------
``DualHeatLoRALinear`` protects the adapter output but cannot protect an output
independently, because ``lora_A`` is shared across every output row. The BERT
helper ``build_exact_slowheat_lora`` fixes that by freezing ``A`` entirely,
which costs all input-subspace plasticity. The mechanisms here explore the
space between those two points.

Every mechanism keeps PEFT as the authority for the adapter forward: the
instrumentation only attaches forward hooks and reads ``lora_A`` / ``lora_B``
weights. No LoRA forward is reimplemented, so adapter dropout, dtype casting,
disabled adapters and merged adapters keep PEFT semantics.

Mechanisms
----------
``rank``
    The protected unit is a bottleneck direction ``z_j = (A x)_j``, the LoRA's
    own functional unit. Importance is ``|z_j * dL/dz_j|`` over ``r`` units.
    Protecting ``j`` masks row ``j`` of ``A`` (producer) and column ``j`` of
    ``B`` (consumer), so the whole term ``B[:, j] z_j`` is frozen under a hard
    mask. The protection is exact per component, not per output.

``leak``
    The protected unit is the module output ``i``, and ``A`` stays trainable.
    A row ``j`` of ``A`` is only as plastic as the most-protected output it can
    reach through ``B``, which closes the leak that ``DualHeatLoRALinear``
    leaves open. ``min`` combination with ``tau = 0`` gives exact per-output
    protection; ``weighted`` trades exactness for retained plasticity.

``slice``
    The rank is partitioned across task boundaries. Dimensions owned by a past
    task are hard-frozen in both factors, which is exact for the previous
    tasks' contributions, at a fixed total rank (no adapter growth).

Reference arms
--------------
``vanilla``
    Plain LoRA, no tracker and no mask.

``exact``
    Solution A: ``A`` frozen, rows of ``B`` masked by an output-space tracker.
    This mirrors ``dual_heater.bert.build_exact_slowheat_lora`` on a Qwen host.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .optim import PlasticityMaskBinding
from .transformer import SlowHeatFFNTracker

LoRAMethod = Literal["vanilla", "exact", "rank", "leak", "slice"]
LeakCombination = Literal["min", "weighted"]

#: Mechanisms that build a per-unit importance tracker.
TRACKED_METHODS: frozenset[str] = frozenset({"exact", "rank", "leak"})

#: Mechanisms whose tracker lives in the rank bottleneck instead of the output.
RANK_SPACE_METHODS: frozenset[str] = frozenset({"rank"})


@dataclass(frozen=True)
class LoRASlowHeatConfig:
    """Configuration shared by every adapted module."""

    method: LoRAMethod = "rank"
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    slow_strength: float = 3.0
    #: Fraction of units guaranteed to stay plastic after consolidation.
    plasticity_budget: float = 0.5
    importance_decay: float = 0.99
    importance_eps: float = 1e-8
    #: Hard masks are exactly 0/1; soft masks use 1/(1 + beta * heat).
    hard: bool = False
    leak_tau: float = 0.0
    leak_combination: LeakCombination = "min"
    target_modules: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
    #: Number of task boundaries the ``slice`` schedule must cover.
    task_count: int = 2
    consolidation_strategy: Literal["max", "mean", "sum"] = "max"
    adapter_name: str = "default"

    def __post_init__(self) -> None:
        if self.method not in {"vanilla", "exact", "rank", "leak", "slice"}:
            raise ValueError(
                "method deve ser 'vanilla', 'exact', 'rank', 'leak' ou 'slice'"
            )
        for name in ("rank", "task_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} deve ser um inteiro positivo")
        floats = {
            "alpha": self.alpha,
            "dropout": self.dropout,
            "slow_strength": self.slow_strength,
            "plasticity_budget": self.plasticity_budget,
            "importance_decay": self.importance_decay,
            "importance_eps": self.importance_eps,
            "leak_tau": self.leak_tau,
        }
        if not all(math.isfinite(value) for value in floats.values()):
            raise ValueError("hiperparâmetros LoRA SlowHeat devem ser finitos")
        if self.alpha <= 0.0:
            raise ValueError("alpha deve ser > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout deve estar em [0, 1)")
        if self.slow_strength < 0.0:
            raise ValueError("slow_strength deve ser >= 0")
        if not 0.0 <= self.plasticity_budget <= 1.0:
            raise ValueError("plasticity_budget deve estar em [0, 1]")
        if not 0.0 <= self.importance_decay < 1.0:
            raise ValueError("importance_decay deve estar em [0, 1)")
        if self.importance_eps <= 0.0:
            raise ValueError("importance_eps deve ser > 0")
        if self.leak_tau < 0.0:
            raise ValueError("leak_tau deve ser >= 0")
        if self.leak_combination not in {"min", "weighted"}:
            raise ValueError("leak_combination deve ser 'min' ou 'weighted'")
        if self.consolidation_strategy not in {"max", "mean", "sum"}:
            raise ValueError("consolidation_strategy deve ser 'max', 'mean' ou 'sum'")
        if not self.target_modules:
            raise ValueError("target_modules não pode ser vazio")
        if not isinstance(self.hard, bool):
            raise TypeError("hard deve ser booleano")
        if not self.adapter_name:
            raise ValueError("adapter_name não pode ser vazio")
        if self.method == "slice" and self.task_count > self.rank:
            raise ValueError(
                "slice requer rank >= task_count para dar ao menos uma dimensão "
                "por tarefa"
            )


def _factor(tracker: SlowHeatFFNTracker, hard: bool) -> Tensor:
    """Return the per-unit plasticity factor in ``[0, 1]``."""

    if hard:
        return (tracker.slow_heat <= 0.0).to(dtype=tracker.slow_heat.dtype)
    return tracker.get_lr_scales()


def rank_slice_bounds(rank: int, task_count: int, task_index: int) -> tuple[int, int]:
    """Return the ``[start, end)`` rank slice owned by ``task_index``.

    Remainder dimensions go to the earliest tasks, so every task owns at least
    ``rank // task_count`` dimensions and the partition is exact.
    """

    if task_count < 1 or rank < task_count:
        raise ValueError("rank deve ser >= task_count >= 1")
    if not 0 <= task_index < task_count:
        raise ValueError("task_index fora do intervalo do schedule")
    base, remainder = divmod(rank, task_count)
    start = task_index * base + min(task_index, remainder)
    width = base + (1 if task_index < remainder else 0)
    return start, start + width


@dataclass
class _AdaptedModule:
    """One PEFT ``LoraLayer`` plus the state the mechanism needs for it."""

    name: str
    layer: nn.Module
    lora_a: nn.Linear
    lora_b: nn.Linear
    tracker: SlowHeatFFNTracker | None


class QwenLoRASlowHeat:
    """Instrumentation and mask bindings for a PEFT-wrapped LoRA model.

    The trackers are deliberately NOT registered as submodules of the wrapped
    model: PEFT and Transformers both walk ``named_modules`` when saving and
    when deciding what to train, and mechanism state is neither. Call
    :meth:`to` to co-locate them with the model.
    """

    def __init__(
        self,
        peft_model: nn.Module,
        config: LoRASlowHeatConfig,
    ) -> None:
        self.model = peft_model
        self.config = config
        self.modules: list[_AdaptedModule] = []
        self._handles: list[Any] = []
        self._validity_mask: Tensor | None = None
        self._task_index = 0
        self._frozen_rank: Tensor = torch.zeros(config.rank, dtype=torch.bool)
        self._collect_modules()
        if config.method in TRACKED_METHODS:
            self._install_hooks()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _collect_modules(self) -> None:
        from peft.tuners.lora import LoraLayer

        adapter = self.config.adapter_name
        for name, module in self.model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            # nn.ModuleDict has no .get(); membership must be tested first.
            if adapter not in module.lora_A or adapter not in module.lora_B:
                continue
            lora_a = module.lora_A[adapter]
            lora_b = module.lora_B[adapter]
            if not isinstance(lora_a, nn.Linear) or not isinstance(lora_b, nn.Linear):
                raise TypeError("este mecanismo requer adapters LoRA lineares")
            if lora_a.weight.shape[0] != self.config.rank:
                raise ValueError(
                    f"{name}: rank do adapter diverge da configuração "
                    f"({lora_a.weight.shape[0]} != {self.config.rank})"
                )
            units = (
                self.config.rank
                if self.config.method in RANK_SPACE_METHODS
                else lora_b.weight.shape[0]
            )
            tracker = (
                self._new_tracker(units)
                if self.config.method in TRACKED_METHODS
                else None
            )
            self.modules.append(
                _AdaptedModule(
                    name=name,
                    layer=module,
                    lora_a=lora_a,
                    lora_b=lora_b,
                    tracker=tracker,
                )
            )
        if not self.modules:
            raise RuntimeError(
                "nenhum LoraLayer encontrado; verifique target_modules e adapter_name"
            )

    def _new_tracker(self, units: int) -> SlowHeatFFNTracker:
        return SlowHeatFFNTracker(
            units,
            slow_strength=self.config.slow_strength,
            plasticity_budget=self.config.plasticity_budget,
            importance_decay=self.config.importance_decay,
            importance_eps=self.config.importance_eps,
        )

    def _install_hooks(self) -> None:
        rank_space = self.config.method in RANK_SPACE_METHODS
        for adapted in self.modules:
            tracker = adapted.tracker
            assert tracker is not None

            def observe(_module, _inputs, output, *, state=tracker):
                mask = self._validity_mask
                if not isinstance(output, Tensor):
                    raise TypeError("hook SlowHeat recebeu saída não tensorial")
                state.observe(output, mask)
                return output

            # Rank-space methods watch the bottleneck (output of A); output-space
            # methods watch the module output, which is base + adapter delta.
            target = adapted.lora_a if rank_space else adapted.layer
            self._handles.append(target.register_forward_hook(observe))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def to(self, device) -> QwenLoRASlowHeat:
        for adapted in self.modules:
            if adapted.tracker is not None:
                adapted.tracker.to(device)
        self._frozen_rank = self._frozen_rank.to(device)
        return self

    def train(self, mode: bool = True) -> QwenLoRASlowHeat:
        for adapted in self.modules:
            if adapted.tracker is not None:
                adapted.tracker.train(mode)
        return self

    def eval(self) -> QwenLoRASlowHeat:
        return self.train(False)

    # ------------------------------------------------------------------
    # forward-time context
    # ------------------------------------------------------------------

    @contextmanager
    def validity(self, attention_mask: Tensor | None) -> Iterator[None]:
        """Scope the padding mask used to discount pad positions."""

        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("attention_mask deve ter forma [B, T]")
        previous = self._validity_mask
        self._validity_mask = (
            attention_mask.detach() if attention_mask is not None else None
        )
        try:
            yield
        finally:
            self._validity_mask = previous

    # ------------------------------------------------------------------
    # task lifecycle
    # ------------------------------------------------------------------

    @property
    def task_index(self) -> int:
        return self._task_index

    def begin_task(self, task_index: int) -> None:
        """Move the rank schedule to ``task_index``; other methods are no-ops."""

        if task_index < 0:
            raise ValueError("task_index deve ser >= 0")
        self._task_index = task_index
        if self.config.method != "slice":
            return
        frozen = torch.zeros(
            self.config.rank,
            dtype=torch.bool,
            device=self._frozen_rank.device,
        )
        for previous in range(min(task_index, self.config.task_count)):
            start, end = rank_slice_bounds(
                self.config.rank, self.config.task_count, previous
            )
            frozen[start:end] = True
        # Beyond the declared schedule every slice is spent; the run must not
        # silently reuse a past task's dimensions.
        if task_index >= self.config.task_count:
            raise RuntimeError(
                "schedule de slice esgotado: task_index >= task_count"
            )
        self._frozen_rank = frozen

    def consolidate(self) -> None:
        """Fold the finished task's evidence into persistent importance."""

        if self.config.method not in TRACKED_METHODS:
            return
        for adapted in self.modules:
            assert adapted.tracker is not None
            adapted.tracker.consolidate(strategy=self.config.consolidation_strategy)

    # ------------------------------------------------------------------
    # mask bindings
    # ------------------------------------------------------------------

    def _rank_factor_source(self, adapted: _AdaptedModule) -> Callable[[], Tensor]:
        tracker = adapted.tracker
        assert tracker is not None
        hard = self.config.hard

        def factor() -> Tensor:
            return _factor(tracker, hard)

        return factor

    def _leak_row_source(self, adapted: _AdaptedModule) -> Callable[[], Tensor]:
        """Plasticity of each ``A`` row, bounded by the outputs it reaches."""

        tracker = adapted.tracker
        assert tracker is not None
        hard = self.config.hard
        tau = self.config.leak_tau
        combination = self.config.leak_combination
        eps = self.config.importance_eps
        lora_b = adapted.lora_b

        def factor() -> Tensor:
            output_factor = _factor(tracker, hard)
            weight = lora_b.weight.detach()
            output_factor = output_factor.to(
                device=weight.device, dtype=weight.dtype
            )
            if combination == "min":
                reaches = weight.abs() > tau
                candidates = torch.where(
                    reaches,
                    output_factor.reshape(-1, 1),
                    torch.ones_like(weight),
                )
                rank_factor = candidates.min(dim=0).values
            else:
                weights = weight.pow(2)
                weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(eps)
                rank_factor = (weights * output_factor.reshape(-1, 1)).sum(dim=0)
            return rank_factor.clamp(0.0, 1.0).reshape(-1, 1)

        return factor

    def _slice_source(self, *, columns: bool) -> Callable[[], Tensor]:
        def factor() -> Tensor:
            plastic = (~self._frozen_rank).to(dtype=torch.float32)
            return plastic.reshape(1, -1) if columns else plastic.reshape(-1, 1)

        return factor

    def mask_bindings(self) -> list[PlasticityMaskBinding]:
        """Return exactly one binding per masked trainable parameter."""

        method = self.config.method
        bindings: list[PlasticityMaskBinding] = []
        if method == "vanilla":
            return bindings

        for adapted in self.modules:
            prefix = f"lora_{method}_{adapted.name}"
            if method == "rank":
                source = self._rank_factor_source(adapted)

                def rows(get=source) -> Tensor:
                    return get().reshape(-1, 1)

                def columns(get=source) -> Tensor:
                    return get().reshape(1, -1)

                bindings.append(
                    PlasticityMaskBinding(
                        adapted.lora_a.weight, rows, f"{prefix}_A_rank_rows"
                    )
                )
                bindings.append(
                    PlasticityMaskBinding(
                        adapted.lora_b.weight, columns, f"{prefix}_B_rank_columns"
                    )
                )
            elif method == "leak":
                source = self._rank_factor_source(adapted)

                def output_rows(get=source) -> Tensor:
                    return get().reshape(-1, 1)

                bindings.append(
                    PlasticityMaskBinding(
                        adapted.lora_b.weight, output_rows, f"{prefix}_B_output_rows"
                    )
                )
                bindings.append(
                    PlasticityMaskBinding(
                        adapted.lora_a.weight,
                        self._leak_row_source(adapted),
                        f"{prefix}_A_leak_rows",
                    )
                )
            elif method == "slice":
                bindings.append(
                    PlasticityMaskBinding(
                        adapted.lora_a.weight,
                        self._slice_source(columns=False),
                        f"{prefix}_A_slice_rows",
                    )
                )
                bindings.append(
                    PlasticityMaskBinding(
                        adapted.lora_b.weight,
                        self._slice_source(columns=True),
                        f"{prefix}_B_slice_columns",
                    )
                )
            elif method == "exact":
                # A is frozen by the builder, so only B carries a binding.
                source = self._rank_factor_source(adapted)

                def exact_rows(get=source) -> Tensor:
                    return get().reshape(-1, 1)

                bindings.append(
                    PlasticityMaskBinding(
                        adapted.lora_b.weight, exact_rows, f"{prefix}_B_output_rows"
                    )
                )
        return bindings

    def register_plasticity_masks(self, optimizer) -> None:
        register = getattr(optimizer, "register_mask_bindings", None)
        if not callable(register):
            raise TypeError("optimizer deve expor register_mask_bindings()")
        register(self.mask_bindings())

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def effective_plasticity(self) -> float:
        """Mean mask value over every masked element.

        This is the measured quantity the iso-plasticity protocol controls, so
        arms can be compared at matched retained plasticity instead of at
        matched ``slow_strength``.
        """

        bindings = self.mask_bindings()
        if not bindings:
            return 1.0
        total = 0.0
        count = 0
        for binding in bindings:
            mask = binding.mask() if callable(binding.mask) else binding.mask
            expanded = torch.broadcast_to(
                mask.to(binding.parameter.device), binding.parameter.shape
            )
            total += float(expanded.sum().item())
            count += expanded.numel()
        return total / count if count else 1.0

    def leak_collapse_fraction(self, threshold: float = 0.1) -> float | None:
        """Fraction of ``A`` rows driven below ``threshold`` by the leak bound.

        This is the falsifier for the ``leak`` mechanism: once ``B`` is dense,
        almost every rank row reaches some protected output, the bound collapses
        and the method degenerates into "freeze A" with extra cost.
        """

        if self.config.method != "leak":
            return None
        collapsed = 0
        total = 0
        for adapted in self.modules:
            factor = self._leak_row_source(adapted)().flatten()
            collapsed += int((factor < threshold).sum().item())
            total += factor.numel()
        return collapsed / total if total else 0.0

    def protected_unit_fraction(self) -> float | None:
        if self.config.method not in TRACKED_METHODS:
            return None
        protected = 0
        total = 0
        for adapted in self.modules:
            assert adapted.tracker is not None
            protected += int((adapted.tracker.slow_heat > 0.0).sum().item())
            total += adapted.tracker.slow_heat.numel()
        return protected / total if total else 0.0

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": self.config.method,
            "adapted_modules": len(self.modules),
            "tracker_units": (
                self.modules[0].tracker.units
                if self.modules[0].tracker is not None
                else None
            ),
            "effective_plasticity": self.effective_plasticity(),
            "protected_unit_fraction": self.protected_unit_fraction(),
            "leak_collapse_fraction": self.leak_collapse_fraction(),
            "frozen_rank_dims": int(self._frozen_rank.sum().item()),
            "config": asdict(self.config),
        }


def build_lora_slowheat(
    model: nn.Module,
    config: LoRASlowHeatConfig,
    *,
    modules_to_save: list[str] | None = None,
) -> tuple[nn.Module, QwenLoRASlowHeat]:
    """Wrap ``model`` with PEFT LoRA and attach the configured mechanism.

    ``modules_to_save`` keeps the randomly-initialized classification head
    trainable. Freezing it would make a Class-IL run structurally unable to
    learn any label, so it stays trainable and unmasked by design.
    """

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise ImportError("LoRA SlowHeat requer o extra opcional 'nlp' (peft)") from error

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        modules_to_save=list(modules_to_save or ["score"]),
        bias="none",
    )
    wrapped = get_peft_model(model, peft_config, adapter_name=config.adapter_name)

    if config.method == "exact":
        # Solution A: with A fixed, masking row i of B protects exactly the
        # effective row of output i.
        found = False
        for name, parameter in wrapped.named_parameters():
            if ".lora_A." in name:
                parameter.requires_grad_(False)
                found = True
        if not found:
            raise RuntimeError("PEFT não criou matrizes LoRA A nos módulos alvo")

    instrumentation = QwenLoRASlowHeat(wrapped, config)
    return wrapped, instrumentation
