"""Hugging Face BERT integration for Functional SlowHeat.

This module is optional: importing the base :mod:`dual_heater` package does not
require Transformers or PEFT. Install ``dual-heater[nlp]`` before importing it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal

import torch
from torch import Tensor, nn

try:
    from transformers import BertForSequenceClassification
except ImportError as error:  # pragma: no cover - exercised in minimal installs
    raise ImportError(
        "dual_heater.bert requer o extra opcional 'nlp' (transformers)"
    ) from error

from .optim import PlasticityMaskBinding
from .transformer import (
    AttentionCombination,
    SlowHeatAttentionTracker,
    SlowHeatFFNTracker,
)


@dataclass(frozen=True)
class BertSlowHeatConfig:
    """Functional SlowHeat configuration shared by every BERT encoder block."""

    slow_strength: float = 3.0
    ffn_plasticity_budget: float = 0.25
    attention_plasticity_budget: float = 0.25
    importance_decay: float = 0.99
    importance_eps: float = 1e-8
    attention_combination: AttentionCombination = "max"
    track_ffn: bool = True
    track_attention: bool = True
    protect_classifier: bool = False

    def __post_init__(self) -> None:
        values = {
            "slow_strength": self.slow_strength,
            "ffn_plasticity_budget": self.ffn_plasticity_budget,
            "attention_plasticity_budget": self.attention_plasticity_budget,
            "importance_decay": self.importance_decay,
            "importance_eps": self.importance_eps,
        }
        if not all(torch.isfinite(torch.tensor(value)).item() for value in values.values()):
            raise ValueError("parâmetros SlowHeat do BERT devem ser finitos")
        if self.slow_strength < 0.0:
            raise ValueError("slow_strength deve ser >= 0")
        if not 0.0 <= self.ffn_plasticity_budget <= 1.0:
            raise ValueError("ffn_plasticity_budget deve estar em [0, 1]")
        if not 0.0 <= self.attention_plasticity_budget <= 1.0:
            raise ValueError("attention_plasticity_budget deve estar em [0, 1]")
        if not 0.0 <= self.importance_decay < 1.0:
            raise ValueError("importance_decay deve estar em [0, 1)")
        if self.importance_eps <= 0.0:
            raise ValueError("importance_eps deve ser > 0")
        if self.attention_combination not in {"max", "mean", "sum"}:
            raise ValueError(
                "attention_combination deve ser 'max', 'mean' ou 'sum'"
            )
        if not self.track_ffn and not self.track_attention and not self.protect_classifier:
            raise ValueError("ao menos uma família SlowHeat deve estar habilitada")


@dataclass(frozen=True)
class ExactSlowHeatLoRAConfig:
    """Producer-only LoRA configuration with frozen A and masked B."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    adapter_name: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError("rank deve ser um inteiro positivo")
        if not torch.isfinite(torch.tensor(self.alpha)).item() or self.alpha <= 0.0:
            raise ValueError("alpha deve ser finito e > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout deve estar em [0, 1)")
        if not self.adapter_name:
            raise ValueError("adapter_name não pode ser vazio")


def _factor(tracker: SlowHeatFFNTracker, hard: bool) -> Tensor:
    if hard:
        return (tracker.slow_heat <= 0.0).to(dtype=tracker.slow_heat.dtype)
    return tracker.get_lr_scales()


class SlowHeatBertForSequenceClassification(BertForSequenceClassification):
    """BERT classifier instrumented with post-GELU and per-head SlowHeat."""

    slowheat_schema_version = 1
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [
        r"_slowheat_signature",
        r"ffn_trackers\..*",
        r"attention_trackers\..*",
        r"classifier_tracker\..*",
    ]

    def __init__(
        self,
        config,
        slowheat_config: BertSlowHeatConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.slowheat_config = slowheat_config or BertSlowHeatConfig()
        self.ffn_trackers = nn.ModuleList()
        self.attention_trackers = nn.ModuleList()
        self.classifier_tracker: SlowHeatFFNTracker | None = None
        self._slowheat_validity_mask: Tensor | None = None
        self._slowheat_hook_handles: list[Any] = []
        self._install_slowheat_instrumentation()
        self.register_buffer(
            "_slowheat_signature",
            self._build_slowheat_signature(),
        )

    def _build_slowheat_signature(self) -> Tensor:
        combination = {"max": 0.0, "mean": 1.0, "sum": 2.0}[
            self.slowheat_config.attention_combination
        ]
        return torch.tensor(
            [
                self.slowheat_schema_version,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_hidden_layers,
                self.config.num_attention_heads,
                self.config.num_labels,
                float(self.slowheat_config.track_ffn),
                float(self.slowheat_config.track_attention),
                float(self.slowheat_config.protect_classifier),
                combination,
                self.slowheat_config.slow_strength,
                self.slowheat_config.ffn_plasticity_budget,
                self.slowheat_config.attention_plasticity_budget,
                self.slowheat_config.importance_decay,
                self.slowheat_config.importance_eps,
            ],
            dtype=torch.float64,
        )

    def _new_ffn_tracker(self, units: int, budget: float) -> SlowHeatFFNTracker:
        config = self.slowheat_config
        return SlowHeatFFNTracker(
            units,
            slow_strength=config.slow_strength,
            plasticity_budget=budget,
            importance_decay=config.importance_decay,
            importance_eps=config.importance_eps,
        )

    def _install_slowheat_instrumentation(self) -> None:
        slow_config = self.slowheat_config
        attention_caches: list[dict[str, Tensor]] = []
        ffn_index = 0
        attention_index = 0
        for layer_index, layer in enumerate(self.bert.encoder.layer):
            if slow_config.track_ffn:
                if ffn_index < len(self.ffn_trackers):
                    tracker = self.ffn_trackers[ffn_index]
                else:
                    tracker = self._new_ffn_tracker(
                        self.config.intermediate_size,
                        slow_config.ffn_plasticity_budget,
                    )
                    self.ffn_trackers.append(tracker)
                ffn_index += 1

                def observe_ffn(_module, _inputs, output, *, state=tracker):
                    mask = self._slowheat_validity_mask
                    if mask is not None:
                        state.observe(output, mask)

                self._slowheat_hook_handles.append(
                    layer.intermediate.register_forward_hook(observe_ffn)
                )

            if slow_config.track_attention:
                self_attention = layer.attention.self
                head_dim = int(
                    getattr(
                        self_attention,
                        "attention_head_size",
                        self.config.hidden_size // self.config.num_attention_heads,
                    )
                )
                if attention_index < len(self.attention_trackers):
                    tracker = self.attention_trackers[attention_index]
                else:
                    tracker = SlowHeatAttentionTracker(
                        self.config.num_attention_heads,
                        head_dim,
                        slow_strength=slow_config.slow_strength,
                        plasticity_budget=slow_config.attention_plasticity_budget,
                        importance_decay=slow_config.importance_decay,
                        importance_eps=slow_config.importance_eps,
                        combination=slow_config.attention_combination,
                    )
                    self.attention_trackers.append(tracker)
                attention_index += 1
                cache: dict[str, Tensor] = {}
                attention_caches.append(cache)

                def capture(name: str, target: dict[str, Tensor]):
                    def hook(_module, _inputs, output):
                        if self._slowheat_validity_mask is not None:
                            target[name] = output

                    return hook

                for name in ("query", "key", "value"):
                    projection = getattr(self_attention, name)
                    self._slowheat_hook_handles.append(
                        projection.register_forward_hook(capture(name, cache))
                    )

                def observe_attention(
                    _module,
                    _inputs,
                    output,
                    *,
                    state=tracker,
                    target=cache,
                    index=layer_index,
                ):
                    mask = self._slowheat_validity_mask
                    try:
                        if mask is None:
                            return
                        if not all(name in target for name in ("query", "key", "value")):
                            raise RuntimeError(
                                f"projeções Q/K/V ausentes na camada BERT {index}"
                            )
                        merged = output[0] if isinstance(output, tuple) else output
                        state.observe(
                            target["query"],
                            target["key"],
                            target["value"],
                            merged,
                            mask,
                        )
                    finally:
                        target.clear()

                self._slowheat_hook_handles.append(
                    self_attention.register_forward_hook(observe_attention)
                )

        if slow_config.protect_classifier:
            if self.classifier_tracker is None:
                self.classifier_tracker = self._new_ffn_tracker(
                    self.config.num_labels,
                    slow_config.ffn_plasticity_budget,
                )

            def observe_classifier(_module, _inputs, output):
                assert self.classifier_tracker is not None
                self.classifier_tracker.observe(output)

            self._slowheat_hook_handles.append(
                self.classifier.register_forward_hook(observe_classifier)
            )

    def reinstall_slowheat_instrumentation(self) -> None:
        """Rebind hooks after wrappers such as PEFT replace Linear modules."""

        self.remove_slowheat_instrumentation()
        self._install_slowheat_instrumentation()

    def remove_slowheat_instrumentation(self) -> None:
        """Remove hooks explicitly so discarded GPU models cannot form cycles."""

        for handle in self._slowheat_hook_handles:
            handle.remove()
        self._slowheat_hook_handles.clear()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs,
    ):
        if self.training and self.is_gradient_checkpointing:
            raise RuntimeError(
                "activation checkpointing ainda não é compatível com os hooks SlowHeat"
            )
        if attention_mask is None:
            source = input_ids if input_ids is not None else inputs_embeds
            if source is None:
                raise ValueError("input_ids ou inputs_embeds deve ser fornecido")
            attention_mask = torch.ones(
                source.shape[:2], dtype=torch.long, device=source.device
            )
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask deve ter forma [B, T]")
        self._slowheat_validity_mask = attention_mask.detach()
        try:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                **kwargs,
            )
        finally:
            self._slowheat_validity_mask = None

    def get_ffn_trackers(self) -> list[SlowHeatFFNTracker]:
        return list(self.ffn_trackers)

    def get_attention_trackers(self) -> list[SlowHeatAttentionTracker]:
        return list(self.attention_trackers)

    def get_slow_states(
        self,
    ) -> list[SlowHeatFFNTracker | SlowHeatAttentionTracker]:
        states: list[SlowHeatFFNTracker | SlowHeatAttentionTracker] = []
        for index in range(len(self.bert.encoder.layer)):
            if self.slowheat_config.track_ffn:
                states.append(self.ffn_trackers[index])
            if self.slowheat_config.track_attention:
                states.append(self.attention_trackers[index])
        if self.classifier_tracker is not None:
            states.append(self.classifier_tracker)
        return states

    def consolidate(self, strategy: Literal["max", "mean", "sum"] = "max") -> None:
        for state in self.get_slow_states():
            state.consolidate(strategy=strategy)

    def capacity_metrics(self) -> list[dict[str, float]]:
        return [state.capacity_metrics() for state in self.get_slow_states()]

    def mask_bindings(self, *, hard: bool = False) -> list[PlasticityMaskBinding]:
        bindings: list[PlasticityMaskBinding] = []
        ffn_index = 0
        attention_index = 0
        for layer_index, layer in enumerate(self.bert.encoder.layer):
            if self.slowheat_config.track_ffn:
                tracker = self.ffn_trackers[ffn_index]
                ffn_index += 1

                def rows(state=tracker):
                    return _factor(state, hard).reshape(-1, 1)

                def vector(state=tracker):
                    return _factor(state, hard)

                def columns(state=tracker):
                    return _factor(state, hard).reshape(1, -1)

                prefix = f"bert_layer_{layer_index}_ffn_{self.config.intermediate_size}"
                bindings.extend(
                    (
                        PlasticityMaskBinding(
                            layer.intermediate.dense.weight,
                            rows,
                            f"{prefix}_producer_rows",
                        ),
                        PlasticityMaskBinding(
                            layer.intermediate.dense.bias,
                            vector,
                            f"{prefix}_producer_bias",
                        ),
                        PlasticityMaskBinding(
                            layer.output.dense.weight,
                            columns,
                            f"{prefix}_consumer_columns",
                        ),
                    )
                )
            if self.slowheat_config.track_attention:
                tracker = self.attention_trackers[attention_index]
                attention_index += 1

                def expanded(state=tracker):
                    return state.expanded_head_scales(hard=hard)

                def attention_rows(state=tracker):
                    return state.expanded_head_scales(hard=hard).reshape(-1, 1)

                def attention_columns(state=tracker):
                    return state.expanded_head_scales(hard=hard).reshape(1, -1)

                prefix = (
                    f"bert_layer_{layer_index}_attention_"
                    f"{tracker.num_heads}x{tracker.head_dim}"
                )
                self_attention = layer.attention.self
                for name in ("query", "key", "value"):
                    projection = getattr(self_attention, name)
                    bindings.extend(
                        (
                            PlasticityMaskBinding(
                                projection.weight,
                                attention_rows,
                                f"{prefix}_{name}_rows",
                            ),
                            PlasticityMaskBinding(
                                projection.bias,
                                expanded,
                                f"{prefix}_{name}_bias",
                            ),
                        )
                    )
                bindings.append(
                    PlasticityMaskBinding(
                        layer.attention.output.dense.weight,
                        attention_columns,
                        f"{prefix}_output_columns",
                    )
                )
        if self.classifier_tracker is not None:
            tracker = self.classifier_tracker

            def classifier_rows(state=tracker):
                return _factor(state, hard).reshape(-1, 1)

            def classifier_vector(state=tracker):
                return _factor(state, hard)

            bindings.extend(
                (
                    PlasticityMaskBinding(
                        self.classifier.weight,
                        classifier_rows,
                        f"bert_classifier_{self.config.num_labels}_rows",
                    ),
                    PlasticityMaskBinding(
                        self.classifier.bias,
                        classifier_vector,
                        f"bert_classifier_{self.config.num_labels}_bias",
                    ),
                )
            )
        return bindings

    def register_plasticity_masks(self, optimizer, *, hard: bool = False) -> None:
        register = getattr(optimizer, "register_mask_bindings", None)
        if not callable(register):
            raise TypeError("optimizer deve expor register_mask_bindings()")
        register(self.mask_bindings(hard=hard))

    def slowheat_topology(self) -> dict[str, Any]:
        return {
            "schema_version": self.slowheat_schema_version,
            "model_type": self.config.model_type,
            "hidden_size": self.config.hidden_size,
            "intermediate_size": self.config.intermediate_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "num_labels": self.config.num_labels,
            "slowheat_config": asdict(self.slowheat_config),
        }

    def load_state_dict(self, state_dict, *args, **kwargs):
        signature = state_dict.get("_slowheat_signature")
        if signature is not None and not torch.equal(
            signature.detach().cpu(), self._slowheat_signature.detach().cpu()
        ):
            raise RuntimeError(
                "checkpoint SlowHeat incompatível com a topologia/configuração BERT"
            )
        return super().load_state_dict(state_dict, *args, **kwargs)


def _producer_target_names(model: SlowHeatBertForSequenceClassification) -> list[str]:
    names: list[str] = []
    for index in range(len(model.bert.encoder.layer)):
        prefix = f"bert.encoder.layer.{index}"
        if model.slowheat_config.track_attention:
            names.extend(
                f"{prefix}.attention.self.{name}"
                for name in ("query", "key", "value")
            )
        if model.slowheat_config.track_ffn:
            names.append(f"{prefix}.intermediate.dense")
    return names


def build_exact_slowheat_lora(
    model: SlowHeatBertForSequenceClassification,
    config: ExactSlowHeatLoRAConfig | None = None,
):
    """Wrap BERT with producer-only PEFT LoRA and freeze every A matrix."""

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise ImportError("LoRA exata requer o extra opcional 'nlp' (peft)") from error
    lora = config or ExactSlowHeatLoRAConfig()
    if model.slowheat_config.protect_classifier:
        raise ValueError(
            "LoRA exata mantém o classificador plástico; protect_classifier deve ser False"
        )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=_producer_target_names(model),
        modules_to_save=["classifier"],
        bias="none",
    )
    wrapped = get_peft_model(model, peft_config, adapter_name=lora.adapter_name)
    found_a = False
    for name, parameter in wrapped.named_parameters():
        if ".lora_A." in name:
            parameter.requires_grad_(False)
            found_a = True
    if not found_a:
        raise RuntimeError("PEFT não criou matrizes LoRA A nos módulos produtores")
    model.reinstall_slowheat_instrumentation()
    return wrapped


def _unwrap_slowheat_bert(model: nn.Module) -> SlowHeatBertForSequenceClassification:
    if isinstance(model, SlowHeatBertForSequenceClassification):
        return model
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        candidate = getter()
        if isinstance(candidate, SlowHeatBertForSequenceClassification):
            return candidate
    for module in model.modules():
        if isinstance(module, SlowHeatBertForSequenceClassification):
            return module
    raise TypeError("modelo PEFT não contém SlowHeatBertForSequenceClassification")


def _single_lora_b(module: nn.Module) -> tuple[str, nn.Parameter]:
    lora_b = getattr(module, "lora_B", None)
    if lora_b is None or len(lora_b) != 1:
        raise ValueError("a variante exata requer exatamente um adapter LoRA ativo")
    name, projection = next(iter(lora_b.items()))
    if not isinstance(projection, nn.Linear):
        raise TypeError("lora_B deve ser uma projeção linear")
    return str(name), projection.weight


def exact_lora_mask_bindings(
    model: nn.Module,
    *,
    hard: bool = False,
) -> list[PlasticityMaskBinding]:
    """Build row masks for LoRA B in producer projections only."""

    base = _unwrap_slowheat_bert(model)
    bindings: list[PlasticityMaskBinding] = []
    ffn_index = 0
    attention_index = 0
    for layer_index, layer in enumerate(base.bert.encoder.layer):
        if base.slowheat_config.track_attention:
            tracker = base.attention_trackers[attention_index]
            attention_index += 1

            def attention_rows(state=tracker):
                return state.expanded_head_scales(hard=hard).reshape(-1, 1)

            for name in ("query", "key", "value"):
                adapter_name, parameter = _single_lora_b(
                    getattr(layer.attention.self, name)
                )
                bindings.append(
                    PlasticityMaskBinding(
                        parameter,
                        attention_rows,
                        f"bert_lora_{adapter_name}_layer_{layer_index}_{name}_rows",
                    )
                )
        if base.slowheat_config.track_ffn:
            tracker = base.ffn_trackers[ffn_index]
            ffn_index += 1

            def ffn_rows(state=tracker):
                return _factor(state, hard).reshape(-1, 1)

            adapter_name, parameter = _single_lora_b(layer.intermediate.dense)
            bindings.append(
                PlasticityMaskBinding(
                    parameter,
                    ffn_rows,
                    f"bert_lora_{adapter_name}_layer_{layer_index}_ffn_rows",
                )
            )
    return bindings


def register_exact_lora_masks(model: nn.Module, optimizer, *, hard: bool = False) -> None:
    register = getattr(optimizer, "register_mask_bindings", None)
    if not callable(register):
        raise TypeError("optimizer deve expor register_mask_bindings()")
    register(exact_lora_mask_bindings(model, hard=hard))
