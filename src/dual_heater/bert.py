"""Hugging Face BERT integration for Functional SlowHeat.

This module is optional: importing the base :mod:`dual_heater` package does not
require Transformers or PEFT. Install ``dual-heater[nlp]`` before importing it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
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

from .fast_heat import (
    FastHeatActivation,
    FastHeatConfig,
    FastHeatGate,
    fast_heat_states,
    reset_fast_heat,
)
from .optim import PlasticityMaskBinding
from .slow_heat import _SlowHeatImportanceMixin
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
    residual_plasticity_budget: float = 0.25
    pooler_plasticity_budget: float = 0.25
    importance_decay: float = 0.99
    importance_eps: float = 1e-8
    attention_combination: AttentionCombination = "max"
    track_ffn: bool = True
    track_attention: bool = True
    track_embeddings: bool = False
    track_residual: bool = False
    protect_layer_norm: bool = False
    protect_pooler: bool = False
    protect_classifier: bool = False
    fast_heat: FastHeatConfig | None = None
    freeze_unbound_parameters: bool = False
    capacity_scope: Literal["local", "global", "hierarchical"] = "local"

    def __post_init__(self) -> None:
        values = {
            "slow_strength": self.slow_strength,
            "ffn_plasticity_budget": self.ffn_plasticity_budget,
            "attention_plasticity_budget": self.attention_plasticity_budget,
            "residual_plasticity_budget": self.residual_plasticity_budget,
            "pooler_plasticity_budget": self.pooler_plasticity_budget,
            "importance_decay": self.importance_decay,
            "importance_eps": self.importance_eps,
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("parâmetros SlowHeat do BERT devem ser finitos")
        if self.slow_strength < 0.0:
            raise ValueError("slow_strength deve ser >= 0")
        if not 0.0 <= self.ffn_plasticity_budget <= 1.0:
            raise ValueError("ffn_plasticity_budget deve estar em [0, 1]")
        if not 0.0 <= self.attention_plasticity_budget <= 1.0:
            raise ValueError("attention_plasticity_budget deve estar em [0, 1]")
        if not 0.0 <= self.residual_plasticity_budget <= 1.0:
            raise ValueError("residual_plasticity_budget deve estar em [0, 1]")
        if not 0.0 <= self.pooler_plasticity_budget <= 1.0:
            raise ValueError("pooler_plasticity_budget deve estar em [0, 1]")
        if not 0.0 <= self.importance_decay < 1.0:
            raise ValueError("importance_decay deve estar em [0, 1)")
        if self.importance_eps <= 0.0:
            raise ValueError("importance_eps deve ser > 0")
        if self.attention_combination not in {"max", "mean", "sum"}:
            raise ValueError(
                "attention_combination deve ser 'max', 'mean' ou 'sum'"
            )
        if self.fast_heat is not None and not isinstance(self.fast_heat, FastHeatConfig):
            raise TypeError("fast_heat deve ser FastHeatConfig ou None")
        if not isinstance(self.freeze_unbound_parameters, bool):
            raise TypeError("freeze_unbound_parameters deve ser booleano")
        if self.capacity_scope not in {"local", "global", "hierarchical"}:
            raise ValueError(
                "capacity_scope deve ser 'local', 'global' ou 'hierarchical'"
            )
        switches = {
            "track_embeddings": self.track_embeddings,
            "track_residual": self.track_residual,
            "protect_layer_norm": self.protect_layer_norm,
            "protect_pooler": self.protect_pooler,
        }
        if any(not isinstance(value, bool) for value in switches.values()):
            raise TypeError("opções de cobertura BERT devem ser booleanas")
        if (
            not self.track_ffn
            and not self.track_attention
            and not self.track_embeddings
            and not self.track_residual
            and not self.protect_layer_norm
            and not self.protect_pooler
            and not self.protect_classifier
            and self.fast_heat is None
        ):
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
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha deve ser finito e > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout deve estar em [0, 1)")
        if not self.adapter_name:
            raise ValueError("adapter_name não pode ser vazio")


_PROTOCOL_CONFIG_KEY = "dual_heater_slowheat"


def _protocol_metadata(
    config: BertSlowHeatConfig,
    schema_version: int,
) -> dict[str, Any]:
    return {"schema_version": schema_version, "config": asdict(config)}


def _config_from_protocol_metadata(
    raw: Any,
    schema_version: int,
) -> BertSlowHeatConfig:
    try:
        if not isinstance(raw, dict) or raw.get("schema_version") != schema_version:
            raise ValueError("schema incompatível")
        values = dict(raw["config"])
        fast = values.get("fast_heat")
        values["fast_heat"] = FastHeatConfig(**fast) if fast is not None else None
        return BertSlowHeatConfig(**values)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("protocolo DualHeat inválido no config.json") from error


def _factor(tracker: SlowHeatFFNTracker, hard: bool) -> Tensor:
    if hard:
        return (tracker.slow_heat <= 0.0).to(dtype=tracker.slow_heat.dtype)
    return tracker.get_lr_scales()


FactorSource = Callable[[], Tensor]


def _dynamic_matrix_mask(
    row_source: FactorSource | None,
    column_source: FactorSource | None,
) -> FactorSource:
    """Build one dynamic matrix mask from zero, one, or two endpoint factors."""

    if row_source is None and column_source is None:
        raise ValueError("ao menos um endpoint deve fornecer uma máscara")

    def mask() -> Tensor:
        rows = row_source().reshape(-1, 1) if row_source is not None else None
        columns = (
            column_source().reshape(1, -1) if column_source is not None else None
        )
        if rows is None:
            assert columns is not None
            return columns
        if columns is None:
            return rows
        if rows.device != columns.device:
            columns = columns.to(rows.device)
        return torch.minimum(rows, columns)

    return mask


class SlowHeatBertForSequenceClassification(BertForSequenceClassification):
    """BERT classifier instrumented with post-GELU and per-head SlowHeat."""

    slowheat_schema_version = 3
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [
        r"_slowheat_signature",
        r"ffn_trackers\..*",
        r"attention_trackers\..*",
        r"residual_trackers\..*",
        r"pooler_tracker\..*",
        r"classifier_tracker\..*",
        r"bert\.encoder\.layer\..*\.intermediate\.intermediate_act_fn\.gate\.fast_heat",
    ]

    def _init_weights(self, module) -> None:
        """Reset mechanism state that `from_pretrained` leaves uninitialized.

        Transformers materializes modules lazily and only initializes what
        `_init_weights` knows about, so FastHeat/SlowHeat buffers would otherwise
        start from arbitrary memory instead of zero when a native BERT
        checkpoint is loaded.
        """

        super()._init_weights(module)
        if isinstance(module, FastHeatGate):
            module.reset_fast_heat()
        elif isinstance(module, _SlowHeatImportanceMixin):
            with torch.no_grad():
                module.importance_memory.zero_()
                module.slow_heat.zero_()
                module.task_ema.zero_()
                module.task_step.zero_()
                module.consolidated_tasks.zero_()
        elif module is self:
            # FastHeat gates carry `_is_hf_initialized`, so Transformers never
            # visits them individually. Their reset happens in
            # `_adjust_missing_and_unexpected_keys`, which is the first hook
            # that knows which buffers were genuinely absent from the
            # checkpoint; resetting here would clobber a restored `fast_heat`.
            pass

    def _adjust_missing_and_unexpected_keys(
        self,
        missing_keys: Any,
        unexpected_keys: list[str] | None = None,
        loading_task_model_from_base_state_dict: bool | None = None,
    ):
        """Zero mechanism buffers that the checkpoint did not provide.

        Transformers materializes modules lazily, so a buffer absent from the
        checkpoint keeps arbitrary memory instead of its registered zeros. This
        runs after weights are loaded and is the only hook that distinguishes a
        genuinely missing `fast_heat` from one the checkpoint restored.
        """

        if (
            unexpected_keys is None
            and loading_task_model_from_base_state_dict is None
        ):
            loading_info = missing_keys
            checkpoint_missing_keys = (
                loading_info.get("missing_keys", [])
                if isinstance(loading_info, dict)
                else getattr(loading_info, "missing_keys", ()) or ()
            )
            super_args = (loading_info,)
        else:
            if (
                unexpected_keys is None
                or loading_task_model_from_base_state_dict is None
            ):
                raise TypeError(
                    "a assinatura atual requer missing_keys, unexpected_keys e "
                    "loading_task_model_from_base_state_dict"
                )
            checkpoint_missing_keys = missing_keys
            super_args = (
                missing_keys,
                unexpected_keys,
                loading_task_model_from_base_state_dict,
            )
        missing = set(checkpoint_missing_keys)
        if missing:
            with torch.no_grad():
                for name, gate in self.named_modules():
                    if isinstance(gate, FastHeatGate) and (
                        f"{name}.fast_heat" in missing
                    ):
                        gate.reset_fast_heat()
        return super()._adjust_missing_and_unexpected_keys(*super_args)

    def __init__(
        self,
        config,
        slowheat_config: BertSlowHeatConfig | None = None,
    ) -> None:
        stored_metadata = getattr(config, _PROTOCOL_CONFIG_KEY, None)
        stored_config = (
            _config_from_protocol_metadata(
                stored_metadata,
                self.slowheat_schema_version,
            )
            if stored_metadata is not None
            else None
        )
        if (
            stored_config is not None
            and slowheat_config is not None
            and stored_config != slowheat_config
        ):
            raise RuntimeError(
                "slowheat_config diverge do protocolo persistido no config.json"
            )
        effective_config = slowheat_config or stored_config or BertSlowHeatConfig()
        super().__init__(config)
        self.slowheat_config = effective_config
        setattr(
            self.config,
            _PROTOCOL_CONFIG_KEY,
            _protocol_metadata(effective_config, self.slowheat_schema_version),
        )
        self.ffn_trackers = nn.ModuleList()
        self.attention_trackers = nn.ModuleList()
        self.residual_trackers = nn.ModuleList()
        self.pooler_tracker: SlowHeatFFNTracker | None = None
        self.classifier_tracker: SlowHeatFFNTracker | None = None
        self._slowheat_validity_mask: Tensor | None = None
        self._slowheat_hook_handles: list[Any] = []
        self._install_slowheat_instrumentation()
        if self.slowheat_config.freeze_unbound_parameters:
            self.freeze_parameters_without_bindings()
        self.register_buffer(
            "_slowheat_signature",
            self._build_slowheat_signature(),
        )

    def _slowheat_identity_payload(self) -> dict[str, Any]:
        """Canonical, JSON-serializable identity of topology plus protocol."""

        return {
            "schema_version": self.slowheat_schema_version,
            "hidden_size": self.config.hidden_size,
            "intermediate_size": self.config.intermediate_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "num_labels": self.config.num_labels,
            "slowheat_config": asdict(self.slowheat_config),
        }

    def _build_slowheat_signature(self) -> Tensor:
        """Digest the identity into int64 so no dtype cast can alter it.

        The previous float vector lost small hyperparameters (importance_eps of
        1e-8 and 1e-9 both flush to zero in fp16), which made two different
        protocols compare equal on a half-precision model.
        """

        canonical = json.dumps(
            self._slowheat_identity_payload(), sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        return torch.tensor(
            [
                int.from_bytes(digest[index : index + 8], "big", signed=True)
                for index in range(0, 32, 8)
            ],
            dtype=torch.int64,
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

    def _needs_embedding_state(self) -> bool:
        config = self.slowheat_config
        return any(
            (config.track_embeddings, config.track_residual, config.protect_layer_norm)
        )

    def _needs_layer_residual_states(self) -> bool:
        config = self.slowheat_config
        return config.track_residual or config.protect_layer_norm

    def _new_residual_tracker(self) -> SlowHeatFFNTracker:
        return self._new_ffn_tracker(
            self.config.hidden_size,
            self.slowheat_config.residual_plasticity_budget,
        )

    def _install_slowheat_instrumentation(self) -> None:
        slow_config = self.slowheat_config
        attention_caches: list[dict[str, Tensor]] = []
        ffn_index = 0
        attention_index = 0
        residual_index = 0

        def install_residual_hook(module: nn.Module) -> None:
            nonlocal residual_index
            if residual_index < len(self.residual_trackers):
                tracker = self.residual_trackers[residual_index]
            else:
                tracker = self._new_residual_tracker()
                self.residual_trackers.append(tracker)
            residual_index += 1

            def observe_residual(_module, _inputs, output, *, state=tracker):
                mask = self._slowheat_validity_mask
                if mask is not None:
                    state.observe(output, mask)

            self._slowheat_hook_handles.append(
                module.register_forward_hook(observe_residual)
            )

        if self._needs_embedding_state():
            install_residual_hook(self.bert.embeddings)
        for layer_index, layer in enumerate(self.bert.encoder.layer):
            if slow_config.fast_heat is not None:
                activation = layer.intermediate.intermediate_act_fn
                if not isinstance(activation, FastHeatActivation):
                    layer.intermediate.intermediate_act_fn = FastHeatActivation(
                        activation,
                        self.config.intermediate_size,
                        unit_dim=-1,
                        config=slow_config.fast_heat,
                    )
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

            if self._needs_layer_residual_states():
                install_residual_hook(layer.attention.output)
                install_residual_hook(layer.output)

        if slow_config.protect_pooler:
            if self.bert.pooler is None:
                raise RuntimeError("protect_pooler requer um pooler BERT")
            if self.pooler_tracker is None:
                self.pooler_tracker = self._new_ffn_tracker(
                    self.config.hidden_size,
                    slow_config.pooler_plasticity_budget,
                )

            def observe_pooler(_module, _inputs, output):
                assert self.pooler_tracker is not None
                self.pooler_tracker.observe(output)

            self._slowheat_hook_handles.append(
                self.bert.pooler.register_forward_hook(observe_pooler)
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
        self.prepare_fast_heat()
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

    def get_residual_trackers(self) -> list[SlowHeatFFNTracker]:
        return list(self.residual_trackers)

    def get_fast_states(self):
        return fast_heat_states(self)

    def reset_fast_heat(self) -> None:
        reset_fast_heat(self)

    @torch.no_grad()
    def prepare_fast_heat(self) -> None:
        gates = self.get_fast_states()
        if not gates:
            return
        config = gates[0].config
        if config.competition != "global_topk":
            for gate in gates:
                gate.set_external_scale(None)
            return
        heat = torch.cat([gate.fast_heat for gate in gates])
        scale = torch.ones_like(heat)
        selected_count = min(
            math.floor(config.topk_fraction * heat.numel() + 1e-12),
            int(torch.count_nonzero(heat > 0.0).item()),
        )
        if config.fast_strength > 0.0 and selected_count:
            selected = torch.argsort(heat, descending=True, stable=True)[:selected_count]
            scale[selected] = 1.0 / (
                1.0 + config.fast_strength * heat[selected]
            )
        offset = 0
        for gate in gates:
            width = gate.fast_heat.numel()
            gate.set_external_scale(scale[offset : offset + width])
            offset += width

    def get_slow_states(
        self,
    ) -> list[SlowHeatFFNTracker | SlowHeatAttentionTracker]:
        states: list[SlowHeatFFNTracker | SlowHeatAttentionTracker] = []
        for index in range(len(self.bert.encoder.layer)):
            if self.slowheat_config.track_ffn:
                states.append(self.ffn_trackers[index])
            if self.slowheat_config.track_attention:
                states.append(self.attention_trackers[index])
        states.extend(self.residual_trackers)
        if self.pooler_tracker is not None:
            states.append(self.pooler_tracker)
        if self.classifier_tracker is not None:
            states.append(self.classifier_tracker)
        return states

    @staticmethod
    def _merge_task_importance(
        states: list[SlowHeatFFNTracker | SlowHeatAttentionTracker],
        strategy: Literal["max", "mean", "sum"],
    ) -> None:
        if strategy not in {"max", "mean", "sum"}:
            raise ValueError("strategy deve ser 'max', 'mean' ou 'sum'")
        if any(state.task_step.item() == 0 for state in states):
            raise RuntimeError("não é possível consolidar uma task sem backward")
        with torch.no_grad():
            for state in states:
                if strategy == "max":
                    state.importance_memory.copy_(
                        torch.maximum(state.importance_memory, state.task_ema)
                    )
                elif strategy == "mean":
                    count = int(state.consolidated_tasks.item()) + 1
                    state.importance_memory.add_(
                        (state.task_ema - state.importance_memory) / count
                    )
                else:
                    state.importance_memory.add_(state.task_ema)
                state.consolidated_tasks.add_(1)
                state.task_ema.zero_()
                state.task_step.zero_()

    @staticmethod
    @torch.no_grad()
    def _apply_global_capacity(
        states: list[SlowHeatFFNTracker | SlowHeatAttentionTracker],
    ) -> None:
        if not states:
            return
        budget = states[0].plasticity_budget
        if any(state.plasticity_budget != budget for state in states[1:]):
            raise RuntimeError("o budget global deve ser uniforme dentro da família")
        importance = torch.cat([state.importance_memory for state in states])
        protected = min(
            math.floor((1.0 - budget) * importance.numel() + 1e-12),
            int(torch.count_nonzero(importance > 0.0).item()),
        )
        for state in states:
            state.slow_heat.zero_()
        if protected == 0:
            return
        selected = torch.argsort(importance, descending=True, stable=True)[:protected]
        heat = torch.zeros_like(importance)
        heat[selected] = importance[selected] / importance[selected].max().clamp_min(
            states[0].importance_eps
        )
        offset = 0
        for state in states:
            width = state.slow_heat.numel()
            state.slow_heat.copy_(heat[offset : offset + width])
            offset += width

    @staticmethod
    @torch.no_grad()
    def _apply_hierarchical_capacity(
        states: list[SlowHeatFFNTracker | SlowHeatAttentionTracker],
    ) -> None:
        if not states:
            return
        budget = states[0].plasticity_budget
        if any(state.plasticity_budget != budget for state in states[1:]):
            raise RuntimeError("o budget hierárquico deve ser uniforme dentro da família")
        capacities = [
            int(torch.count_nonzero(state.importance_memory > 0.0).item())
            for state in states
        ]
        protected = min(
            math.floor(
                (1.0 - budget)
                * sum(state.importance_memory.numel() for state in states)
                + 1e-12
            ),
            sum(capacities),
        )
        for state in states:
            state.slow_heat.zero_()
        if protected == 0:
            return
        weights = [float(state.importance_memory.mean()) for state in states]
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            weights = [float(capacity) for capacity in capacities]
            weight_sum = sum(weights)
        ideals = [protected * weight / weight_sum for weight in weights]
        quotas = [min(capacity, math.floor(ideal)) for capacity, ideal in zip(capacities, ideals, strict=True)]
        remaining = protected - sum(quotas)
        priority = sorted(
            range(len(states)),
            key=lambda index: (ideals[index] - math.floor(ideals[index]), weights[index], -index),
            reverse=True,
        )
        while remaining:
            progressed = False
            for index in priority:
                if quotas[index] < capacities[index]:
                    quotas[index] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                raise RuntimeError("não foi possível distribuir o budget hierárquico")
        maxima = [
            state.importance_memory[
                torch.argsort(state.importance_memory, descending=True, stable=True)[:quota]
            ].max()
            for state, quota in zip(states, quotas, strict=True)
            if quota
        ]
        normalizer = torch.stack(maxima).max().clamp_min(states[0].importance_eps)
        for state, quota in zip(states, quotas, strict=True):
            if quota == 0:
                continue
            selected = torch.argsort(
                state.importance_memory, descending=True, stable=True
            )[:quota]
            state.slow_heat[selected] = state.importance_memory[selected] / normalizer

    def consolidate(self, strategy: Literal["max", "mean", "sum"] = "max") -> None:
        scope = self.slowheat_config.capacity_scope
        if scope == "local":
            for state in self.get_slow_states():
                state.consolidate(strategy=strategy)
            return
        families: list[list[SlowHeatFFNTracker | SlowHeatAttentionTracker]] = [
            list(self.ffn_trackers),
            list(self.attention_trackers),
            list(self.residual_trackers),
            [self.pooler_tracker] if self.pooler_tracker is not None else [],
            [self.classifier_tracker] if self.classifier_tracker is not None else [],
        ]
        for states in families:
            if not states:
                continue
            self._merge_task_importance(states, strategy)
            if scope == "global":
                self._apply_global_capacity(states)
            else:
                self._apply_hierarchical_capacity(states)
    def capacity_metrics(self) -> list[dict[str, float]]:
        return [state.capacity_metrics() for state in self.get_slow_states()]

    def mask_bindings(self, *, hard: bool = False) -> list[PlasticityMaskBinding]:
        bindings: list[PlasticityMaskBinding] = []
        extended_coverage = any(
            (
                self.slowheat_config.track_embeddings,
                self.slowheat_config.protect_layer_norm,
                self.slowheat_config.protect_pooler,
            )
        )

        def source(state):
            return lambda: _factor(state, hard)

        def append(parameter, mask, kind):
            if parameter is None:
                return
            if any(binding.parameter is parameter for binding in bindings):
                raise RuntimeError(f"parâmetro BERT recebeu binding duplicado: {kind}")
            bindings.append(PlasticityMaskBinding(parameter, mask, kind))

        has_embedding_state = self._needs_embedding_state()
        has_layer_residual_states = self._needs_layer_residual_states()
        embedding_state = self.residual_trackers[0] if has_embedding_state else None
        layer_residual_offset = 1 if has_embedding_state else 0

        if self.slowheat_config.track_embeddings:
            assert embedding_state is not None

            def embedding_columns(state=embedding_state):
                return _factor(state, hard).reshape(1, -1)

            for name in ("word_embeddings", "position_embeddings", "token_type_embeddings"):
                append(
                    getattr(self.bert.embeddings, name).weight,
                    embedding_columns,
                    f"bert_embeddings_{name}_embedding_columns",
                )
        if self.slowheat_config.protect_layer_norm:
            assert embedding_state is not None
            embedding_factor = source(embedding_state)
            append(
                self.bert.embeddings.LayerNorm.weight,
                embedding_factor,
                "bert_embeddings_layernorm_residual_vector",
            )
            append(
                self.bert.embeddings.LayerNorm.bias,
                embedding_factor,
                "bert_embeddings_layernorm_residual_bias",
            )

        for layer_index, layer in enumerate(self.bert.encoder.layer):
            attention_state = None
            block_state = None
            if has_layer_residual_states:
                attention_state = self.residual_trackers[
                    layer_residual_offset + 2 * layer_index
                ]
                block_state = self.residual_trackers[
                    layer_residual_offset + 2 * layer_index + 1
                ]
            input_state = (
                embedding_state
                if layer_index == 0
                else self.residual_trackers[
                    layer_residual_offset + 2 * layer_index - 1
                ]
            ) if has_layer_residual_states else None

            attention_source: FactorSource | None = None
            attention_prefix: str | None = None
            if self.slowheat_config.track_attention:
                tracker = self.attention_trackers[layer_index]

                def attention_factor(state=tracker):
                    return state.expanded_head_scales(hard=hard)

                attention_source = attention_factor
                attention_prefix = (
                    f"bert_layer_{layer_index}_attention_"
                    f"{tracker.num_heads}x{tracker.head_dim}"
                )
            input_source = (
                source(input_state)
                if self.slowheat_config.track_residual and input_state is not None
                else None
            )
            attention_residual_source = (
                source(attention_state)
                if self.slowheat_config.track_residual and attention_state is not None
                else None
            )

            self_attention = layer.attention.self
            for name in ("query", "key", "value"):
                projection = getattr(self_attention, name)
                if attention_source is not None or input_source is not None:
                    kind = f"bert_layer_{layer_index}_{name}"
                    if attention_source is not None:
                        kind += "_attention_rows"
                    if input_source is not None:
                        kind += "_residual_columns"
                    if (
                        input_source is None
                        and attention_prefix is not None
                        and not extended_coverage
                    ):
                        kind = f"{attention_prefix}_{name}_rows"
                    append(
                        projection.weight,
                        _dynamic_matrix_mask(attention_source, input_source),
                        kind,
                    )
                if attention_source is not None:
                    bias_kind = (
                        f"{attention_prefix}_{name}_bias"
                        if input_source is None
                        else f"bert_layer_{layer_index}_{name}_attention_bias"
                    )
                    append(projection.bias, attention_source, bias_kind)

            if attention_residual_source is not None or attention_source is not None:
                if attention_residual_source is None and attention_prefix is not None:
                    output_kind = f"{attention_prefix}_output_columns"
                else:
                    output_kind = f"bert_layer_{layer_index}_attention_output"
                    if attention_residual_source is not None:
                        output_kind += "_residual_rows"
                    if attention_source is not None:
                        output_kind += "_attention_columns"
                append(
                    layer.attention.output.dense.weight,
                    _dynamic_matrix_mask(
                        attention_residual_source,
                        attention_source,
                    ),
                    output_kind,
                )
            if attention_residual_source is not None:
                append(
                    layer.attention.output.dense.bias,
                    attention_residual_source,
                    f"bert_layer_{layer_index}_attention_output_residual_bias",
                )
            if self.slowheat_config.protect_layer_norm:
                assert attention_state is not None
                attention_norm_source = source(attention_state)
                append(
                    layer.attention.output.LayerNorm.weight,
                    attention_norm_source,
                    f"bert_layer_{layer_index}_attention_layernorm_residual_vector",
                )
                append(
                    layer.attention.output.LayerNorm.bias,
                    attention_norm_source,
                    f"bert_layer_{layer_index}_attention_layernorm_residual_bias",
                )

            ffn_source = None
            ffn_prefix = f"bert_layer_{layer_index}_ffn_{self.config.intermediate_size}"
            if self.slowheat_config.track_ffn:
                ffn_source = source(self.ffn_trackers[layer_index])
            if ffn_source is not None or attention_residual_source is not None:
                kind = f"bert_layer_{layer_index}_intermediate"
                if ffn_source is not None:
                    kind += "_ffn_rows"
                if attention_residual_source is not None:
                    kind += "_residual_columns"
                if attention_residual_source is None and not extended_coverage:
                    kind = f"{ffn_prefix}_producer_rows"
                append(
                    layer.intermediate.dense.weight,
                    _dynamic_matrix_mask(ffn_source, attention_residual_source),
                    kind,
                )
            if ffn_source is not None:
                append(
                    layer.intermediate.dense.bias,
                    ffn_source,
                    f"{ffn_prefix}_producer_bias",
                )

            block_source = (
                source(block_state)
                if self.slowheat_config.track_residual and block_state is not None
                else None
            )
            if block_source is not None or ffn_source is not None:
                kind = f"bert_layer_{layer_index}_output"
                if block_source is not None:
                    kind += "_residual_rows"
                if ffn_source is not None:
                    kind += "_ffn_columns"
                if block_source is None and not extended_coverage:
                    kind = f"{ffn_prefix}_consumer_columns"
                append(
                    layer.output.dense.weight,
                    _dynamic_matrix_mask(block_source, ffn_source),
                    kind,
                )
            if block_source is not None:
                append(
                    layer.output.dense.bias,
                    block_source,
                    f"bert_layer_{layer_index}_output_residual_bias",
                )
            if self.slowheat_config.protect_layer_norm:
                assert block_state is not None
                block_norm_source = source(block_state)
                append(
                    layer.output.LayerNorm.weight,
                    block_norm_source,
                    f"bert_layer_{layer_index}_output_layernorm_residual_vector",
                )
                append(
                    layer.output.LayerNorm.bias,
                    block_norm_source,
                    f"bert_layer_{layer_index}_output_layernorm_residual_bias",
                )

        last_block_state = (
            self.residual_trackers[
                layer_residual_offset + 2 * len(self.bert.encoder.layer) - 1
            ]
            if has_layer_residual_states and len(self.bert.encoder.layer) > 0
            else None
        )
        last_block_source = (
            source(last_block_state)
            if self.slowheat_config.track_residual and last_block_state is not None
            else None
        )
        pooler_source = (
            source(self.pooler_tracker)
            if self.slowheat_config.protect_pooler
            and self.pooler_tracker is not None
            else None
        )
        if self.bert.pooler is not None and (
            pooler_source is not None or last_block_source is not None
        ):
            kind = "bert_pooler_dense"
            if pooler_source is not None:
                kind += "_pooler_rows"
            if last_block_source is not None:
                kind += "_residual_columns"
            append(
                self.bert.pooler.dense.weight,
                _dynamic_matrix_mask(pooler_source, last_block_source),
                kind,
            )
        if self.bert.pooler is not None and pooler_source is not None:
            append(
                self.bert.pooler.dense.bias,
                pooler_source,
                "bert_pooler_dense_pooler_bias",
            )

        classifier_source = (
            source(self.classifier_tracker)
            if self.classifier_tracker is not None
            else None
        )
        if classifier_source is not None or pooler_source is not None:
            if classifier_source is not None and pooler_source is None:
                kind = f"bert_classifier_{self.config.num_labels}_rows"
            else:
                kind = "bert_classifier"
                if classifier_source is not None:
                    kind += "_classifier_rows"
                if pooler_source is not None:
                    kind += "_pooler_columns"
            append(
                self.classifier.weight,
                _dynamic_matrix_mask(classifier_source, pooler_source),
                kind,
            )
        if self.classifier_tracker is not None:
            append(
                self.classifier.bias,
                classifier_source,
                f"bert_classifier_{self.config.num_labels}_bias",
            )
        return bindings

    def register_plasticity_masks(self, optimizer, *, hard: bool = False) -> None:
        if self.slowheat_config.freeze_unbound_parameters:
            self.validate_trainable_mask_coverage()
        register = getattr(optimizer, "register_mask_bindings", None)
        if not callable(register):
            raise TypeError("optimizer deve expor register_mask_bindings()")
        register(self.mask_bindings(hard=hard))

    def uncovered_trainable_parameters(self) -> list[str]:
        """Return trainable parameter names absent from every mask binding."""

        bound = {id(binding.parameter) for binding in self.mask_bindings()}
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and id(parameter) not in bound
        ]

    def mask_coverage_summary(self) -> dict[str, int | float]:
        bindings = self.mask_bindings()
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        trainable_ids = {id(parameter) for parameter in trainable}
        masked = [
            binding.parameter
            for binding in bindings
            if id(binding.parameter) in trainable_ids
        ]
        trainable_count = sum(parameter.numel() for parameter in trainable)
        masked_count = sum(parameter.numel() for parameter in masked)
        return {
            "binding_count": len(bindings),
            "trainable_parameter_count": trainable_count,
            "masked_parameter_count": masked_count,
            "masked_fraction": (
                masked_count / trainable_count if trainable_count else 0.0
            ),
        }

    def validate_trainable_mask_coverage(self) -> None:
        """Fail when any trainable parameter can bypass plasticity masking."""

        uncovered = self.uncovered_trainable_parameters()
        if uncovered:
            raise RuntimeError(
                "parâmetros treináveis sem máscara: " + ", ".join(uncovered)
            )

    def freeze_parameters_without_bindings(self) -> None:
        """Freeze embeddings, norms, pooler, output biases and any other leak path."""

        bound = {id(binding.parameter) for binding in self.mask_bindings()}
        for parameter in self.parameters():
            parameter.requires_grad_(id(parameter) in bound)
        self.validate_trainable_mask_coverage()

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
