"""Hugging Face Qwen2 integration for Functional SlowHeat.

Scope of this module
--------------------
Only the gated feed-forward family is instrumented. Two architectural facts
drive that decision and both are load-bearing, so they are stated here instead
of being rediscovered later:

1. Qwen2 uses a *gated* MLP, ``down_proj(act(gate_proj(x)) * up_proj(x))``. The
   functional unit is the elementwise product, observed through a forward
   pre-hook on ``down_proj``.

   The ``|z * dL/dz|`` estimator is in fact invariant along this chain: for
   ``z = a * u`` we have ``dL/da = dL/dz * u``, hence
   ``|a * dL/da| = |z * dL/dz|``, and symmetrically for ``u``. Scoring the
   activation output, the up projection output or their product therefore
   yields the same ranking (verified numerically to ~1e-7 in
   ``tests/test_slow_heat_qwen.py``). The pre-hook placement is still the
   correct one because ``down_proj``'s input is exactly the tensor that the
   column mask on ``down_proj.weight`` multiplies, so tracker and mask address
   the same object; but no ranking bug is being avoided by it.

2. Qwen2 uses grouped-query attention. ``Qwen2.5-0.5B`` has 14 query heads and
   only 2 key/value heads, so Q, K and V do not share a hidden size and one
   head-indexed importance vector cannot address all three projections.
   :class:`~dual_heater.transformer.SlowHeatAttentionTracker` assumes they do,
   therefore attention is intentionally out of scope. Extending it requires a
   GQA-aware tracker that maps each KV group to its query heads.

This module is optional: importing the base :mod:`dual_heater` package does not
require Transformers. Install ``dual-heater[nlp]`` before importing it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal

import torch
from torch import Tensor, nn

try:
    from transformers import Qwen2ForSequenceClassification
except ImportError as error:  # pragma: no cover - exercised in minimal installs
    raise ImportError(
        "dual_heater.qwen requer o extra opcional 'nlp' (transformers)"
    ) from error

from .optim import PlasticityMaskBinding
from .slow_heat import _SlowHeatImportanceMixin
from .transformer import (
    CapacityScope,
    FactorSource,
    SlowHeatFFNTracker,
    apply_family_capacity,
    dynamic_matrix_mask,
)


@dataclass(frozen=True)
class QwenSlowHeatConfig:
    """Functional SlowHeat configuration shared by every Qwen2 decoder block."""

    slow_strength: float = 3.0
    ffn_plasticity_budget: float = 0.25
    score_plasticity_budget: float = 0.25
    importance_decay: float = 0.99
    importance_eps: float = 1e-8
    track_ffn: bool = True
    protect_score: bool = False
    freeze_unbound_parameters: bool = False
    # The sequence-classification head is randomly initialized, so freezing it
    # would make a Class-IL run structurally unable to learn any label. When
    # `freeze_unbound_parameters` is on, this keeps `score` trainable and
    # unmasked, and coverage validation reports it as a declared exemption.
    keep_score_plastic: bool = True
    capacity_scope: CapacityScope = "local"
    consolidation_strategy: Literal["max", "mean", "sum"] = "max"

    def __post_init__(self) -> None:
        values = {
            "slow_strength": self.slow_strength,
            "ffn_plasticity_budget": self.ffn_plasticity_budget,
            "score_plasticity_budget": self.score_plasticity_budget,
            "importance_decay": self.importance_decay,
            "importance_eps": self.importance_eps,
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("parâmetros SlowHeat do Qwen devem ser finitos")
        if self.slow_strength < 0.0:
            raise ValueError("slow_strength deve ser >= 0")
        for name in ("ffn_plasticity_budget", "score_plasticity_budget"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} deve estar em [0, 1]")
        if not 0.0 <= self.importance_decay < 1.0:
            raise ValueError("importance_decay deve estar em [0, 1)")
        if self.importance_eps <= 0.0:
            raise ValueError("importance_eps deve ser > 0")
        switches = {
            "track_ffn": self.track_ffn,
            "protect_score": self.protect_score,
            "freeze_unbound_parameters": self.freeze_unbound_parameters,
            "keep_score_plastic": self.keep_score_plastic,
        }
        if any(not isinstance(value, bool) for value in switches.values()):
            raise TypeError("opções de cobertura Qwen devem ser booleanas")
        if self.capacity_scope not in {"local", "global", "hierarchical"}:
            raise ValueError(
                "capacity_scope deve ser 'local', 'global' ou 'hierarchical'"
            )
        if self.consolidation_strategy not in {"max", "mean", "sum"}:
            raise ValueError(
                "consolidation_strategy deve ser 'max', 'mean' ou 'sum'"
            )
        if self.protect_score and self.keep_score_plastic:
            raise ValueError(
                "protect_score e keep_score_plastic são mutuamente exclusivos"
            )
        if not self.track_ffn and not self.protect_score:
            raise ValueError("ao menos uma família SlowHeat deve estar habilitada")


_PROTOCOL_CONFIG_KEY = "dual_heater_slowheat"


def _protocol_metadata(
    config: QwenSlowHeatConfig,
    schema_version: int,
) -> dict[str, Any]:
    return {"schema_version": schema_version, "config": asdict(config)}


def _config_from_protocol_metadata(
    raw: Any,
    schema_version: int,
) -> QwenSlowHeatConfig:
    try:
        if not isinstance(raw, dict) or raw.get("schema_version") != schema_version:
            raise ValueError("schema incompatível")
        return QwenSlowHeatConfig(**dict(raw["config"]))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("protocolo SlowHeat inválido no config.json") from error


def _factor(tracker: SlowHeatFFNTracker, hard: bool) -> Tensor:
    if hard:
        return (tracker.slow_heat <= 0.0).to(dtype=tracker.slow_heat.dtype)
    return tracker.get_lr_scales()


class SlowHeatQwen2ForSequenceClassification(Qwen2ForSequenceClassification):
    """Qwen2 classifier instrumented with per-unit SlowHeat on the gated FFN."""

    slowheat_schema_version = 1
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [
        r"_slowheat_signature",
        r"ffn_trackers\..*",
        r"score_tracker\..*",
    ]

    def _init_weights(self, module) -> None:
        """Reset mechanism state that ``from_pretrained`` leaves uninitialized.

        Transformers materializes modules lazily and only initializes what
        ``_init_weights`` knows about, so SlowHeat buffers would otherwise start
        from arbitrary memory instead of zero when a native Qwen2 checkpoint is
        loaded.
        """

        super()._init_weights(module)
        if isinstance(module, _SlowHeatImportanceMixin):
            with torch.no_grad():
                module.importance_memory.zero_()
                module.slow_heat.zero_()
                module.task_ema.zero_()
                module.task_step.zero_()
                module.consolidated_tasks.zero_()

    def __init__(
        self,
        config,
        slowheat_config: QwenSlowHeatConfig | None = None,
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
        effective_config = slowheat_config or stored_config or QwenSlowHeatConfig()
        super().__init__(config)
        self.slowheat_config = effective_config
        setattr(
            self.config,
            _PROTOCOL_CONFIG_KEY,
            _protocol_metadata(effective_config, self.slowheat_schema_version),
        )
        self.ffn_trackers = nn.ModuleList()
        self.score_tracker: SlowHeatFFNTracker | None = None
        self._slowheat_validity_mask: Tensor | None = None
        self._slowheat_hook_handles: list[Any] = []
        self._install_slowheat_instrumentation()
        if self.slowheat_config.freeze_unbound_parameters:
            self.freeze_parameters_without_bindings()
        self.register_buffer(
            "_slowheat_signature",
            self._build_slowheat_signature(),
        )

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------

    def _slowheat_identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.slowheat_schema_version,
            "model_type": self.config.model_type,
            "hidden_size": self.config.hidden_size,
            "intermediate_size": self.config.intermediate_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": self.config.num_key_value_heads,
            "num_labels": self.config.num_labels,
            "slowheat_config": asdict(self.slowheat_config),
        }

    def _build_slowheat_signature(self) -> Tensor:
        """Digest the identity into int64 so no dtype cast can alter it.

        A float vector would lose small hyperparameters (``importance_eps`` of
        1e-8 and 1e-9 both flush to zero in fp16), making two different
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

    # ------------------------------------------------------------------
    # instrumentation
    # ------------------------------------------------------------------

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
        for layer_index, layer in enumerate(self.model.layers):
            if not slow_config.track_ffn:
                break
            if layer_index < len(self.ffn_trackers):
                tracker = self.ffn_trackers[layer_index]
            else:
                tracker = self._new_ffn_tracker(
                    self.config.intermediate_size,
                    slow_config.ffn_plasticity_budget,
                )
                self.ffn_trackers.append(tracker)

            # The gated product `act(gate_proj(x)) * up_proj(x)` is only
            # materialized as `down_proj`'s input, so a pre-hook is the single
            # placement that observes the true functional unit.
            def observe_gated_product(_module, inputs, *, state=tracker):
                mask = self._slowheat_validity_mask
                if mask is None:
                    return
                if not inputs:
                    raise RuntimeError(
                        "down_proj recebeu entrada vazia; hook SlowHeat inválido"
                    )
                state.observe(inputs[0], mask)
                return

            self._slowheat_hook_handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(observe_gated_product)
            )

        if slow_config.protect_score:
            if self.score_tracker is None:
                self.score_tracker = self._new_ffn_tracker(
                    self.config.num_labels,
                    slow_config.score_plasticity_budget,
                )

            def observe_score(_module, _inputs, output):
                assert self.score_tracker is not None
                self.score_tracker.observe(output)

            self._slowheat_hook_handles.append(
                self.score.register_forward_hook(observe_score)
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

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values=None,
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
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                **kwargs,
            )
        finally:
            self._slowheat_validity_mask = None

    # ------------------------------------------------------------------
    # state access and consolidation
    # ------------------------------------------------------------------

    def get_ffn_trackers(self) -> list[SlowHeatFFNTracker]:
        return list(self.ffn_trackers)

    def get_slow_states(self) -> list[SlowHeatFFNTracker]:
        states: list[SlowHeatFFNTracker] = list(self.ffn_trackers)
        if self.score_tracker is not None:
            states.append(self.score_tracker)
        return states

    def _slow_state_families(self) -> list[list[SlowHeatFFNTracker]]:
        return [
            list(self.ffn_trackers),
            [self.score_tracker] if self.score_tracker is not None else [],
        ]

    def consolidate(self, strategy: Literal["max", "mean", "sum"] | None = None) -> None:
        apply_family_capacity(
            self._slow_state_families(),
            scope=self.slowheat_config.capacity_scope,
            strategy=strategy or self.slowheat_config.consolidation_strategy,
        )

    def capacity_metrics(self) -> list[dict[str, float]]:
        return [state.capacity_metrics() for state in self.get_slow_states()]

    # ------------------------------------------------------------------
    # optimizer bindings
    # ------------------------------------------------------------------

    def mask_bindings(self, *, hard: bool = False) -> list[PlasticityMaskBinding]:
        bindings: list[PlasticityMaskBinding] = []

        def source(state: SlowHeatFFNTracker) -> FactorSource:
            return lambda: _factor(state, hard)

        def append(parameter, mask, kind: str) -> None:
            if parameter is None:
                return
            if any(binding.parameter is parameter for binding in bindings):
                raise RuntimeError(f"parâmetro Qwen recebeu binding duplicado: {kind}")
            bindings.append(PlasticityMaskBinding(parameter, mask, kind))

        if self.slowheat_config.track_ffn:
            for layer_index, layer in enumerate(self.model.layers):
                ffn_source = source(self.ffn_trackers[layer_index])
                prefix = (
                    f"qwen_layer_{layer_index}_ffn_{self.config.intermediate_size}"
                )
                # Producers write into the unit: mask their output rows.
                append(
                    layer.mlp.gate_proj.weight,
                    dynamic_matrix_mask(ffn_source, None),
                    f"{prefix}_gate_producer_rows",
                )
                append(
                    layer.mlp.up_proj.weight,
                    dynamic_matrix_mask(ffn_source, None),
                    f"{prefix}_up_producer_rows",
                )
                # The consumer reads the unit: mask its input columns.
                append(
                    layer.mlp.down_proj.weight,
                    dynamic_matrix_mask(None, ffn_source),
                    f"{prefix}_down_consumer_columns",
                )

        if self.score_tracker is not None:
            score_source = source(self.score_tracker)
            append(
                self.score.weight,
                dynamic_matrix_mask(score_source, None),
                f"qwen_score_{self.config.num_labels}_rows",
            )
            append(
                getattr(self.score, "bias", None),
                score_source,
                f"qwen_score_{self.config.num_labels}_bias",
            )
        return bindings

    def register_plasticity_masks(self, optimizer, *, hard: bool = False) -> None:
        if self.slowheat_config.freeze_unbound_parameters:
            self.validate_trainable_mask_coverage()
        register = getattr(optimizer, "register_mask_bindings", None)
        if not callable(register):
            raise TypeError("optimizer deve expor register_mask_bindings()")
        register(self.mask_bindings(hard=hard))

    # ------------------------------------------------------------------
    # coverage accounting
    # ------------------------------------------------------------------

    def exempt_parameter_names(self) -> list[str]:
        """Names deliberately left trainable and unmasked.

        Only the classification head qualifies, and only when
        ``keep_score_plastic`` is set. Reporting it explicitly stops a
        structurally required exemption from reading as accidental leakage.
        """

        if not self.slowheat_config.keep_score_plastic:
            return []
        return [
            name
            for name, _ in self.named_parameters()
            if name == "score.weight" or name.startswith("score.")
        ]

    def uncovered_trainable_parameters(self) -> list[str]:
        """Return trainable parameter names absent from every mask binding."""

        bound = {id(binding.parameter) for binding in self.mask_bindings()}
        exempt = set(self.exempt_parameter_names())
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and id(parameter) not in bound
            and name not in exempt
        ]

    def mask_coverage_summary(self) -> dict[str, int | float]:
        bindings = self.mask_bindings()
        trainable = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        trainable_ids = {id(parameter) for parameter in trainable}
        masked = [
            binding.parameter
            for binding in bindings
            if id(binding.parameter) in trainable_ids
        ]
        exempt_names = set(self.exempt_parameter_names())
        exempt_count = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name in exempt_names and parameter.requires_grad
        )
        trainable_count = sum(parameter.numel() for parameter in trainable)
        masked_count = sum(parameter.numel() for parameter in masked)
        return {
            "binding_count": len(bindings),
            "trainable_parameter_count": trainable_count,
            "masked_parameter_count": masked_count,
            "exempt_parameter_count": exempt_count,
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
        """Freeze embeddings, norms, attention and any other leak path."""

        bound = {id(binding.parameter) for binding in self.mask_bindings()}
        exempt_names = set(self.exempt_parameter_names())
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(id(parameter) in bound or name in exempt_names)
        self.validate_trainable_mask_coverage()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def slowheat_topology(self) -> dict[str, Any]:
        return self._slowheat_identity_payload()

    def load_state_dict(self, state_dict, *args, **kwargs):
        signature = state_dict.get("_slowheat_signature")
        if signature is not None and not torch.equal(
            signature.detach().cpu(), self._slowheat_signature.detach().cpu()
        ):
            raise RuntimeError(
                "checkpoint SlowHeat incompatível com a topologia/configuração Qwen"
            )
        return super().load_state_dict(state_dict, *args, **kwargs)
