"""Paired class-incremental CLINC150 benchmark for BERT SlowHeat.

The default run uses the balanced in-scope portion of CLINC150 as ten domain
tasks. It keeps one 150-way head, masks unseen logits, and never supplies a task
identifier to the model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import random
import time
from collections import deque
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dual_heater.bert import (
    BertSlowHeatConfig,
    ExactSlowHeatLoRAConfig,
    SlowHeatBertForSequenceClassification,
    build_exact_slowheat_lora,
    register_exact_lora_masks,
)
from dual_heater.metrics import compute_cl_metrics
from dual_heater.optim import SlowHeatAdamW
from experiments.artifacts import (
    read_torch_checkpoint,
    write_json_atomic,
    write_torch_atomic,
)
from experiments.confirmatory_statistics import (
    exact_two_sided_sign_test,
    normal_summary,
)
from experiments.live_telemetry import (
    TelemetryWriter,
    cuda_memory_payload,
    finite_or_none,
)
from experiments.peak_memory import PeakMemoryTracker
from experiments.provenance import write_environment_manifest

CLINC150_DOMAINS: dict[str, tuple[str, ...]] = {
    "banking": (
        "freeze_account", "routing", "pin_change", "bill_due", "pay_bill",
        "account_blocked", "interest_rate", "min_payment", "bill_balance",
        "transfer", "order_checks", "balance", "spending_history",
        "transactions", "report_fraud",
    ),
    "credit_cards": (
        "replacement_card_duration", "expiration_date", "damaged_card",
        "improve_credit_score", "report_lost_card", "card_declined",
        "credit_limit_change", "apr", "redeem_rewards", "credit_limit",
        "rewards_balance", "application_status", "credit_score", "new_card",
        "international_fees",
    ),
    "kitchen_and_dining": (
        "food_last", "confirm_reservation", "how_busy", "ingredients_list",
        "calories", "nutrition_info", "recipe", "restaurant_reviews",
        "restaurant_reservation", "meal_suggestion", "restaurant_suggestion",
        "cancel_reservation", "ingredient_substitution", "cook_time",
        "accept_reservations",
    ),
    "home": (
        "what_song", "play_music", "todo_list_update", "reminder",
        "reminder_update", "calendar_update", "order_status", "update_playlist",
        "shopping_list", "calendar", "next_song", "order", "todo_list",
        "shopping_list_update", "smart_home",
    ),
    "auto_and_commute": (
        "current_location", "oil_change_when", "oil_change_how", "uber",
        "traffic", "tire_pressure", "schedule_maintenance", "gas", "mpg",
        "distance", "directions", "last_maintenance", "gas_type",
        "tire_change", "jump_start",
    ),
    "travel": (
        "plug_type", "travel_notification", "translate", "flight_status",
        "international_visa", "timezone", "exchange_rate", "travel_suggestion",
        "travel_alert", "vaccines", "lost_luggage", "book_flight",
        "book_hotel", "carry_on", "car_rental",
    ),
    "utility": (
        "weather", "alarm", "date", "find_phone", "share_location", "timer",
        "make_call", "calculator", "definition", "measurement_conversion",
        "flip_coin", "spelling", "time", "roll_dice", "text",
    ),
    "work": (
        "pto_request_status", "next_holiday", "insurance_change", "insurance",
        "meeting_schedule", "payday", "taxes", "income", "rollover_401k",
        "pto_balance", "pto_request", "w2", "schedule_meeting",
        "direct_deposit", "pto_used",
    ),
    "small_talk": (
        "who_made_you", "meaning_of_life", "who_do_you_work_for",
        "do_you_have_pets", "what_are_your_hobbies", "fun_fact",
        "what_is_your_name", "where_are_you_from", "goodbye", "thank_you",
        "greeting", "tell_joke", "are_you_a_bot", "how_old_are_you",
        "what_can_i_ask_you",
    ),
    "meta": (
        "change_speed", "user_name", "whisper_mode", "yes", "change_volume",
        "no", "change_language", "repeat", "change_accent", "cancel",
        "sync_device", "change_user_name", "change_ai_name", "reset_settings",
        "maybe",
    ),
}

SUPPORTED_METHODS = (
    "vanilla",
    "slowheat_none",
    "slowheat_ffn",
    "slowheat",
    "replay",
    "slowheat_replay",
    "lora_replay",
    "slowheat_lora_replay",
)
SLOWHEAT_METHODS = {
    "slowheat_none", "slowheat_ffn", "slowheat", "slowheat_replay",
    "slowheat_lora_replay",
}
REPLAY_METHODS = {
    "replay", "slowheat_replay", "lora_replay", "slowheat_lora_replay",
}
LORA_METHODS = {"lora_replay", "slowheat_lora_replay"}
CHECKPOINT_SCHEMA_VERSION = 1
BERT_MINI_MODEL = "google/bert_uncased_L-4_H-256_A-4"
BERT_BASE_MODEL = "google-bert/bert-base-uncased"
DEFAULT_CALIBRATION_GRID = tuple(
    {
        "slow_strength": strength,
        "ffn_plasticity_budget": ffn_budget,
        "attention_plasticity_budget": attention_budget,
    }
    for strength, ffn_budget, attention_budget in itertools.product(
        (1.0, 3.0, 10.0, 30.0),
        (0.25, 0.50),
        (0.25, 0.50),
    )
)


@dataclass(frozen=True)
class SplitCLINC150Config:
    seed: int = 42
    model_name: str = BERT_MINI_MODEL
    model_revision: str | None = None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    dataset_name: str = "clinc/clinc_oos"
    dataset_config: str = "plus"
    dataset_revision: str | None = None
    max_length: int = 128
    batch_size: int = 32
    replay_batch_size: int = 32
    replay_per_class: int = 20
    epochs_per_task: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    slow_strength: float = 3.0
    ffn_plasticity_budget: float = 0.25
    attention_plasticity_budget: float = 0.25
    importance_decay: float = 0.99
    importance_eps: float = 1e-8
    attention_combination: str = "max"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    evaluate_test: bool = True
    methods: tuple[str, ...] = (
        "vanilla", "slowheat_none", "slowheat_ffn", "slowheat", "replay",
        "slowheat_replay",
    )
    device: str = "cpu"

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed deve ser não negativa")
        if not self.model_name or not self.dataset_name:
            raise ValueError("model_name e dataset_name não podem ser vazios")
        integers = {
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "replay_batch_size": self.replay_batch_size,
            "replay_per_class": self.replay_per_class,
            "epochs_per_task": self.epochs_per_task,
            "lora_rank": self.lora_rank,
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in integers.values()
        ):
            raise ValueError("contagens do protocolo CLINC150 devem ser positivas")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio deve estar em [0, 1)")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate deve ser > 0 e weight_decay >= 0")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm deve ser > 0")
        if not isinstance(self.evaluate_test, bool):
            raise TypeError("evaluate_test deve ser booleano")
        unknown = set(self.methods) - set(SUPPORTED_METHODS)
        if unknown or not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError(f"lista de métodos inválida: {sorted(unknown)}")
        BertSlowHeatConfig(
            slow_strength=self.slow_strength,
            ffn_plasticity_budget=self.ffn_plasticity_budget,
            attention_plasticity_budget=self.attention_plasticity_budget,
            importance_decay=self.importance_decay,
            importance_eps=self.importance_eps,
            attention_combination=self.attention_combination,  # type: ignore[arg-type]
        )
        ExactSlowHeatLoRAConfig(rank=self.lora_rank, alpha=self.lora_alpha)


@dataclass(frozen=True)
class TokenizedTextSplit:
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
    labels: Tensor
    source_indices: Tensor

    def __post_init__(self) -> None:
        count = len(self.labels)
        if any(
            tensor.dtype != torch.long
            for tensor in (
                self.input_ids, self.attention_mask, self.token_type_ids,
                self.labels, self.source_indices,
            )
        ):
            raise TypeError("tensores textuais devem usar dtype int64")
        if self.input_ids.ndim != 2 or self.input_ids.shape != self.attention_mask.shape:
            raise ValueError("input_ids e attention_mask devem ter forma [N, T]")
        if self.token_type_ids.shape != self.input_ids.shape:
            raise ValueError("token_type_ids deve alinhar input_ids")
        if self.labels.shape != (count,) or self.source_indices.shape != (count,):
            raise ValueError("labels/source_indices devem ter forma [N]")
        if self.input_ids.shape[0] != count:
            raise ValueError("campos do split textual estão desalinhados")
        if any(tensor.device.type != "cpu" for tensor in self.tensors()):
            raise ValueError("splits textuais materializados devem permanecer em CPU")

    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self.input_ids, self.attention_mask, self.token_type_ids,
            self.labels, self.source_indices,
        )

    def select(self, indices: Tensor) -> TokenizedTextSplit:
        indices = indices.detach().cpu().to(torch.long)
        return TokenizedTextSplit(*(tensor[indices] for tensor in self.tensors()))


@dataclass(frozen=True)
class CLINC150Task:
    domain: str
    classes: tuple[int, ...]
    train: TokenizedTextSplit
    validation: TokenizedTextSplit
    test: TokenizedTextSplit


class TextReplayBuffer:
    """CPU int64 replay memory with tokenizer-compatible fields."""

    def __init__(self, max_length: int) -> None:
        self.max_length = max_length
        self.input_ids = torch.empty((0, max_length), dtype=torch.long)
        self.attention_mask = torch.empty((0, max_length), dtype=torch.long)
        self.token_type_ids = torch.empty((0, max_length), dtype=torch.long)
        self.labels = torch.empty(0, dtype=torch.long)
        self.source_tasks = torch.empty(0, dtype=torch.long)
        self.source_indices = torch.empty(0, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def append(self, split: TokenizedTextSplit, *, task_index: int) -> None:
        if split.input_ids.shape[1] != self.max_length:
            raise ValueError("max_length do replay é incompatível")
        self.input_ids = torch.cat((self.input_ids, split.input_ids))
        self.attention_mask = torch.cat((self.attention_mask, split.attention_mask))
        self.token_type_ids = torch.cat((self.token_type_ids, split.token_type_ids))
        self.labels = torch.cat((self.labels, split.labels))
        self.source_tasks = torch.cat(
            (self.source_tasks, torch.full((len(split.labels),), task_index, dtype=torch.long))
        )
        self.source_indices = torch.cat((self.source_indices, split.source_indices))

    def batch(self, indices: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.input_ids[indices], self.attention_mask[indices],
            self.token_type_ids[indices], self.labels[indices],
        )

    @property
    def memory_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.state_dict().values())

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "token_type_ids": self.token_type_ids,
            "labels": self.labels,
            "source_tasks": self.source_tasks,
            "source_indices": self.source_indices,
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        required = set(self.state_dict())
        if set(state) != required:
            raise ValueError("estado do replay textual está incompleto")
        if state["input_ids"].shape[1:] != (self.max_length,):
            raise ValueError("checkpoint de replay usa max_length incompatível")
        count = len(state["labels"])
        if any(len(state[name]) != count for name in required):
            raise ValueError("checkpoint de replay contém campos desalinhados")
        for name in required:
            tensor = state[name]
            if tensor.dtype != torch.long:
                raise TypeError("checkpoint de replay deve usar int64")
            setattr(self, name, tensor.detach().cpu().contiguous())


def _intent_order() -> tuple[str, ...]:
    return tuple(intent for intents in CLINC150_DOMAINS.values() for intent in intents)


def _label_names(dataset_split, label_column: str) -> list[str] | None:
    feature = getattr(dataset_split, "features", {}).get(label_column)
    names = getattr(feature, "names", None)
    return list(names) if names is not None else None


def _tokenize_records(
    texts: list[str],
    labels: list[int],
    source_indices: list[int],
    *,
    tokenizer,
    max_length: int,
) -> TokenizedTextSplit:
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].detach().cpu().to(torch.long)
    attention_mask = encoded["attention_mask"].detach().cpu().to(torch.long)
    token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))
    return TokenizedTextSplit(
        input_ids,
        attention_mask,
        token_type_ids.detach().cpu().to(torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(source_indices, dtype=torch.long),
    )


def build_clinc150_tasks(
    dataset,
    tokenizer,
    *,
    max_length: int = 128,
) -> list[CLINC150Task]:
    """Materialize ten official CLINC150 domains and exclude OOS records."""

    intent_to_id = {name: index for index, name in enumerate(_intent_order())}
    if len(intent_to_id) != 150:
        raise RuntimeError("mapa oficial CLINC150 deve conter 150 intenções únicas")
    split_names = {"train": "train", "validation": "validation", "test": "test"}
    tasks: list[CLINC150Task] = []
    for domain, intents in CLINC150_DOMAINS.items():
        materialized: dict[str, TokenizedTextSplit] = {}
        for target_name, source_name in split_names.items():
            source = dataset[source_name]
            columns = set(getattr(source, "column_names", ()))
            label_column = "intent" if "intent" in columns else "label"
            names = _label_names(source, label_column)
            texts: list[str] = []
            labels: list[int] = []
            indices: list[int] = []
            for source_index, row in enumerate(source):
                raw_label = row[label_column]
                intent = names[int(raw_label)] if names is not None else str(raw_label)
                if intent not in intents:
                    continue
                texts.append(str(row["text"]))
                labels.append(intent_to_id[intent])
                indices.append(source_index)
            if not texts:
                raise RuntimeError(f"domínio {domain} não possui exemplos em {source_name}")
            materialized[target_name] = _tokenize_records(
                texts,
                labels,
                indices,
                tokenizer=tokenizer,
                max_length=max_length,
            )
        tasks.append(
            CLINC150Task(
                domain=domain,
                classes=tuple(intent_to_id[intent] for intent in intents),
                train=materialized["train"],
                validation=materialized["validation"],
                test=materialized["test"],
            )
        )
    return tasks


def load_clinc150_tasks(
    config: SplitCLINC150Config,
) -> tuple[list[CLINC150Task], dict[str, Any]]:
    """Download/tokenize the configured dataset and return provenance metadata."""

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise ImportError("CLINC150 requer o extra opcional 'nlp'") from error
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        revision=config.dataset_revision,
    )
    tokenizer_name = config.tokenizer_name or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=config.tokenizer_revision or config.model_revision,
        use_fast=True,
    )
    tasks = build_clinc150_tasks(dataset, tokenizer, max_length=config.max_length)
    metadata = {
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config,
        "dataset_revision": config.dataset_revision,
        "dataset_fingerprints": {
            name: getattr(split, "_fingerprint", None) for name, split in dataset.items()
        },
        "tokenizer_name": getattr(tokenizer, "name_or_path", tokenizer_name),
        "tokenizer_revision": config.tokenizer_revision or config.model_revision,
        "tokenizer_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "max_length": config.max_length,
        "domain_order": list(CLINC150_DOMAINS),
        "domain_intents": {key: list(value) for key, value in CLINC150_DOMAINS.items()},
        "oos_policy": "excluded",
    }
    return tasks, metadata


def text_task_fingerprint(tasks: list[CLINC150Task]) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        digest.update(task.domain.encode())
        digest.update(json.dumps(task.classes).encode())
        for split_name in ("train", "validation", "test"):
            split = getattr(task, split_name)
            for tensor in split.tensors():
                contiguous = tensor.contiguous()
                digest.update(str(contiguous.dtype).encode())
                digest.update(json.dumps(list(contiguous.shape)).encode())
                digest.update(memoryview(contiguous.numpy()).cast("B"))
    return digest.hexdigest()


def select_replay_examples(
    task: CLINC150Task,
    *,
    per_class: int,
) -> TokenizedTextSplit:
    selected: list[Tensor] = []
    for label in task.classes:
        indices = torch.nonzero(task.train.labels == label, as_tuple=False).flatten()
        if len(indices) < per_class:
            raise ValueError(
                f"classe {label} possui {len(indices)} exemplos; replay requer {per_class}"
            )
        selected.append(indices[:per_class])
    return task.train.select(torch.cat(selected))


def _trimmed_batch(
    input_ids: Tensor,
    attention_mask: Tensor,
    token_type_ids: Tensor,
    labels: Tensor,
    *,
    device: str,
) -> dict[str, Tensor]:
    width = max(1, int(attention_mask.sum(dim=1).max().item()))
    return {
        "input_ids": input_ids[:, :width].to(device),
        "attention_mask": attention_mask[:, :width].to(device),
        "token_type_ids": token_type_ids[:, :width].to(device),
        "labels": labels.to(device),
    }


def _seen_classes(tasks: list[CLINC150Task], stage: int) -> tuple[int, ...]:
    return tuple(label for task in tasks[: stage + 1] for label in task.classes)


def _mask_unseen_logits(logits: Tensor, seen_classes: tuple[int, ...]) -> Tensor:
    unseen = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    unseen[list(seen_classes)] = False
    return logits.masked_fill(unseen, torch.finfo(logits.dtype).min)


def _macro_f1(predictions: Tensor, targets: Tensor, classes: tuple[int, ...]) -> float:
    scores: list[float] = []
    for label in classes:
        true_positive = int(((predictions == label) & (targets == label)).sum())
        false_positive = int(((predictions == label) & (targets != label)).sum())
        false_negative = int(((predictions != label) & (targets == label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    split: TokenizedTextSplit,
    *,
    task_classes: tuple[int, ...],
    seen_classes: tuple[int, ...],
    batch_size: int,
    device: str,
) -> tuple[float, float, float]:
    was_training = model.training
    model.eval()
    class_predictions: list[Tensor] = []
    task_predictions: list[Tensor] = []
    targets: list[Tensor] = []
    for start in range(0, len(split.labels), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(split.labels)))
        batch = _trimmed_batch(
            split.input_ids[indices], split.attention_mask[indices],
            split.token_type_ids[indices], split.labels[indices], device=device,
        )
        labels = batch.pop("labels")
        logits = model(**batch).logits
        class_predictions.append(_mask_unseen_logits(logits, seen_classes).argmax(-1).cpu())
        task_indices = torch.tensor(task_classes, device=device)
        local = logits.index_select(-1, task_indices).argmax(-1)
        task_predictions.append(task_indices[local].cpu())
        targets.append(labels.cpu())
    if was_training:
        model.train()
    predicted = torch.cat(class_predictions)
    task_predicted = torch.cat(task_predictions)
    expected = torch.cat(targets)
    return (
        float((predicted == expected).float().mean()),
        float((task_predicted == expected).float().mean()),
        _macro_f1(predicted, expected, task_classes),
    )


def _json_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in matrix
    ]


def _slowheat_config(config: SplitCLINC150Config, method: str) -> BertSlowHeatConfig:
    return BertSlowHeatConfig(
        slow_strength=config.slow_strength,
        ffn_plasticity_budget=config.ffn_plasticity_budget,
        attention_plasticity_budget=config.attention_plasticity_budget,
        importance_decay=config.importance_decay,
        importance_eps=config.importance_eps,
        attention_combination=config.attention_combination,  # type: ignore[arg-type]
        track_ffn=True,
        track_attention=method != "slowheat_ffn",
        protect_classifier=False,
    )


def _find_slowheat_model(model: nn.Module) -> SlowHeatBertForSequenceClassification | None:
    return next(
        (
            module
            for module in model.modules()
            if isinstance(module, SlowHeatBertForSequenceClassification)
        ),
        None,
    )


def _build_model(
    method: str,
    config: SplitCLINC150Config,
    model_config,
    initial_state: dict[str, Any],
):
    from transformers import BertForSequenceClassification

    torch.manual_seed(config.seed)
    if method in {"vanilla", "replay"}:
        model = BertForSequenceClassification(deepcopy(model_config))
        model.load_state_dict(initial_state)
        return model
    model = SlowHeatBertForSequenceClassification(
        deepcopy(model_config), _slowheat_config(config, method)
    )
    model.load_state_dict(initial_state, strict=False)
    if method in LORA_METHODS:
        model = build_exact_slowheat_lora(
            model,
            ExactSlowHeatLoRAConfig(rank=config.lora_rank, alpha=config.lora_alpha),
        )
    return model


def _build_optimizer_and_scheduler(
    model: nn.Module,
    method: str,
    config: SplitCLINC150Config,
    total_steps: int,
):
    from transformers import get_linear_schedule_with_warmup

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if method in SLOWHEAT_METHODS:
        optimizer = SlowHeatAdamW(parameters, **kwargs)
        slow_model = _find_slowheat_model(model)
        assert slow_model is not None
        if method == "slowheat_lora_replay":
            register_exact_lora_masks(model, optimizer)
        else:
            slow_model.register_plasticity_masks(optimizer)
    else:
        optimizer = torch.optim.AdamW(parameters, **kwargs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def _stage_schedule(count: int, *, seed: int) -> list[Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randperm(count, generator=generator)]


def _checkpoint_identity(
    config: SplitCLINC150Config,
    metadata: dict[str, Any],
    data_sha256: str,
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "data_sha256": data_sha256,
        "metadata": metadata,
        "task_order": list(CLINC150_DOMAINS),
    }


def _run_split_clinc150(
    config: SplitCLINC150Config,
    tasks: list[CLINC150Task],
    *,
    metadata: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    telemetry: bool = False,
    telemetry_every: int = 10,
    _telemetry_holder: list[TelemetryWriter] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all configured methods with paired initialization and schedules."""

    config.validate()
    if len(tasks) != 10 or any(len(task.classes) != 15 for task in tasks):
        raise ValueError("CLINC150 requer dez tasks de quinze classes")
    if tuple(task.domain for task in tasks) != tuple(CLINC150_DOMAINS):
        raise ValueError("ordem de domínios diverge do protocolo")
    metadata = {} if metadata is None else dict(metadata)
    data_sha256 = text_task_fingerprint(tasks)
    destination = None if output_dir is None else Path(output_dir)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    from transformers import BertForSequenceClassification

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    template = BertForSequenceClassification.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        num_labels=150,
        ignore_mismatched_sizes=True,
    )
    model_config = deepcopy(template.config)
    initial_state = deepcopy(template.state_dict())
    metadata["resolved_model_commit"] = getattr(template.config, "_commit_hash", None)
    del template
    identity = _checkpoint_identity(config, metadata, data_sha256)
    if destination is not None:
        write_json_atomic(destination / "protocol.json", identity)
    if telemetry and destination is None:
        raise ValueError("telemetria requer output_dir")
    telemetry_writer = (
        TelemetryWriter(
            destination,
            identity=identity,
            every=telemetry_every,
            resumed=resume,
        )
        if telemetry and destination is not None
        else None
    )
    if telemetry_writer is not None:
        if _telemetry_holder is not None:
            _telemetry_holder.append(telemetry_writer)
        telemetry_writer.emit(
            "run_start",
            seed=config.seed,
            methods=list(config.methods),
            task_count=len(tasks),
            model_name=config.model_name,
            device=config.device,
        )
    task_steps = [
        math.ceil(len(task.train.labels) / config.batch_size)
        * config.epochs_per_task
        for task in tasks
    ]
    total_steps = sum(task_steps)
    results: dict[str, dict[str, Any]] = {}

    for method in config.methods:
        method_started = time.perf_counter()
        telemetry_overhead_start = (
            telemetry_writer.overhead_seconds
            if telemetry_writer is not None
            else 0.0
        )
        model = _build_model(method, config, model_config, initial_state).to(config.device)
        optimizer, scheduler = _build_optimizer_and_scheduler(
            model, method, config, total_steps
        )
        replay = TextReplayBuffer(config.max_length)
        accuracy = np.full((len(tasks), len(tasks)), np.nan)
        validation_accuracy = np.full_like(accuracy, np.nan)
        task_aware = np.full_like(accuracy, np.nan)
        macro_f1 = np.full_like(accuracy, np.nan)
        training_losses: list[list[float]] = []
        capacity_history: list[list[dict[str, float]]] = []
        tokens_processed = 0
        next_stage = 0
        checkpoint_path = (
            None if destination is None else destination / method / "checkpoint.pt"
        )
        if checkpoint_path is not None and checkpoint_path.is_file():
            if not resume:
                raise FileExistsError(
                    f"checkpoint existente para {method}; use resume=True"
                )
            checkpoint = read_torch_checkpoint(checkpoint_path)
            if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise RuntimeError("versão de checkpoint CLINC150 incompatível")
            if checkpoint.get("identity") != identity:
                raise RuntimeError("checkpoint CLINC150 não corresponde ao protocolo")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            replay.load_state_dict(checkpoint["replay"])
            accuracy = checkpoint["accuracy"].numpy()
            validation_accuracy = checkpoint["validation_accuracy"].numpy()
            task_aware = checkpoint["task_aware"].numpy()
            macro_f1 = checkpoint["macro_f1"].numpy()
            training_losses = checkpoint["training_losses"]
            capacity_history = checkpoint["capacity_history"]
            tokens_processed = int(checkpoint["tokens_processed"])
            next_stage = int(checkpoint["next_stage"])

        slow_model = _find_slowheat_model(model)
        method_step = sum(task_steps[:next_stage])
        session_start_step = method_step
        session_start_tokens = tokens_processed
        telemetry_method_started = time.perf_counter()
        recent_losses: deque[float] = deque(
            (
                value
                for task_loss in training_losses[-2:]
                for value in task_loss[-50:]
            ),
            maxlen=50,
        )
        if telemetry_writer is not None:
            telemetry_writer.emit(
                "method_start",
                seed=config.seed,
                method=method,
                resumed=next_stage > 0,
                next_stage=next_stage,
                total_stages=len(tasks),
                method_step=method_step,
                total_steps=total_steps,
                tokens_processed=tokens_processed,
            )
            telemetry_writer.publish_heat(
                slow_model,
                context={
                    "seed": config.seed,
                    "method": method,
                    "stage": max(0, next_stage - 1),
                    "phase": "method_start",
                },
            )
        memory_tracker = PeakMemoryTracker(config.device).start()
        for stage in range(next_stage, len(tasks)):
            task = tasks[stage]
            seen = _seen_classes(tasks, stage)
            stage_losses: list[float] = []
            if telemetry_writer is not None:
                telemetry_writer.emit(
                    "task_start",
                    seed=config.seed,
                    method=method,
                    stage=stage,
                    domain=task.domain,
                    total_stages=len(tasks),
                    seen_class_count=len(seen),
                    method_step=method_step,
                    total_steps=total_steps,
                )
            model.train()
            for epoch in range(config.epochs_per_task):
                epoch_loss_start = len(stage_losses)
                generator = torch.Generator().manual_seed(
                    config.seed * 1_000_003 + stage * 10_007 + epoch
                )
                order = torch.randperm(len(task.train.labels), generator=generator)
                replay_order = (
                    torch.randperm(len(replay), generator=generator)
                    if method in REPLAY_METHODS and len(replay)
                    else torch.empty(0, dtype=torch.long)
                )
                replay_cursor = 0
                batches_in_epoch = math.ceil(len(order) / config.batch_size)
                if telemetry_writer is not None:
                    telemetry_writer.emit(
                        "epoch_start",
                        seed=config.seed,
                        method=method,
                        stage=stage,
                        domain=task.domain,
                        epoch=epoch,
                        total_epochs=config.epochs_per_task,
                        batches_in_epoch=batches_in_epoch,
                        method_step=method_step,
                        total_steps=total_steps,
                    )
                for start in range(0, len(order), config.batch_size):
                    current = order[start : start + config.batch_size]
                    input_ids = task.train.input_ids[current]
                    attention_mask = task.train.attention_mask[current]
                    token_type_ids = task.train.token_type_ids[current]
                    labels = task.train.labels[current]
                    if len(replay_order):
                        if replay_cursor + config.replay_batch_size > len(replay_order):
                            replay_order = torch.randperm(len(replay), generator=generator)
                            replay_cursor = 0
                        selected = replay_order[
                            replay_cursor : replay_cursor + config.replay_batch_size
                        ]
                        replay_cursor += len(selected)
                        remembered = replay.batch(selected)
                        input_ids = torch.cat((input_ids, remembered[0]))
                        attention_mask = torch.cat((attention_mask, remembered[1]))
                        token_type_ids = torch.cat((token_type_ids, remembered[2]))
                        labels = torch.cat((labels, remembered[3]))
                    batch = _trimmed_batch(
                        input_ids, attention_mask, token_type_ids, labels,
                        device=config.device,
                    )
                    labels = batch.pop("labels")
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(**batch).logits
                    loss = F.cross_entropy(_mask_unseen_logits(logits, seen), labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        config.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    loss_value = float(loss.detach())
                    stage_losses.append(loss_value)
                    recent_losses.append(loss_value)
                    tokens_processed += int(batch["attention_mask"].sum())
                    method_step += 1
                    if (
                        telemetry_writer is not None
                        and telemetry_writer.should_publish_batch(method_step)
                    ):
                        elapsed = max(
                            time.perf_counter() - telemetry_method_started,
                            1e-12,
                        )
                        completed_in_session = method_step - session_start_step
                        steps_per_second = completed_in_session / elapsed
                        tokens_per_second = (
                            tokens_processed - session_start_tokens
                        ) / elapsed
                        eta = (
                            (total_steps - method_step) / steps_per_second
                            if steps_per_second > 0.0
                            else math.nan
                        )
                        context = {
                            "seed": config.seed,
                            "method": method,
                            "stage": stage,
                            "domain": task.domain,
                            "epoch": epoch,
                            "batch": start // config.batch_size + 1,
                            "batches_in_epoch": batches_in_epoch,
                            "method_step": method_step,
                            "total_steps": total_steps,
                            "progress": method_step / total_steps,
                            "tokens_processed": tokens_processed,
                        }
                        telemetry_writer.emit(
                            "batch",
                            **context,
                            loss=loss_value,
                            rolling_loss=sum(recent_losses) / len(recent_losses),
                            learning_rate=float(optimizer.param_groups[0]["lr"]),
                            tokens_per_second=tokens_per_second,
                            steps_per_second=steps_per_second,
                            eta_seconds=finite_or_none(eta),
                            **cuda_memory_payload(config.device),
                        )
                        telemetry_writer.publish_heat(
                            slow_model,
                            context={**context, "phase": "training"},
                        )
                if telemetry_writer is not None:
                    epoch_losses = stage_losses[epoch_loss_start:]
                    telemetry_writer.emit(
                        "epoch_end",
                        seed=config.seed,
                        method=method,
                        stage=stage,
                        domain=task.domain,
                        epoch=epoch,
                        total_epochs=config.epochs_per_task,
                        mean_loss=(
                            sum(epoch_losses) / len(epoch_losses)
                            if epoch_losses
                            else None
                        ),
                        method_step=method_step,
                        total_steps=total_steps,
                    )
            training_losses.append(stage_losses)

            if method in REPLAY_METHODS:
                replay.append(
                    select_replay_examples(task, per_class=config.replay_per_class),
                    task_index=stage,
                )
            slow_model = _find_slowheat_model(model)
            if method in SLOWHEAT_METHODS and method != "slowheat_none":
                assert slow_model is not None
                slow_model.consolidate(strategy="max")
                capacity_history.append(slow_model.capacity_metrics())
                if telemetry_writer is not None:
                    telemetry_writer.emit(
                        "consolidation",
                        seed=config.seed,
                        method=method,
                        stage=stage,
                        domain=task.domain,
                        strategy="max",
                        capacity=capacity_history[-1],
                    )

            if telemetry_writer is not None:
                telemetry_writer.publish_heat(
                    slow_model,
                    context={
                        "seed": config.seed,
                        "method": method,
                        "stage": stage,
                        "domain": task.domain,
                        "phase": "task_boundary",
                        "method_step": method_step,
                        "total_steps": total_steps,
                    },
                    stage_snapshot=True,
                )
                telemetry_writer.emit(
                    "evaluation_start",
                    seed=config.seed,
                    method=method,
                    stage=stage,
                    domain=task.domain,
                    evaluated_tasks=stage + 1,
                )

            for task_index in range(stage + 1):
                validation_class_il, _, _ = _evaluate(
                    model,
                    tasks[task_index].validation,
                    task_classes=tasks[task_index].classes,
                    seen_classes=seen,
                    batch_size=config.batch_size,
                    device=config.device,
                )
                validation_accuracy[stage, task_index] = validation_class_il
                if config.evaluate_test:
                    class_il, aware, f1 = _evaluate(
                        model,
                        tasks[task_index].test,
                        task_classes=tasks[task_index].classes,
                        seen_classes=seen,
                        batch_size=config.batch_size,
                        device=config.device,
                    )
                    accuracy[stage, task_index] = class_il
                    task_aware[stage, task_index] = aware
                    macro_f1[stage, task_index] = f1

            if telemetry_writer is not None:
                telemetry_writer.emit(
                    "evaluation_end",
                    seed=config.seed,
                    method=method,
                    stage=stage,
                    domain=task.domain,
                    validation_accuracy_matrix=_json_matrix(validation_accuracy),
                    accuracy_matrix=(
                        _json_matrix(accuracy) if config.evaluate_test else None
                    ),
                    task_aware_accuracy_matrix=(
                        _json_matrix(task_aware) if config.evaluate_test else None
                    ),
                    macro_f1_matrix=(
                        _json_matrix(macro_f1) if config.evaluate_test else None
                    ),
                )

            if checkpoint_path is not None:
                write_torch_atomic(
                    checkpoint_path,
                    {
                        "schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "identity": identity,
                        "next_stage": stage + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "replay": replay.state_dict(),
                        "accuracy": torch.from_numpy(accuracy.copy()),
                        "validation_accuracy": torch.from_numpy(
                            validation_accuracy.copy()
                        ),
                        "task_aware": torch.from_numpy(task_aware.copy()),
                        "macro_f1": torch.from_numpy(macro_f1.copy()),
                        "training_losses": training_losses,
                        "capacity_history": capacity_history,
                        "tokens_processed": tokens_processed,
                    },
                )
                if telemetry_writer is not None:
                    telemetry_writer.emit(
                        "checkpoint",
                        seed=config.seed,
                        method=method,
                        stage=stage,
                        domain=task.domain,
                        next_stage=stage + 1,
                        method_step=method_step,
                    )
            if telemetry_writer is not None:
                telemetry_writer.emit(
                    "task_end",
                    seed=config.seed,
                    method=method,
                    stage=stage,
                    domain=task.domain,
                    method_step=method_step,
                    total_steps=total_steps,
                    mean_loss=(
                        sum(stage_losses) / len(stage_losses)
                        if stage_losses
                        else None
                    ),
                )

        memory = memory_tracker.stop()
        validation_metrics = asdict(compute_cl_metrics(validation_accuracy))
        metrics = (
            asdict(compute_cl_metrics(accuracy)) if config.evaluate_test else None
        )
        task_aware_metrics = (
            asdict(compute_cl_metrics(task_aware)) if config.evaluate_test else None
        )
        result = {
            "validation_accuracy_matrix": _json_matrix(validation_accuracy),
            "validation_metrics": validation_metrics,
            "accuracy_matrix": _json_matrix(accuracy) if config.evaluate_test else None,
            "task_aware_accuracy_matrix": (
                _json_matrix(task_aware) if config.evaluate_test else None
            ),
            "macro_f1_matrix": _json_matrix(macro_f1) if config.evaluate_test else None,
            "metrics": metrics,
            "task_aware_metrics": task_aware_metrics,
            "final_macro_f1": (
                float(np.mean(macro_f1[-1])) if config.evaluate_test else None
            ),
            "classifier_gap": (
                task_aware_metrics["final_average_accuracy"]
                - metrics["final_average_accuracy"]
                if metrics is not None and task_aware_metrics is not None
                else None
            ),
            "training_losses": training_losses,
            "capacity_history": capacity_history,
            "tokens_processed": tokens_processed,
            "replay_memory_bytes": replay.memory_bytes,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "elapsed_seconds": time.perf_counter() - method_started,
            "peak_memory": memory,
        }
        results[method] = result
        if destination is not None:
            write_json_atomic(destination / method / "results.json", result)
        if telemetry_writer is not None:
            telemetry_overhead = (
                telemetry_writer.overhead_seconds - telemetry_overhead_start
            )
            telemetry_writer.emit(
                "method_end",
                seed=config.seed,
                method=method,
                status="complete",
                method_step=method_step,
                total_steps=total_steps,
                tokens_processed=tokens_processed,
                elapsed_seconds=result["elapsed_seconds"],
                telemetry_overhead_seconds=telemetry_overhead,
                telemetry_overhead_ratio=(
                    telemetry_overhead / result["elapsed_seconds"]
                    if result["elapsed_seconds"] > 0.0
                    else 0.0
                ),
                validation_metrics=validation_metrics,
                metrics=metrics,
                peak_memory=memory,
            )
        slow_model = _find_slowheat_model(model)
        if slow_model is not None:
            slow_model.remove_slowheat_instrumentation()
        del model, optimizer, scheduler
        gc.collect()
        if torch.device(config.device).type == "cuda":
            torch.cuda.empty_cache()
    if telemetry_writer is not None:
        telemetry_writer.emit(
            "run_end",
            seed=config.seed,
            status="complete",
            completed_methods=list(results),
        )
        telemetry_writer.close()
    return results


def run_split_clinc150(
    config: SplitCLINC150Config,
    tasks: list[CLINC150Task],
    *,
    metadata: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    telemetry: bool = False,
    telemetry_every: int = 10,
) -> dict[str, dict[str, Any]]:
    """Run paired methods and optionally publish read-only live telemetry."""

    holder: list[TelemetryWriter] = []
    try:
        return _run_split_clinc150(
            config,
            tasks,
            metadata=metadata,
            output_dir=output_dir,
            resume=resume,
            telemetry=telemetry,
            telemetry_every=telemetry_every,
            _telemetry_holder=holder,
        )
    except BaseException as error:
        if holder and not holder[-1].is_closed:
            writer = holder[-1]
            memory_payload = {}
            with suppress(Exception):
                memory_payload = cuda_memory_payload(config.device)
            with suppress(Exception):
                writer.emit(
                    "run_error",
                    seed=config.seed,
                    status="failed",
                    error_type=type(error).__name__,
                    error_message=str(error)[:2_000],
                    **memory_payload,
                )
            with suppress(Exception):
                writer.close(error=error)
        raise


def run_split_clinc150_multi_seed(
    base_config: SplitCLINC150Config,
    tasks: list[CLINC150Task],
    *,
    seeds: list[int],
    metadata: dict[str, Any],
    output_dir: str | Path,
    resume: bool = False,
    telemetry: bool = False,
    telemetry_every: int = 10,
) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds deve ser não vazio e sem duplicatas")
    if not base_config.evaluate_test:
        raise ValueError("agregação final requer evaluate_test=True")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_environment_manifest(
        destination, project_root=Path(__file__).resolve().parents[1]
    )
    raw: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        config = replace(base_config, seed=seed)
        raw[seed] = run_split_clinc150(
            config,
            tasks,
            metadata=metadata,
            output_dir=destination / f"seed_{seed}",
            resume=resume,
            telemetry=telemetry,
            telemetry_every=telemetry_every,
        )
    aggregate: dict[str, Any] = {
        "seeds": seeds,
        "primary_endpoint": "final_average_accuracy",
        "methods": {},
        "paired_differences": {},
    }
    metric_names = (
        "final_average_accuracy", "average_forgetting", "backward_transfer"
    )
    for method in base_config.methods:
        aggregate["methods"][method] = {
            metric: normal_summary(
                [raw[seed][method]["metrics"][metric] for seed in seeds]
            )
            for metric in metric_names
        }
    pairs = (
        ("replay", "slowheat_replay"),
        ("lora_replay", "slowheat_lora_replay"),
    )
    for reference, candidate in pairs:
        if reference not in base_config.methods or candidate not in base_config.methods:
            continue
        differences = [
            raw[seed][candidate]["metrics"]["final_average_accuracy"]
            - raw[seed][reference]["metrics"]["final_average_accuracy"]
            for seed in seeds
        ]
        aggregate["paired_differences"][f"{candidate}_minus_{reference}"] = {
            **normal_summary(differences),
            "exact_sign_test_p": exact_two_sided_sign_test(differences),
        }
    write_json_atomic(destination / "aggregate.json", aggregate)
    write_json_atomic(
        destination / "multi_seed_config.json",
        {"config": asdict(base_config), "seeds": seeds, "metadata": metadata},
    )
    return aggregate


def calibrate_bert_mini_slowheat(
    base_config: SplitCLINC150Config,
    tasks: list[CLINC150Task],
    *,
    metadata: dict[str, Any],
    seeds: tuple[int, ...] = (698_971_273, 1_288_660_088, 1_181_804_493),
    candidates: tuple[dict[str, float], ...] = DEFAULT_CALIBRATION_GRID,
    output_dir: str | Path,
    resume: bool = False,
    telemetry: bool = False,
    telemetry_every: int = 10,
) -> dict[str, Any]:
    """Select SlowHeat hyperparameters using validation matrices only."""

    if base_config.model_name != BERT_MINI_MODEL:
        raise ValueError("calibração deve usar o BERT-Mini pré-declarado")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds de calibração devem ser únicas e não vazias")
    if not candidates:
        raise ValueError("a grade de calibração não pode ser vazia")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        differences: list[float] = []
        reference_scores: list[float] = []
        candidate_scores: list[float] = []
        for seed in seeds:
            config = replace(
                base_config,
                seed=seed,
                methods=("replay", "slowheat_replay"),
                evaluate_test=False,
                slow_strength=float(candidate["slow_strength"]),
                ffn_plasticity_budget=float(candidate["ffn_plasticity_budget"]),
                attention_plasticity_budget=float(
                    candidate["attention_plasticity_budget"]
                ),
            )
            result = run_split_clinc150(
                config,
                tasks,
                metadata=metadata,
                output_dir=(
                    destination / f"candidate_{candidate_index}" / f"seed_{seed}"
                ),
                resume=resume,
                telemetry=telemetry,
                telemetry_every=telemetry_every,
            )
            reference = float(
                result["replay"]["validation_metrics"]["final_average_accuracy"]
            )
            slowheat = float(
                result["slowheat_replay"]["validation_metrics"][
                    "final_average_accuracy"
                ]
            )
            reference_scores.append(reference)
            candidate_scores.append(slowheat)
            differences.append(slowheat - reference)
        summaries.append(
            {
                "candidate_index": candidate_index,
                "hyperparameters": dict(candidate),
                "reference_validation": normal_summary(reference_scores),
                "slowheat_validation": normal_summary(candidate_scores),
                "paired_validation_difference": normal_summary(differences),
            }
        )
    selected = min(
        summaries,
        key=lambda item: (
            -item["paired_validation_difference"]["mean"],
            item["hyperparameters"]["slow_strength"],
            -item["hyperparameters"]["ffn_plasticity_budget"],
            -item["hyperparameters"]["attention_plasticity_budget"],
        ),
    )
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_test_evaluation",
        "selection_rule": (
            "maximize mean paired SlowHeat+Replay minus Replay final validation "
            "accuracy; ties prefer lower strength then greater plastic capacity"
        ),
        "model_name": BERT_MINI_MODEL,
        "test_evaluated": False,
        "seeds": list(seeds),
        "selected": selected,
        "candidates": summaries,
        "metadata": metadata,
    }
    write_json_atomic(destination / "frozen_slowheat_manifest.json", manifest)
    return manifest


def apply_frozen_slowheat_manifest(
    config: SplitCLINC150Config,
    manifest: dict[str, Any],
    *,
    model_name: str | None = None,
) -> SplitCLINC150Config:
    """Apply a validation-frozen candidate to Mini or Base benchmark config."""

    if manifest.get("schema_version") != 1 or manifest.get("status") != (
        "frozen_before_test_evaluation"
    ):
        raise ValueError("manifesto SlowHeat não está congelado ou é incompatível")
    if manifest.get("test_evaluated") is not False:
        raise ValueError("manifesto de calibração não pode ter consultado o teste")
    selected = manifest.get("selected", {}).get("hyperparameters")
    required = {
        "slow_strength",
        "ffn_plasticity_budget",
        "attention_plasticity_budget",
    }
    if not isinstance(selected, dict) or set(selected) != required:
        raise ValueError("manifesto não contém hiperparâmetros SlowHeat completos")
    return replace(
        config,
        model_name=model_name or config.model_name,
        slow_strength=float(selected["slow_strength"]),
        ffn_plasticity_budget=float(selected["ffn_plasticity_budget"]),
        attention_plasticity_budget=float(
            selected["attention_plasticity_budget"]
        ),
        evaluate_test=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/split_clinc150")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS)
    parser.add_argument("--model-name", default=SplitCLINC150Config.model_name)
    parser.add_argument(
        "--batch-size", type=int, default=SplitCLINC150Config.batch_size
    )
    parser.add_argument(
        "--replay-batch-size",
        type=int,
        default=SplitCLINC150Config.replay_batch_size,
    )
    parser.add_argument(
        "--max-length", type=int, default=SplitCLINC150Config.max_length
    )
    parser.add_argument(
        "--epochs-per-task",
        type=int,
        default=SplitCLINC150Config.epochs_per_task,
    )
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--frozen-manifest")
    parser.add_argument("--bert-base", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--telemetry-every", type=int, default=10)
    args = parser.parse_args()
    config = SplitCLINC150Config(
        model_name=args.model_name,
        methods=(
            tuple(args.methods)
            if args.methods is not None
            else SplitCLINC150Config.methods
        ),
        device=args.device,
        batch_size=args.batch_size,
        replay_batch_size=args.replay_batch_size,
        max_length=args.max_length,
        epochs_per_task=args.epochs_per_task,
    )
    tasks, metadata = load_clinc150_tasks(config)
    if args.calibrate:
        if args.frozen_manifest or args.bert_base:
            parser.error("--calibrate não pode ser combinado com manifesto/BERT-base")
        calibrate_bert_mini_slowheat(
            config,
            tasks,
            metadata=metadata,
            seeds=tuple(args.seeds),
            output_dir=args.output_dir,
            resume=args.resume,
            telemetry=args.telemetry,
            telemetry_every=args.telemetry_every,
        )
        return
    if args.bert_base and not args.frozen_manifest:
        parser.error("--bert-base requer --frozen-manifest")
    if args.frozen_manifest:
        with Path(args.frozen_manifest).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        config = apply_frozen_slowheat_manifest(
            config,
            manifest,
            model_name=BERT_BASE_MODEL if args.bert_base else None,
        )
        if args.bert_base:
            # The tokenizer follows the base model unless explicitly configured.
            config = replace(config, tokenizer_name=BERT_BASE_MODEL)
            tasks, metadata = load_clinc150_tasks(config)
    run_split_clinc150_multi_seed(
        config,
        tasks,
        seeds=args.seeds,
        metadata=metadata,
        output_dir=args.output_dir,
        resume=args.resume,
        telemetry=args.telemetry,
        telemetry_every=args.telemetry_every,
    )


if __name__ == "__main__":
    main()
