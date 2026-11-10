"""Generative continual-learning runner for Fase 3b: SuperNI-15 (SAPT) on Qwen2.5.

Protocol
--------
The 15 SuperNI tasks of SAPT (``CL_Benchmark/SuperNI``) in a declared order
(SAPT order 1 by default, order 2 on request), trained one after another with
a causal-LM LoRA on Qwen2.5-0.5B. Every arm shares the data stream,
tokenization, optimizer family, budget and evaluation code; only the LoRA
plasticity mechanism changes.

* Loss on the ANSWER tokens only: prompt and padding positions are labelled
  ``-100``. Qwen's pad token is its eos token, so masking is by position,
  never by token id.
* Prompt = ``build_prompt`` (definition first). Over ``max_source_length``
  tokens the BODY is cut at its end, so the start (the definition) survives,
  and the ``"\\nOutput:"`` marker is kept so the answer position stays defined.
* Target = first reference, at most ``max_target_length`` tokens, then eos.
* After task ``t``: greedy generation (``max_new_tokens``, stop at eos) on the
  eval split of tasks ``0..t`` only. ROUGE-L is the in-repo
  ``dual_heater.superni_metrics.rouge_l``, maximum over all references (as in
  the official SuperNI scorer). Scores are fractions in [0, 1], not x100.
* Endpoints AP, forgetting and BWT exactly as SAPT ``score.py``.
* Zero-shot ROUGE-L of the base model on every task before any training,
  once per run (every arm starts from the same base and LoRA B = 0).

Arms
----
``vanilla`` (SeqLoRA), ``exact`` (SlowHeat on LoRA, A frozen + rows of B
masked), ``frozen_a_control`` (LoRA-FA) and ``lr_control`` (uniform lr scale,
which must be given explicitly so it can be set to a measured E of another
arm). E is measured as in ``experiments/qwen_lora_slowheat.py``:
``mask_coverage`` + ``plasticity_record`` per arm, then
``attach_surface_plasticity`` against ``vanilla``.

Gates (study doc section 4) are COMPUTED AND RECORDED, never enforced here:
G-aq (score right after training task t minus zero-shot of t), G-imp
(importance alive in every adapted module, checked before each
consolidation) and a G-disp dispersion summary of the primary.

Precision
---------
fp32 by default. ``--precision fp16`` (CUDA only, Pascal: never bf16) loads the
base in fp16, keeps every LoRA adapter in fp32, runs the forward under
autocast fp16 with a GradScaler, and calls ``scaler.unscale_`` BEFORE gradient
clipping and the masked step, so masks act on true gradients. A non-finite
step is skipped. fp16 is known to underflow the importance estimator in full
fine-tuning (``experiments/qwen_iso_plasticity.py``); G-imp is what detects it.

Status: EXPLORATORY runner. No protocol is frozen by this file.

Run (CPU smoke):
    CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src HF_HOME=.hf-cache .venv/bin/python \\
        experiments/superni_cl.py --output tmp/superni_smoke --tasks 2 \\
        --train-per-task 20 --eval-per-task 5 --steps-per-task 10 \\
        --learning-rate-scale 0.6 --device cpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dual_heater.lora_slowheat import (
    LoRASlowHeatConfig,
    QwenLoRASlowHeat,
    attach_surface_plasticity,
    build_lora_slowheat,
)
from dual_heater.optim import SlowHeatAdamW
from dual_heater.superni_data import SuperNIExample, SuperNITask, load_sapt_superni
from dual_heater.superni_metrics import exact_match, rouge_l

ARMS: tuple[str, ...] = ("vanilla", "exact", "frozen_a_control", "lr_control")
IGNORE_INDEX = -100
OUTPUT_MARKER = "\nOutput:"
PRECISIONS: tuple[str, ...] = ("fp32", "fp16")
DEFAULT_DATA_ROOT = "/mnt/B-SSD/fachel/datasets/sapt/CL_Benchmark/SuperNI"


@dataclass(frozen=True)
class RunConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    data_root: str = DEFAULT_DATA_ROOT
    order: int = 1
    tasks: int | None = None
    eval_split: str = "test"
    train_per_task: int | None = None
    eval_per_task: int | None = None
    seed: int = 0
    max_source_length: int = 1024
    max_target_length: int = 50
    max_new_tokens: int = 50
    batch_size: int = 4
    eval_batch_size: int = 8
    epochs_per_task: int = 1
    #: When set, overrides ``epochs_per_task`` with a fixed number of
    #: optimizer steps per task (data cycled in a seeded order).
    steps_per_task: int | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    rank: int = 8
    alpha: float = 16.0
    slow_strength: float = 3.0
    plasticity_budget: float = 0.5
    hard: bool = False
    target_modules: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
    #: Uniform lr scale of ``lr_control``. Required whenever that arm runs.
    learning_rate_scale: float | None = None
    precision: str = "fp32"
    #: G-aq: an arm passes on task t when score[t][t] - zero_shot[t] > margin.
    acquisition_margin: float = 0.0
    #: G-disp: minimum vanilla forgetting for the regime to have dispersion.
    min_vanilla_forgetting: float = 0.0
    #: G-disp: |E_X - E_lr_control| at or below this counts as matched.
    matched_tolerance: float = 0.01
    device: str = "cpu"

    def validate(self, *, arms: Sequence[str]) -> None:
        unknown = [arm for arm in arms if arm not in ARMS]
        if unknown:
            raise ValueError(f"braço desconhecido: {unknown}")
        if len(set(arms)) != len(arms):
            raise ValueError("braços repetidos")
        if "lr_control" in arms and self.learning_rate_scale is None:
            raise ValueError(
                "lr_control exige --learning-rate-scale explícito (o E medido de "
                "outro braço); sem ele o falsificador roda sem tratamento"
            )
        if self.learning_rate_scale is not None and not 0.0 < self.learning_rate_scale <= 1.0:
            raise ValueError("learning_rate_scale deve estar em (0, 1]")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision deve ser um de {PRECISIONS}")
        if self.precision == "fp16" and not self.device.startswith("cuda"):
            raise ValueError("fp16 só é suportado em CUDA")
        for name in ("max_source_length", "max_target_length", "max_new_tokens",
                     "batch_size", "eval_batch_size", "epochs_per_task", "rank"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} deve ser positivo")
        if self.steps_per_task is not None and self.steps_per_task < 1:
            raise ValueError("steps_per_task deve ser positivo")
        if self.max_source_length <= len(OUTPUT_MARKER):
            raise ValueError("max_source_length curto demais para o marcador")


# ---------------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------------


class HFTokenizer:
    """Adapter exposing the four members the runner uses from an HF tokenizer."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self.eos_token_id = int(tokenizer.eos_token_id)
        pad = tokenizer.pad_token_id
        self.pad_token_id = int(pad if pad is not None else tokenizer.eos_token_id)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=True)


@dataclass(frozen=True)
class EncodedExample:
    prompt_ids: list[int]
    target_ids: list[int]
    references: tuple[str, ...]
    example_id: str = ""
    prompt_truncated: bool = False


@dataclass(frozen=True)
class EncodedTask:
    name: str
    train: tuple[EncodedExample, ...]
    evaluation: tuple[EncodedExample, ...]
    ids_digest: str = ""


def encode_prompt(tokenizer, prompt: str, *, max_source_length: int) -> tuple[list[int], bool]:
    """Token ids of ``prompt`` cut to ``max_source_length``, keeping the start.

    Body and output marker are tokenized separately (identically for training
    and generation), and only the body is cut, at its END.
    """

    if prompt.endswith(OUTPUT_MARKER):
        body, marker = prompt[: -len(OUTPUT_MARKER)], OUTPUT_MARKER
    else:
        body, marker = prompt, ""
    body_ids = tokenizer.encode(body)
    marker_ids = tokenizer.encode(marker) if marker else []
    room = max_source_length - len(marker_ids)
    if room < 1:
        raise ValueError("max_source_length não comporta o marcador")
    truncated = len(body_ids) > room
    return body_ids[:room] + marker_ids, truncated


def encode_target(tokenizer, target: str, *, max_target_length: int) -> list[int]:
    """`` target`` (leading space, as it follows ``Output:``), cut, then eos."""

    return tokenizer.encode(" " + target.strip())[:max_target_length] + [tokenizer.eos_token_id]


def encode_example(
    tokenizer, example: SuperNIExample, *, max_source_length: int, max_target_length: int
) -> EncodedExample:
    prompt_ids, truncated = encode_prompt(
        tokenizer, example.prompt, max_source_length=max_source_length
    )
    return EncodedExample(
        prompt_ids=prompt_ids,
        target_ids=encode_target(tokenizer, example.target, max_target_length=max_target_length),
        references=tuple(example.references) or (example.target,),
        example_id=example.example_id,
        prompt_truncated=truncated,
    )


def encode_task(tokenizer, task: SuperNITask, config: RunConfig) -> EncodedTask:
    def encode(examples):
        return tuple(
            encode_example(
                tokenizer,
                example,
                max_source_length=config.max_source_length,
                max_target_length=config.max_target_length,
            )
            for example in examples
        )

    digest = hashlib.sha256()
    for split_name, examples in (("train", task.train), ("eval", task.evaluation)):
        for example in examples:
            digest.update(f"{task.name}|{split_name}|{example.example_id}|{example.prompt}".encode())
    return EncodedTask(
        name=task.name,
        train=encode(task.train),
        evaluation=encode(task.evaluation),
        ids_digest=digest.hexdigest(),
    )


def collate_training(features: Sequence[EncodedExample], *, pad_token_id: int) -> dict[str, Tensor]:
    """Right-padded batch; labels are the answer tokens and nothing else."""

    width = max(len(f.prompt_ids) + len(f.target_ids) for f in features)
    input_ids = torch.full((len(features), width), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(features), width), dtype=torch.long)
    labels = torch.full((len(features), width), IGNORE_INDEX, dtype=torch.long)
    for row, feature in enumerate(features):
        sequence = feature.prompt_ids + feature.target_ids
        start = len(feature.prompt_ids)
        input_ids[row, : len(sequence)] = torch.tensor(sequence)
        attention[row, : len(sequence)] = 1
        labels[row, start : len(sequence)] = torch.tensor(feature.target_ids)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def answer_only_loss(model, batch: dict[str, Tensor]) -> Tensor:
    """Token-mean cross-entropy over answer tokens, ``lm_head`` on those only.

    Same value as ``model(**batch).loss`` (HF shifts labels by one and averages
    over non-ignored tokens), but the vocabulary projection runs on the N
    answer positions instead of all B x T: at 4 x 1075 tokens and a 151,936
    vocabulary the full fp32 logits alone are 2.6 GB and OOM an 11-12 GB card.
    """

    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = inner.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    ).last_hidden_state
    targets = batch["labels"][:, 1:]
    keep = targets != IGNORE_INDEX
    logits = inner.lm_head(hidden[:, :-1][keep])
    return torch.nn.functional.cross_entropy(logits.float(), targets[keep])


# ---------------------------------------------------------------------------
# generation and scoring
# ---------------------------------------------------------------------------


def cut_at_eos(ids: Sequence[int], *, eos_token_id: int) -> list[int]:
    ids = [int(value) for value in ids]
    if eos_token_id in ids:
        return ids[: ids.index(eos_token_id)]
    return ids


@torch.no_grad()
def generate_answers(
    model,
    prompts: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    batch_size: int,
    device: str,
    autocast_dtype: torch.dtype | None = None,
) -> list[list[int]]:
    """Greedy continuation of each prompt, left-padded batches, cut at eos."""

    from transformers import GenerationConfig

    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
    )
    # Batches are formed over prompts sorted by length, so long and short
    # prompts do not share a batch and padding stays small; answers are put
    # back in input order at the end. Greedy decoding is per-row, so grouping
    # does not change any answer (pinned by test).
    order = sorted(range(len(prompts)), key=lambda index: len(prompts[index]))
    by_position: dict[int, list[int]] = {}
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        chunk = [prompts[index] for index in indices]
        width = max(len(prompt) for prompt in chunk)
        input_ids = torch.full((len(chunk), width), pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, prompt in enumerate(chunk):
            input_ids[row, width - len(prompt) :] = torch.tensor(list(prompt))
            attention[row, width - len(prompt) :] = 1
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else _nullcontext()
        )
        with context:
            output = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
                generation_config=generation_config,
            )
        for row, index in enumerate(indices):
            by_position[index] = cut_at_eos(output[row, width:].tolist(), eos_token_id=eos_token_id)
    return [by_position[index] for index in range(len(prompts))]


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


def score_predictions(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> dict[str, Any]:
    """Mean ROUGE-L and EM, each the max over an example's references."""

    if len(predictions) != len(references):
        raise ValueError("predições e referências com tamanhos diferentes")
    per_rouge = [
        max(rouge_l(prediction, reference) for reference in refs)
        for prediction, refs in zip(predictions, references)
    ]
    per_em = [
        max(exact_match(prediction, reference) for reference in refs)
        for prediction, refs in zip(predictions, references)
    ]
    count = len(per_rouge)
    return {
        "rouge_l": sum(per_rouge) / count if count else float("nan"),
        "exact_match": sum(per_em) / count if count else float("nan"),
        "per_example_rouge_l": per_rouge,
    }


def _autocast_dtype(config: RunConfig) -> torch.dtype | None:
    return torch.float16 if config.precision == "fp16" else None


def evaluate_task(model, encoded: EncodedTask, config: RunConfig, *, tokenizer, stage=None) -> dict[str, Any]:
    """Greedy-generate the eval split of ONE task and score it."""

    was_training = model.training
    model.eval()
    started = time.perf_counter()
    generated = generate_answers(
        model,
        [example.prompt_ids for example in encoded.evaluation],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.eval_batch_size,
        device=config.device,
        autocast_dtype=_autocast_dtype(config),
    )
    _synchronize(config.device)
    seconds = time.perf_counter() - started
    predictions = [tokenizer.decode(ids).strip() for ids in generated]
    scores = score_predictions(predictions, [example.references for example in encoded.evaluation])
    if was_training:
        model.train()
    return {
        "task": encoded.name,
        "stage": stage,
        "rouge_l": scores["rouge_l"],
        "exact_match": scores["exact_match"],
        "examples": len(predictions),
        "generated_tokens": sum(len(ids) for ids in generated),
        "seconds": seconds,
        "samples": [
            {"prediction": prediction, "reference": example.references[0]}
            for prediction, example in list(zip(predictions, encoded.evaluation))[:3]
        ],
    }


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# endpoints and gates
# ---------------------------------------------------------------------------


def sapt_endpoints(matrix: Sequence[Sequence[float | None]]) -> dict[str, Any]:
    """AP, forgetting and BWT exactly as SAPT ``score.py``.

    ``matrix[s][t]`` is the score on task ``t`` after training task ``s``;
    entries with ``t > s`` are ``None`` (SAPT pads them with 0).

    * AP = mean of the last row.
    * forgetting = mean over ``t < T-1`` of (best score on ``t`` over rows
      ``0..T-2``, i.e. EXCLUDING the final row) minus the final score.
    * BWT = sum over ALL ``t`` of (final - just-trained) divided by ``T``.
    """

    total = len(matrix)
    if total == 0 or any(len(row) != total for row in matrix):
        raise ValueError("matriz deve ser T x T")
    for stage in range(total):
        for task in range(stage + 1):
            if matrix[stage][task] is None:
                raise ValueError(f"entrada vista vazia: [{stage}][{task}]")
    final = [float(value) for value in matrix[-1]]
    per_task_forgetting = [
        max(float(matrix[stage][task]) for stage in range(task, total - 1)) - final[task]
        for task in range(total - 1)
    ]
    return {
        "ap": sum(final) / total,
        "forgetting": (
            sum(per_task_forgetting) / len(per_task_forgetting) if per_task_forgetting else None
        ),
        "per_task_forgetting": per_task_forgetting,
        "bwt": sum(final[task] - float(matrix[task][task]) for task in range(total)) / total,
        "scale": "fraction",
    }


def acquisition_gate(
    matrix: Sequence[Sequence[float | None]], zero_shot: Sequence[float], *, margin: float
) -> dict[str, Any]:
    """G-aq: score right after training task t minus zero-shot of task t."""

    rows = []
    for task, base in enumerate(zero_shot):
        after = float(matrix[task][task])
        delta = after - float(base)
        rows.append(
            {"task": task, "after_training": after, "zero_shot": float(base),
             "delta": delta, "passes": delta > margin}
        )
    return {"margin": margin, "per_task": rows, "all_pass": all(row["passes"] for row in rows)}


def dispersion_summary(
    results: Sequence[dict[str, Any]],
    *,
    min_vanilla_forgetting: float,
    matched_tolerance: float,
    reference: str = "vanilla",
    control: str = "lr_control",
) -> dict[str, Any]:
    """G-disp: does the primary vary between arms, and does vanilla forget?"""

    by_arm = {row["method"]: row for row in results}
    ap = {arm: row["endpoints"]["ap"] for arm, row in by_arm.items()}
    forgetting = {arm: row["endpoints"]["forgetting"] for arm, row in by_arm.items()}
    plasticity = {arm: row.get("surface_plasticity") for arm, row in by_arm.items()}
    values = list(ap.values())
    mean = sum(values) / len(values)
    known_forgetting = [value for value in forgetting.values() if value is not None]
    vanilla_forgetting = forgetting.get(reference)
    treated = [arm for arm in by_arm if arm not in (reference, control)]
    d_unmatched = (
        {arm: ap[arm] - ap[reference] for arm in treated} if reference in ap else {}
    )
    d_matched: dict[str, float | None] = {}
    for arm in treated:
        e_arm, e_control = plasticity.get(arm), plasticity.get(control)
        if control in ap and e_arm is not None and e_control is not None and (
            abs(e_arm - e_control) <= matched_tolerance
        ):
            d_matched[arm] = ap[arm] - ap[control]
        else:
            d_matched[arm] = None
    ap_range = max(values) - min(values)
    return {
        "ap_by_arm": ap,
        "forgetting_by_arm": forgetting,
        "surface_plasticity_by_arm": plasticity,
        "ap_range": ap_range,
        "ap_std": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
        "forgetting_range": (
            max(known_forgetting) - min(known_forgetting) if known_forgetting else None
        ),
        "vanilla_forgetting": vanilla_forgetting,
        "min_vanilla_forgetting": min_vanilla_forgetting,
        "vanilla_forgetting_ok": (
            None if vanilla_forgetting is None else vanilla_forgetting >= min_vanilla_forgetting
        ),
        "d_unmatched": d_unmatched,
        "d_matched": d_matched,
        "matched_tolerance": matched_tolerance,
        "primary_varies": len(values) > 1 and ap_range > 0.0,
    }


_LAYER = re.compile(r"layers\.(\d+)\.")


def importance_liveness(instrumentation: QwenLoRASlowHeat) -> dict[str, Any]:
    """G-imp: fraction of units with recorded importance, per adapted module.

    Must be read BEFORE ``consolidate()``, which zeroes ``task_ema``. A module
    whose ``task_ema`` is identically zero recorded nothing (the fp16
    underflow mode of ``assert_importance_is_live``); this records rather
    than raises, the gate is read from the manifest.
    """

    trackers = [(m.name, m.tracker) for m in instrumentation.modules if m.tracker is not None]
    if not trackers:
        return {"applicable": False}
    densities = {
        name: float((tracker.task_ema.detach() > 0.0).float().mean().item())
        for name, tracker in trackers
    }
    dead = [name for name, value in densities.items() if value == 0.0]
    layers: dict[int, list[float]] = {}
    for name, value in densities.items():
        match = _LAYER.search(name)
        layers.setdefault(int(match.group(1)) if match else -1, []).append(value)
    layers_dead = sorted(index for index, values in layers.items() if max(values) == 0.0)
    return {
        "applicable": True,
        "live": not dead,
        "modules_total": len(densities),
        "modules_dead": len(dead),
        "dead_modules": dead,
        "layers_total": len(layers),
        "layers_dead": layers_dead,
        "density_min": min(densities.values()),
        "density_mean": sum(densities.values()) / len(densities),
    }


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


def resolve_effective_lr(method: str, config: RunConfig) -> float:
    """Only ``lr_control`` is scaled, and only by an explicit scale."""

    if method != "lr_control":
        return config.learning_rate
    if config.learning_rate_scale is None:
        raise ValueError("lr_control exige learning_rate_scale explícito")
    return config.learning_rate * config.learning_rate_scale


def plasticity_record(
    *, masked_mean: float, masked_parameters: int, effective_lr: float, learning_rate: float
) -> dict[str, Any]:
    """Same three fields as ``experiments.qwen_lora_slowheat.plasticity_record``."""

    if learning_rate <= 0.0:
        raise ValueError("learning_rate deve ser > 0")
    return {
        "masked_mean": masked_mean,
        "masked_parameters": masked_parameters,
        "learning_rate_scale": effective_lr / learning_rate,
    }


def configure_precision(model, precision: str):
    """fp32: everything fp32. fp16: base fp16, every LoRA adapter kept fp32."""

    if precision == "fp32":
        return model.float()
    if precision == "fp16":
        model.half()
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
        return model
    raise ValueError(f"precision deve ser um de {PRECISIONS}")


def scaled_optimizer_step(scaler, optimizer, parameters: Sequence[Tensor], *, max_grad_norm: float) -> bool:
    """fp16 step: unscale, check finiteness, clip, masked step, update scale.

    ``unscale_`` comes first so clipping and the plasticity masks (applied
    inside ``SlowHeatAdamW.step``) see the true gradients. ``scaler.step`` is
    not used: it would hand the optimizer a ``grad_scale`` the masked step
    rejects. Returns whether the step was taken.
    """

    scaler.unscale_(optimizer)
    finite = all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
        if parameter.grad is not None
    )
    if finite:
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scaler.update()
    return finite


@dataclass
class Arm:
    method: str
    model: Any
    instrumentation: QwenLoRASlowHeat
    optimizer: SlowHeatAdamW
    trainable: list[Tensor]
    effective_lr: float
    scaler: Any = None
    skipped_steps: int = 0
    train_tokens: int = 0
    steps: int = 0
    step_seconds: list[float] = field(default_factory=list)


def build_arm(method: str, config: RunConfig, *, model_factory: Callable[[], Any], task_count: int) -> Arm:
    torch.manual_seed(config.seed)
    base = model_factory()
    lora_config = LoRASlowHeatConfig(
        method=method,  # type: ignore[arg-type]
        rank=config.rank,
        alpha=config.alpha,
        slow_strength=config.slow_strength,
        plasticity_budget=config.plasticity_budget,
        hard=config.hard,
        target_modules=tuple(config.target_modules),
        task_count=max(task_count, 1),
    )
    model, instrumentation = build_lora_slowheat(base, lora_config, task_type="CAUSAL_LM")
    configure_precision(model, config.precision)
    model.to(config.device)
    instrumentation.to(config.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    effective_lr = resolve_effective_lr(method, config)
    optimizer = SlowHeatAdamW(trainable, lr=effective_lr, weight_decay=config.weight_decay)
    instrumentation.register_plasticity_masks(optimizer)
    scaler = (
        torch.amp.GradScaler("cuda") if config.precision == "fp16" else None
    )
    return Arm(method, model, instrumentation, optimizer, trainable, effective_lr, scaler)


def _batches(count: int, config: RunConfig, stage: int) -> list[Tensor]:
    generator = torch.Generator().manual_seed(config.seed * 1000 + stage)
    if config.steps_per_task is None:
        batches = []
        for _ in range(config.epochs_per_task):
            order = torch.randperm(count, generator=generator)
            batches.extend(
                order[start : start + config.batch_size]
                for start in range(0, count, config.batch_size)
            )
        return batches
    stream: list[int] = []
    needed = config.steps_per_task * config.batch_size
    while len(stream) < needed:
        stream.extend(torch.randperm(count, generator=generator).tolist())
    return [
        torch.tensor(stream[step * config.batch_size : (step + 1) * config.batch_size])
        for step in range(config.steps_per_task)
    ]


def train_task(arm: Arm, train: Sequence[EncodedExample], config: RunConfig, *, stage: int, pad_token_id: int) -> int:
    """Train one task under the arm's budget; returns optimizer steps taken."""

    if not train:
        raise ValueError("tarefa sem exemplos de treino")
    arm.model.train()
    arm.instrumentation.train()
    steps = 0
    for indices in _batches(len(train), config, stage):
        started = time.perf_counter()
        batch = collate_training([train[int(i)] for i in indices], pad_token_id=pad_token_id)
        batch = {key: value.to(config.device) for key, value in batch.items()}
        arm.train_tokens += int(batch["attention_mask"].sum())
        with arm.instrumentation.validity(batch["attention_mask"]):
            if arm.scaler is None:
                loss = answer_only_loss(arm.model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(arm.trainable, config.max_grad_norm)
                arm.optimizer.step()
                arm.optimizer.zero_grad(set_to_none=True)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = answer_only_loss(arm.model, batch)
                arm.scaler.scale(loss).backward()
                if not scaled_optimizer_step(
                    arm.scaler, arm.optimizer, arm.trainable, max_grad_norm=config.max_grad_norm
                ):
                    arm.skipped_steps += 1
        _synchronize(config.device)
        arm.step_seconds.append(time.perf_counter() - started)
        steps += 1
    arm.steps += steps
    return steps


def _freeze_snapshot(arm: Arm) -> dict[str, Any]:
    """What must NOT move during the next task: zero-mask entries and frozen A."""

    masked = []
    for binding in arm.instrumentation.mask_bindings():
        mask = binding.mask() if callable(binding.mask) else binding.mask
        parameter = binding.parameter
        zero = torch.broadcast_to(mask.to(parameter.device), parameter.shape) == 0
        masked.append((parameter, zero, parameter.detach().clone()))
    frozen = [
        (parameter, parameter.detach().clone())
        for name, parameter in arm.model.named_parameters()
        if "lora_" in name and not parameter.requires_grad
    ]
    return {"masked": masked, "frozen": frozen}


def _freeze_audit(snapshot: dict[str, Any]) -> dict[str, Any]:
    zero_entries = 0
    zero_change = 0.0
    plastic_entries = 0
    plastic_changed = 0
    for parameter, zero, before in snapshot["masked"]:
        delta = (parameter.detach() - before).abs()
        zero_entries += int(zero.sum())
        if bool(zero.any()):
            zero_change = max(zero_change, float(delta[zero].max()))
        plastic_entries += int((~zero).sum())
        plastic_changed += int((delta[~zero] > 0).sum())
    frozen_change = 0.0
    frozen_count = 0
    for parameter, before in snapshot["frozen"]:
        frozen_count += parameter.numel()
        frozen_change = max(frozen_change, float((parameter.detach() - before).abs().max()))
    return {
        "zero_mask_entries": zero_entries,
        "zero_mask_max_abs_change": zero_change,
        "plastic_entries_changed_fraction": (
            plastic_changed / plastic_entries if plastic_entries else None
        ),
        "frozen_parameters": frozen_count,
        "frozen_max_abs_change": frozen_change,
    }


def _release(arm: Arm, device: str) -> None:
    import gc

    arm.instrumentation.remove_hooks()
    arm.model = None
    arm.optimizer = None
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def run_arm(
    method: str,
    tasks: Sequence[EncodedTask],
    config: RunConfig,
    *,
    tokenizer,
    model_factory: Callable[[], Any],
) -> dict[str, Any]:
    device = config.device
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    arm = build_arm(method, config, model_factory=model_factory, task_count=len(tasks))
    total = len(tasks)
    matrix: list[list[float | None]] = [[None] * total for _ in range(total)]
    em_matrix: list[list[float | None]] = [[None] * total for _ in range(total)]
    stages: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    started = time.perf_counter()
    generation_seconds = 0.0
    generated_examples = 0
    generated_tokens = 0
    try:
        for stage, task in enumerate(tasks):
            arm.instrumentation.begin_task(stage)
            in_force_mean, in_force_count = arm.instrumentation.mask_coverage()
            stage_started = time.perf_counter()
            first_step = len(arm.step_seconds)
            snapshot = _freeze_snapshot(arm)
            steps = train_task(arm, task.train, config, stage=stage, pad_token_id=tokenizer.pad_token_id)
            train_seconds = time.perf_counter() - stage_started
            freeze_audit = _freeze_audit(snapshot)
            del snapshot
            liveness = importance_liveness(arm.instrumentation)
            arm.instrumentation.consolidate()
            eval_started = time.perf_counter()
            for evaluated in range(stage + 1):
                result = evaluate_task(arm.model, tasks[evaluated], config, tokenizer=tokenizer, stage=stage)
                matrix[stage][evaluated] = result["rouge_l"]
                em_matrix[stage][evaluated] = result["exact_match"]
                generation_seconds += result["seconds"]
                generated_examples += result["examples"]
                generated_tokens += result["generated_tokens"]
                evaluations.append(result)
            step_times = arm.step_seconds[first_step:]
            stages.append(
                {
                    "stage": stage,
                    "task": task.name,
                    "train_steps": steps,
                    "train_seconds": train_seconds,
                    "seconds_per_step_median": sorted(step_times)[len(step_times) // 2],
                    "eval_seconds": time.perf_counter() - eval_started,
                    "mask_mean_in_force": in_force_mean,
                    "masked_parameters_in_force": in_force_count,
                    "mask_mean_after_consolidation": arm.instrumentation.effective_plasticity(),
                    "importance_liveness": liveness,
                    "freeze_audit": freeze_audit,
                    "peak_cuda_allocated_mib": (
                        torch.cuda.max_memory_allocated(device) / 2**20
                        if device.startswith("cuda") else None
                    ),
                }
            )
        masked_mean, masked_parameters = arm.instrumentation.mask_coverage()
        trainable_count = sum(p.numel() for p in arm.trainable)
        lora_dtypes = sorted(
            {str(p.dtype) for n, p in arm.model.named_parameters() if "lora_" in n}
        )
    finally:
        _release(arm, device)
    endpoints = sapt_endpoints(matrix)
    liveness_rows = [stage["importance_liveness"] for stage in stages]
    return {
        "method": method,
        "rouge_l_matrix": matrix,
        "exact_match_matrix": em_matrix,
        "endpoints": endpoints,
        "exact_match_endpoints": sapt_endpoints(em_matrix),
        "trainable_parameters": trainable_count,
        "effective_learning_rate": arm.effective_lr,
        **plasticity_record(
            masked_mean=masked_mean,
            masked_parameters=masked_parameters,
            effective_lr=arm.effective_lr,
            learning_rate=config.learning_rate,
        ),
        "mean_mask_in_force": sum(s["mask_mean_in_force"] for s in stages) / len(stages),
        "optimizer_steps": arm.steps,
        "skipped_steps": arm.skipped_steps,
        "train_tokens": arm.train_tokens,
        "lora_dtypes": lora_dtypes,
        "wall_clock_seconds": time.perf_counter() - started,
        "train_seconds": sum(arm.step_seconds),
        "seconds_per_train_step": sum(arm.step_seconds) / max(len(arm.step_seconds), 1),
        "generation_seconds": generation_seconds,
        "generated_examples": generated_examples,
        "generated_tokens": generated_tokens,
        "seconds_per_generated_example": (
            generation_seconds / generated_examples if generated_examples else None
        ),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.startswith("cuda") else None
        ),
        "importance_live_all_stages": (
            None
            if not liveness_rows or not liveness_rows[0].get("applicable")
            else all(row["live"] for row in liveness_rows)
        ),
        "stages": stages,
        "evaluations": evaluations,
    }


def zero_shot_scores(
    tasks: Sequence[EncodedTask], config: RunConfig, *, tokenizer, model_factory: Callable[[], Any]
) -> dict[str, Any]:
    """Base model, no adapter, no training: the G-aq floor per task."""

    torch.manual_seed(config.seed)
    model = configure_precision(model_factory(), config.precision).to(config.device)
    started = time.perf_counter()
    results = [evaluate_task(model, task, config, tokenizer=tokenizer, stage=-1) for task in tasks]
    seconds = time.perf_counter() - started
    del model
    if config.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "rouge_l": [row["rouge_l"] for row in results],
        "exact_match": [row["exact_match"] for row in results],
        "seconds": seconds,
        "examples": sum(row["examples"] for row in results),
        "per_task": results,
    }


# ---------------------------------------------------------------------------
# protocol and manifest
# ---------------------------------------------------------------------------


def build_protocol(config: RunConfig, *, task_names: Sequence[str], arms: Sequence[str]) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("device")
    payload["target_modules"] = list(config.target_modules)
    payload.update(
        {
            "task_names": list(task_names),
            "arms": list(arms),
            "metric": "rouge_l max over references (dual_heater.superni_metrics), fraction",
            "training_target": "first reference + eos, answer-only loss",
            "prompt_truncation": "keep start, cut body end, keep output marker",
            "endpoints": "SAPT score.py (AP, Fgt, BWT)",
            "generation": "greedy, stop at eos",
            "importance_liveness_read": "before consolidation",
        }
    )
    return payload


def protocol_hash(protocol: dict[str, Any]) -> str:
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_experiment(
    config: RunConfig,
    tasks: Sequence[SuperNITask],
    *,
    arms: Sequence[str],
    tokenizer,
    model_factory: Callable[[], Any],
) -> dict[str, Any]:
    config.validate(arms=arms)
    encoded = [encode_task(tokenizer, task, config) for task in tasks]
    protocol = build_protocol(config, task_names=[task.name for task in tasks], arms=arms)
    zero_shot = zero_shot_scores(encoded, config, tokenizer=tokenizer, model_factory=model_factory)
    results = []
    for method in arms:
        print(f"arm {method} ...", flush=True)
        record = run_arm(method, encoded, config, tokenizer=tokenizer, model_factory=model_factory)
        record["gates"] = {
            "acquisition": acquisition_gate(
                record["rouge_l_matrix"], zero_shot["rouge_l"], margin=config.acquisition_margin
            )
        }
        print(
            f"    AP {record['endpoints']['ap']:.4f} | forgetting "
            f"{record['endpoints']['forgetting']} | {record['wall_clock_seconds']:.1f}s",
            flush=True,
        )
        results.append(record)
    if any(row["method"] == "vanilla" for row in results):
        attach_surface_plasticity(results, reference_arm="vanilla")
    step_times = [row["seconds_per_train_step"] for row in results]
    generation = [
        row["seconds_per_generated_example"] for row in results
        if row["seconds_per_generated_example"] is not None
    ]
    return {
        "status": "exploratory_runner_smoke",
        "claim_scope": "wiring, cost and gate measurement; no inferential comparison",
        "protocol": protocol,
        "protocol_hash": protocol_hash(protocol),
        "protocol_document": "goals/estudo_fase3b.md",
        "seed": config.seed,
        "device": config.device,
        "task_order": [task.name for task in tasks],
        "task_fingerprint": hashlib.sha256(
            "".join(task.ids_digest for task in encoded).encode()
        ).hexdigest(),
        "train_examples_per_task": [len(task.train) for task in encoded],
        "eval_examples_per_task": [len(task.evaluation) for task in encoded],
        "truncated_prompts_per_task": [
            sum(e.prompt_truncated for e in task.train + task.evaluation) for task in encoded
        ],
        "zero_shot": zero_shot,
        "results": results,
        "gates": {
            "G-aq": {row["method"]: row["gates"]["acquisition"] for row in results},
            "G-imp": {row["method"]: row["importance_live_all_stages"] for row in results},
            "G-disp": dispersion_summary(
                results,
                min_vanilla_forgetting=config.min_vanilla_forgetting,
                matched_tolerance=config.matched_tolerance,
            ),
        },
        "timings": {
            "seconds_per_train_step": sum(step_times) / len(step_times),
            "seconds_per_generated_example": (
                sum(generation) / len(generation) if generation else None
            ),
            "zero_shot_seconds": zero_shot["seconds"],
            "per_arm": {
                row["method"]: {
                    "seconds_per_train_step": row["seconds_per_train_step"],
                    "seconds_per_generated_example": row["seconds_per_generated_example"],
                    "wall_clock_seconds": row["wall_clock_seconds"],
                    "peak_cuda_allocated_mib": row["peak_cuda_allocated_mib"],
                }
                for row in results
            },
        },
        "git_commit": _git_commit(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(config.device)
                if config.device.startswith("cuda") else None
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--order", type=int, default=1, choices=(1, 2))
    parser.add_argument("--tasks", type=int, default=None, help="first N tasks of the order")
    parser.add_argument("--task-names", nargs="+", default=None,
                        help="explicit task sequence, overrides --order")
    parser.add_argument("--eval-split", default="test", choices=("dev", "test"))
    parser.add_argument("--train-per-task", type=int, default=None)
    parser.add_argument("--eval-per-task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs-per-task", type=int, default=1)
    parser.add_argument("--steps-per-task", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--slow-strength", type=float, default=3.0)
    parser.add_argument("--plasticity-budget", type=float, default=0.5)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--target-modules", nargs="+", default=["gate_proj", "up_proj", "down_proj"])
    parser.add_argument(
        "--learning-rate-scale", type=float, default=None,
        help="uniform lr scale of lr_control; required when that arm runs. Set it "
        "to the measured surface E of the arm it falsifies.",
    )
    parser.add_argument("--precision", default="fp32", choices=PRECISIONS,
                        help="fp16 is CUDA-only; bf16 is not offered (Pascal)")
    parser.add_argument("--acquisition-margin", type=float, default=0.0)
    parser.add_argument("--min-vanilla-forgetting", type=float, default=0.0)
    parser.add_argument("--matched-tolerance", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    return parser


def build_run_config(arguments: argparse.Namespace) -> RunConfig:
    return RunConfig(
        model_name=arguments.model,
        data_root=arguments.data_root,
        order=arguments.order,
        tasks=arguments.tasks,
        eval_split=arguments.eval_split,
        train_per_task=arguments.train_per_task,
        eval_per_task=arguments.eval_per_task,
        seed=arguments.seed,
        max_source_length=arguments.max_source_length,
        max_target_length=arguments.max_target_length,
        max_new_tokens=arguments.max_new_tokens,
        batch_size=arguments.batch_size,
        eval_batch_size=arguments.eval_batch_size,
        epochs_per_task=arguments.epochs_per_task,
        steps_per_task=arguments.steps_per_task,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        max_grad_norm=arguments.max_grad_norm,
        rank=arguments.rank,
        alpha=arguments.alpha,
        slow_strength=arguments.slow_strength,
        plasticity_budget=arguments.plasticity_budget,
        hard=arguments.hard,
        target_modules=tuple(arguments.target_modules),
        learning_rate_scale=arguments.learning_rate_scale,
        precision=arguments.precision,
        acquisition_margin=arguments.acquisition_margin,
        min_vanilla_forgetting=arguments.min_vanilla_forgetting,
        matched_tolerance=arguments.matched_tolerance,
        device=arguments.device,
    )


def main() -> None:
    arguments = build_parser().parse_args()
    config = build_run_config(arguments)
    config.validate(arms=arguments.arms)
    tasks = load_sapt_superni(
        config.data_root,
        order=config.order,
        task_names=arguments.task_names,
        tasks=config.tasks,
        eval_split=config.eval_split,
        train_per_task=config.train_per_task,
        eval_per_task=config.eval_per_task,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(config.model_name))

    def model_factory():
        return AutoModelForCausalLM.from_pretrained(config.model_name, dtype=torch.float32)

    manifest = run_experiment(
        config, tasks, arms=arguments.arms, tokenizer=tokenizer, model_factory=model_factory
    )
    path = arguments.output / "manifest.json"
    write_manifest(path, manifest)
    print(f"manifest written: {path}")


if __name__ == "__main__":
    main()
