"""TRACE continual instruction tuning: one model, eight tasks, LoRA arms.

Protocol source: Wang et al. 2023, "TRACE: A Comprehensive Benchmark for
Continual Learning in Large Language Models" (arXiv 2310.06762), and the
official repository ``BeyonderXX/TRACE@master`` (``training/main.py``,
``model/base_model.py``, ``inference/infer_single.py``,
``scripts/train_lora.sh``). Defaults below are the official LoRA baseline.

Status: IMPLEMENTATION ONLY. No result produced by this runner is
confirmatory: there is no preregistration, no frozen seed band and no declared
endpoint gate yet (roadmap 3b.3). Use it for cost/dispersion calibration
(3b.2) until a protocol exists.

What is official (and where it comes from)
------------------------------------------
* Task order C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE-cm,
  NumGLUE-ds, 20Minuten (``train_lora.sh``).
* LoRA ``r=8, alpha=32, dropout=0.1``, PEFT default target modules
  (``q_proj``, ``v_proj`` for Qwen2/LLaMA) (``training/main.py``).
* Adam ``betas=(0.9, 0.95)``, ``weight_decay=0``, constant LR ``1e-4`` with no
  warmup (``get_constant_schedule_with_warmup``; the ``cosine`` flag in the
  script is never read), gradient clipping 1.0 (``utils/ds_utils.py``).
* Epochs per task ``5,3,7,5,3,5,5,7`` (``train_lora.sh``; the paper's appendix
  agrees for the LoRA baselines).
* ``max_prompt_len=1024``, ``max_ans_len=512``, left padding, left truncation,
  ``pad_to_multiple_of=8``, the collator's label mask
  (:mod:`dual_heater.trace_data`).
* Evaluation after task ``t`` on the TEST split of tasks ``0..t``, sampling
  with ``temperature=0.1`` and ``max_new_tokens=max_ans_len``; decode with
  ``skip_special_tokens=True`` and no stripping (``infer_single.py``).
* Per-task scoring and OP/BWT (:mod:`dual_heater.trace_metrics`).

Declared deviations
-------------------
R1. Model: Qwen2.5-0.5B(-Instruct) instead of LLaMA-2-7B-chat. Numbers are
    NOT comparable to the paper's tables.
R2. Precision fp32 instead of DeepSpeed fp16/ZeRO-2. Under fp16 the SlowHeat
    importance ``|z * dL/dz|`` underflows to zero in early layers on this
    hardware (measured on Qwen2.5-0.5B, see the continual-learning skill,
    pitfall 20), and :class:`SlowHeatAdamW` rejects ``GradScaler``.
R3. Gradient accumulation is flushed at the end of each task. DeepSpeed counts
    micro-steps globally, so upstream a partial accumulation window can spill
    into the next task's first update. Here no update mixes two tasks.
R4. Global batch is ``micro_batch * grad_accum`` on ONE GPU. The paper reports
    128; ``train_lora.sh`` gives 1 * 8 * 4 GPUs = 32. Default here: 1 * 32.
R5. ``pad_token_id`` for generation is EOS. Upstream passes ``unk_token_id``,
    which is ``None`` for Qwen; HF then falls back to EOS anyway.
R6. Sampling is seeded per (stage, task) so a run is reproducible. Upstream
    seeds once at process start. ``--decoding greedy`` is offered for
    calibration; it is NOT the official decoding.
R7. ``--train-per-task`` / ``--test-per-task`` subsample (seeded, order
    preserving). The official release has 5000 train / 2000 test per task;
    anything smaller is not the official benchmark and is recorded in the
    manifest.
R8. No chat template, exactly as upstream: TRACE prompts carry their own
    instruction text. This contradicts the house rule for instruct models and
    is deliberate, for faithfulness; ``--chat-template`` is NOT implemented.
R9. Decoding does NOT inherit the checkpoint's ``generation_config``. Upstream
    passes only ``temperature``/``do_sample``, so HF merges in the model's own
    defaults (LLaMA-2-chat adds ``top_p=0.9``; Qwen2.5-Instruct would add
    ``top_p=0.8, top_k=20, repetition_penalty=1.1``). Here the config is built
    from scratch: pure temperature-0.1 sampling, ``top_p=1``, ``top_k=0``, no
    repetition penalty, so the decoding rule cannot change with the checkpoint.
R10. Label tokenization ``split`` (default) instead of the official joint
    ``tokenize(prompt + answer)``. With Qwen's BPE the official mask merges
    the prompt's final ``":"`` with the answer's first letter in 100% of FOMC
    and ScienceQA train examples, so the model is trained on a continuation
    it never meets at inference. See ``dual_heater.trace_data``.
    ``--label-tokenization official`` restores upstream behaviour.

Data
----
``data/TRACE`` -> ``data/TRACE_raw/TRACE-Benchmark/LLM-CL-Benchmark_5000``, the
official release (Google Drive id ``1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV``,
archive sha256 ``11152d50...68a8``). Audit of what is in it:
``docs/audits/trace_data_audit.md``.

Arms
----
``vanilla`` (official LoRA SeqFT), ``lr_control``, ``frozen_a_control``,
``exact``, ``rank``, ``leak``, ``slice``: the same arms as
``experiments.qwen_lora_slowheat``, built by
:func:`dual_heater.lora_slowheat.build_lora_slowheat` with
``task_type="CAUSAL_LM"``. Plasticity bookkeeping (``masked_mean``,
``masked_parameters``, ``learning_rate_scale``, ``trainable_parameters``) is
written for every arm so ``surface_plasticity`` can be recomputed offline.

``--learning-rate-scale frozen_a_surface`` sets the ``lr_control`` scale to the
fraction of the vanilla adapter that survives freezing ``A``, measured from the
built model's parameter counts (no accuracy is read). It is the pairing used
for ``frozen_a_control - lr_control``.

Run (CPU dry run, no model, no GPU; reports token lengths for cost planning)::

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \\
        .venv/bin/python -m experiments.trace_cl --data-root DATA/TRACE \\
        --output results/trace_dry --dry-run

GPU runs need explicit authorization in this repository.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dual_heater.lora_slowheat import (
    FROZEN_A_METHODS,
    TRACKED_METHODS,
    LoRASlowHeatConfig,
    attach_surface_plasticity,
    build_lora_slowheat,
)
from dual_heater.optim import SlowHeatAdamW
from dual_heater.trace_data import (
    LABEL_TOKENIZATIONS,
    SpecialTokens,
    TraceTask,
    collate_inference,
    collate_train,
    load_trace_tasks,
    special_token_ids,
    tokenize_official,
    trace_data_identity,
)
from dual_heater.trace_metrics import (
    TRACE_TASKS,
    evaluate_task,
    primary_score,
    trace_cl_metrics,
)
from experiments.artifacts import source_fingerprint, write_json_atomic

ARMS: tuple[str, ...] = (
    "vanilla",
    "lr_control",
    "frozen_a_control",
    "exact",
    "rank",
    "leak",
    "slice",
)

#: Official LoRA epochs per task, in TRACE_TASKS order (scripts/train_lora.sh).
OFFICIAL_LORA_EPOCHS: dict[str, int] = dict(zip(TRACE_TASKS, (5, 3, 7, 5, 3, 5, 5, 7)))

FROZEN_A_SURFACE = "frozen_a_surface"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TraceRunConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tasks: tuple[str, ...] = TRACE_TASKS
    epochs: tuple[int, ...] = tuple(OFFICIAL_LORA_EPOCHS[name] for name in TRACE_TASKS)
    seed: int = 1234
    data_seed: int = 0
    train_per_task: int | None = None
    test_per_task: int | None = None
    max_prompt_len: int = 1024
    max_ans_len: int = 512
    max_new_tokens: int = 512
    pad_to_multiple_of: int = 8
    label_tokenization: str = "split"
    micro_batch: int = 1
    grad_accum: int = 32
    eval_batch: int = 8
    learning_rate: float = 1e-4
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    rank: int = 8
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    decoding: str = "official"
    temperature: float = 0.1
    # SlowHeat knobs (ignored by unmasked arms)
    slow_strength: float = 3.0
    plasticity_budget: float = 0.5
    hard: bool = False
    leak_combination: str = "min"
    target_plasticity: float | None = None
    learning_rate_scale: float | str | None = None
    gradient_checkpointing: bool = False
    device: str = "cuda:0"
    dtype: str = "float32"
    #: Copied into the manifest so a reader sees the deviation list in force.
    deviations: tuple[str, ...] = field(
        default=("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"), compare=False
    )

    def validate(self) -> None:
        if len(self.tasks) < 2:
            raise ValueError("TRACE requer ao menos 2 tarefas na sequência")
        if len(self.epochs) != len(self.tasks):
            raise ValueError("epochs deve ter um valor por tarefa")
        if any(int(value) < 1 for value in self.epochs):
            raise ValueError("epochs devem ser >= 1")
        for name in (
            "max_prompt_len", "max_ans_len", "max_new_tokens", "pad_to_multiple_of",
            "micro_batch", "grad_accum", "eval_batch", "rank",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} deve ser >= 1")
        if self.label_tokenization not in LABEL_TOKENIZATIONS:
            raise ValueError(f"label_tokenization deve ser um de {LABEL_TOKENIZATIONS}")
        if self.decoding not in {"official", "greedy"}:
            raise ValueError("decoding deve ser 'official' ou 'greedy'")
        if self.decoding == "official" and not self.temperature > 0.0:
            raise ValueError("temperature deve ser > 0 na decodificação oficial")
        if self.dtype != "float32":
            raise ValueError(
                "apenas float32 (desvio R2): fp16 zera a importância SlowHeat em Pascal"
            )
        if isinstance(self.learning_rate_scale, str):
            if self.learning_rate_scale != FROZEN_A_SURFACE:
                raise ValueError(
                    f"learning_rate_scale deve ser número ou '{FROZEN_A_SURFACE}'"
                )
        elif self.learning_rate_scale is not None and not (
            0.0 < float(self.learning_rate_scale) <= 1.0
        ):
            raise ValueError("learning_rate_scale deve estar em (0, 1]")
        if self.target_plasticity is not None and not 0.0 < self.target_plasticity <= 1.0:
            raise ValueError("target_plasticity deve estar em (0, 1]")
        if not self.target_modules:
            raise ValueError("target_modules não pode ser vazio")

    def payload(self) -> dict[str, Any]:
        """JSON-stable configuration (tuples become lists exactly once)."""

        return json.loads(json.dumps(asdict(self)))


# ---------------------------------------------------------------------------
# plasticity pairing
# ---------------------------------------------------------------------------


def frozen_a_surface_scale(model: torch.nn.Module) -> float:
    """Fraction of the vanilla LoRA surface that is NOT an ``A`` matrix.

    Pure structure: counts ``lora_A`` vs all LoRA factors. Pairing
    ``lr_control`` at this scale is the TRACE analogue of
    ``FROZEN_A_LR_SCALE`` (which is specific to the CLINC150 host at r=16 and
    must not be reused here).
    """

    a_count = 0
    total = 0
    for name, parameter in model.named_parameters():
        if ".lora_A." in name:
            a_count += parameter.numel()
            total += parameter.numel()
        elif ".lora_B." in name:
            total += parameter.numel()
    if total == 0:
        raise RuntimeError("modelo sem fatores LoRA: não há superfície para parear")
    return (total - a_count) / total


def resolve_effective_lr(method: str, config: TraceRunConfig, model: torch.nn.Module) -> float:
    if method != "lr_control":
        return config.learning_rate
    scale = config.learning_rate_scale
    if scale == FROZEN_A_SURFACE:
        return config.learning_rate * frozen_a_surface_scale(model)
    if scale is not None:
        return config.learning_rate * float(scale)
    if config.target_plasticity is not None:
        return config.learning_rate * config.target_plasticity
    return config.learning_rate


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


ModelFactory = Callable[[TraceRunConfig], torch.nn.Module]


def default_model_factory(config: TraceRunConfig) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(config.seed)
    return AutoModelForCausalLM.from_pretrained(config.model_name, dtype=torch.float32)


def build_arm_model(method: str, config: TraceRunConfig, factory: ModelFactory):
    base = factory(config)
    if hasattr(base, "config"):
        # Qwen's published config declares bf16; Pascal has none (R2).
        base.config.torch_dtype = torch.float32
    lora_config = LoRASlowHeatConfig(
        method=method,  # type: ignore[arg-type]
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        slow_strength=config.slow_strength,
        plasticity_budget=config.plasticity_budget,
        hard=config.hard,
        leak_combination=config.leak_combination,  # type: ignore[arg-type]
        target_modules=tuple(config.target_modules),
        task_count=len(config.tasks),
    )
    model, instrumentation = build_lora_slowheat(base, lora_config, task_type="CAUSAL_LM")
    if config.gradient_checkpointing:
        if method in TRACKED_METHODS:
            # Recomputation re-runs the forward hooks, registering the
            # importance hook twice per step and double-counting evidence.
            raise ValueError(
                "gradient checkpointing não é suportado em braços com tracker"
            )
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, instrumentation


# ---------------------------------------------------------------------------
# generation and evaluation
# ---------------------------------------------------------------------------


def build_generation_config(config: TraceRunConfig, special: SpecialTokens):
    """Fresh GenerationConfig (R9): nothing is inherited from the checkpoint."""

    from transformers import GenerationConfig

    common = {
        "max_new_tokens": config.max_new_tokens,
        "eos_token_id": special.eos,
        "pad_token_id": special.pad,
        "bos_token_id": special.bos,
        "use_cache": True,
        "repetition_penalty": 1.0,
        "num_return_sequences": 1,
    }
    if config.decoding == "official":
        return GenerationConfig(
            do_sample=True, temperature=config.temperature, top_p=1.0, top_k=0, **common
        )
    return GenerationConfig(do_sample=False, temperature=None, top_p=None, top_k=None, **common)


def _decode(tokenizer: Any, rows: torch.Tensor) -> list[str]:
    return list(
        tokenizer.batch_decode(
            rows, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    )


@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    tokenizer: Any,
    task: TraceTask,
    special: SpecialTokens,
    config: TraceRunConfig,
    *,
    generation_seed: int,
) -> list[str]:
    model.eval()
    torch.manual_seed(generation_seed)
    generation_config = build_generation_config(config, special)
    predictions: list[str] = []
    examples = list(task.test)
    for start in range(0, len(examples), config.eval_batch):
        chunk = examples[start : start + config.eval_batch]
        batch = collate_inference(
            tokenizer,
            chunk,
            special,
            max_prompt_len=config.max_prompt_len,
            pad_to_multiple_of=config.pad_to_multiple_of,
        )
        batch = {key: value.to(config.device) for key, value in batch.items()}
        generated = model.generate(**batch, generation_config=generation_config)
        width = batch["input_ids"].shape[1]
        predictions.extend(_decode(tokenizer, generated[:, width:]))
    return predictions


def score_task(task: TraceTask, predictions: Sequence[str]) -> dict[str, Any]:
    sources = [example.prompt for example in task.test]
    truths = [example.answer for example in task.test]
    evaluation = evaluate_task(task.name, sources, predictions, truths)
    return {
        "evaluation": evaluation,
        "primary": primary_score(task.name, evaluation),
        "empty_predictions": sum(1 for item in predictions if item == ""),
        "n": len(predictions),
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _set_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mib(device: str) -> float | None:
    if device.startswith("cuda"):
        return float(torch.cuda.max_memory_allocated(device)) / 2**20
    return None


def _assert_importance_alive(instrumentation, method: str, stage: int) -> None:
    """Fail loudly if a tracker recorded no evidence in a whole task.

    The fp16 failure mode (pitfall 20) and the zero-init ``B`` blindness of the
    ``rank`` estimator (pitfall 30) both look like a plausible manifest. ``rank``
    is exempt on stage 0 only, where ``B`` starts at zero by construction.
    """

    if method not in TRACKED_METHODS:
        return
    dead: list[str] = []
    for adapted in instrumentation.modules:
        tracker = adapted.tracker
        # Per-task evidence lives in task_ema until consolidate() zeroes it
        # (dual_heater.slow_heat._SlowHeatImportanceMixin). No fallback: a
        # renamed buffer must break this guard, not disable it.
        signal = tracker.task_ema
        steps = int(tracker.task_step.item())
        if steps == 0 or not bool(torch.any(signal != 0)):
            dead.append(adapted.name)
    if dead and not (method == "rank" and stage == 0):
        raise RuntimeError(
            f"{method}: {len(dead)} módulos sem importância registrada no estágio {stage} "
            f"(ex.: {dead[0]}); a máscara seria construída sobre zero"
        )


def run_arm(
    method: str,
    tasks: Sequence[TraceTask],
    config: TraceRunConfig,
    tokenizer: Any,
    *,
    model_factory: ModelFactory = default_model_factory,
    on_stage: Callable[[dict[str, Any]], None] | None = None,
    inspect: Callable[[torch.nn.Module, Any, Any, str], None] | None = None,
) -> dict[str, Any]:
    """Train ``method`` over ``tasks`` and evaluate after every task.

    ``inspect(model, instrumentation, optimizer, when)`` is a test seam, called
    with ``when="start"`` after the optimizer exists and ``when="end"`` before
    the model is released. It must not mutate anything.
    """

    if method not in ARMS:
        raise ValueError(f"braço desconhecido: {method}")
    config.validate()
    if [task.name for task in tasks] != list(config.tasks):
        raise ValueError("as tarefas carregadas não batem com config.tasks")

    torch.manual_seed(config.seed)
    _set_device(config.device)
    special = special_token_ids(tokenizer)
    model, instrumentation = build_arm_model(method, config, model_factory)
    model.to(config.device)
    instrumentation.to(config.device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    effective_lr = resolve_effective_lr(method, config, model)
    optimizer = SlowHeatAdamW(
        trainable,
        lr=effective_lr,
        betas=tuple(config.adam_betas),
        weight_decay=config.weight_decay,
    )
    instrumentation.register_plasticity_masks(optimizer)
    if inspect is not None:
        inspect(model, instrumentation, optimizer, "start")

    task_count = len(tasks)
    matrix: list[list[float | None]] = [[None] * task_count for _ in range(task_count)]
    evaluations: list[list[dict[str, Any] | None]] = [[None] * task_count for _ in range(task_count)]
    predictions_log: dict[str, list[str]] = {}
    stages: list[dict[str, Any]] = []
    calibration: list[dict[str, float]] = []
    train_tokens = 0
    answer_tokens = 0
    optimizer_steps = 0
    started = time.perf_counter()

    for stage, task in enumerate(tasks):
        stage_started = time.perf_counter()
        instrumentation.begin_task(stage)
        model.train()
        instrumentation.train()
        generator = torch.Generator().manual_seed(config.seed * 1000 + stage)
        examples = list(task.train)
        pending = 0
        stage_losses: list[float] = []

        for _ in range(int(config.epochs[stage])):
            order = torch.randperm(len(examples), generator=generator).tolist()
            for start in range(0, len(order), config.micro_batch):
                chunk = [examples[index] for index in order[start : start + config.micro_batch]]
                batch = collate_train(
                    tokenizer,
                    chunk,
                    special,
                    max_prompt_len=config.max_prompt_len,
                    max_ans_len=config.max_ans_len,
                    pad_to_multiple_of=config.pad_to_multiple_of,
                    label_tokenization=config.label_tokenization,
                )
                batch = {key: value.to(config.device) for key, value in batch.items()}
                train_tokens += int(batch["attention_mask"].sum())
                answer_tokens += int((batch["labels"] != -100).sum())
                with instrumentation.validity(batch["attention_mask"]):
                    loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"{method}: loss não finita no estágio {stage} ({task.name})"
                    )
                stage_losses.append(float(loss.detach()))
                (loss / config.grad_accum).backward()
                pending += 1
                if pending == config.grad_accum:
                    torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    pending = 0
        if pending:
            # R3: flush at the task boundary. Rescale so the partial window is
            # a mean over its own micro-batches, like a full one.
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.mul_(config.grad_accum / pending)
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        _assert_importance_alive(instrumentation, method, stage)
        instrumentation.consolidate()
        if config.target_plasticity is not None:
            solved = instrumentation.calibrate_to_target_plasticity(config.target_plasticity)
            calibration.append({"stage": float(stage), **solved})
        train_seconds = time.perf_counter() - stage_started

        eval_started = time.perf_counter()
        instrumentation.eval()
        for evaluated in range(stage + 1):
            predicted = generate_predictions(
                model,
                tokenizer,
                tasks[evaluated],
                special,
                config,
                generation_seed=config.seed * 10_000 + stage * 100 + evaluated,
            )
            scored = score_task(tasks[evaluated], predicted)
            matrix[stage][evaluated] = scored["primary"]
            evaluations[stage][evaluated] = {
                key: value for key, value in scored.items() if key != "primary"
            } | {"primary": scored["primary"]}
            predictions_log[f"{stage}:{tasks[evaluated].name}"] = predicted
        model.train()
        instrumentation.train()

        diagnostics = instrumentation.diagnostics()
        record = {
            "stage": stage,
            "task": task.name,
            "epochs": int(config.epochs[stage]),
            "train_examples": len(examples),
            "mean_train_loss": statistics.fmean(stage_losses) if stage_losses else None,
            "last_train_loss": stage_losses[-1] if stage_losses else None,
            "train_seconds": train_seconds,
            "eval_seconds": time.perf_counter() - eval_started,
            "cumulative_peak_memory_mib": _peak_mib(config.device),
            "effective_plasticity": diagnostics.get("effective_plasticity"),
            "protected_unit_fraction": diagnostics.get("protected_unit_fraction"),
            "row": matrix[stage][: stage + 1],
        }
        stages.append(record)
        if on_stage is not None:
            on_stage(record)

    elapsed = time.perf_counter() - started
    metrics = trace_cl_metrics(matrix)
    standard = None
    if task_count > 1:
        from dual_heater.metrics import compute_cl_metrics

        dense = np.array(
            [[np.nan if value is None else value for value in row] for row in matrix],
            dtype=np.float64,
        )
        standard = compute_cl_metrics(dense)
    masked_mean, masked_parameters = instrumentation.mask_coverage()
    if inspect is not None:
        inspect(model, instrumentation, optimizer, "end")
    instrumentation.remove_hooks()

    result = {
        "method": method,
        "overall_performance": metrics.overall_performance,
        "backward_transfer": metrics.backward_transfer,
        "backward_transfer_old_mean": metrics.backward_transfer_old_mean,
        "average_forgetting": standard.average_forgetting if standard else None,
        "per_task_forgetting": list(standard.per_task_forgetting) if standard else None,
        "per_task_final": list(metrics.per_task_final),
        "accuracy_matrix": matrix,
        "evaluations": evaluations,
        "stages": stages,
        "calibration": calibration,
        "wall_clock_seconds": elapsed,
        "train_tokens": train_tokens,
        "answer_tokens": answer_tokens,
        "optimizer_steps": optimizer_steps,
        "peak_memory_mib": _peak_mib(config.device),
        "trainable_parameters": trainable_count,
        "effective_learning_rate": effective_lr,
        "masked_mean": masked_mean,
        "masked_parameters": masked_parameters,
        "learning_rate_scale": effective_lr / config.learning_rate,
        "frozen_a": method in FROZEN_A_METHODS,
    }
    result["_predictions"] = predictions_log

    del model, optimizer, instrumentation
    if config.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _identity(config: TraceRunConfig, data_identity: dict[str, Any], method: str) -> dict[str, Any]:
    return {
        "runner": "experiments.trace_cl",
        "method": method,
        "source_sha256": source_fingerprint(PROJECT_ROOT),
        "config": config.payload(),
        "data": data_identity,
    }


def run_arms(
    arms: Sequence[str],
    tasks: Sequence[TraceTask],
    config: TraceRunConfig,
    tokenizer: Any,
    output: Path,
    *,
    data_identity: dict[str, Any],
    model_factory: ModelFactory = default_model_factory,
) -> list[dict[str, Any]]:
    """Run each arm once, skipping arms whose manifest matches this identity.

    A manifest with a DIFFERENT identity (other code, config or data) is an
    error, never overwritten: reusing a path across runs is how a stale run
    silently wins (skill pitfalls 21 and 23).
    """

    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for method in arms:
        arm_dir = output / method
        manifest_path = arm_dir / "manifest.json"
        identity = _identity(config, data_identity, method)
        if manifest_path.is_file():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved.get("identity") != identity:
                raise RuntimeError(
                    f"{manifest_path} existe com identidade diferente; use outro --output"
                )
            results.append(saved["result"])
            print(f"[{method}] já completo, pulando", flush=True)
            continue

        progress_path = arm_dir / "progress.json"
        progress: list[dict[str, Any]] = []

        def report(record: dict[str, Any], *, name: str = method) -> None:
            # Written after EVERY stage: a multi-day arm that dies at stage 6
            # must still leave the five finished rows on disk. There is no
            # resume; this is evidence, not a checkpoint.
            progress.append(record)
            write_json_atomic(
                progress_path, {"identity": identity, "complete": False, "stages": progress}
            )
            row = " ".join(f"{value:.3f}" for value in record["row"])
            print(
                f"[{name}] estágio {record['stage']} {record['task']}: {row} "
                f"(treino {record['train_seconds']:.0f}s, eval {record['eval_seconds']:.0f}s)",
                flush=True,
            )

        result = run_arm(
            method, tasks, config, tokenizer, model_factory=model_factory, on_stage=report
        )
        predictions = result.pop("_predictions")
        write_json_atomic(arm_dir / "predictions.json", predictions)
        write_json_atomic(manifest_path, {"identity": identity, "result": result})
        write_json_atomic(
            progress_path, {"identity": identity, "complete": True, "stages": progress}
        )
        results.append(result)

    if any(row["method"] == "vanilla" for row in results):
        attach_surface_plasticity(results)
    return results


def token_length_report(
    tasks: Sequence[TraceTask], tokenizer: Any, config: TraceRunConfig
) -> dict[str, Any]:
    """CPU-only cost probe: token lengths and truncation rates per task."""

    special = special_token_ids(tokenizer)
    report: dict[str, Any] = {}
    big = 10**9
    for task in tasks:
        prompt_lengths = [
            len(tokenize_official(tokenizer, ex.prompt, big, special, add_bos_token=True, add_eos_token=False))
            for ex in task.test
        ]
        answer_lengths = [
            len(tokenize_official(tokenizer, ex.answer, big, special, add_bos_token=False, add_eos_token=True))
            for ex in task.train
        ]
        train_lengths = [
            len(tokenize_official(
                tokenizer, ex.prompt + ex.answer, big, special, add_bos_token=True, add_eos_token=True
            ))
            for ex in task.train
        ]
        limit = config.max_prompt_len + config.max_ans_len
        report[task.name] = {
            "train_examples": len(task.train),
            "test_examples": len(task.test),
            "train_seq_mean": statistics.fmean(train_lengths) if train_lengths else 0.0,
            "train_seq_p95": float(np.percentile(train_lengths, 95)) if train_lengths else 0.0,
            "train_seq_max": max(train_lengths, default=0),
            "train_truncated_fraction": (
                sum(1 for value in train_lengths if value > limit) / len(train_lengths)
                if train_lengths else 0.0
            ),
            "test_prompt_truncated_fraction": (
                sum(1 for value in prompt_lengths if value > config.max_prompt_len) / len(prompt_lengths)
                if prompt_lengths else 0.0
            ),
            "answer_tokens_p95": float(np.percentile(answer_lengths, 95)) if answer_lengths else 0.0,
            "answer_longer_than_max_new_tokens": (
                sum(1 for value in answer_lengths if value > config.max_new_tokens) / len(answer_lengths)
                if answer_lengths else 0.0
            ),
            "train_tokens_per_epoch": sum(min(value, limit) for value in train_lengths),
        }
    total = sum(
        report[task.name]["train_tokens_per_epoch"] * int(epochs)
        for task, epochs in zip(tasks, config.epochs)
    )
    return {"per_task": report, "total_train_tokens_all_epochs": total}


def _parse_scale(value: str | None) -> float | str | None:
    if value is None:
        return None
    if value == FROZEN_A_SURFACE:
        return value
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRACE continual instruction tuning (LoRA arms)")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=TraceRunConfig.model_name)
    parser.add_argument("--arms", nargs="+", default=["vanilla"], choices=ARMS)
    parser.add_argument("--tasks", nargs="+", default=list(TRACE_TASKS), choices=TRACE_TASKS)
    parser.add_argument(
        "--epochs", default=None,
        help="vírgula-separado, um por tarefa; default = épocas LoRA oficiais",
    )
    parser.add_argument("--seed", type=int, default=TraceRunConfig.seed)
    parser.add_argument("--data-seed", type=int, default=TraceRunConfig.data_seed)
    parser.add_argument("--train-per-task", type=int, default=None)
    parser.add_argument("--test-per-task", type=int, default=None)
    parser.add_argument("--max-prompt-len", type=int, default=TraceRunConfig.max_prompt_len)
    parser.add_argument("--max-ans-len", type=int, default=TraceRunConfig.max_ans_len)
    parser.add_argument("--max-new-tokens", type=int, default=None, help="default = max-ans-len")
    parser.add_argument(
        "--label-tokenization", choices=LABEL_TOKENIZATIONS, default="split",
        help="'split' (R10, default) ou 'official' (máscara conjunta do upstream)",
    )
    parser.add_argument("--micro-batch", type=int, default=TraceRunConfig.micro_batch)
    parser.add_argument("--grad-accum", type=int, default=TraceRunConfig.grad_accum)
    parser.add_argument("--eval-batch", type=int, default=TraceRunConfig.eval_batch)
    parser.add_argument("--learning-rate", type=float, default=TraceRunConfig.learning_rate)
    parser.add_argument("--rank", type=int, default=TraceRunConfig.rank)
    parser.add_argument("--alpha", type=float, default=TraceRunConfig.alpha)
    parser.add_argument("--dropout", type=float, default=TraceRunConfig.dropout)
    parser.add_argument("--target-modules", nargs="+", default=list(TraceRunConfig.target_modules))
    parser.add_argument("--decoding", choices=["official", "greedy"], default="official")
    parser.add_argument("--temperature", type=float, default=TraceRunConfig.temperature)
    parser.add_argument("--slow-strength", type=float, default=TraceRunConfig.slow_strength)
    parser.add_argument("--plasticity-budget", type=float, default=TraceRunConfig.plasticity_budget)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--leak-combination", choices=["min", "weighted"], default="min")
    parser.add_argument("--target-plasticity", type=float, default=None)
    parser.add_argument(
        "--learning-rate-scale", default=None,
        help=f"escala do lr_control: número em (0,1] ou '{FROZEN_A_SURFACE}'",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="só tokeniza e reporta comprimentos; não carrega modelo nem usa GPU",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> TraceRunConfig:
    tasks = tuple(args.tasks)
    if args.epochs is None:
        epochs = tuple(OFFICIAL_LORA_EPOCHS[name] for name in tasks)
    else:
        epochs = tuple(int(value) for value in str(args.epochs).split(","))
    config = TraceRunConfig(
        model_name=args.model,
        tasks=tasks,
        epochs=epochs,
        seed=args.seed,
        data_seed=args.data_seed,
        train_per_task=args.train_per_task,
        test_per_task=args.test_per_task,
        max_prompt_len=args.max_prompt_len,
        max_ans_len=args.max_ans_len,
        max_new_tokens=args.max_new_tokens or args.max_ans_len,
        label_tokenization=args.label_tokenization,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        eval_batch=args.eval_batch,
        learning_rate=args.learning_rate,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=tuple(args.target_modules),
        decoding=args.decoding,
        temperature=args.temperature,
        slow_strength=args.slow_strength,
        plasticity_budget=args.plasticity_budget,
        hard=args.hard,
        leak_combination=args.leak_combination,
        target_plasticity=args.target_plasticity,
        learning_rate_scale=_parse_scale(args.learning_rate_scale),
        gradient_checkpointing=args.gradient_checkpointing,
        device=args.device,
    )
    config.validate()
    return config


def _format_summary(results: Sequence[dict[str, Any]]) -> str:
    lines = ["| arm | OP | BWT (paper, /t) | forgetting | E_surface | steps | peak MiB |",
             "|---|---|---|---|---|---|---|"]
    for row in results:
        peak = row.get("peak_memory_mib")
        lines.append(
            "| {m} | {op:.4f} | {bwt:+.4f} | {fg} | {e} | {s} | {p} |".format(
                m=row["method"],
                op=row["overall_performance"],
                bwt=row["backward_transfer"],
                fg=f"{row['average_forgetting']:.4f}" if row.get("average_forgetting") is not None else "-",
                e=f"{row['surface_plasticity']:.4f}" if "surface_plasticity" in row else "-",
                s=row["optimizer_steps"],
                p=f"{peak:.0f}" if peak is not None else "-",
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tasks = load_trace_tasks(
        args.data_root,
        task_names=config.tasks,
        train_per_task=config.train_per_task,
        test_per_task=config.test_per_task,
        data_seed=config.data_seed,
    )
    data_identity = trace_data_identity(args.data_root, tasks)
    overlap = {name: count for name, count in data_identity["train_test_prompt_overlap"].items() if count}
    if overlap:
        print(f"AVISO: prompts de teste presentes no treino: {overlap}", flush=True)

    if args.dry_run:
        report = {
            "kind": "dry_run_token_lengths",
            "config": config.payload(),
            "data": data_identity,
            "lengths": token_length_report(tasks, tokenizer, config),
        }
        write_json_atomic(args.output / "dry_run.json", report)
        for name, row in report["lengths"]["per_task"].items():
            print(
                f"{name:12s} seq média {row['train_seq_mean']:7.1f} p95 {row['train_seq_p95']:7.1f} "
                f"trunc treino {row['train_truncated_fraction']:.3f} "
                f"trunc teste {row['test_prompt_truncated_fraction']:.3f} "
                f"resp p95 {row['answer_tokens_p95']:.0f}",
                flush=True,
            )
        total = report["lengths"]["total_train_tokens_all_epochs"]
        print(f"tokens de treino (todas as épocas): {total:,}", flush=True)
        return 0

    results = run_arms(
        args.arms, tasks, config, tokenizer, args.output, data_identity=data_identity
    )
    write_json_atomic(args.output / "summary.json", {"results": results})
    print(_format_summary(results), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
