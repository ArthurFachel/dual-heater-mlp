"""Smoke benchmark: three SlowHeat-in-LoRA mechanisms on Qwen2.5-0.5B.

Protocol
--------
Class-incremental Split-CLINC150, first ``--tasks`` official domains, one seed,
one GPU. Every arm shares the identical data stream, tokenization, optimizer
family, learning rate, epoch count and evaluation code; only the LoRA masking
mechanism changes. The classification head stays trainable and unmasked in
every arm, including the baselines, because a randomly-initialized head must
stay plastic or a Class-IL run cannot learn any label at all.

Reported per arm
----------------
final_average_accuracy, average_forgetting, backward_transfer (shared metric
code in ``dual_heater.metrics``), wall-clock seconds, total training tokens
(padding excluded), and peak memory (CUDA allocator).

Status: EXPLORATORY. One seed cannot support an inferential claim. The numbers
are a feasibility and cost measurement, not evidence that any arm is better.

Run:
    HF_HOME=.hf-cache CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
        experiments/qwen_lora_slowheat.py --output results/qwen_lora_slowheat
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from dual_heater.lora_slowheat import (
    LoRASlowHeatConfig,
    QwenLoRASlowHeat,
    build_lora_slowheat,
)
from dual_heater.metrics import compute_cl_metrics
from dual_heater.optim import SlowHeatAdamW
from experiments.peak_memory import PeakMemoryTracker
from experiments.split_clinc150 import (
    CLINC150Task,
    SplitCLINC150Config,
    load_clinc150_tasks,
)

ARMS: tuple[str, ...] = (
    "vanilla", "exact", "rank", "leak", "slice", "lr_control",
)


@dataclass(frozen=True)
class RunConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    seed: int = 0
    tasks: int = 2
    train_per_class: int = 20
    eval_per_class: int = 10
    max_length: int = 48
    batch_size: int = 8
    epochs_per_task: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    rank: int = 8
    alpha: float = 16.0
    slow_strength: float = 3.0
    plasticity_budget: float = 0.5
    hard: bool = False
    leak_combination: str = "min"
    #: When set, each arm's knob is solved so measured plasticity equals this
    #: value at every task boundary, making arms cost-matched in retained
    #: plasticity instead of in nominal protection strength.
    target_plasticity: float | None = None
    device: str = "cuda:0"

    def validate(self) -> None:
        if self.tasks < 2:
            raise ValueError("o benchmark requer ao menos 2 tarefas")
        if self.rank < self.tasks:
            raise ValueError("rank deve ser >= tasks para o braço 'slice'")
        for name in ("train_per_class", "eval_per_class", "batch_size", "epochs_per_task"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} deve ser positivo")


def _subsample(split, *, per_class: int, generator: torch.Generator):
    """Take up to ``per_class`` examples per label, deterministically."""

    selected: list[Tensor] = []
    for label in sorted(set(split.labels.tolist())):
        indices = torch.nonzero(split.labels == label, as_tuple=False).flatten()
        permutation = torch.randperm(len(indices), generator=generator)
        selected.append(indices[permutation[:per_class]])
    chosen = torch.cat(selected)
    order = torch.randperm(len(chosen), generator=generator)
    return split.select(chosen[order])


def _trimmed_batch(split, indices: Tensor, device: str) -> dict[str, Tensor]:
    attention = split.attention_mask[indices]
    width = max(1, int(attention.sum(dim=1).max().item()))
    return {
        "input_ids": split.input_ids[indices][:, :width].to(device),
        "attention_mask": attention[:, :width].to(device),
        "labels": split.labels[indices].to(device),
    }


def _seen_classes(tasks: list[CLINC150Task], stage: int) -> list[int]:
    return [label for task in tasks[: stage + 1] for label in task.classes]


@torch.no_grad()
def _evaluate(
    model,
    instrumentation: QwenLoRASlowHeat,
    split,
    *,
    seen: list[int],
    batch_size: int,
    device: str,
) -> float:
    """Class-IL accuracy: logits are restricted to classes seen so far."""

    was_training = model.training
    model.eval()
    instrumentation.eval()
    correct = 0
    total = 0
    unseen_fill = torch.finfo(torch.float32).min
    for start in range(0, len(split.labels), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(split.labels)))
        batch = _trimmed_batch(split, indices, device)
        labels = batch.pop("labels")
        with instrumentation.validity(batch["attention_mask"]):
            logits = model(**batch).logits.float()
        mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
        mask[seen] = False
        predicted = logits.masked_fill(mask, unseen_fill).argmax(-1)
        correct += int((predicted == labels).sum())
        total += len(labels)
    if was_training:
        model.train()
        instrumentation.train()
    return correct / total if total else 0.0


def _build_model(config: RunConfig, num_labels: int, method: str):
    from transformers import AutoConfig, Qwen2ForSequenceClassification

    model_config = AutoConfig.from_pretrained(config.model_name)
    model_config.num_labels = num_labels
    # Pascal cards have no native bf16 and Qwen's published config declares it.
    # Loading fp32 also keeps SlowHeatAdamW off the GradScaler path, which the
    # masked pointwise step explicitly rejects.
    model_config.torch_dtype = torch.float32
    torch.manual_seed(config.seed)
    model = Qwen2ForSequenceClassification.from_pretrained(
        config.model_name,
        config=model_config,
        dtype=torch.float32,
    )
    lora_config = LoRASlowHeatConfig(
        method=method,  # type: ignore[arg-type]
        rank=config.rank,
        alpha=config.alpha,
        slow_strength=config.slow_strength,
        plasticity_budget=config.plasticity_budget,
        hard=config.hard,
        leak_combination=config.leak_combination,  # type: ignore[arg-type]
        task_count=config.tasks,
    )
    return build_lora_slowheat(model, lora_config)


def run_arm(
    method: str,
    tasks: list[CLINC150Task],
    config: RunConfig,
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    device = config.device
    if device.startswith("cuda"):
        # CUDA initializes lazily; the allocator stats API rejects the device
        # until the context exists.
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model, instrumentation = _build_model(config, 150, method)
    model.config.pad_token_id = pad_token_id
    if hasattr(model, "base_model"):
        model.base_model.model.config.pad_token_id = pad_token_id
    model.to(device)
    instrumentation.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)

    tracker = PeakMemoryTracker(device).start()
    started = time.perf_counter()

    # The lr_control arm removes plasticity uniformly through the learning
    # rate instead of selectively through a mask. Scaling lr by the same
    # target E makes it the cost-matched falsifier for every masked arm.
    effective_lr = config.learning_rate
    if method == "lr_control" and config.target_plasticity is not None:
        effective_lr = config.learning_rate * config.target_plasticity

    optimizer = SlowHeatAdamW(
        trainable,
        lr=effective_lr,
        weight_decay=config.weight_decay,
    )
    instrumentation.register_plasticity_masks(optimizer)
    calibration: list[dict[str, float]] = []

    matrix = np.full((len(tasks), len(tasks)), np.nan)
    train_tokens = 0
    steps = 0
    per_stage: list[dict[str, Any]] = []

    for stage, task in enumerate(tasks):
        instrumentation.begin_task(stage)
        model.train()
        instrumentation.train()
        split = task.train
        generator = torch.Generator().manual_seed(config.seed * 1000 + stage)

        for _ in range(config.epochs_per_task):
            order = torch.randperm(len(split.labels), generator=generator)
            for start in range(0, len(order), config.batch_size):
                indices = order[start : start + config.batch_size]
                batch = _trimmed_batch(split, indices, device)
                train_tokens += int(batch["attention_mask"].sum())
                with instrumentation.validity(batch["attention_mask"]):
                    loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1

        # Consolidation happens at the task boundary, before evaluation, so the
        # protection an arm carries into the next task is the one measured here.
        instrumentation.consolidate()
        # Solve the knob so measured plasticity hits the declared target. This
        # happens AFTER consolidation (the mask exists) and BEFORE evaluation,
        # and never looks at accuracy.
        if config.target_plasticity is not None:
            solved = instrumentation.calibrate_to_target_plasticity(
                config.target_plasticity
            )
            calibration.append({"stage": float(stage), **solved})
        stage_diagnostics = instrumentation.diagnostics()
        stage_memory = tracker.snapshot()
        per_stage.append(
            {
                "stage": stage,
                "domain": task.domain,
                "effective_plasticity": stage_diagnostics["effective_plasticity"],
                "protected_unit_fraction": stage_diagnostics["protected_unit_fraction"],
                "leak_collapse_fraction": stage_diagnostics["leak_collapse_fraction"],
                "frozen_rank_dims": stage_diagnostics["frozen_rank_dims"],
                # Cumulative peak up to the end of this stage: the tracker is
                # monotonic, so a per-stage increase is what this reveals.
                "cumulative_peak_memory_mib": (
                    float(stage_memory["peak_memory_bytes"]) / 2**20
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )

        seen = _seen_classes(tasks, stage)
        for evaluated in range(stage + 1):
            matrix[stage, evaluated] = _evaluate(
                model,
                instrumentation,
                tasks[evaluated].validation,
                seen=seen,
                batch_size=config.batch_size,
                device=device,
            )

    elapsed = time.perf_counter() - started
    memory = tracker.stop()
    metrics = compute_cl_metrics(matrix)
    instrumentation.remove_hooks()

    result = {
        "method": method,
        "final_average_accuracy": metrics.final_average_accuracy,
        "average_forgetting": metrics.average_forgetting,
        "backward_transfer": metrics.backward_transfer,
        "per_task_forgetting": list(metrics.per_task_forgetting),
        "accuracy_matrix": [
            [None if np.isnan(value) else float(value) for value in row]
            for row in matrix
        ],
        "wall_clock_seconds": elapsed,
        "train_tokens": train_tokens,
        "optimizer_steps": steps,
        "tokens_per_second": train_tokens / elapsed if elapsed else None,
        "peak_memory_bytes": memory["peak_memory_bytes"],
        "peak_memory_mib": float(memory["peak_memory_bytes"]) / 2**20,
        "peak_memory_backend": memory["peak_memory_backend"],
        "peak_cuda_reserved_mib": (
            float(memory["peak_cuda_reserved_bytes"]) / 2**20
            if memory["peak_cuda_reserved_bytes"] is not None
            else None
        ),
        "trainable_parameters": trainable_count,
        "effective_learning_rate": effective_lr,
        "target_plasticity": config.target_plasticity,
        "calibration": calibration,
        "stages": per_stage,
    }

    del model, optimizer, instrumentation
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _format_table(results: list[dict[str, Any]]) -> str:
    header = (
        "| arm | FAA | forgetting | BWT | seconds | tokens | tok/s | "
        "peak MiB | trainable |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|"
    rows = [
        "| {method} | {faa:.4f} | {forget:+.4f} | {bwt:+.4f} | {sec:.1f} | "
        "{tok:,} | {tps:.0f} | {mem:.1f} | {par:,} |".format(
            method=item["method"],
            faa=item["final_average_accuracy"],
            forget=item["average_forgetting"],
            bwt=item["backward_transfer"],
            sec=item["wall_clock_seconds"],
            tok=item["train_tokens"],
            tps=item["tokens_per_second"] or 0.0,
            mem=item["peak_memory_mib"],
            par=item["trainable_parameters"],
        )
        for item in results
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--eval-per-class", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-per-task", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--slow-strength", type=float, default=3.0)
    parser.add_argument("--plasticity-budget", type=float, default=0.5)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument(
        "--target-plasticity",
        type=float,
        default=None,
        help="Solve each arm's knob so measured plasticity equals this value.",
    )
    parser.add_argument("--leak-combination", default="min", choices=["min", "weighted"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    arguments = parser.parse_args()

    config = RunConfig(
        seed=arguments.seed,
        tasks=arguments.tasks,
        train_per_class=arguments.train_per_class,
        eval_per_class=arguments.eval_per_class,
        max_length=arguments.max_length,
        batch_size=arguments.batch_size,
        epochs_per_task=arguments.epochs_per_task,
        rank=arguments.rank,
        slow_strength=arguments.slow_strength,
        plasticity_budget=arguments.plasticity_budget,
        hard=arguments.hard,
        leak_combination=arguments.leak_combination,
        target_plasticity=arguments.target_plasticity,
        device=arguments.device,
    )
    config.validate()

    data_config = SplitCLINC150Config(
        seed=config.seed,
        model_name=config.model_name,
        max_length=config.max_length,
    )
    tasks, data_metadata = load_clinc150_tasks(data_config, include_test=False)
    tasks = tasks[: config.tasks]

    generator = torch.Generator().manual_seed(config.seed)
    tasks = [
        CLINC150Task(
            domain=task.domain,
            classes=task.classes,
            train=_subsample(
                task.train, per_class=config.train_per_class, generator=generator
            ),
            validation=_subsample(
                task.validation, per_class=config.eval_per_class, generator=generator
            ),
            test=None,
        )
        for task in tasks
    ]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    pad_token_id = tokenizer.pad_token_id

    results = [
        run_arm(method, tasks, config, pad_token_id=pad_token_id)
        for method in arguments.arms
    ]

    manifest = {
        "status": "exploratory_single_seed",
        "claim_scope": (
            "feasibility and cost measurement; one seed supports no inferential "
            "comparison between arms"
        ),
        "config": asdict(config),
        "arms": list(arguments.arms),
        "task_domains": [task.domain for task in tasks],
        "train_examples_per_task": [len(task.train.labels) for task in tasks],
        "eval_examples_per_task": [len(task.validation.labels) for task in tasks],
        "data": data_metadata,
        "git_commit": _git_commit(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(config.device)
                if config.device.startswith("cuda")
                else None
            ),
        },
        "results": results,
    }

    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    table = _format_table(results)
    (arguments.output / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
