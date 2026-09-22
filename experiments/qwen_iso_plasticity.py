"""Iso-plasticity ablation runner for SlowHeat on Qwen2.

The question
------------
Given the same amount of effective plasticity removed, does it matter *how*
that removal is distributed across units? Every arm in a family is matched on
effective plasticity ``E``; only the spread changes. A low budget protects many
units weakly, a high budget protects few units hard.

The protocol is frozen in `goals/protocol_iso_plasticity.md`. This runner
implements it and nothing else: no hyperparameter is chosen here, and the
selection rule for ``beta`` is bisection against a declared target, never a
grid search with accuracy in the loop.

Controls
--------
Three controls per family, all matched on ``E*``:

``permuted``
    The learned heat vector is permuted. This preserves the multiset of heat
    values exactly, so ``E`` is identical by construction and only the identity
    of the protected units changes. It answers "does the ranking inform?".
    Zeroing the heat and re-protecting random units at ``h=1`` is *not* an
    iso-``E`` control: it changes the distribution and therefore ``E``.

``reduced_lr``
    ``lr * E*`` with no mask registered at all. It answers "is the mechanism
    reducible to a smaller learning rate?". This is the control the BERT
    diagnostic lacked, and it is mandatory.

``vanilla``
    ``E = 1``: no consolidation, no mask, unchanged learning rate. Forgetting
    anchor.

Precision
---------
Weights are always held in fp32 and fp16 is applied through autocast plus a
gradient scaler. Loading fp16 weights *and* enabling the scaler is the one
combination that breaks, because the scaler expects fp32 master weights.

The gradient scaler does not bias the importance estimator: the tracker
normalizes each step's signal by its own mean before the EMA, so a uniform
scale factor cancels exactly, even when the scaler changes scale mid-task.

Run (CPU smoke, tiny budget):
    CUDA_VISIBLE_DEVICES= HF_HOME=.hf-cache PYTHONPATH=. \\
        .venv/bin/python experiments/qwen_iso_plasticity.py --seed 0
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from dual_heater.optim import SlowHeatAdamW
from dual_heater.qwen import (
    QwenSlowHeatConfig,
    SlowHeatQwen2ForSequenceClassification,
)
from experiments.capacity_calibration import (
    IsoPlasticityPoint,
    effective_plasticity,
    free_fraction_scoped,
    protected_heat_scoped,
    solve_strength_for_plasticity_scoped,
)
from experiments.split_clinc150 import (
    CLINC150_DOMAINS,
    CLINC150Task,
    _evaluate,
    build_clinc150_tasks,
    capture_parameter_drift_reference,
    summarize_parameter_drift,
    text_task_fingerprint,
)

ArmKind = Literal["iso", "permuted", "reduced_lr", "vanilla"]

# Declared in the frozen protocol, section E.
ISO_BUDGETS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)
ISO_TARGETS: tuple[float, ...] = (0.75, 0.50)


@dataclass(frozen=True)
class ArmSpec:
    """One declared arm. Nothing here is chosen after seeing accuracy."""

    name: str
    kind: ArmKind
    target_plasticity: float
    # `budget` is None exactly for the two arms that register no mask.
    budget: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.target_plasticity <= 1.0:
            raise ValueError("target_plasticity deve estar em (0, 1]")
        masked = self.kind in {"iso", "permuted"}
        if masked and self.budget is None:
            raise ValueError(f"braço {self.name} com máscara exige budget")
        if not masked and self.budget is not None:
            raise ValueError(f"braço {self.name} sem máscara não pode ter budget")


def build_arms(
    *,
    target_plasticity: float,
    budgets: Sequence[float],
    permuted_budget: float,
) -> list[ArmSpec]:
    """Declared arms for one family: every budget plus the three controls.

    Reachability is NOT decided here. A budget whose floor exceeds the target
    cannot be reached by any ``beta``, but the floor depends on the measured
    importance, which does not exist until the first consolidation. Arms are
    therefore declared for every budget and discarded at the first boundary if
    the solver returns ``None``; the discard is recorded in the manifest rather
    than saturating ``beta`` at a value nobody declared.
    """

    if permuted_budget not in set(budgets):
        raise ValueError("permuted_budget deve ser um dos budgets declarados")
    arms = [
        ArmSpec(
            name=f"iso_b{budget:g}",
            kind="iso",
            target_plasticity=target_plasticity,
            budget=budget,
        )
        for budget in budgets
    ]
    arms.append(
        ArmSpec(
            name=f"permuted_b{permuted_budget:g}",
            kind="permuted",
            target_plasticity=target_plasticity,
            budget=permuted_budget,
        )
    )
    arms.append(
        ArmSpec(
            name="reduced_lr",
            kind="reduced_lr",
            target_plasticity=target_plasticity,
        )
    )
    arms.append(
        ArmSpec(
            name="vanilla",
            kind="vanilla",
            target_plasticity=target_plasticity,
        )
    )
    return arms


def reachable_iso_points(
    importances: Sequence[Tensor],
    *,
    target_plasticity: float,
    budgets: Sequence[float],
    scope: str,
) -> tuple[list[IsoPlasticityPoint], list[dict[str, float]]]:
    """Solve ``beta`` per budget, separating reachable from unreachable.

    Returns the solved points and the budgets that were discarded, each with
    the floor that made the target unreachable. Saturating ``beta`` instead
    would silently report a configuration that was never declared.
    """

    points: list[IsoPlasticityPoint] = []
    discarded: list[dict[str, float]] = []
    for budget in budgets:
        strength = solve_strength_for_plasticity_scoped(
            importances, budget, target_plasticity, scope=scope
        )
        floor = free_fraction_scoped(importances, budget, scope=scope)
        if strength is None:
            discarded.append(
                {
                    "budget": budget,
                    "floor": floor,
                    "target_plasticity": target_plasticity,
                }
            )
            continue
        heat = protected_heat_scoped(importances, budget, scope=scope)
        points.append(
            IsoPlasticityPoint(
                target_plasticity=target_plasticity,
                budget=budget,
                slow_strength=strength,
                protected_units=int(torch.count_nonzero(heat > 0.0).item()),
                protected_fraction=float(
                    torch.count_nonzero(heat > 0.0).item() / heat.numel()
                ),
                achieved_plasticity=effective_plasticity(heat, strength),
            )
        )
    return points, discarded


def permute_slowheat_heat(model, *, seed: int) -> bool:
    """Shuffle each tracker's heat vector in place, preserving ``E`` exactly.

    Permuting the WHOLE vector, zeros included, preserves both the protected
    count and the multiset of heat values, so ``E = mean 1/(1 + beta*h)`` is
    identical by construction. Only the identity of the protected units moves.

    Returns whether any permutation was a real reordering, so a caller can
    refuse to report a "random" control that happened to be the identity.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    reordered = False
    with torch.no_grad():
        for tracker in model.get_slow_states():
            heat = tracker.slow_heat
            if int(torch.count_nonzero(heat).item()) == 0:
                continue
            permutation = torch.randperm(heat.numel(), generator=generator)
            if not torch.equal(permutation, torch.arange(heat.numel())):
                reordered = True
            heat.copy_(heat[permutation.to(heat.device)])
    return reordered


def layer_importances(model) -> list[Tensor]:
    """Per-tracker importance memory, as a LIST.

    Capacity scope is part of the mechanism: under ``local`` each layer
    normalizes to its own maximum, so pooling the family into one vector and
    dividing by a single global maximum answers a different question.
    """

    return [
        tracker.importance_memory.detach().float().cpu().clone()
        for tracker in model.get_ffn_trackers()
    ]


def measured_plasticity(model) -> float:
    """Effective plasticity read off the heat the model really holds.

    This is deliberately not recomputed from the importance vector: it reads
    ``slow_heat`` and ``slow_strength`` as the optimizer sees them, so a
    mismatch between the arithmetic used to solve ``beta`` and the arithmetic
    the model applies shows up as a failed target instead of passing silently.
    """

    trackers = model.get_ffn_trackers()
    strengths = {float(tracker.slow_strength) for tracker in trackers}
    if len(strengths) != 1:
        raise RuntimeError("slow_strength divergiu entre trackers do mesmo braço")
    heat = torch.cat([tracker.slow_heat.detach().float().cpu() for tracker in trackers])
    return effective_plasticity(heat, strengths.pop())


def resolve_beta(
    model,
    *,
    arm: ArmSpec,
    scope: str,
) -> float | None:
    """Solve ``E(beta) = E*`` on the importance consolidated so far.

    Returns ``None`` when the arm's budget cannot reach the target, which is
    the signal to discard the arm rather than saturate it.
    """

    if arm.budget is None:
        return None
    return solve_strength_for_plasticity_scoped(
        layer_importances(model),
        arm.budget,
        arm.target_plasticity,
        scope=scope,
    )


def register_reduced_lr_arm(model, optimizer) -> None:
    """Confirm the reduced-LR control registered no mask at all.

    The control is only a control if the optimizer is untouched: the whole
    point is to ask whether a smaller learning rate reproduces the mechanism.
    This asserts the invariant instead of trusting the call site.
    """

    registered = getattr(optimizer, "_plasticity_masks", {})
    if registered:
        raise RuntimeError(
            "braço de LR reduzido não pode registrar máscara; registrou "
            f"{len(registered)}"
        )


def _build_model(
    *,
    model_name: str,
    arm: ArmSpec,
    scope: str,
    num_labels: int,
    pad_token_id: int | None,
    device: torch.device,
):
    slowheat = QwenSlowHeatConfig(
        # beta starts at zero: no protection exists before the first
        # consolidation, and the run resolves it against the declared target
        # at every boundary. A nonzero default here would apply protection
        # nobody declared during task 1.
        slow_strength=0.0,
        ffn_plasticity_budget=arm.budget if arm.budget is not None else 0.25,
        capacity_scope=scope,
        freeze_unbound_parameters=True,
    )
    model = SlowHeatQwen2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        slowheat_config=slowheat,
        dtype=torch.float32,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = pad_token_id
    model.to(device)
    return model


def _train_task(
    model,
    optimizer,
    scaler,
    task: CLINC150Task,
    *,
    steps: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    autocast_dtype: torch.dtype | None,
) -> list[float]:
    split = task.train
    count = len(split.labels)
    order = torch.randperm(count, generator=generator)
    model.train()
    losses: list[float] = []
    for step in range(steps):
        start = (step * batch_size) % max(count - batch_size, 1)
        index = order[start : start + batch_size]
        if len(index) == 0:
            break
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else _NullContext()
        )
        with context:
            output = model(
                input_ids=split.input_ids[index].to(device),
                attention_mask=split.attention_mask[index].to(device),
                labels=split.labels[index].to(device),
            )
        if scaler is not None:
            scaler.scale(output.loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output.loss.backward()
            optimizer.step()
        losses.append(float(output.loss.item()))
    return losses


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def _seen_classes(tasks: Sequence[CLINC150Task], stage: int) -> tuple[int, ...]:
    return tuple(label for task in tasks[: stage + 1] for label in task.classes)


def _evaluate_seen(
    model,
    tasks: Sequence[CLINC150Task],
    *,
    stage: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, float]]:
    """Class-IL evaluation on every task seen so far.

    ``seen_classes`` masks the logits of classes no task has introduced yet.
    Omitting it inflates every number, because the model would be credited for
    not predicting labels it structurally cannot have learned.
    """

    seen = _seen_classes(tasks, stage)
    results: list[dict[str, float]] = []
    for index in range(stage + 1):
        task = tasks[index]
        class_acc, task_acc, macro_f1 = _evaluate(
            model,
            task.validation,
            task_classes=task.classes,
            seen_classes=seen,
            batch_size=batch_size,
            device=str(device),
        )
        results.append(
            {
                "task": index,
                "domain": task.domain,
                "class_il_accuracy": class_acc,
                "task_il_accuracy": task_acc,
                "macro_f1": macro_f1,
            }
        )
    return results


def _endpoints(matrix: list[list[float | None]]) -> dict[str, float | None]:
    """FAA, retention, acquisition, forgetting and BWT from the accuracy matrix.

    ``matrix[t][k]`` is the accuracy on task ``k`` after training task ``t``.
    Forgetting deliberately EXCLUDES the last task: a task that was just
    learned has not had the chance to be forgotten, and including it reports a
    number that is structurally near zero and dilutes the real effect.
    """

    total = len(matrix)
    final = matrix[-1]
    seen_final = [value for value in final if value is not None]
    faa = sum(seen_final) / len(seen_final) if seen_final else None
    forgetting: list[float] = []
    backward: list[float] = []
    for task in range(total - 1):
        history = [
            matrix[stage][task]
            for stage in range(total)
            if matrix[stage][task] is not None
        ]
        if not history or final[task] is None:
            continue
        forgetting.append(max(history) - final[task])
        backward.append(final[task] - history[0])
    return {
        "final_average_accuracy": faa,
        "retention_first_task": final[0],
        "acquisition_last_task": final[-1],
        "mean_forgetting": (
            sum(forgetting) / len(forgetting) if forgetting else None
        ),
        "backward_transfer": (
            sum(backward) / len(backward) if backward else None
        ),
    }


def _release(model, optimizer, device: torch.device) -> None:
    """Free an arm's GPU memory before the next arm allocates its own.

    `del` alone is not enough. The SlowHeat forward hooks close over the model
    and the mask bindings close over both the trackers and the optimizer, so a
    finished arm sits in a reference cycle that only the cycle collector can
    break. Without this, arm 2 allocates while arm 1 is still resident and a
    0.5B model at fp32 with Adam state runs a 11 GB card out of memory.
    """

    if optimizer is not None:
        clear = getattr(optimizer, "clear_plasticity_masks", None)
        if callable(clear):
            clear()
        optimizer.state.clear()
        for group in optimizer.param_groups:
            group["params"] = []
    if model is not None:
        model.remove_slowheat_instrumentation()
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def run_arm(
    arm: ArmSpec,
    tasks: Sequence[CLINC150Task],
    *,
    model_name: str,
    scope: str,
    learning_rate: float,
    batch_size: int,
    steps_per_task: int,
    seed: int,
    device: torch.device,
    pad_token_id: int | None,
    beta_policy: str,
    autocast_dtype: torch.dtype | None,
) -> dict[str, Any]:
    """Train one arm through the task sequence and report mechanism + endpoints."""

    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = None
    # The optimizer is built inside the body but must be released here, so the
    # body hands it back through this holder rather than being forced to own a
    # try/finally of its own.
    holder: list[Any] = []
    try:
        model = _build_model(
            model_name=model_name,
            arm=arm,
            scope=scope,
            num_labels=150,
            pad_token_id=pad_token_id,
            device=device,
        )
        return _run_arm_body(
            arm,
            tasks,
            model=model,
            scope=scope,
            learning_rate=learning_rate,
            batch_size=batch_size,
            steps_per_task=steps_per_task,
            seed=seed,
            device=device,
            beta_policy=beta_policy,
            autocast_dtype=autocast_dtype,
            generator=generator,
            optimizer_sink=holder.append,
        )
    finally:
        _release(model, holder[0] if holder else None, device)


def _run_arm_body(
    arm: ArmSpec,
    tasks: Sequence[CLINC150Task],
    *,
    model,
    scope: str,
    learning_rate: float,
    batch_size: int,
    steps_per_task: int,
    seed: int,
    device: torch.device,
    beta_policy: str,
    autocast_dtype: torch.dtype | None,
    generator: torch.Generator,
    optimizer_sink,
) -> dict[str, Any]:
    masked = arm.kind in {"iso", "permuted"}
    lr = (
        learning_rate * arm.target_plasticity
        if arm.kind == "reduced_lr"
        else learning_rate
    )
    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    optimizer_sink(optimizer)
    if masked:
        # The mask source is a callable reading `slow_heat` and `slow_strength`
        # live, so registering once is enough: re-resolved betas take effect
        # without re-registering.
        model.register_plasticity_masks(optimizer, hard=False)
    else:
        register_reduced_lr_arm(model, optimizer)

    scaler = (
        torch.amp.GradScaler(device.type)
        if autocast_dtype == torch.float16 and device.type == "cuda"
        else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total = len(tasks)
    matrix: list[list[float | None]] = []
    boundaries: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    resolved_beta: float | None = None
    started = time.perf_counter()

    for stage, task in enumerate(tasks):
        drift_reference = (
            capture_parameter_drift_reference(model.mask_bindings(hard=True))
            if masked
            else None
        )
        losses = _train_task(
            model,
            optimizer,
            scaler,
            task,
            steps=steps_per_task,
            batch_size=batch_size,
            device=device,
            generator=generator,
            autocast_dtype=autocast_dtype,
        )
        model.eval()
        evaluations = _evaluate_seen(
            model, tasks, stage=stage, batch_size=batch_size, device=device
        )
        row: list[float | None] = [None] * total
        for entry in evaluations:
            row[int(entry["task"])] = entry["class_il_accuracy"]
        matrix.append(row)
        stages.append(
            {
                "stage": stage,
                "domain": task.domain,
                "steps": len(losses),
                "first_loss": losses[0] if losses else None,
                "last_loss": losses[-1] if losses else None,
                "evaluations": evaluations,
                # Drift is split by protected identity (heat > 0), taken from
                # the hard bindings, while the optimizer applies the soft mask.
                # Under a soft mask no factor is exactly zero, so asking for
                # "protected drift" against the soft mask would report an
                # empty set.
                "drift": (
                    summarize_parameter_drift(drift_reference)
                    if drift_reference is not None
                    else None
                ),
            }
        )

        if stage == total - 1 or not masked:
            continue

        model.consolidate(strategy="max")
        if beta_policy == "per_boundary" or resolved_beta is None:
            candidate = resolve_beta(model, arm=arm, scope=scope)
            if candidate is None:
                model.train()
                return {
                    "arm": asdict(arm),
                    "discarded": True,
                    "reason": (
                        f"budget {arm.budget} não alcança E*="
                        f"{arm.target_plasticity} na fronteira {stage}->{stage + 1}"
                    ),
                    "floor": free_fraction_scoped(
                        layer_importances(model), arm.budget, scope=scope
                    ),
                }
            resolved_beta = candidate
        for tracker in model.get_ffn_trackers():
            tracker.slow_strength = resolved_beta
        permuted = None
        if arm.kind == "permuted":
            permuted = permute_slowheat_heat(model, seed=seed + 1000 * stage)
        achieved = measured_plasticity(model)
        boundaries.append(
            {
                "boundary": f"{stage}->{stage + 1}",
                "slow_strength": resolved_beta,
                "achieved_plasticity": achieved,
                "target_plasticity": arm.target_plasticity,
                "plasticity_error": abs(achieved - arm.target_plasticity),
                "protected_units": sum(
                    int(torch.count_nonzero(tracker.slow_heat > 0.0).item())
                    for tracker in model.get_ffn_trackers()
                ),
                "permutation_reordered": permuted,
            }
        )
        model.train()

    elapsed = time.perf_counter() - started
    peak = {}
    if device.type == "cuda":
        peak = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    record = {
        "arm": asdict(arm),
        "discarded": False,
        "learning_rate": lr,
        "learning_rate_scaled": arm.kind == "reduced_lr",
        "registered_mask_count": len(getattr(optimizer, "_plasticity_masks", {})),
        "accuracy_matrix": matrix,
        "endpoints": _endpoints(matrix),
        "boundaries": boundaries,
        "stages": stages,
        "wall_time_seconds": elapsed,
        **peak,
    }
    # Releasing is the caller's job (`run_arm`'s finally), so it also runs on
    # the discarded-arm return path and on an exception.
    return record


def protocol_hash(protocol: dict[str, Any]) -> str:
    """Stable digest of the declared protocol, for auditing the manifest."""

    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset", default="clinc/clinc_oos")
    parser.add_argument("--dataset-config", default="plus")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["banking", "credit_cards"],
        help="task sequence in training order, as declared in the protocol",
    )
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps-per-task", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-plasticity",
        type=float,
        action="append",
        default=None,
        help="E* family; repeatable. Defaults to the declared 0.75 and 0.50.",
    )
    parser.add_argument(
        "--permuted-budget",
        type=float,
        default=0.25,
        help=(
            "budget of the permuted control. Declared, not tuned: 0.25 is the "
            "budget the capacity criterion is evaluated at and is reachable in "
            "both families."
        ),
    )
    parser.add_argument(
        "--beta-policy",
        default="per_boundary",
        choices=("per_boundary", "first"),
        help="protocol A2 declares per_boundary; 'first' is for comparison only",
    )
    parser.add_argument("--capacity-scope", default="local", choices=("local", "global", "hierarchical"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="fp32", choices=("fp32", "fp16"))
    parser.add_argument(
        "--output", default="results/qwen_iso_plasticity/manifest.json"
    )
    args = parser.parse_args()

    unknown = [name for name in args.domains if name not in CLINC150_DOMAINS]
    if unknown:
        raise ValueError("domínio desconhecido: " + ", ".join(unknown))
    if len(set(args.domains)) != len(args.domains):
        raise ValueError("--domains não pode repetir domínios")
    if len(args.domains) < 2:
        raise ValueError("--domains exige ao menos dois domínios")

    targets = tuple(args.target_plasticity or ISO_TARGETS)
    device = torch.device(args.device)
    autocast_dtype = torch.float16 if args.precision == "fp16" else None
    if autocast_dtype is not None and device.type != "cuda":
        raise ValueError("fp16 exige --device cuda")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset(args.dataset, args.dataset_config)
    by_domain = {
        task.domain: task
        for task in build_clinc150_tasks(
            dataset, tokenizer, max_length=args.max_length, include_test=False
        )
    }
    tasks = [by_domain[name] for name in args.domains]

    protocol = {
        "model": args.model,
        "dataset": f"{args.dataset}:{args.dataset_config}",
        "tasks": list(args.domains),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "steps_per_task": args.steps_per_task,
        "learning_rate": args.learning_rate,
        "capacity_scope": args.capacity_scope,
        "beta_policy": args.beta_policy,
        "targets": list(targets),
        "budgets": list(ISO_BUDGETS),
        "permuted_budget": args.permuted_budget,
        "precision": args.precision,
        "freeze_unbound_parameters": True,
        "consolidation_strategy": "max",
        "evaluation": "class-il with seen-class logit masking",
    }

    families: dict[str, Any] = {}
    for target in targets:
        arms = build_arms(
            target_plasticity=target,
            budgets=ISO_BUDGETS,
            permuted_budget=args.permuted_budget,
        )
        records = []
        for arm in arms:
            print(f"[E*={target:g}] arm {arm.name} ({arm.kind}) ...", flush=True)
            record = run_arm(
                arm,
                tasks,
                model_name=args.model,
                scope=args.capacity_scope,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                steps_per_task=args.steps_per_task,
                seed=args.seed,
                device=device,
                pad_token_id=tokenizer.pad_token_id,
                beta_policy=args.beta_policy,
                autocast_dtype=autocast_dtype,
            )
            records.append(record)
            if record["discarded"]:
                print(f"    discarded: {record['reason']}", flush=True)
            else:
                endpoints = record["endpoints"]
                print(
                    f"    FAA {endpoints['final_average_accuracy']:.4f} | "
                    f"retention {endpoints['retention_first_task']:.4f} | "
                    f"forgetting {endpoints['mean_forgetting']:.4f}",
                    flush=True,
                )
        families[f"{target}"] = {
            "target_plasticity": target,
            "declared_arms": [asdict(arm) for arm in arms],
            "discarded_arms": [
                {"arm": record["arm"], "reason": record["reason"]}
                for record in records
                if record["discarded"]
            ],
            "arms": records,
        }

    manifest = {
        "protocol": protocol,
        "protocol_hash": protocol_hash(protocol),
        "protocol_document": "goals/protocol_iso_plasticity.md",
        "capacity_scope": args.capacity_scope,
        "seed": args.seed,
        "task_fingerprint": text_task_fingerprint(list(tasks)),
        "families": families,
    }
    path = Path(args.output)
    write_manifest(path, manifest)
    print(f"manifest written: {path}")


if __name__ == "__main__":
    main()
