# BERT Replay Memory-Efficiency Diagnostic Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a validation-only two-task diagnostic that compares replay against hard SlowHeat+replay at exactly 1, 5, and 10 stored examples per class, with the 1-example condition as the predeclared primary test of low-memory benefit.

**Architecture:** Build a separate six-condition runner on top of `run_split_clinc150()` so model initialization, data, task order, minibatches, replay selection, checkpoints, and telemetry remain shared. Because this is now the third BERT diagnostic, first extract the already-tested endpoint/statistics formatters into one small public helper module; keep experiment-specific conditions, contrasts, reports, and decision gates in their own runners. The new runner must compare equal replay memories within each budget and must not choose a “best” budget after inspecting results.

**Tech stack:** Python 3.12, PyTorch, Transformers, NumPy, pytest, Ruff, and existing atomic artifact/provenance helpers.

---

## Current context / assumptions

- The completed 20-example-per-class diagnostic is in `results/bert_slowheat_replay_diagnostic/`.
- At 20 examples/class, replay alone reached 91.22% final average accuracy and hard+replay reached 90.39%. The primary paired difference was negative in every seed: -1.17, -0.33, and -1.00 percentage points.
- Hard+replay retained T1 slightly better (+1.67 points) but reduced T2 acquisition by 3.33 points and took 2.21× as long. Therefore, do not run a ten-task hard+replay experiment at memory 20.
- The remaining hypothesis is narrower: hard SlowHeat may help when episodic memory is scarce.
- The predeclared replay budgets are exactly `(1, 5, 10)` examples per class. Do not add 20 to the new grid; it already has complete artifacts and is contextual evidence, not part of the new primary analysis.
- The primary endpoint is validation final average accuracy at `replay_per_class=1`.
- The primary contrast is `hard_replay_minus_replay_memory_1`.
- Conditions at budgets 5 and 10 are secondary/exploratory. Do not promote whichever one looks best to primary after the run.
- Use seeds 11, 22, and 33 because this remains a mechanism diagnostic. Any confirmatory ten-task run must use held-out seeds in a later plan.
- Every condition remains validation-only and loads only train/validation data.
- Replay selection remains the existing frozen selector in `experiments/split_clinc150.py:604-617`: the first `per_class` examples for each class. It is model-independent, so replay and hard+replay receive identical memories at a given seed/budget.
- Training defaults remain `batch_size=2`, `replay_batch_size=2`, `epochs_per_task=4`, `max_length=128`, and task order `banking → credit_cards`.
- Do not run GPU training, download data, install dependencies, commit, or push during implementation unless separately authorized. Lightweight CPU tests are allowed.
- The working tree is already dirty with the previous diagnostic implementation. Preserve all unrelated changes and stage files explicitly; never use `git add .`.

## Files likely to change

- Create: `experiments/bert_slowheat_diagnostic_common.py`
- Create: `tests/test_bert_slowheat_diagnostic_common.py`
- Modify: `experiments/bert_slowheat_diagnostic.py`
- Modify: `experiments/bert_slowheat_replay_diagnostic.py`
- Create: `experiments/bert_slowheat_replay_budget_diagnostic.py`
- Create: `tests/test_bert_slowheat_replay_budget_diagnostic.py`
- Modify: `README.md`

Do not modify `experiments/split_clinc150.py`; it already supports both required methods and arbitrary positive `replay_per_class` values.

---

## Task 1: Extract shared BERT diagnostic reporting helpers

**Objective:** Remove the third-runner duplication point without changing existing summaries or Markdown output.

**Files:**
- Create: `experiments/bert_slowheat_diagnostic_common.py`
- Create: `tests/test_bert_slowheat_diagnostic_common.py`
- Modify: `experiments/bert_slowheat_diagnostic.py:54-107,192-208`
- Modify: `experiments/bert_slowheat_replay_diagnostic.py:10-16`

**Step 1: Write the characterization test**

Create `tests/test_bert_slowheat_diagnostic_common.py`:

```python
import pytest

pytest.importorskip("torch")

from experiments.bert_slowheat_diagnostic_common import (  # noqa: E402
    condition_endpoints,
    format_mean_std,
    format_mean_std_scientific,
    summarize_values,
)


def _result():
    return {
        "validation_accuracy_matrix": [[0.8, None], [0.6, 0.9]],
        "validation_metrics": {
            "final_average_accuracy": 0.75,
            "average_forgetting": 0.2,
            "backward_transfer": -0.2,
            "forward_transfer": None,
            "per_task_forgetting": [0.2, 0.0],
        },
        "parameter_drift_history": [
            {
                "stage": 0,
                "protected_count": 0,
                "protected_rms": None,
                "protected_max_abs": None,
                "plastic_count": 0,
                "plastic_rms": None,
                "plastic_max_abs": None,
            },
            {
                "stage": 1,
                "protected_count": 4,
                "protected_rms": 8.0e-4,
                "protected_max_abs": 1.0e-3,
                "plastic_count": 8,
                "plastic_rms": 2.0e-3,
                "plastic_max_abs": 3.0e-3,
            },
        ],
        "elapsed_seconds": 10.0,
        "tokens_processed": 1_000,
        "replay_memory_bytes": 2**20,
        "peak_memory": {
            "peak_memory_bytes": 2 * 2**20,
            "peak_cuda_reserved_bytes": 3 * 2**20,
        },
    }


def test_common_helpers_preserve_existing_endpoint_and_format_contract():
    endpoints = condition_endpoints(_result())

    assert endpoints["final_average_accuracy"] == pytest.approx(0.75)
    assert endpoints["task1_retention"] == pytest.approx(0.6)
    assert endpoints["tokens_per_second"] == pytest.approx(100.0)
    assert endpoints["replay_memory_mib"] == pytest.approx(1.0)
    assert summarize_values([1.0, 3.0]) == {
        "n": 2,
        "mean": 2.0,
        "sample_std": pytest.approx(2**0.5),
    }
    assert format_mean_std(
        {"n": 1, "mean": 0.75, "sample_std": 0.0}, 100.0
    ) == "75.00 ± 0.00"
    assert format_mean_std_scientific(
        {"n": 1, "mean": 8.0e-4, "sample_std": 0.0}
    ) == "8.000e-04 ± 0.000e+00"
```

**Step 2: Verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_diagnostic_common.py -v
```

Expected: collection ERROR because `experiments.bert_slowheat_diagnostic_common` does not exist.

**Step 3: Create the complete common helper module**

Create `experiments/bert_slowheat_diagnostic_common.py`:

```python
"""Shared endpoint aggregation and formatting for BERT diagnostics."""

from __future__ import annotations

from statistics import mean, stdev
from typing import Any


def summarize_values(
    values: list[float | None],
) -> dict[str, float | int | None]:
    finite = [value for value in values if value is not None]
    return {
        "n": len(finite),
        "mean": mean(finite) if finite else None,
        "sample_std": stdev(finite) if len(finite) > 1 else (0.0 if finite else None),
    }


def condition_endpoints(result: dict[str, Any]) -> dict[str, float | None]:
    matrix = result["validation_accuracy_matrix"]
    drift = result["parameter_drift_history"][1]
    peak_reserved = result["peak_memory"]["peak_cuda_reserved_bytes"]
    validation_metrics = result["validation_metrics"]
    elapsed_seconds = float(result["elapsed_seconds"])
    tokens_processed = float(result["tokens_processed"])
    if elapsed_seconds <= 0.0:
        raise ValueError("elapsed_seconds deve ser positivo")
    return {
        "task1_acquisition": float(matrix[0][0]),
        "task1_retention": float(matrix[1][0]),
        "task1_forgetting": float(matrix[0][0] - matrix[1][0]),
        "task2_acquisition": float(matrix[1][1]),
        "final_average_accuracy": float(
            validation_metrics["final_average_accuracy"]
        ),
        "backward_transfer": float(validation_metrics["backward_transfer"]),
        "protected_rms_drift": (
            None if drift["protected_rms"] is None else float(drift["protected_rms"])
        ),
        "protected_max_abs_drift": (
            None
            if drift["protected_max_abs"] is None
            else float(drift["protected_max_abs"])
        ),
        "protected_count": float(drift["protected_count"]),
        "plastic_count": float(drift["plastic_count"]),
        "plastic_rms_drift": (
            None if drift["plastic_rms"] is None else float(drift["plastic_rms"])
        ),
        "plastic_max_abs_drift": (
            None
            if drift["plastic_max_abs"] is None
            else float(drift["plastic_max_abs"])
        ),
        "elapsed_seconds": elapsed_seconds,
        "tokens_processed": tokens_processed,
        "tokens_per_second": tokens_processed / elapsed_seconds,
        "replay_memory_mib": float(result["replay_memory_bytes"]) / 2**20,
        "peak_memory_mib": float(result["peak_memory"]["peak_memory_bytes"]) / 2**20,
        "peak_reserved_mib": (
            None if peak_reserved is None else float(peak_reserved) / 2**20
        ),
    }


def format_mean_std(
    summary: dict[str, float | int | None],
    scale: float = 1.0,
) -> str:
    mean_value = summary["mean"]
    std_value = summary["sample_std"]
    if mean_value is None or std_value is None:
        return "—"
    return (
        f"{float(mean_value) * scale:.2f} ± "
        f"{float(std_value) * scale:.2f}"
    )


def format_mean_std_scientific(
    summary: dict[str, float | int | None],
) -> str:
    mean_value = summary["mean"]
    std_value = summary["sample_std"]
    if mean_value is None or std_value is None:
        return "—"
    return f"{float(mean_value):.3e} ± {float(std_value):.3e}"
```

**Step 4: Replace definitions with imports in existing runners**

In `experiments/bert_slowheat_diagnostic.py`:

- remove `from statistics import mean, stdev`;
- delete `_summary`, `_condition_endpoints`, `_mean_std`, and `_mean_std_scientific`;
- add:

```python
from experiments.bert_slowheat_diagnostic_common import (
    condition_endpoints,
    format_mean_std,
    format_mean_std_scientific,
    summarize_values,
)
```

Replace names throughout that file:

```text
_summary                    -> summarize_values
_condition_endpoints        -> condition_endpoints
_mean_std                   -> format_mean_std
_mean_std_scientific        -> format_mean_std_scientific
```

In `experiments/bert_slowheat_replay_diagnostic.py`, replace the imports from `bert_slowheat_diagnostic` with the same public imports from `bert_slowheat_diagnostic_common`, then apply the same four name replacements.

**Step 5: Verify GREEN and no behavior regressions**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_diagnostic_common.py \
  tests/test_bert_slowheat_diagnostic.py \
  tests/test_bert_slowheat_replay_diagnostic.py -q
```

Expected: all selected tests PASS. Existing rendered summaries/tables must remain byte-equivalent except that zero/negative elapsed input now raises a clear error.

**Step 6: Commit**

```bash
git add \
  experiments/bert_slowheat_diagnostic_common.py \
  experiments/bert_slowheat_diagnostic.py \
  experiments/bert_slowheat_replay_diagnostic.py \
  tests/test_bert_slowheat_diagnostic_common.py
git commit -m "refactor: share BERT diagnostic reporting helpers"
```

---

## Task 2: Define the exact six-condition memory grid

**Objective:** Encode the predeclared treatment/budget matrix without a user-extensible budget flag.

**Files:**
- Create: `experiments/bert_slowheat_replay_budget_diagnostic.py`
- Create: `tests/test_bert_slowheat_replay_budget_diagnostic.py`

**Step 1: Write the failing matrix test**

Create `tests/test_bert_slowheat_replay_budget_diagnostic.py`:

```python
import pytest

pytest.importorskip("transformers")

from experiments import bert_slowheat_replay_budget_diagnostic as budget  # noqa: E402


def test_budget_diagnostic_has_exactly_the_predeclared_six_conditions():
    conditions = budget.replay_budget_conditions()

    assert [(item.name, item.method, item.replay_per_class) for item in conditions] == [
        ("replay_memory_1", "replay", 1),
        ("slowheat_hard_replay_memory_1", "slowheat_ffn_attention_replay", 1),
        ("replay_memory_5", "replay", 5),
        ("slowheat_hard_replay_memory_5", "slowheat_ffn_attention_replay", 5),
        ("replay_memory_10", "replay", 10),
        ("slowheat_hard_replay_memory_10", "slowheat_ffn_attention_replay", 10),
    ]
    assert [item.mask_mode for item in conditions] == [
        "soft", "hard", "soft", "hard", "soft", "hard"
    ]
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_replay_budget_diagnostic.py::test_budget_diagnostic_has_exactly_the_predeclared_six_conditions -v
```

Expected: collection ERROR because the module does not exist.

**Step 3: Create the module and matrix**

Create `experiments/bert_slowheat_replay_budget_diagnostic.py` with:

```python
"""Low-memory replay diagnostic for BERT SlowHeat on two CLINC150 tasks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from experiments.artifacts import write_json_atomic
from experiments.bert_slowheat_diagnostic_common import (
    condition_endpoints,
    format_mean_std,
    format_mean_std_scientific,
    summarize_values,
)
from experiments.provenance import write_environment_manifest
from experiments.split_clinc150 import (
    PlasticityMaskMode,
    SplitCLINC150Config,
    load_clinc150_tasks,
    run_split_clinc150,
)

REPLAY_BUDGETS = (1, 5, 10)
PRIMARY_BUDGET = 1


@dataclass(frozen=True)
class ReplayBudgetCondition:
    name: str
    method: str
    mask_mode: PlasticityMaskMode
    replay_per_class: int


def replay_budget_conditions() -> tuple[ReplayBudgetCondition, ...]:
    conditions: list[ReplayBudgetCondition] = []
    for replay_per_class in REPLAY_BUDGETS:
        conditions.extend(
            [
                ReplayBudgetCondition(
                    f"replay_memory_{replay_per_class}",
                    "replay",
                    "soft",
                    replay_per_class,
                ),
                ReplayBudgetCondition(
                    f"slowheat_hard_replay_memory_{replay_per_class}",
                    "slowheat_ffn_attention_replay",
                    "hard",
                    replay_per_class,
                ),
            ]
        )
    return tuple(conditions)
```

**Step 4: Verify GREEN and commit**

Run the command from Step 2.

Expected: PASS.

```bash
git add \
  experiments/bert_slowheat_replay_budget_diagnostic.py \
  tests/test_bert_slowheat_replay_budget_diagnostic.py
git commit -m "feat: define BERT replay memory grid"
```

---

## Task 3: Orchestrate paired runs for every seed and budget

**Objective:** Run each condition with the correct fixed replay budget while preserving all other protocol fields.

**Files:**
- Modify: `experiments/bert_slowheat_replay_budget_diagnostic.py`
- Test: `tests/test_bert_slowheat_replay_budget_diagnostic.py`

**Step 1: Add a complete fake result helper**

Append to the test file:

```python
from experiments.split_clinc150 import SplitCLINC150Config  # noqa: E402


def _fake_result(*, replay_memory_bytes: int = 1_000):
    return {
        "validation_accuracy_matrix": [[0.8, None], [0.6, 0.9]],
        "validation_metrics": {
            "final_average_accuracy": 0.75,
            "average_forgetting": 0.2,
            "backward_transfer": -0.2,
            "forward_transfer": None,
            "per_task_forgetting": [0.2, 0.0],
        },
        "parameter_drift_history": [
            {
                "stage": 0,
                "protected_count": 0,
                "protected_rms": None,
                "protected_max_abs": None,
                "plastic_count": 0,
                "plastic_rms": None,
                "plastic_max_abs": None,
            },
            {
                "stage": 1,
                "protected_count": 4,
                "protected_rms": 0.0,
                "protected_max_abs": 0.0,
                "plastic_count": 8,
                "plastic_rms": 0.1,
                "plastic_max_abs": 0.2,
            },
        ],
        "elapsed_seconds": 10.0,
        "tokens_processed": 1_000,
        "replay_memory_bytes": replay_memory_bytes,
        "peak_memory": {
            "peak_memory_bytes": 2**20,
            "peak_cuda_reserved_bytes": 2 * 2**20,
        },
    }
```

**Step 2: Write the failing orchestration test**

```python
def test_run_budget_diagnostic_builds_paired_validation_only_configs(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run(config, tasks, **kwargs):
        calls.append((config, kwargs["output_dir"]))
        return {
            config.methods[0]: _fake_result(
                replay_memory_bytes=config.replay_per_class * 1_000
            )
        }

    monkeypatch.setattr(budget, "run_split_clinc150", fake_run)
    monkeypatch.setattr(budget, "write_environment_manifest", lambda *a, **k: None)

    raw = budget.run_replay_budget_diagnostic(
        SplitCLINC150Config(device="cpu"),
        tasks=[object()] * 10,
        metadata={},
        seeds=[11, 22],
        output_dir=tmp_path,
    )

    assert len(calls) == 12
    assert set(raw) == {11, 22}
    assert [config.replay_per_class for config, _ in calls[:6]] == [1, 1, 5, 5, 10, 10]
    assert all(config.task_limit == 2 for config, _ in calls)
    assert all(config.replay_batch_size == 2 for config, _ in calls)
    assert all(config.evaluate_test is False for config, _ in calls)
    assert calls[0][1] == tmp_path / "seed_11" / "replay_memory_1"
    assert calls[-1][1] == (
        tmp_path / "seed_22" / "slowheat_hard_replay_memory_10"
    )
```

**Step 3: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_replay_budget_diagnostic.py::test_run_budget_diagnostic_builds_paired_validation_only_configs -v
```

Expected: FAIL because `run_replay_budget_diagnostic` does not exist.

**Step 4: Implement orchestration**

Append to the production module:

```python
def run_replay_budget_diagnostic(
    base_config: SplitCLINC150Config,
    tasks,
    *,
    metadata: dict[str, Any],
    seeds: list[int],
    output_dir: str | Path,
    resume: bool = False,
    telemetry: bool = False,
    telemetry_every: int = 10,
) -> dict[int, dict[str, dict[str, Any]]]:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds deve ser não vazio e sem duplicatas")
    if base_config.evaluate_test:
        raise ValueError("diagnóstico deve permanecer validation-only")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_environment_manifest(
        destination,
        project_root=Path(__file__).resolve().parents[1],
    )
    raw: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        raw[seed] = {}
        for condition in replay_budget_conditions():
            config = replace(
                base_config,
                seed=seed,
                methods=(condition.method,),
                plasticity_mask_mode=condition.mask_mode,
                replay_per_class=condition.replay_per_class,
                replay_batch_size=2,
                task_limit=2,
                evaluate_test=False,
            )
            result = run_split_clinc150(
                config,
                tasks,
                metadata=metadata,
                output_dir=destination / f"seed_{seed}" / condition.name,
                resume=resume,
                telemetry=telemetry,
                telemetry_every=telemetry_every,
            )[condition.method]
            raw[seed][condition.name] = result
    return raw
```

Do not write reports yet; Task 4 first tests memory-pair validation and contrast identity.

**Step 5: Verify GREEN and commit**

Run the command from Step 3.

Expected: PASS.

```bash
git add \
  experiments/bert_slowheat_replay_budget_diagnostic.py \
  tests/test_bert_slowheat_replay_budget_diagnostic.py
git commit -m "feat: orchestrate replay memory diagnostic"
```

---

## Task 4: Validate equal replay memory and compute predeclared contrasts

**Objective:** Refuse scientifically invalid unequal-memory pairs and aggregate one paired contrast per fixed budget.

**Files:**
- Modify: `experiments/bert_slowheat_replay_budget_diagnostic.py`
- Test: `tests/test_bert_slowheat_replay_budget_diagnostic.py`

**Step 1: Write the failing summary test**

Append:

```python
def test_budget_summary_declares_primary_budget_and_three_fixed_contrasts():
    raw = {}
    for seed in (11, 22):
        raw[seed] = {}
        for condition in budget.replay_budget_conditions():
            raw[seed][condition.name] = _fake_result(
                replay_memory_bytes=condition.replay_per_class * 1_000
            )

    summary = budget.summarize_replay_budget_diagnostic(raw)

    assert summary["primary_endpoint"] == "final_average_accuracy"
    assert summary["primary_budget_per_class"] == 1
    assert summary["primary_contrast"] == "hard_replay_minus_replay_memory_1"
    assert set(summary["paired_contrasts"]) == {
        "hard_replay_minus_replay_memory_1",
        "hard_replay_minus_replay_memory_5",
        "hard_replay_minus_replay_memory_10",
    }
    assert summary["paired_contrasts"][
        "hard_replay_minus_replay_memory_1"
    ]["final_average_accuracy"]["by_seed"] == {"11": 0.0, "22": 0.0}
```

**Step 2: Write the unequal-memory rejection test**

```python
def test_budget_summary_rejects_unequal_memory_within_a_pair():
    raw = {
        11: {
            condition.name: _fake_result(
                replay_memory_bytes=condition.replay_per_class * 1_000
            )
            for condition in budget.replay_budget_conditions()
        }
    }
    raw[11]["slowheat_hard_replay_memory_1"]["replay_memory_bytes"] += 1

    with pytest.raises(ValueError, match="memória de replay desigual"):
        budget.summarize_replay_budget_diagnostic(raw)
```

**Step 3: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_bert_slowheat_replay_budget_diagnostic.py \
  -k 'budget_summary' -v
```

Expected: both tests FAIL because the summary function does not exist.

**Step 4: Implement the complete summary**

Append to the production module:

```python
def summarize_replay_budget_diagnostic(
    raw: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not raw:
        raise ValueError("resultados diagnósticos não podem ser vazios")
    seeds = sorted(raw)
    names = [condition.name for condition in replay_budget_conditions()]
    for seed in seeds:
        if set(raw[seed]) != set(names):
            raise ValueError(f"seed {seed} não contém as seis condições")
        for replay_per_class in REPLAY_BUDGETS:
            replay_name = f"replay_memory_{replay_per_class}"
            hard_name = f"slowheat_hard_replay_memory_{replay_per_class}"
            replay_bytes = int(raw[seed][replay_name]["replay_memory_bytes"])
            hard_bytes = int(raw[seed][hard_name]["replay_memory_bytes"])
            if replay_bytes != hard_bytes:
                raise ValueError(
                    "memória de replay desigual para "
                    f"seed={seed}, replay_per_class={replay_per_class}: "
                    f"{replay_bytes} != {hard_bytes}"
                )

    endpoints = {
        seed: {
            name: condition_endpoints(raw[seed][name])
            for name in names
        }
        for seed in seeds
    }
    metric_names = tuple(endpoints[seeds[0]][names[0]])
    conditions = {
        name: {
            metric: summarize_values(
                [endpoints[seed][name][metric] for seed in seeds]
            )
            for metric in metric_names
        }
        for name in names
    }
    paired_contrasts: dict[str, dict[str, dict[str, Any]]] = {}
    for replay_per_class in REPLAY_BUDGETS:
        replay_name = f"replay_memory_{replay_per_class}"
        hard_name = f"slowheat_hard_replay_memory_{replay_per_class}"
        contrast_name = f"hard_replay_minus_replay_memory_{replay_per_class}"
        paired_contrasts[contrast_name] = {}
        for metric in metric_names:
            by_seed: dict[str, float | None] = {}
            for seed in seeds:
                left = endpoints[seed][hard_name][metric]
                right = endpoints[seed][replay_name][metric]
                by_seed[str(seed)] = (
                    None if left is None or right is None else left - right
                )
            paired_contrasts[contrast_name][metric] = {
                **summarize_values(list(by_seed.values())),
                "by_seed": by_seed,
            }
    return {
        "schema_version": 1,
        "endpoint_source": "validation",
        "primary_endpoint": "final_average_accuracy",
        "primary_budget_per_class": PRIMARY_BUDGET,
        "primary_contrast": "hard_replay_minus_replay_memory_1",
        "secondary_budgets_per_class": [5, 10],
        "task_count": 2,
        "seeds": seeds,
        "conditions": conditions,
        "raw_by_seed": {str(seed): endpoints[seed] for seed in seeds},
        "paired_contrasts": paired_contrasts,
    }
```

**Step 5: Verify GREEN and commit**

Run the command from Step 3.

Expected: both tests PASS.

```bash
git add \
  experiments/bert_slowheat_replay_budget_diagnostic.py \
  tests/test_bert_slowheat_replay_budget_diagnostic.py
git commit -m "feat: aggregate paired replay memory contrasts"
```

---

## Task 5: Write the memory-efficiency report and explicit gate

**Objective:** Present final accuracy and resource costs per memory budget without hiding the T1/T2 trade-off.

**Files:**
- Modify: `experiments/bert_slowheat_replay_budget_diagnostic.py`
- Test: `tests/test_bert_slowheat_replay_budget_diagnostic.py`

**Step 1: Write the failing Markdown test**

Append:

```python
def test_budget_markdown_labels_primary_and_secondary_budgets():
    raw = {
        11: {
            condition.name: _fake_result(
                replay_memory_bytes=condition.replay_per_class * 1_000
            )
            for condition in budget.replay_budget_conditions()
        }
    }

    rendered = budget.replay_budget_markdown(
        budget.summarize_replay_budget_diagnostic(raw)
    )

    assert "Primary budget: 1 replay example per class" in rendered
    assert "Secondary budgets: 5 and 10" in rendered
    assert "Final average (%)" in rendered
    assert "Actual replay memory (MiB)" in rendered
    assert "hard_replay_minus_replay_memory_1" in rendered
    assert "Do not select the best-looking secondary budget" in rendered
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_replay_budget_diagnostic.py::test_budget_markdown_labels_primary_and_secondary_budgets -v
```

Expected: FAIL because `replay_budget_markdown` does not exist.

**Step 3: Implement the report**

Append:

```python
def replay_budget_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BERT SlowHeat low-memory replay diagnostic",
        "",
        "Validation means ± sample standard deviation across paired seeds.",
        "Primary budget: 1 replay example per class.",
        "Secondary budgets: 5 and 10 replay examples per class.",
        "Primary endpoint: final average accuracy.",
        "",
        "| Condition | Examples/class | Final average (%) | T1 retention (%) | "
        "T1 forgetting (%) | T2 acquisition (%) | Actual replay memory (MiB) | "
        "Protected RMS drift | Tokens | Tokens/s | Time (s) | "
        "Peak allocated (MiB) | Peak reserved (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in replay_budget_conditions():
        item = summary["conditions"][condition.name]
        lines.append(
            f"| {condition.name} | {condition.replay_per_class} | "
            f"{format_mean_std(item['final_average_accuracy'], 100.0)} | "
            f"{format_mean_std(item['task1_retention'], 100.0)} | "
            f"{format_mean_std(item['task1_forgetting'], 100.0)} | "
            f"{format_mean_std(item['task2_acquisition'], 100.0)} | "
            f"{format_mean_std(item['replay_memory_mib'])} | "
            f"{format_mean_std_scientific(item['protected_rms_drift'])} | "
            f"{format_mean_std(item['tokens_processed'])} | "
            f"{format_mean_std(item['tokens_per_second'])} | "
            f"{format_mean_std(item['elapsed_seconds'])} | "
            f"{format_mean_std(item['peak_memory_mib'])} | "
            f"{format_mean_std(item['peak_reserved_mib'])} |"
        )
    lines.extend(
        [
            "",
            "Decision gate:",
            "",
            "- Primary contrast: hard_replay_minus_replay_memory_1 on final average accuracy.",
            "- Require a positive paired difference in every seed.",
            "- Require mean T2 acquisition loss no worse than 2 percentage points.",
            "- Require exactly zero protected RMS and max drift for hard+replay.",
            "- Budgets 5 and 10 are secondary diagnostics only.",
            "- Do not select the best-looking secondary budget as a confirmatory result.",
            "",
        ]
    )
    return "\n".join(lines)
```

At the end of `run_replay_budget_diagnostic()`, before `return raw`, add:

```python
    summary = summarize_replay_budget_diagnostic(raw)
    write_json_atomic(destination / "diagnostic_summary.json", summary)
    (destination / "diagnostic_table.md").write_text(
        replay_budget_markdown(summary), encoding="utf-8"
    )
```

**Step 4: Verify GREEN and report artifacts**

Run the command from Step 2, then extend the orchestration test with:

```python
    assert (tmp_path / "diagnostic_summary.json").is_file()
    assert (tmp_path / "diagnostic_table.md").is_file()
```

Run:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_bert_slowheat_replay_budget_diagnostic.py -q
```

Expected: all tests in the file PASS.

**Step 5: Commit**

```bash
git add \
  experiments/bert_slowheat_replay_budget_diagnostic.py \
  tests/test_bert_slowheat_replay_budget_diagnostic.py
git commit -m "feat: report replay memory efficiency"
```

---

## Task 6: Add a frozen validation-only CLI

**Objective:** Expose the predeclared experiment without permitting test evaluation or post-hoc budget changes.

**Files:**
- Modify: `experiments/bert_slowheat_replay_budget_diagnostic.py`
- Test: `tests/test_bert_slowheat_replay_budget_diagnostic.py`

**Step 1: Write failing CLI tests**

Append:

```python
def test_budget_cli_defaults_are_frozen():
    args = budget.build_parser().parse_args([])

    assert args.seeds == [11, 22, 33]
    assert args.batch_size == 2
    assert args.replay_batch_size == 2
    assert args.task_limit == 2
    assert args.evaluate_test is False
    assert not hasattr(args, "replay_per_class")


def test_budget_cli_rejects_protocol_escape_hatches():
    parser = budget.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--evaluate-test"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--replay-per-class", "20"])
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_bert_slowheat_replay_budget_diagnostic.py \
  -k 'budget_cli' -v
```

Expected: FAIL because `build_parser` does not exist.

**Step 3: Implement the parser and main**

Append:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/bert_slowheat_replay_budget_diagnostic",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--replay-batch-size", type=int, choices=[2], default=2)
    parser.add_argument("--epochs-per-task", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--task-limit", type=int, choices=[2], default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--telemetry-every", type=int, default=10)
    parser.set_defaults(evaluate_test=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SplitCLINC150Config(
        device=args.device,
        batch_size=args.batch_size,
        replay_batch_size=args.replay_batch_size,
        epochs_per_task=args.epochs_per_task,
        max_length=args.max_length,
        task_limit=args.task_limit,
        evaluate_test=False,
    )
    tasks, metadata = load_clinc150_tasks(config, include_test=False)
    run_replay_budget_diagnostic(
        config,
        tasks,
        metadata=metadata,
        seeds=args.seeds,
        output_dir=args.output_dir,
        resume=args.resume,
        telemetry=args.telemetry,
        telemetry_every=args.telemetry_every,
    )


if __name__ == "__main__":
    main()
```

**Step 4: Add a test proving the test split remains closed**

```python
def test_budget_main_loads_only_train_and_validation(monkeypatch, tmp_path):
    received = []
    monkeypatch.setattr(
        budget,
        "load_clinc150_tasks",
        lambda config, *, include_test: (
            received.append(include_test) or ([object()] * 10, {})
        ),
    )
    monkeypatch.setattr(
        budget,
        "run_replay_budget_diagnostic",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["bert_slowheat_replay_budget_diagnostic", "--output-dir", str(tmp_path)],
    )

    budget.main()

    assert received == [False]
```

**Step 5: Verify GREEN and commit**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_bert_slowheat_replay_budget_diagnostic.py -q
python3 -m experiments.bert_slowheat_replay_budget_diagnostic --help
```

Expected: all tests PASS; help does not list `--evaluate-test` or `--replay-per-class`.

```bash
git add \
  experiments/bert_slowheat_replay_budget_diagnostic.py \
  tests/test_bert_slowheat_replay_budget_diagnostic.py
git commit -m "feat: add frozen replay memory CLI"
```

---

## Task 7: Document the exact command and hypothesis

**Objective:** Make the low-memory protocol discoverable while preventing reinterpretation of secondary budgets.

**Files:**
- Modify: `README.md:274-290`

**Step 1: Add this subsection after “Replay interaction follow-up”**

```markdown
#### Low-memory replay efficiency

The 20-example-per-class follow-up favored replay alone. The next diagnostic
asks whether learned hard protection helps when episodic memory is scarce:

```bash
CUDA_VISIBLE_DEVICES=2 python3 -m experiments.bert_slowheat_replay_budget_diagnostic \
  --device cuda --batch-size 2 --replay-batch-size 2 \
  --seeds 11 22 33 --telemetry --telemetry-every 10 \
  --output-dir results/bert_slowheat_replay_budget_diagnostic
```

The runner executes replay and hard SlowHeat+replay at exactly 1, 5, and 10
stored examples per class. The primary endpoint is final average validation
accuracy at one example per class. Budgets 5 and 10 are secondary diagnostics;
they must not be searched post hoc for a favorable result.
```

Use an outer four-backtick fence while editing so the nested shell block remains valid Markdown.

**Step 2: Verify documentation references real flags**

```bash
python3 -m experiments.bert_slowheat_replay_budget_diagnostic --help
```

Expected: exit code 0; output includes `--device`, `--batch-size`, `--replay-batch-size`, `--seeds`, `--telemetry`, `--telemetry-every`, `--resume`, and `--output-dir`; it excludes budget/test escape hatches.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add low-memory replay protocol"
```

---

## Task 8: Run complete lightweight validation

**Objective:** Verify the new runner and shared-helper refactor without starting CUDA training.

**Files:** No production changes unless a test exposes a defect; every defect fix begins with a focused failing test.

**Step 1: Run focused diagnostic tests**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_diagnostic_common.py \
  tests/test_bert_slowheat_diagnostic.py \
  tests/test_bert_slowheat_replay_diagnostic.py \
  tests/test_bert_slowheat_replay_budget_diagnostic.py -q
```

Expected: all selected tests PASS.

**Step 2: Run CLINC runner regressions**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_split_clinc150.py -q
```

Expected: all CLINC150 tests PASS, including exact zero hard-protected drift and resume behavior.

**Step 3: Run optimizer/BERT regressions**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_optim.py tests/test_slow_heat_bert.py -q
```

Expected: all selected tests PASS.

**Step 4: Run the complete suite**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest -q
```

Expected: all tests PASS.

**Step 5: Run Ruff on touched Python files**

```bash
python3 -m ruff check \
  experiments/bert_slowheat_diagnostic_common.py \
  experiments/bert_slowheat_diagnostic.py \
  experiments/bert_slowheat_replay_diagnostic.py \
  experiments/bert_slowheat_replay_budget_diagnostic.py \
  tests/test_bert_slowheat_diagnostic_common.py \
  tests/test_bert_slowheat_diagnostic.py \
  tests/test_bert_slowheat_replay_diagnostic.py \
  tests/test_bert_slowheat_replay_budget_diagnostic.py
```

Expected: `All checks passed!`

**Step 6: Inspect final scope**

```bash
git diff --check
git status --short
git diff --stat
```

Expected: `git diff --check` emits nothing. Confirm there are no changes under `results/` and no raw artifacts were touched.

**Step 7: Final implementation commit only if uncommitted fixes remain**

```bash
git add \
  experiments/bert_slowheat_diagnostic_common.py \
  experiments/bert_slowheat_diagnostic.py \
  experiments/bert_slowheat_replay_diagnostic.py \
  experiments/bert_slowheat_replay_budget_diagnostic.py \
  tests/test_bert_slowheat_diagnostic_common.py \
  tests/test_bert_slowheat_diagnostic.py \
  tests/test_bert_slowheat_replay_diagnostic.py \
  tests/test_bert_slowheat_replay_budget_diagnostic.py \
  README.md
git commit -m "test: verify replay memory diagnostic"
```

Do not execute the CUDA benchmark in the implementation session.

---

## Task 9: Hand off the CUDA benchmark and apply the predeclared gate

**Objective:** Produce the new evidence only after implementation checks pass and explicit GPU authorization is given.

**Files:** No code changes.

**Step 1: Give this exact command to the user; do not run it automatically**

```bash
CUDA_VISIBLE_DEVICES=2 python3 -m experiments.bert_slowheat_replay_budget_diagnostic \
  --device cuda --batch-size 2 --replay-batch-size 2 \
  --seeds 11 22 33 --telemetry --telemetry-every 10 \
  --output-dir results/bert_slowheat_replay_budget_diagnostic
```

Expected layout:

```text
results/bert_slowheat_replay_budget_diagnostic/
├── environment.json
├── diagnostic_summary.json
├── diagnostic_table.md
├── seed_11/
│   ├── replay_memory_1/replay/{checkpoint.pt,results.json}
│   ├── slowheat_hard_replay_memory_1/slowheat_ffn_attention_replay/{checkpoint.pt,results.json}
│   ├── replay_memory_5/replay/{checkpoint.pt,results.json}
│   ├── slowheat_hard_replay_memory_5/slowheat_ffn_attention_replay/{checkpoint.pt,results.json}
│   ├── replay_memory_10/replay/{checkpoint.pt,results.json}
│   └── slowheat_hard_replay_memory_10/slowheat_ffn_attention_replay/{checkpoint.pt,results.json}
├── seed_22/...
└── seed_33/...
```

**Step 2: Verify artifacts after the user runs it**

Use read-only checks to assert:

- 18/18 `results.json` files exist;
- no JSON contains NaN or infinity;
- all telemetry streams end with `method_end`, `run_end`, and `session_end`;
- all protocols share data/source hashes, model commit, and task order;
- replay and hard+replay have exactly equal `replay_memory_bytes` within every `(seed, budget)` pair;
- replay memory grows strictly from budget 1 to 5 to 10;
- hard+replay protected RMS and max drift are exactly zero in every run;
- all per-seed primary differences are present in `diagnostic_summary.json`.

**Step 3: Apply the predeclared primary gate**

Primary contrast: `hard_replay_minus_replay_memory_1`.

Proceed to a separate confirmatory plan only if all conditions hold:

1. final average accuracy difference is positive in every seed at budget 1;
2. mean T2 acquisition loss at budget 1 is no worse than 2 percentage points;
3. protected RMS and max drift are exactly zero;
4. replay memories are byte-identical within each pair;
5. absolute allocated/reserved memory, elapsed time, tokens, and throughput are reported.

Budgets 5 and 10 explain the memory-response curve but cannot rescue a failed primary gate. If budget 1 fails, conclude that hard SlowHeat does not provide demonstrated replay-memory efficiency in this two-task protocol and stop expanding this mechanism on CLINC150. If budget 1 passes, use held-out seeds and a separately frozen budget in the next full-sequence plan.

---

## Tests / validation acceptance criteria

Implementation is complete only when:

1. Shared endpoint/statistics formatting exists in one module and both previous runners retain their output contract.
2. The new matrix contains exactly six conditions: two treatments at budgets 1, 5, and 10.
3. The CLI exposes no `--replay-per-class` or `--evaluate-test` escape hatch.
4. Every condition uses task limit 2, replay batch size 2, and validation-only data.
5. Summary generation rejects unequal replay bytes within a paired comparison.
6. The primary endpoint, primary budget, primary contrast, and secondary budgets are stored explicitly in JSON.
7. Reports include final accuracy, retention, forgetting, T2 acquisition, actual replay memory, protected drift, tokens, throughput, time, allocated memory, and reserved memory.
8. The report forbids post-hoc selection among budgets 5 and 10.
9. Focused tests, CLINC regressions, optimizer/BERT regressions, full pytest, Ruff, and `git diff --check` pass.
10. No GPU training, dataset download, dependency installation, result mutation, commit, or push occurs without explicit authorization during execution.

## Risks, tradeoffs, and open questions

- **Primary gate may fail despite secondary wins:** That is the intended protection against budget fishing. Report secondary curves, but do not reinterpret them as confirmatory.
- **Very small memory is noisy:** One example per class may depend strongly on example order. The selector is deliberately frozen and common; changing it after results would alter the treatment.
- **Replay batch composition:** `replay_batch_size=2` stays constant while memory capacity changes. This isolates storage capacity, not replay sampling intensity.
- **Memory and compute are separate:** Actual replay storage increases with budget, while per-step replay batch size remains fixed. Report stored bytes, tokens, time, and throughput together.
- **Three seeds are diagnostic:** Sign consistency is a mechanism gate, not publication-grade inference.
- **Two tasks are not scalability evidence:** A positive low-memory result only justifies a held-out full-sequence plan; it does not establish ten-task efficacy.
- **Source provenance remains dirty:** Existing runs record `git.dirty=true`. New runs must retain environment/source hashes; do not describe commit ID alone as sufficient provenance.
- **Common-helper refactor:** This is justified because three runners now share the same endpoint contract. Keep the module limited to reporting primitives; do not move experiment-specific conditions or gates into it.
