# BERT SlowHeat Mechanism Diagnostic Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a reproducible two-task CLINC150 diagnostic that compares vanilla, soft SlowHeat FFN+attention at β ∈ {3, 10, 30}, learned hard protection, and equal-coverage random hard protection while reporting retention, acquisition, parameter drift, runtime, and total peak VRAM.

**Architecture:** Extend the existing `experiments/split_clinc150.py` protocol with two orthogonal controls: a task prefix limit and a plasticity-mask mode (`soft`, `hard`, or `random_hard`). Keep the core trainer responsible for mask registration, deterministic random-mask assignment, drift telemetry, checkpoint/resume, and raw results; put the six-condition experiment matrix and its aggregate report in a dedicated `experiments/bert_slowheat_diagnostic.py` runner. Reuse `SlowHeatAdamW` and BERT mask bindings rather than adding another optimizer or duplicating training logic.

**Tech stack:** Python 3.12, PyTorch, Transformers, NumPy, pytest, existing atomic artifact helpers.

---

## Current context / assumptions

- The relevant implementation is `experiments/split_clinc150.py`; BERT mask bindings already support `hard=True` in `src/dual_heater/bert.py`.
- `SlowHeatAdamW` masks the final AdamW update, including decoupled weight decay, so no optimizer rewrite is needed.
- `slowheat_ffn_attention` tracks only FFN units and attention heads. Embeddings, residual paths, LayerNorm, pooler, and classifier remain unprotected in every SlowHeat diagnostic condition.
- The diagnostic remains validation-only (`evaluate_test=False`) and must never read the held-out test split.
- The input task list remains the full official ten-domain CLINC150 protocol. `task_limit=2` selects the first two tasks only after validating the full dataset and order.
- Random hard protection preserves the learned method’s per-state protected count but replaces the selected unit identities with a deterministic random sample after each task boundary. It is re-sampled at each boundary from a local CPU generator and does not perturb training RNG state.
- Parameter drift is measured from immediately before training task k to immediately after training task k and before the next consolidation. Stage 0 has no previously protected parameters and therefore reports null drift values.
- Do not run GPU training, download datasets, or install dependencies during implementation. Only lightweight CPU tests are approved. The final benchmark command is prepared for the user to run.

## Files to change

- Modify: `experiments/split_clinc150.py`
- Create: `experiments/bert_slowheat_diagnostic.py`
- Modify: `tests/test_split_clinc150.py`
- Create: `tests/test_bert_slowheat_diagnostic.py`
- Modify: `README.md`

Do not modify `src/dual_heater/optim.py` or `src/dual_heater/bert.py`; the required hard-mask behavior already exists.

---

## Task 1: Add validated protocol controls

**Objective:** Represent task-prefix and mask-mode choices explicitly in `SplitCLINC150Config` so they enter protocol identity and checkpoint compatibility automatically.

**Files:**
- Modify: `experiments/split_clinc150.py:24,216-306`
- Test: `tests/test_split_clinc150.py`

**Step 1: Write the failing tests**

Append to `tests/test_split_clinc150.py`:

```python
@pytest.mark.parametrize("mode", ["soft", "hard", "random_hard"])
def test_clinc_config_accepts_diagnostic_mask_modes(mode):
    SplitCLINC150Config(plasticity_mask_mode=mode).validate()


def test_clinc_config_rejects_unknown_diagnostic_mask_mode():
    with pytest.raises(ValueError, match="plasticity_mask_mode"):
        SplitCLINC150Config(plasticity_mask_mode="unknown").validate()


@pytest.mark.parametrize("task_limit", [2, 10, None])
def test_clinc_config_accepts_valid_task_limits(task_limit):
    SplitCLINC150Config(task_limit=task_limit).validate()


@pytest.mark.parametrize("task_limit", [0, 1, 11, True])
def test_clinc_config_rejects_invalid_task_limits(task_limit):
    with pytest.raises((TypeError, ValueError), match="task_limit"):
        SplitCLINC150Config(task_limit=task_limit).validate()
```

**Step 2: Verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_split_clinc150.py \
  -k 'diagnostic_mask_modes or task_limits' -v
```

Expected: FAIL because `SplitCLINC150Config` does not accept `plasticity_mask_mode` or `task_limit`.

**Step 3: Implement the minimal config fields and validation**

In `experiments/split_clinc150.py`, extend the typing import and add an alias:

```python
from typing import Any, Literal

PlasticityMaskMode = Literal["soft", "hard", "random_hard"]
```

Add these fields to `SplitCLINC150Config` immediately after `scheduler_scope`:

```python
    plasticity_mask_mode: PlasticityMaskMode = "soft"
    task_limit: int | None = None
```

Add this validation immediately after the existing scheduler validation:

```python
        if self.plasticity_mask_mode not in ("soft", "hard", "random_hard"):
            raise ValueError(
                "plasticity_mask_mode deve ser 'soft', 'hard' ou 'random_hard'"
            )
        if self.task_limit is not None:
            if not isinstance(self.task_limit, int) or isinstance(self.task_limit, bool):
                raise TypeError("task_limit deve ser inteiro ou None")
            if not 2 <= self.task_limit <= len(CLINC150_DOMAINS):
                raise ValueError("task_limit deve estar entre 2 e 10")
```

**Step 4: Verify GREEN**

Run the command from Step 2.

Expected: `7 passed` and no failures.

**Step 5: Commit**

```bash
git add experiments/split_clinc150.py tests/test_split_clinc150.py
git commit -m "feat: add CLINC diagnostic protocol controls"
```

---

## Task 2: Register soft and hard masks from the protocol

**Objective:** Make `plasticity_mask_mode` select existing soft or hard BERT optimizer masks without changing model coverage.

**Files:**
- Modify: `experiments/split_clinc150.py:775-804`
- Test: `tests/test_split_clinc150.py`

**Step 1: Write the failing test**

Append:

```python
def test_optimizer_uses_hard_bindings_for_hard_diagnostic_modes():
    model_config = transformers.BertConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=12,
        num_labels=4,
    )
    for mode, expected_fragment in [
        ("soft", "slowheat"),
        ("hard", "hard"),
        ("random_hard", "hard"),
    ]:
        config = SplitCLINC150Config(
            methods=("slowheat_ffn_attention",),
            plasticity_mask_mode=mode,
        )
        model = clinc_module.SlowHeatBertForSequenceClassification(
            model_config,
            clinc_module._slowheat_config(config, "slowheat_ffn_attention"),
        )
        optimizer, _ = clinc_module._build_optimizer_and_scheduler(
            model, "slowheat_ffn_attention", config, total_steps=4
        )
        kinds = [kind for _, _, kind in optimizer._plasticity_masks.values()]
        assert kinds
        assert all(expected_fragment in kind for kind in kinds)
        if mode == "soft":
            assert all("hard" not in kind for kind in kinds)
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_split_clinc150.py::test_optimizer_uses_hard_bindings_for_hard_diagnostic_modes -v
```

Expected: FAIL because `_build_optimizer_and_scheduler()` always registers soft masks.

**Step 3: Implement hard registration**

Replace the non-LoRA registration branch in `_build_optimizer_and_scheduler()` with:

```python
        if method == "slowheat_lora_replay":
            register_exact_lora_masks(
                model,
                optimizer,
                hard=config.plasticity_mask_mode != "soft",
            )
        else:
            slow_model.register_plasticity_masks(
                optimizer,
                hard=config.plasticity_mask_mode != "soft",
            )
```

**Step 4: Verify GREEN**

Run the command from Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add experiments/split_clinc150.py tests/test_split_clinc150.py
git commit -m "feat: select hard SlowHeat optimizer masks"
```

---

## Task 3: Add deterministic equal-coverage random protection

**Objective:** Replace learned protected identities with deterministic random identities while preserving each SlowHeat state’s protected count.

**Files:**
- Modify: `experiments/split_clinc150.py` near `_find_slowheat_model()`
- Test: `tests/test_split_clinc150.py`

**Step 1: Write the failing test**

Append:

```python
def test_random_protection_is_deterministic_and_preserves_each_state_count():
    model_config = transformers.BertConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=12,
        num_labels=4,
    )
    config = SplitCLINC150Config(methods=("slowheat_ffn_attention",))

    def prepared_model():
        model = clinc_module.SlowHeatBertForSequenceClassification(
            model_config,
            clinc_module._slowheat_config(config, "slowheat_ffn_attention"),
        )
        for index, state in enumerate(model.get_slow_states()):
            count = min(index + 1, state.slow_heat.numel())
            state.slow_heat.zero_()
            state.slow_heat[:count] = torch.linspace(1.0, 0.5, count)
        return model

    first = prepared_model()
    second = prepared_model()
    expected_counts = [
        int(torch.count_nonzero(state.slow_heat))
        for state in first.get_slow_states()
    ]

    clinc_module.randomize_slowheat_protection(first, seed=123)
    clinc_module.randomize_slowheat_protection(second, seed=123)

    assert [
        int(torch.count_nonzero(state.slow_heat))
        for state in first.get_slow_states()
    ] == expected_counts
    assert all(
        torch.equal(left.slow_heat, right.slow_heat)
        for left, right in zip(first.get_slow_states(), second.get_slow_states())
    )
    assert all(
        set(torch.unique(state.slow_heat).tolist()) <= {0.0, 1.0}
        for state in first.get_slow_states()
    )
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_split_clinc150.py::test_random_protection_is_deterministic_and_preserves_each_state_count -v
```

Expected: FAIL with `AttributeError` because `randomize_slowheat_protection` does not exist.

**Step 3: Implement the helper**

Add after `_find_slowheat_model()`:

```python
def randomize_slowheat_protection(
    model: SlowHeatBertForSequenceClassification,
    *,
    seed: int,
) -> None:
    """Randomize protected identities without changing per-state capacity."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for state in model.get_slow_states():
            protected = int(torch.count_nonzero(state.slow_heat).item())
            state.slow_heat.zero_()
            if protected == 0:
                continue
            indices = torch.randperm(
                state.slow_heat.numel(), generator=generator
            )[:protected].to(state.slow_heat.device)
            state.slow_heat[indices] = 1.0
```

Use this deterministic boundary seed formula later:

```python
config.seed * 1_000_003 + stage * 10_007 + 97
```

**Step 4: Verify GREEN**

Run the command from Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add experiments/split_clinc150.py tests/test_split_clinc150.py
git commit -m "feat: add deterministic random SlowHeat control"
```

---

## Task 4: Add protected-versus-plastic parameter drift measurement

**Objective:** Measure actual parameter movement under the protection mask, independently of the selected optimizer mode.

**Files:**
- Modify: `experiments/split_clinc150.py` near the diagnostic helper from Task 3
- Test: `tests/test_split_clinc150.py`

**Step 1: Write the failing test**

Append:

```python
def test_parameter_drift_separates_protected_and_plastic_entries():
    parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    binding = clinc_module.PlasticityMaskBinding(
        parameter=parameter,
        mask=torch.tensor([[0.0], [1.0]]),
        kind="test_weight",
    )
    reference = clinc_module.capture_parameter_drift_reference([binding])

    with torch.no_grad():
        parameter.add_(torch.tensor([[0.0, 0.0], [3.0, 4.0]]))

    summary = clinc_module.summarize_parameter_drift(reference)

    assert summary["protected_count"] == 2
    assert summary["plastic_count"] == 2
    assert summary["protected_rms"] == pytest.approx(0.0)
    assert summary["protected_max_abs"] == pytest.approx(0.0)
    assert summary["plastic_rms"] == pytest.approx(math.sqrt(12.5))
    assert summary["plastic_max_abs"] == pytest.approx(4.0)
```

Update the import from `dual_heater.bert` in `experiments/split_clinc150.py` only if necessary; `PlasticityMaskBinding` is already imported indirectly nowhere, so add:

```python
from dual_heater.optim import PlasticityMaskBinding, SlowHeatAdamW
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_split_clinc150.py::test_parameter_drift_separates_protected_and_plastic_entries -v
```

Expected: FAIL because the drift helpers do not exist.

**Step 3: Implement complete drift helpers**

Add:

```python
@dataclass(frozen=True)
class ParameterDriftReference:
    kind: str
    parameter: nn.Parameter
    before: Tensor
    protected: Tensor


def capture_parameter_drift_reference(
    bindings: Sequence[PlasticityMaskBinding],
) -> list[ParameterDriftReference]:
    references: list[ParameterDriftReference] = []
    for binding in bindings:
        raw_mask = binding.mask() if callable(binding.mask) else binding.mask
        mask = torch.broadcast_to(
            raw_mask.detach().to(binding.parameter.device),
            binding.parameter.shape,
        )
        references.append(
            ParameterDriftReference(
                kind=binding.kind,
                parameter=binding.parameter,
                before=binding.parameter.detach().cpu().float().clone(),
                protected=(mask <= 0.0).detach().cpu(),
            )
        )
    return references


def summarize_parameter_drift(
    references: Sequence[ParameterDriftReference],
) -> dict[str, float | int | None]:
    totals = {
        "protected": {"count": 0, "sum_sq": 0.0, "max_abs": 0.0},
        "plastic": {"count": 0, "sum_sq": 0.0, "max_abs": 0.0},
    }
    for reference in references:
        delta = reference.parameter.detach().cpu().float() - reference.before
        for name, selected in (
            ("protected", reference.protected),
            ("plastic", ~reference.protected),
        ):
            values = delta[selected]
            if values.numel() == 0:
                continue
            totals[name]["count"] += values.numel()
            totals[name]["sum_sq"] += float(values.square().sum())
            totals[name]["max_abs"] = max(
                totals[name]["max_abs"], float(values.abs().max())
            )

    result: dict[str, float | int | None] = {}
    for name in ("protected", "plastic"):
        count = int(totals[name]["count"])
        result[f"{name}_count"] = count
        result[f"{name}_rms"] = (
            math.sqrt(float(totals[name]["sum_sq"]) / count) if count else None
        )
        result[f"{name}_max_abs"] = (
            float(totals[name]["max_abs"]) if count else None
        )
    return result
```

**Step 4: Verify GREEN**

Run the command from Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add experiments/split_clinc150.py tests/test_split_clinc150.py
git commit -m "feat: measure SlowHeat parameter drift"
```

---

## Task 5: Limit the run to an official task prefix

**Objective:** Run a two-task diagnostic without weakening validation of the full CLINC150 dataset or corrupting protocol identity.

**Files:**
- Modify: `experiments/split_clinc150.py:877-940,903-969`
- Test: `tests/test_split_clinc150.py`

**Step 1: Write the failing test**

Append:

```python
def test_task_limit_runs_only_the_official_prefix(monkeypatch, tmp_path):
    _patch_tiny_bert(monkeypatch)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    config = SplitCLINC150Config(
        max_length=12,
        batch_size=30,
        replay_batch_size=5,
        replay_per_class=1,
        epochs_per_task=1,
        methods=("vanilla",),
        task_limit=2,
    )

    result = run_split_clinc150(config, tasks, output_dir=tmp_path / "run")["vanilla"]
    protocol = json.loads(
        (tmp_path / "run" / "protocol.json").read_text(encoding="utf-8")
    )

    assert len(result["validation_accuracy_matrix"]) == 2
    assert all(len(row) == 2 for row in result["validation_accuracy_matrix"])
    assert protocol["task_order"] == ["banking", "credit_cards"]
    assert protocol["config"]["task_limit"] == 2
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_split_clinc150.py::test_task_limit_runs_only_the_official_prefix -v
```

Expected: FAIL because the runner still executes all ten tasks and records all ten domains.

**Step 3: Make checkpoint identity accept the actual task sequence**

Replace `_checkpoint_identity()` with:

```python
def _checkpoint_identity(
    config: SplitCLINC150Config,
    metadata: dict[str, Any],
    data_sha256: str,
    tasks: Sequence[CLINC150Task],
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "data_sha256": data_sha256,
        "metadata": metadata,
        "task_order": [task.domain for task in tasks],
    }
```

**Step 4: Validate full input, then select the prefix**

At the beginning of `_run_split_clinc150()`, retain the existing full ten-task validation, then add:

```python
    if config.task_limit is not None:
        tasks = tasks[: config.task_limit]
```

Place it before `data_sha256 = text_task_fingerprint(tasks)`. Pass `tasks` into `_checkpoint_identity()`:

```python
        **_checkpoint_identity(config, metadata, data_sha256, tasks),
```

All existing matrix allocation, schedules, telemetry, and metrics then use the selected local list.

**Step 5: Verify GREEN**

Run the command from Step 2.

Expected: PASS and the test completes using exactly two stages.

**Step 6: Commit**

```bash
git add experiments/split_clinc150.py tests/test_split_clinc150.py
git commit -m "feat: support official CLINC task prefixes"
```

---

## Task 6: Integrate randomization, drift, and resume state into training

**Objective:** Apply the random control at task boundaries, record drift for every stage, and preserve it exactly across resume.

**Files:**
- Modify: `experiments/split_clinc150.py:199,972-1414`
- Test: `tests/test_split_clinc150.py`

**Step 1: Write the failing integration test**

Append:

```python
def test_two_task_random_hard_run_records_zero_protected_drift_and_resumes(
    monkeypatch, tmp_path
):
    _patch_tiny_bert(monkeypatch)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    config = SplitCLINC150Config(
        max_length=12,
        batch_size=30,
        replay_batch_size=5,
        replay_per_class=1,
        epochs_per_task=1,
        methods=("slowheat_ffn_attention",),
        plasticity_mask_mode="random_hard",
        task_limit=2,
    )
    output_dir = tmp_path / "run"

    first = run_split_clinc150(config, tasks, output_dir=output_dir)
    resumed = run_split_clinc150(config, tasks, output_dir=output_dir, resume=True)
    drift = first["slowheat_ffn_attention"]["parameter_drift_history"]

    assert len(drift) == 2
    assert drift[0]["protected_rms"] is None
    assert drift[1]["protected_count"] > 0
    assert drift[1]["protected_rms"] == pytest.approx(0.0)
    assert drift[1]["protected_max_abs"] == pytest.approx(0.0)
    assert resumed["slowheat_ffn_attention"]["parameter_drift_history"] == drift
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_split_clinc150.py::test_two_task_random_hard_run_records_zero_protected_drift_and_resumes -v
```

Expected: FAIL because results and checkpoints do not contain `parameter_drift_history`, and randomization is not wired into task boundaries.

**Step 3: Bump checkpoint schema and initialize/restore drift history**

Change:

```python
CHECKPOINT_SCHEMA_VERSION = 3
```

Initialize beside `capacity_history`:

```python
        parameter_drift_history: list[dict[str, float | int | None]] = []
```

Restore beside `capacity_history`:

```python
            parameter_drift_history = checkpoint["parameter_drift_history"]
```

Replace the version-specific rejection text with:

```python
                raise RuntimeError(
                    "versão de checkpoint CLINC150 incompatível; "
                    "reinicie a execução com um diretório novo"
                )
```

**Step 4: Capture masks before each post-initial task**

Immediately after entering the stage loop and before training:

```python
            drift_reference = (
                capture_parameter_drift_reference(
                    slow_model.mask_bindings(hard=True)
                )
                if slow_model is not None and stage > 0
                else []
            )
```

Immediately after `training_losses.append(stage_losses)` and before consolidation:

```python
            empty_drift = {
                "protected_count": 0,
                "protected_rms": None,
                "protected_max_abs": None,
                "plastic_count": 0,
                "plastic_rms": None,
                "plastic_max_abs": None,
            }
            parameter_drift_history.append(
                {
                    "stage": stage,
                    **(
                        summarize_parameter_drift(drift_reference)
                        if drift_reference
                        else empty_drift
                    ),
                }
            )
```

**Step 5: Randomize after learned consolidation**

Immediately after `slow_model.consolidate(strategy="max")`:

```python
                if config.plasticity_mask_mode == "random_hard":
                    randomize_slowheat_protection(
                        slow_model,
                        seed=config.seed * 1_000_003 + stage * 10_007 + 97,
                    )
```

This ordering preserves learned protected counts while replacing only identities. Record `capacity_history` after randomization.

**Step 6: Persist and expose drift**

Add to the checkpoint payload:

```python
                        "parameter_drift_history": parameter_drift_history,
```

Add to `result`:

```python
            "parameter_drift_history": parameter_drift_history,
```

**Step 7: Update the old-schema regression test**

In `test_checkpoint_without_rng_state_is_rejected`, set:

```python
    payload["schema_version"] = 2
```

The test should continue to expect `RuntimeError` containing `incompatível`.

**Step 8: Verify GREEN and resume regression**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_split_clinc150.py \
  -k 'random_hard_run or resume_with_dropout or checkpoint_without_rng' -v
```

Expected: all selected tests PASS; hard-protected RMS and max drift are exactly zero.

**Step 9: Commit**

```bash
git add experiments/split_clinc150.py tests/test_split_clinc150.py
git commit -m "feat: record and checkpoint SlowHeat drift diagnostics"
```

---

## Task 7: Define the six-condition diagnostic matrix

**Objective:** Give the diagnostic experiment stable condition names and exact protocol settings without duplicating trainer logic.

**Files:**
- Create: `experiments/bert_slowheat_diagnostic.py`
- Create: `tests/test_bert_slowheat_diagnostic.py`

**Step 1: Write the failing test file**

Create `tests/test_bert_slowheat_diagnostic.py`:

```python
import pytest

pytest.importorskip("transformers")

from experiments.bert_slowheat_diagnostic import diagnostic_conditions


def test_diagnostic_matrix_has_exactly_the_predeclared_six_conditions():
    conditions = diagnostic_conditions()

    assert [condition.name for condition in conditions] == [
        "vanilla",
        "slowheat_beta_3",
        "slowheat_beta_10",
        "slowheat_beta_30",
        "slowheat_hard",
        "slowheat_random_hard",
    ]
    assert [condition.method for condition in conditions] == [
        "vanilla",
        "slowheat_ffn_attention",
        "slowheat_ffn_attention",
        "slowheat_ffn_attention",
        "slowheat_ffn_attention",
        "slowheat_ffn_attention",
    ]
    assert [condition.slow_strength for condition in conditions] == [
        3.0, 3.0, 10.0, 30.0, 3.0, 3.0
    ]
    assert [condition.mask_mode for condition in conditions] == [
        "soft", "soft", "soft", "soft", "hard", "random_hard"
    ]
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_bert_slowheat_diagnostic.py -v
```

Expected: collection ERROR because `experiments.bert_slowheat_diagnostic` does not exist.

**Step 3: Create the condition model and matrix**

Create `experiments/bert_slowheat_diagnostic.py` with:

```python
"""Two-task mechanism diagnostic for BERT SlowHeat on CLINC150."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from experiments.artifacts import write_json_atomic
from experiments.provenance import write_environment_manifest
from experiments.split_clinc150 import (
    PlasticityMaskMode,
    SplitCLINC150Config,
    load_clinc150_tasks,
    run_split_clinc150,
)


@dataclass(frozen=True)
class DiagnosticCondition:
    name: str
    method: str
    slow_strength: float
    mask_mode: PlasticityMaskMode


def diagnostic_conditions() -> tuple[DiagnosticCondition, ...]:
    return (
        DiagnosticCondition("vanilla", "vanilla", 3.0, "soft"),
        DiagnosticCondition("slowheat_beta_3", "slowheat_ffn_attention", 3.0, "soft"),
        DiagnosticCondition("slowheat_beta_10", "slowheat_ffn_attention", 10.0, "soft"),
        DiagnosticCondition("slowheat_beta_30", "slowheat_ffn_attention", 30.0, "soft"),
        DiagnosticCondition("slowheat_hard", "slowheat_ffn_attention", 3.0, "hard"),
        DiagnosticCondition(
            "slowheat_random_hard",
            "slowheat_ffn_attention",
            3.0,
            "random_hard",
        ),
    )
```

Leave the imported reporting helpers in place; subsequent tasks use them.

**Step 4: Verify GREEN**

Run the command from Step 2.

Expected: `1 passed`.

**Step 5: Commit**

```bash
git add experiments/bert_slowheat_diagnostic.py tests/test_bert_slowheat_diagnostic.py
git commit -m "feat: define BERT SlowHeat diagnostic matrix"
```

---

## Task 8: Implement one paired diagnostic run per condition and seed

**Objective:** Execute all six conditions with paired initialization/data/schedules and isolated resumable output directories.

**Files:**
- Modify: `experiments/bert_slowheat_diagnostic.py`
- Test: `tests/test_bert_slowheat_diagnostic.py`

**Step 1: Write a failing orchestration test**

Append:

```python
from experiments import bert_slowheat_diagnostic as diagnostic
from experiments.split_clinc150 import SplitCLINC150Config


def test_run_diagnostic_builds_paired_two_task_configs(monkeypatch, tmp_path):
    calls = []

    def fake_run(config, tasks, **kwargs):
        calls.append((config, kwargs["output_dir"]))
        return {
            config.methods[0]: {
                "validation_accuracy_matrix": [[0.8, None], [0.6, 0.9]],
                "validation_metrics": {
                    "final_average_accuracy": 0.75,
                    "average_forgetting": 0.2,
                    "backward_transfer": -0.2,
                    "forward_transfer": None,
                    "per_task_forgetting": [0.2, 0.0],
                },
                "parameter_drift_history": [
                    {"stage": 0, "protected_rms": None, "protected_max_abs": None},
                    {"stage": 1, "protected_rms": 0.0, "protected_max_abs": 0.0},
                ],
                "elapsed_seconds": 1.0,
                "peak_memory": {
                    "peak_memory_bytes": 1024,
                    "peak_cuda_reserved_bytes": 2048,
                },
            }
        }

    monkeypatch.setattr(diagnostic, "run_split_clinc150", fake_run)
    base = SplitCLINC150Config(device="cpu")
    raw = diagnostic.run_diagnostic(
        base,
        tasks=[object()] * 10,
        metadata={},
        seeds=[11, 22],
        output_dir=tmp_path,
    )

    assert len(calls) == 12
    assert set(raw) == {11, 22}
    assert all(config.task_limit == 2 for config, _ in calls)
    assert all(config.evaluate_test is False for config, _ in calls)
    assert calls[0][1] == tmp_path / "seed_11" / "vanilla"
    assert calls[-1][1] == tmp_path / "seed_22" / "slowheat_random_hard"
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_diagnostic.py::test_run_diagnostic_builds_paired_two_task_configs -v
```

Expected: FAIL because `run_diagnostic` does not exist.

**Step 3: Implement orchestration**

Add:

```python
def run_diagnostic(
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
        for condition in diagnostic_conditions():
            config = replace(
                base_config,
                seed=seed,
                methods=(condition.method,),
                slow_strength=condition.slow_strength,
                plasticity_mask_mode=condition.mask_mode,
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

**Step 4: Verify GREEN**

Run the command from Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add experiments/bert_slowheat_diagnostic.py tests/test_bert_slowheat_diagnostic.py
git commit -m "feat: orchestrate paired SlowHeat diagnostics"
```

---

## Task 9: Aggregate scientific endpoints and write reports

**Objective:** Produce a machine-readable summary and a concise Markdown table using total peak memory rather than misleading baseline-relative memory delta.

**Files:**
- Modify: `experiments/bert_slowheat_diagnostic.py`
- Test: `tests/test_bert_slowheat_diagnostic.py`

**Step 1: Write the failing aggregation test**

Append:

```python
def test_summarize_diagnostic_reports_retention_drift_time_and_total_memory():
    def result(task1_after, task1_retained, task2, drift, elapsed, peak):
        return {
            "validation_accuracy_matrix": [
                [task1_after, None],
                [task1_retained, task2],
            ],
            "parameter_drift_history": [
                {"stage": 0, "protected_rms": None, "protected_max_abs": None},
                {"stage": 1, "protected_rms": drift, "protected_max_abs": drift},
            ],
            "elapsed_seconds": elapsed,
            "peak_memory": {
                "peak_memory_bytes": peak,
                "peak_cuda_reserved_bytes": peak + 1024,
            },
        }

    raw = {
        11: {
            name: result(0.8, 0.6, 0.9, 0.0, 10.0, 1024 * 1024)
            for name in [condition.name for condition in diagnostic.diagnostic_conditions()]
        },
        22: {
            name: result(0.9, 0.7, 0.8, 0.0, 14.0, 2 * 1024 * 1024)
            for name in [condition.name for condition in diagnostic.diagnostic_conditions()]
        },
    }

    summary = diagnostic.summarize_diagnostic(raw)
    vanilla = summary["conditions"]["vanilla"]

    assert vanilla["task1_acquisition"]["mean"] == pytest.approx(0.85)
    assert vanilla["task1_retention"]["mean"] == pytest.approx(0.65)
    assert vanilla["task1_forgetting"]["mean"] == pytest.approx(0.20)
    assert vanilla["task2_acquisition"]["mean"] == pytest.approx(0.85)
    assert vanilla["elapsed_seconds"]["mean"] == pytest.approx(12.0)
    assert vanilla["peak_memory_mib"]["mean"] == pytest.approx(1.5)
    assert "paired_vs_vanilla" in summary
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_diagnostic.py::test_summarize_diagnostic_reports_retention_drift_time_and_total_memory -v
```

Expected: FAIL because `summarize_diagnostic` does not exist.

**Step 3: Add complete summary helpers**

Add:

```python
def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
    }


def _condition_endpoints(result: dict[str, Any]) -> dict[str, float]:
    matrix = result["validation_accuracy_matrix"]
    drift = result["parameter_drift_history"][1]
    return {
        "task1_acquisition": float(matrix[0][0]),
        "task1_retention": float(matrix[1][0]),
        "task1_forgetting": float(matrix[0][0] - matrix[1][0]),
        "task2_acquisition": float(matrix[1][1]),
        "protected_rms_drift": float(drift["protected_rms"]),
        "protected_max_abs_drift": float(drift["protected_max_abs"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "peak_memory_mib": float(result["peak_memory"]["peak_memory_bytes"]) / 2**20,
        "peak_reserved_mib": (
            float(result["peak_memory"]["peak_cuda_reserved_bytes"]) / 2**20
        ),
    }


def summarize_diagnostic(
    raw: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    seeds = sorted(raw)
    names = [condition.name for condition in diagnostic_conditions()]
    endpoints = {
        seed: {
            name: _condition_endpoints(raw[seed][name])
            for name in names
        }
        for seed in seeds
    }
    metric_names = tuple(endpoints[seeds[0]][names[0]])
    conditions = {
        name: {
            metric: _summary(
                [endpoints[seed][name][metric] for seed in seeds]
            )
            for metric in metric_names
        }
        for name in names
    }
    paired = {
        name: {
            metric: _summary(
                [
                    endpoints[seed][name][metric]
                    - endpoints[seed]["vanilla"][metric]
                    for seed in seeds
                ]
            )
            for metric in metric_names
        }
        for name in names
        if name != "vanilla"
    }
    return {
        "schema_version": 1,
        "endpoint_source": "validation",
        "task_count": 2,
        "seeds": seeds,
        "conditions": conditions,
        "paired_vs_vanilla": paired,
    }


def _mean_std(summary: dict[str, float | int], scale: float = 1.0) -> str:
    return (
        f"{float(summary['mean']) * scale:.2f} ± "
        f"{float(summary['sample_std']) * scale:.2f}"
    )


def diagnostic_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BERT SlowHeat two-task diagnostic",
        "",
        "Validation means ± sample standard deviation across paired seeds.",
        "",
        "| Condition | T1 acquisition (%) | T1 retention (%) | T1 forgetting (%) | "
        "T2 acquisition (%) | Protected RMS drift | Time (s) | Peak allocated (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in diagnostic_conditions():
        item = summary["conditions"][condition.name]
        lines.append(
            f"| {condition.name} | "
            f"{_mean_std(item['task1_acquisition'], 100.0)} | "
            f"{_mean_std(item['task1_retention'], 100.0)} | "
            f"{_mean_std(item['task1_forgetting'], 100.0)} | "
            f"{_mean_std(item['task2_acquisition'], 100.0)} | "
            f"{_mean_std(item['protected_rms_drift'])} | "
            f"{_mean_std(item['elapsed_seconds'])} | "
            f"{_mean_std(item['peak_memory_mib'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation gates:",
            "",
            "- Learned hard > random hard: the importance ranking contains signal.",
            "- Hard > β=3: the current soft protection is too weak.",
            "- Hard does not retain T1: unit-level FFN/attention protection is insufficient.",
            "- Retention rises while T2 acquisition falls: tune the plasticity budget.",
            "",
        ]
    )
    return "\n".join(lines)
```

**Step 4: Add artifact writing to `run_diagnostic()`**

After all runs complete:

```python
    summary = summarize_diagnostic(raw)
    write_json_atomic(destination / "diagnostic_summary.json", summary)
    (destination / "diagnostic_table.md").write_text(
        diagnostic_markdown(summary), encoding="utf-8"
    )
```

Return `raw` unchanged so tests and callers can inspect per-seed artifacts.

**Step 5: Verify GREEN**

Run the command from Step 2, then:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_bert_slowheat_diagnostic.py -v
```

Expected: all diagnostic tests PASS.

**Step 6: Commit**

```bash
git add experiments/bert_slowheat_diagnostic.py tests/test_bert_slowheat_diagnostic.py
git commit -m "feat: summarize BERT SlowHeat diagnostics"
```

---

## Task 10: Add the standalone CLI

**Objective:** Expose the exact experiment through a validation-only CLI with resume and telemetry support.

**Files:**
- Modify: `experiments/bert_slowheat_diagnostic.py`
- Test: `tests/test_bert_slowheat_diagnostic.py`

**Step 1: Write failing CLI tests**

Append:

```python
def test_diagnostic_cli_defaults_to_the_predeclared_protocol():
    args = diagnostic.build_parser().parse_args([])

    assert args.seeds == [11, 22, 33]
    assert args.task_limit == 2
    assert args.batch_size == 2
    assert args.evaluate_test is False


def test_diagnostic_cli_cannot_open_test_split():
    parser = diagnostic.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--evaluate-test"])
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_bert_slowheat_diagnostic.py \
  -k 'diagnostic_cli' -v
```

Expected: FAIL because `build_parser` does not exist.

**Step 3: Implement parser and main**

Add:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/bert_slowheat_diagnostic",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--batch-size", type=int, default=2)
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
        epochs_per_task=args.epochs_per_task,
        max_length=args.max_length,
        task_limit=args.task_limit,
        evaluate_test=False,
    )
    tasks, metadata = load_clinc150_tasks(config)
    run_diagnostic(
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

Do not add an `--evaluate-test` option. The second test passes because argparse rejects the unknown argument.

**Step 4: Verify GREEN**

Run the command from Step 2.

Expected: `2 passed`.

**Step 5: Commit**

```bash
git add experiments/bert_slowheat_diagnostic.py tests/test_bert_slowheat_diagnostic.py
git commit -m "feat: add BERT SlowHeat diagnostic CLI"
```

---

## Task 11: Document the diagnostic and decision gates

**Objective:** Make the exact user-run command and result interpretation discoverable without opening source code.

**Files:**
- Modify: `README.md:215-252`

**Step 1: Add this section after the existing CLINC150 commands**

```markdown
### Two-task SlowHeat mechanism diagnostic

Before another ten-task coverage ablation, compare protection strength and mask
quality on the first two official CLINC150 domains:

```bash
CUDA_VISIBLE_DEVICES=2 python3 -m experiments.bert_slowheat_diagnostic \
  --device cuda --batch-size 2 --seeds 11 22 33 \
  --telemetry --telemetry-every 10 \
  --output-dir results/bert_slowheat_diagnostic
```

The runner is validation-only and executes six paired conditions: vanilla,
FFN+attention SlowHeat with β=3/10/30, learned hard protection, and deterministic
random hard protection with equal per-state coverage. It writes:

- `diagnostic_summary.json`: machine-readable means, sample standard deviations,
  and paired differences versus vanilla;
- `diagnostic_table.md`: acquisition, retention, forgetting, protected-parameter
  drift, runtime, and total peak allocated memory;
- `seed_<seed>/<condition>/`: protocol, checkpoint, telemetry, and raw result.

Interpretation: learned hard beating random hard validates the importance ranking;
hard beating β=3 indicates weak soft protection; failure of hard protection to
retain task 1 indicates that FFN/attention unit masking alone is insufficient.
```

Use an outer four-backtick fence when editing so the nested shell fence remains valid Markdown.

**Step 2: Verify documentation references real CLI flags**

Run:

```bash
python3 -m experiments.bert_slowheat_diagnostic --help
```

Expected: exit code 0; output lists `--device`, `--batch-size`, `--seeds`, `--telemetry`, `--telemetry-every`, `--resume`, and `--output-dir`; it does not list `--evaluate-test`.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: explain BERT SlowHeat diagnostic"
```

---

## Task 12: Run focused and full CPU validation

**Objective:** Verify all new behavior and detect regressions without starting the real benchmark.

**Files:** No production changes unless a test exposes a defect; fix defects through a new RED→GREEN cycle.

**Step 1: Run focused tests**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_split_clinc150.py \
  tests/test_bert_slowheat_diagnostic.py -q
```

Expected: all selected tests pass; no failures or errors.

**Step 2: Run optimizer and BERT mask regressions**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_optim.py \
  tests/test_slow_heat_bert.py -q
```

Expected: all selected tests pass; existing hard-mask semantics remain unchanged.

**Step 3: Run the complete suite**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest -q
```

Expected: all tests pass.

**Step 4: Run lint only on touched Python files**

```bash
python3 -m ruff check \
  experiments/split_clinc150.py \
  experiments/bert_slowheat_diagnostic.py \
  tests/test_split_clinc150.py \
  tests/test_bert_slowheat_diagnostic.py
```

Expected: `All checks passed!`

**Step 5: Inspect the final diff**

```bash
git diff --check
git status --short
git diff --stat
```

Expected: `git diff --check` emits nothing; status contains only the five planned files plus this plan if it is intentionally tracked.

**Step 6: Final implementation commit**

Only if Steps 1–5 are clean and there are uncommitted fixes:

```bash
git add \
  experiments/split_clinc150.py \
  experiments/bert_slowheat_diagnostic.py \
  tests/test_split_clinc150.py \
  tests/test_bert_slowheat_diagnostic.py \
  README.md
git commit -m "test: verify BERT SlowHeat diagnostic workflow"
```

Do not run the CUDA benchmark in the implementation session. Hand the documented command to the user.

---

## Tests / validation acceptance criteria

Implementation is complete only when all of these hold:

1. The diagnostic matrix contains exactly six conditions with stable names and paired seeds.
2. All conditions use the same initial model, data, minibatch order, task order, epochs, optimizer family, and scheduler; only β/mask mode differ.
3. The runner validates the full official ten-task dataset, then trains only the first two domains.
4. `hard` freezes every protected optimizer entry exactly, including weight decay.
5. `random_hard` preserves each SlowHeat state’s protected count, uses only 0/1 heat, and is deterministic for `(seed, stage)` without consuming global RNG.
6. Stage-1 drift reports protected/plastic counts, RMS drift, and max absolute drift. Hard protected drift is exactly zero in the CPU integration test.
7. Checkpoint/resume reproduces matrices, losses, random masks, and drift history.
8. Reports use `peak_memory_bytes` and `peak_cuda_reserved_bytes`, not only `peak_memory_delta_bytes`.
9. No code path enables test-split evaluation.
10. Focused tests, mask regressions, full `pytest`, Ruff, and `git diff --check` pass.

## Expected experiment artifacts

After the user runs the documented CUDA command:

```text
results/bert_slowheat_diagnostic/
├── environment.json
├── diagnostic_summary.json
├── diagnostic_table.md
├── seed_11/
│   ├── vanilla/vanilla/{checkpoint.pt,results.json}
│   ├── slowheat_beta_3/slowheat_ffn_attention/{checkpoint.pt,results.json}
│   ├── slowheat_beta_10/slowheat_ffn_attention/{checkpoint.pt,results.json}
│   ├── slowheat_beta_30/slowheat_ffn_attention/{checkpoint.pt,results.json}
│   ├── slowheat_hard/slowheat_ffn_attention/{checkpoint.pt,results.json}
│   └── slowheat_random_hard/slowheat_ffn_attention/{checkpoint.pt,results.json}
├── seed_22/...
└── seed_33/...
```

Each condition directory also contains `protocol.json`; telemetry files appear when `--telemetry` is supplied.

## Decision gates after the real run

- If learned hard protection has better task-1 retention than random hard protection across paired seeds, the `|z · ∂L/∂z|` ranking has useful signal.
- If learned hard beats β=3 while task-2 acquisition remains comparable, increase soft protection strength and select β on validation.
- If β=10 or β=30 improves retention but damages task-2 acquisition, add a separate follow-up budget sweep over `{0.10, 0.25, 0.50}`; do not add that sweep to this implementation yet (YAGNI).
- If learned hard and random hard are indistinguishable, stop tuning β and revisit the importance estimator.
- If learned hard still loses task 1 substantially despite exact zero protected drift, FFN/attention unit protection is structurally insufficient; move to `replay` versus `slowheat_ffn_attention + replay` rather than another coverage sweep.

## Risks, tradeoffs, and open questions

- **Only three seeds:** This diagnostic is mechanistic, not confirmatory. Report paired differences and raw seed values; do not claim statistical superiority.
- **Two-task scope:** It isolates immediate retention and acquisition but does not establish ten-task scalability. Run a full sequence only after a decision gate passes.
- **Random baseline definition:** This plan re-samples protected identities after every task boundary while preserving per-state counts. A fixed-for-all-tasks random mask answers a different question and is intentionally excluded.
- **Unprotected paths:** FFN+attention leaves classifier, residual, normalization, embeddings, and pooler unprotected by design so the comparison matches the best completed ablation.
- **Drift coverage:** Drift is measured only for parameters with SlowHeat bindings. Report the bound parameter count alongside protected/plastic counts; do not imply whole-model drift.
- **Memory semantics:** Total peak allocated and reserved memory are primary. Baseline-relative delta may remain in raw results for compatibility but must not drive the “lighter” conclusion.
- **Checkpoint compatibility:** Schema v3 intentionally rejects older checkpoints. Use a new output directory rather than silently migrating incomplete drift history.
- **Artifact nesting:** The dedicated condition directory contains the existing method directory beneath it. Keep this explicit rather than flattening or changing the general CLINC runner’s layout.
