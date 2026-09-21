# Próximos passos do diagnóstico BERT SlowHeat

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Corrigir o relatório atual e executar um diagnóstico fatorial de duas tarefas que determine se a proteção hard aprendida acrescenta retenção útil a um replay comum e pareado.

**Architecture:** Primeiro, tornar o relatório existente cientificamente legível sem repetir o treinamento: incluir acurácia média final/BWT, mostrar drift em notação científica e oferecer reanálise offline dos `results.json`. Depois, adicionar somente a variante ausente `slowheat_ffn_attention_replay` ao runner CLINC150 e criar um runner separado com o fatorial 2x2 `{sem mecanismo, hard aprendido} × {sem replay, com replay}`. O benchmark permanece validation-only, usa o seletor de replay fixo e independente do modelo já existente, e só autoriza uma sequência completa após o novo gate mecanístico.

**Tech stack:** Python 3.12, PyTorch 2.x, Transformers, NumPy, pytest, Ruff e os helpers atômicos existentes em `experiments/artifacts.py`.

---

## Current context / assumptions

- O diagnóstico concluído está em `results/bert_slowheat_diagnostic/` e contém 18/18 runs válidas: 6 condições × seeds 11, 22 e 33.
- O gate atual passou: hard aprendido superou hard aleatório nas três seeds; hard superou β=3; drift protegido hard foi exatamente zero.
- O limite estrutural também apareceu: hard ainda esqueceu 40,67 pontos percentuais de T1. Portanto, não executar outra grade de cobertura ou outra lista de β agora.
- O próximo teste científico é replay contra mecanismo+replay. O fatorial terá exatamente quatro condições:
  1. `vanilla`: AdamW, sem replay;
  2. `replay`: AdamW, replay comum;
  3. `slowheat_hard`: FFN+attention hard aprendido, sem replay;
  4. `slowheat_hard_replay`: a mesma proteção hard aprendida, com o mesmo replay comum.
- O replay existente em `experiments/split_clinc150.py:597-610` escolhe os primeiros `replay_per_class` exemplos de cada classe, sem consultar o modelo. Isso é um seletor comum congelado e atende ao pareamento científico.
- Usar `replay_per_class=20`, `batch_size=2` e `replay_batch_size=2`. O replay é parte do tratamento; não igualar artificialmente tokens processados, mas reportar tokens, tempo e throughput.
- Todos os quatro métodos devem partir do mesmo estado inicial, dados, splits, ordem de tarefas, minibatches atuais e scheduler por tarefa. A única diferença deve ser mecanismo e/ou replay.
- Continuar usando somente os dois primeiros domínios oficiais: `banking` → `credit_cards`, validação apenas, sem carregar o split de teste.
- Seeds 11/22/33 já foram usadas para desenvolvimento e continuam adequadas apenas ao diagnóstico. Se o gate passar, a confirmação de dez tarefas deverá usar seeds não utilizadas, em um plano posterior.
- Não executar treinamento CUDA, baixar dados nem instalar dependências durante a implementação. Apenas testes CPU leves estão pré-aprovados. O comando CUDA final será entregue ao usuário.
- O working tree já contém alterações não commitadas ligadas ao diagnóstico. Antes de cada commit, adicionar somente os arquivos nomeados na tarefa; nunca usar `git add .`.

## Files likely to change

- Modify: `experiments/bert_slowheat_diagnostic.py`
- Modify: `experiments/split_clinc150.py`
- Create: `experiments/bert_slowheat_replay_diagnostic.py`
- Modify: `tests/test_bert_slowheat_diagnostic.py`
- Modify: `tests/test_split_clinc150.py`
- Create: `tests/test_bert_slowheat_replay_diagnostic.py`
- Modify: `README.md`
- Regenerate only after tests pass: `results/bert_slowheat_diagnostic/diagnostic_summary.json`
- Regenerate only after tests pass: `results/bert_slowheat_diagnostic/diagnostic_table.md`

Do not change the raw `seed_*/<condition>/**/results.json`, checkpoints, protocols, telemetry, or heat snapshots.

---

## Task 1: Add the missing scientific endpoints to the current summary

**Objective:** Make final average accuracy, BWT, replay size, token count, and throughput first-class endpoints instead of requiring manual calculation.

**Files:**
- Modify: `experiments/bert_slowheat_diagnostic.py:62-96`
- Test: `tests/test_bert_slowheat_diagnostic.py:46-167`

**Step 1: Write the failing test**

In `_fake_result()` in `tests/test_bert_slowheat_diagnostic.py`, add the existing-schema fields after `elapsed_seconds`:

```python
        "tokens_processed": 1_000,
        "replay_memory_bytes": 2 * 1024 * 1024,
```

Then extend `test_summarize_diagnostic_reports_retention_drift_time_and_total_memory()`:

```python
    assert vanilla["final_average_accuracy"]["mean"] == pytest.approx(0.75)
    assert vanilla["backward_transfer"]["mean"] == pytest.approx(-0.20)
    assert vanilla["tokens_processed"]["mean"] == pytest.approx(1_000.0)
    assert vanilla["tokens_per_second"]["mean"] == pytest.approx(
        (100.0 + 1000.0 / 14.0) / 2.0
    )
    assert vanilla["replay_memory_mib"]["mean"] == pytest.approx(2.0)
```

The throughput expectation is the mean of the two per-run throughputs, not `mean(tokens) / mean(time)`.

**Step 2: Verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_diagnostic.py::test_summarize_diagnostic_reports_retention_drift_time_and_total_memory -v
```

Expected: FAIL with `KeyError: 'final_average_accuracy'`.

**Step 3: Implement the minimal endpoint extraction**

In `_condition_endpoints()` in `experiments/bert_slowheat_diagnostic.py`, read the existing validation metrics and append the new fields:

```python
    validation_metrics = result["validation_metrics"]
    elapsed_seconds = float(result["elapsed_seconds"])
    tokens_processed = float(result["tokens_processed"])
```

Add these entries to the returned dictionary immediately after `task2_acquisition`:

```python
        "final_average_accuracy": float(
            validation_metrics["final_average_accuracy"]
        ),
        "backward_transfer": float(validation_metrics["backward_transfer"]),
```

Replace the existing `elapsed_seconds` entry and append resource fields:

```python
        "elapsed_seconds": elapsed_seconds,
        "tokens_processed": tokens_processed,
        "tokens_per_second": tokens_processed / elapsed_seconds,
        "replay_memory_mib": float(result["replay_memory_bytes"]) / 2**20,
```

Do not recompute final average or BWT from rounded report values; use `validation_metrics` from each raw run.

**Step 4: Verify GREEN**

Run the command from Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add experiments/bert_slowheat_diagnostic.py tests/test_bert_slowheat_diagnostic.py
git commit -m "fix: report complete SlowHeat diagnostic endpoints"
```

---

## Task 2: Fix misleading drift formatting and expose final average in Markdown

**Objective:** Prevent nonzero soft drift from being displayed as `0.00` and make the retention/acquisition trade-off visible in the primary table.

**Files:**
- Modify: `experiments/bert_slowheat_diagnostic.py:181-230`
- Test: `tests/test_bert_slowheat_diagnostic.py`

**Step 1: Write the failing formatting test**

Append to `tests/test_bert_slowheat_diagnostic.py`:

```python
def test_diagnostic_markdown_shows_final_average_and_scientific_drift():
    names = [condition.name for condition in diagnostic.diagnostic_conditions()]
    raw = {
        11: {
            name: _fake_result(
                task1_after=0.8,
                task1_retained=0.6,
                task2=0.9,
                drift=8.038814675802855e-4,
                elapsed=10.0,
                peak=1024 * 1024,
            )
            for name in names
        }
    }

    rendered = diagnostic.diagnostic_markdown(
        diagnostic.summarize_diagnostic(raw)
    )

    assert "Final average (%)" in rendered
    assert "75.00 ± 0.00" in rendered
    assert "8.039e-04 ± 0.000e+00" in rendered
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_diagnostic.py::test_diagnostic_markdown_shows_final_average_and_scientific_drift -v
```

Expected: FAIL because the table has no final-average column and formats drift with two decimal places.

**Step 3: Add a dedicated scientific formatter**

Add after `_mean_std()`:

```python
def _mean_std_scientific(summary: dict[str, float | int | None]) -> str:
    mean_value = summary["mean"]
    std_value = summary["sample_std"]
    if mean_value is None or std_value is None:
        return "—"
    return f"{float(mean_value):.3e} ± {float(std_value):.3e}"
```

**Step 4: Update the table header and rows**

In `diagnostic_markdown()`, use this exact header shape:

```python
        "| Condition | T1 acquisition (%) | T1 retention (%) | T1 forgetting (%) | "
        "T2 acquisition (%) | Final average (%) | Protected entries | "
        "Protected RMS drift | Plastic RMS drift | Time (s) | Tokens/s | "
        "Peak allocated (MiB) | Peak reserved (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
```

Use these expressions in the row, preserving the existing fields around them:

```python
            f"{_mean_std(item['task2_acquisition'], 100.0)} | "
            f"{_mean_std(item['final_average_accuracy'], 100.0)} | "
            f"{_mean_std(item['protected_count'])} | "
            f"{_mean_std_scientific(item['protected_rms_drift'])} | "
            f"{_mean_std_scientific(item['plastic_rms_drift'])} | "
            f"{_mean_std(item['elapsed_seconds'])} | "
            f"{_mean_std(item['tokens_per_second'])} | "
```

**Step 5: Verify GREEN**

Run the command from Step 2, then:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_bert_slowheat_diagnostic.py -q
```

Expected: all tests in the file PASS.

**Step 6: Commit**

```bash
git add experiments/bert_slowheat_diagnostic.py tests/test_bert_slowheat_diagnostic.py
git commit -m "fix: render SlowHeat drift without precision loss"
```

---

## Task 3: Add offline reanalysis of completed raw artifacts

**Objective:** Regenerate the current summary/table after reporting changes without loading CLINC150 or rerunning CUDA training.

**Files:**
- Modify: `experiments/bert_slowheat_diagnostic.py:233-330`
- Test: `tests/test_bert_slowheat_diagnostic.py`

**Step 1: Write the failing loader test**

Append:

```python
def test_load_completed_diagnostic_reads_nested_raw_results(tmp_path):
    for seed in (11, 22):
        for condition in diagnostic.diagnostic_conditions():
            method_dir = tmp_path / f"seed_{seed}" / condition.name / condition.method
            method_dir.mkdir(parents=True)
            (method_dir / "results.json").write_text(
                json.dumps(_fake_result()), encoding="utf-8"
            )

    raw = diagnostic.load_completed_diagnostic(tmp_path)

    assert set(raw) == {11, 22}
    assert set(raw[11]) == {
        condition.name for condition in diagnostic.diagnostic_conditions()
    }
```

Add `import json` at the top of the test file.

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_diagnostic.py::test_load_completed_diagnostic_reads_nested_raw_results -v
```

Expected: FAIL with `AttributeError` because `load_completed_diagnostic` does not exist.

**Step 3: Implement the strict loader and report writer**

Add before `run_diagnostic()`:

```python
def load_completed_diagnostic(
    output_dir: str | Path,
) -> dict[int, dict[str, dict[str, Any]]]:
    source = Path(output_dir)
    seed_dirs = sorted(source.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"nenhuma seed encontrada em {source}")

    raw: dict[int, dict[str, dict[str, Any]]] = {}
    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError as error:
            raise ValueError(f"diretório de seed inválido: {seed_dir.name}") from error
        raw[seed] = {}
        for condition in diagnostic_conditions():
            result_path = (
                seed_dir
                / condition.name
                / condition.method
                / "results.json"
            )
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            raw[seed][condition.name] = json.loads(
                result_path.read_text(encoding="utf-8")
            )
    return raw


def write_diagnostic_reports(
    output_dir: str | Path,
    raw: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    destination = Path(output_dir)
    summary = summarize_diagnostic(raw)
    write_json_atomic(destination / "diagnostic_summary.json", summary)
    (destination / "diagnostic_table.md").write_text(
        diagnostic_markdown(summary), encoding="utf-8"
    )
    return summary
```

Add `import json` to `experiments/bert_slowheat_diagnostic.py`. Replace the duplicated report-writing block at the end of `run_diagnostic()` with:

```python
    write_diagnostic_reports(destination, raw)
```

**Step 4: Add a CLI-only reanalysis branch**

In `build_parser()` add:

```python
    parser.add_argument(
        "--summarize-from",
        help="reanalisar results.json existentes sem carregar dados ou treinar",
    )
```

At the start of `main()`, immediately after parsing:

```python
    if args.summarize_from is not None:
        raw = load_completed_diagnostic(args.summarize_from)
        write_diagnostic_reports(args.summarize_from, raw)
        return
```

**Step 5: Verify GREEN and prove no dataset load**

Add this test:

```python
def test_summarize_from_does_not_load_dataset(monkeypatch, tmp_path):
    for seed in (11,):
        for condition in diagnostic.diagnostic_conditions():
            method_dir = tmp_path / f"seed_{seed}" / condition.name / condition.method
            method_dir.mkdir(parents=True)
            (method_dir / "results.json").write_text(
                json.dumps(_fake_result()), encoding="utf-8"
            )
    monkeypatch.setattr(
        diagnostic,
        "load_clinc150_tasks",
        lambda *args, **kwargs: pytest.fail("dataset must not be loaded"),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["bert_slowheat_diagnostic", "--summarize-from", str(tmp_path)],
    )

    diagnostic.main()

    assert (tmp_path / "diagnostic_summary.json").is_file()
    assert (tmp_path / "diagnostic_table.md").is_file()
```

Run:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_bert_slowheat_diagnostic.py -q
```

Expected: all tests PASS.

**Step 6: Commit**

```bash
git add experiments/bert_slowheat_diagnostic.py tests/test_bert_slowheat_diagnostic.py
git commit -m "feat: reanalyze completed SlowHeat diagnostics offline"
```

---

## Task 4: Register FFN+attention SlowHeat with replay

**Objective:** Add one compositional method name that reuses the existing FFN+attention architecture, hard masking, and replay loop without duplicating training code.

**Files:**
- Modify: `experiments/split_clinc150.py:125-199,729-799`
- Test: `tests/test_split_clinc150.py`

**Step 1: Write the failing registration test**

Append to `tests/test_split_clinc150.py`:

```python
def test_ffn_attention_replay_combines_existing_coverage_and_replay_flags():
    method = "slowheat_ffn_attention_replay"

    assert method in clinc_module.SUPPORTED_METHODS
    assert method in clinc_module.SLOWHEAT_METHODS
    assert method in clinc_module.REPLAY_METHODS

    resolved = clinc_module._slowheat_config(
        SplitCLINC150Config(methods=(method,)), method
    )
    assert resolved.track_ffn is True
    assert resolved.track_attention is True
    assert resolved.track_embeddings is False
    assert resolved.track_residual is False
    assert resolved.protect_layer_norm is False
    assert resolved.protect_pooler is False
    assert resolved.protect_classifier is False
    assert resolved.freeze_unbound_parameters is False
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_split_clinc150.py::test_ffn_attention_replay_combines_existing_coverage_and_replay_flags -v
```

Expected: FAIL because the method is absent from `SUPPORTED_METHODS`.

**Step 3: Register the method exactly once in each behavior set**

In `SUPPORTED_METHODS`, add after `slowheat_ffn_attention`:

```python
    "slowheat_ffn_attention_replay",
```

After `BERT_FULL_COVERAGE_VARIANTS`, define:

```python
FFN_ATTENTION_METHODS = {
    "slowheat_ffn_attention",
    "slowheat_ffn_attention_replay",
}
```

Replace `_EXTENDED_SLOWHEAT_METHODS` with:

```python
_EXTENDED_SLOWHEAT_METHODS = (
    set(BERT_FULL_COVERAGE_VARIANTS) - {"vanilla"}
) | FFN_ATTENTION_METHODS
```

Add the new method to `REPLAY_METHODS`. No separate addition to `SLOWHEAT_METHODS` is needed because it is included through `_EXTENDED_SLOWHEAT_METHODS`.

Change the coverage branch in `_slowheat_config()` from:

```python
    elif method == "slowheat_ffn_attention":
```

to:

```python
    elif method in FFN_ATTENTION_METHODS:
```

Do not add the replay variant to `BERT_FULL_COVERAGE_VARIANTS`; `--full-coverage-variants` must retain its existing experiment matrix.

**Step 4: Verify GREEN**

Run the command from Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add experiments/split_clinc150.py tests/test_split_clinc150.py
git commit -m "feat: add FFN attention SlowHeat replay variant"
```

---

## Task 5: Verify hard protection and replay interact correctly end to end

**Objective:** Prove on the tiny CPU protocol that the new method stores replay, uses it on task 2, and still gives exactly zero protected drift.

**Files:**
- Test: `tests/test_split_clinc150.py`
- Modify production code only if this test exposes a defect; use a new RED→GREEN cycle for any fix.

**Step 1: Write the failing integration test**

Append near the existing two-task random-hard test:

```python
def test_two_task_ffn_attention_hard_replay_records_memory_and_zero_drift(
    monkeypatch, tmp_path
):
    _patch_tiny_bert(monkeypatch)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    method = "slowheat_ffn_attention_replay"
    config = SplitCLINC150Config(
        max_length=12,
        batch_size=30,
        replay_batch_size=5,
        replay_per_class=1,
        epochs_per_task=1,
        methods=(method,),
        plasticity_mask_mode="hard",
        task_limit=2,
    )

    result = run_split_clinc150(
        config,
        tasks,
        output_dir=tmp_path / "run",
    )[method]

    assert result["replay_memory_bytes"] > 0
    assert result["tokens_processed"] > 0
    assert len(result["validation_accuracy_matrix"]) == 2
    assert result["parameter_drift_history"][1]["protected_count"] > 0
    assert result["parameter_drift_history"][1]["protected_rms"] == pytest.approx(0.0)
    assert result["parameter_drift_history"][1]["protected_max_abs"] == pytest.approx(0.0)
```

**Step 2: Verify RED before Task 4 implementation, or GREEN after it**

If implementing tasks sequentially with one commit each, run this test once before Task 4’s production edit to observe RED, then again now:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_split_clinc150.py::test_two_task_ffn_attention_hard_replay_records_memory_and_zero_drift -v
```

Expected after Task 4: PASS. If it fails, fix only the missing method wiring; do not alter optimizer semantics or replay selection.

**Step 3: Run replay/resume regressions**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_split_clinc150.py \
  -k 'ffn_attention_hard_replay or tiny_clinc_runner_is_paired or random_hard_run' -v
```

Expected: all selected tests PASS.

**Step 4: Commit**

```bash
git add tests/test_split_clinc150.py experiments/split_clinc150.py
git commit -m "test: verify hard SlowHeat with replay"
```

---

## Task 6: Create the four-condition replay diagnostic runner

**Objective:** Encode the predeclared 2x2 mechanism/replay comparison in a separate validation-only runner and isolated output directory.

**Files:**
- Create: `experiments/bert_slowheat_replay_diagnostic.py`
- Create: `tests/test_bert_slowheat_replay_diagnostic.py`

**Step 1: Write the failing condition-matrix test**

Create `tests/test_bert_slowheat_replay_diagnostic.py`:

```python
import pytest

pytest.importorskip("transformers")

from experiments import bert_slowheat_replay_diagnostic as followup


def test_replay_diagnostic_is_the_predeclared_two_by_two_factorial():
    conditions = followup.replay_diagnostic_conditions()

    assert [condition.name for condition in conditions] == [
        "vanilla",
        "replay",
        "slowheat_hard",
        "slowheat_hard_replay",
    ]
    assert [condition.method for condition in conditions] == [
        "vanilla",
        "replay",
        "slowheat_ffn_attention",
        "slowheat_ffn_attention_replay",
    ]
    assert [condition.mask_mode for condition in conditions] == [
        "soft",
        "soft",
        "hard",
        "hard",
    ]
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_replay_diagnostic.py::test_replay_diagnostic_is_the_predeclared_two_by_two_factorial -v
```

Expected: collection ERROR because the module does not exist.

**Step 3: Create the runner skeleton and exact matrix**

Create `experiments/bert_slowheat_replay_diagnostic.py`:

```python
"""Two-task replay interaction diagnostic for BERT SlowHeat on CLINC150."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from experiments.artifacts import write_json_atomic
from experiments.bert_slowheat_diagnostic import (
    _condition_endpoints,
    _mean_std,
    _mean_std_scientific,
    _summary,
)
from experiments.provenance import write_environment_manifest
from experiments.split_clinc150 import (
    PlasticityMaskMode,
    SplitCLINC150Config,
    load_clinc150_tasks,
    run_split_clinc150,
)


@dataclass(frozen=True)
class ReplayDiagnosticCondition:
    name: str
    method: str
    mask_mode: PlasticityMaskMode


def replay_diagnostic_conditions() -> tuple[ReplayDiagnosticCondition, ...]:
    return (
        ReplayDiagnosticCondition("vanilla", "vanilla", "soft"),
        ReplayDiagnosticCondition("replay", "replay", "soft"),
        ReplayDiagnosticCondition(
            "slowheat_hard", "slowheat_ffn_attention", "hard"
        ),
        ReplayDiagnosticCondition(
            "slowheat_hard_replay",
            "slowheat_ffn_attention_replay",
            "hard",
        ),
    )
```

The imports of small report helpers are intentional reuse; do not copy endpoint/statistics logic into the new file.

**Step 4: Verify GREEN**

Run the command from Step 2.

Expected: PASS.

**Step 5: Write the failing orchestration test**

Append:

```python
from experiments.split_clinc150 import SplitCLINC150Config


def _fake_result():
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
        "replay_memory_bytes": 0,
        "peak_memory": {
            "peak_memory_bytes": 1024 * 1024,
            "peak_cuda_reserved_bytes": 2 * 1024 * 1024,
        },
    }


def test_run_replay_diagnostic_builds_paired_validation_only_configs(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run(config, tasks, **kwargs):
        calls.append((config, kwargs["output_dir"]))
        return {config.methods[0]: _fake_result()}

    monkeypatch.setattr(followup, "run_split_clinc150", fake_run)
    monkeypatch.setattr(followup, "write_environment_manifest", lambda *a, **k: None)

    raw = followup.run_replay_diagnostic(
        SplitCLINC150Config(device="cpu"),
        tasks=[object()] * 10,
        metadata={},
        seeds=[11, 22],
        output_dir=tmp_path,
    )

    assert len(calls) == 8
    assert set(raw) == {11, 22}
    assert all(config.task_limit == 2 for config, _ in calls)
    assert all(config.evaluate_test is False for config, _ in calls)
    assert calls[0][1] == tmp_path / "seed_11" / "vanilla"
    assert calls[-1][1] == tmp_path / "seed_22" / "slowheat_hard_replay"
```

**Step 6: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_replay_diagnostic.py::test_run_replay_diagnostic_builds_paired_validation_only_configs -v
```

Expected: FAIL because `run_replay_diagnostic` does not exist.

**Step 7: Implement orchestration**

Append to the production file:

```python
def run_replay_diagnostic(
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
        for condition in replay_diagnostic_conditions():
            config = replace(
                base_config,
                seed=seed,
                methods=(condition.method,),
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

Do not write reports yet; Task 7 adds reporting after testing its contrasts.

**Step 8: Verify GREEN and commit**

Run the command from Step 6.

Expected: PASS.

```bash
git add \
  experiments/bert_slowheat_replay_diagnostic.py \
  tests/test_bert_slowheat_replay_diagnostic.py
git commit -m "feat: orchestrate BERT SlowHeat replay diagnostic"
```

---

## Task 7: Aggregate the factorial and predeclare its decision contrasts

**Objective:** Report the primary final-average endpoint and the exact paired contrasts needed to decide whether SlowHeat adds value beyond replay.

**Files:**
- Modify: `experiments/bert_slowheat_replay_diagnostic.py`
- Test: `tests/test_bert_slowheat_replay_diagnostic.py`

**Step 1: Write the failing aggregation test**

Append:

```python
def test_replay_summary_contains_predeclared_paired_contrasts():
    names = [condition.name for condition in followup.replay_diagnostic_conditions()]
    raw = {11: {name: _fake_result() for name in names}}

    summary = followup.summarize_replay_diagnostic(raw)

    assert summary["primary_endpoint"] == "final_average_accuracy"
    assert set(summary["paired_contrasts"]) == {
        "replay_minus_vanilla",
        "hard_minus_vanilla",
        "hard_replay_minus_replay",
        "hard_replay_minus_hard",
    }
    assert summary["paired_contrasts"]["hard_replay_minus_replay"][
        "final_average_accuracy"
    ]["by_seed"] == {"11": 0.0}
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_replay_diagnostic.py::test_replay_summary_contains_predeclared_paired_contrasts -v
```

Expected: FAIL because `summarize_replay_diagnostic` does not exist.

**Step 3: Implement the summary with shared endpoint extraction**

Append:

```python
def summarize_replay_diagnostic(
    raw: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not raw:
        raise ValueError("resultados diagnósticos não podem ser vazios")
    seeds = sorted(raw)
    names = [condition.name for condition in replay_diagnostic_conditions()]
    for seed in seeds:
        if set(raw[seed]) != set(names):
            raise ValueError(f"seed {seed} não contém as quatro condições")

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
    contrast_pairs = {
        "replay_minus_vanilla": ("replay", "vanilla"),
        "hard_minus_vanilla": ("slowheat_hard", "vanilla"),
        "hard_replay_minus_replay": (
            "slowheat_hard_replay",
            "replay",
        ),
        "hard_replay_minus_hard": (
            "slowheat_hard_replay",
            "slowheat_hard",
        ),
    }
    paired_contrasts: dict[str, dict[str, dict[str, Any]]] = {}
    for contrast, (left, right) in contrast_pairs.items():
        paired_contrasts[contrast] = {}
        for metric in metric_names:
            by_seed: dict[str, float | None] = {}
            for seed in seeds:
                left_value = endpoints[seed][left][metric]
                right_value = endpoints[seed][right][metric]
                by_seed[str(seed)] = (
                    None
                    if left_value is None or right_value is None
                    else left_value - right_value
                )
            paired_contrasts[contrast][metric] = {
                **_summary(list(by_seed.values())),
                "by_seed": by_seed,
            }
    return {
        "schema_version": 1,
        "endpoint_source": "validation",
        "primary_endpoint": "final_average_accuracy",
        "task_count": 2,
        "seeds": seeds,
        "conditions": conditions,
        "raw_by_seed": {str(seed): endpoints[seed] for seed in seeds},
        "paired_contrasts": paired_contrasts,
    }
```

**Step 4: Add a concise Markdown report**

Append:

```python
def replay_diagnostic_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BERT SlowHeat replay interaction diagnostic",
        "",
        "Validation means ± sample standard deviation across paired seeds.",
        "Primary endpoint: final average accuracy.",
        "",
        "| Condition | Final average (%) | T1 retention (%) | T1 forgetting (%) | "
        "T2 acquisition (%) | Protected RMS drift | Replay memory (MiB) | "
        "Tokens | Tokens/s | Time (s) | Peak allocated (MiB) | Peak reserved (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in replay_diagnostic_conditions():
        item = summary["conditions"][condition.name]
        lines.append(
            f"| {condition.name} | "
            f"{_mean_std(item['final_average_accuracy'], 100.0)} | "
            f"{_mean_std(item['task1_retention'], 100.0)} | "
            f"{_mean_std(item['task1_forgetting'], 100.0)} | "
            f"{_mean_std(item['task2_acquisition'], 100.0)} | "
            f"{_mean_std_scientific(item['protected_rms_drift'])} | "
            f"{_mean_std(item['replay_memory_mib'])} | "
            f"{_mean_std(item['tokens_processed'])} | "
            f"{_mean_std(item['tokens_per_second'])} | "
            f"{_mean_std(item['elapsed_seconds'])} | "
            f"{_mean_std(item['peak_memory_mib'])} | "
            f"{_mean_std(item['peak_reserved_mib'])} |"
        )
    lines.extend(
        [
            "",
            "Decision gate:",
            "",
            "- Primary contrast: hard_replay_minus_replay on final average accuracy.",
            "- Report all paired seed differences; n=3 remains diagnostic.",
            "- Reject a mechanism benefit if any seed reverses sign or if T2 acquisition "
            "drops by more than 2 percentage points on average.",
            "- Do not start the ten-task sequence until this gate is reviewed.",
            "",
        ]
    )
    return "\n".join(lines)
```

At the end of `run_replay_diagnostic()`, before `return raw`, add:

```python
    summary = summarize_replay_diagnostic(raw)
    write_json_atomic(destination / "diagnostic_summary.json", summary)
    (destination / "diagnostic_table.md").write_text(
        replay_diagnostic_markdown(summary), encoding="utf-8"
    )
```

**Step 5: Verify GREEN**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_bert_slowheat_replay_diagnostic.py -q
```

Expected: all follow-up diagnostic tests PASS.

**Step 6: Commit**

```bash
git add \
  experiments/bert_slowheat_replay_diagnostic.py \
  tests/test_bert_slowheat_replay_diagnostic.py
git commit -m "feat: summarize SlowHeat replay interaction"
```

---

## Task 8: Add a validation-only CLI and document the exact GPU command

**Objective:** Make the follow-up runnable without exposing test evaluation or ambiguous defaults.

**Files:**
- Modify: `experiments/bert_slowheat_replay_diagnostic.py`
- Modify: `tests/test_bert_slowheat_replay_diagnostic.py`
- Modify: `README.md:244-272`

**Step 1: Write the failing parser test**

Append:

```python
def test_replay_diagnostic_cli_defaults_are_frozen():
    args = followup.build_parser().parse_args([])

    assert args.seeds == [11, 22, 33]
    assert args.batch_size == 2
    assert args.replay_batch_size == 2
    assert args.replay_per_class == 20
    assert args.task_limit == 2
    assert args.evaluate_test is False


def test_replay_diagnostic_cli_rejects_test_evaluation():
    with pytest.raises(SystemExit):
        followup.build_parser().parse_args(["--evaluate-test"])
```

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_bert_slowheat_replay_diagnostic.py \
  -k 'cli_' -v
```

Expected: FAIL because `build_parser` does not exist.

**Step 3: Implement the exact parser and main**

Append:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/bert_slowheat_replay_diagnostic",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--replay-batch-size", type=int, default=2)
    parser.add_argument("--replay-per-class", type=int, default=20)
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
        replay_per_class=args.replay_per_class,
        epochs_per_task=args.epochs_per_task,
        max_length=args.max_length,
        task_limit=args.task_limit,
        evaluate_test=False,
    )
    tasks, metadata = load_clinc150_tasks(config, include_test=False)
    run_replay_diagnostic(
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

Do not define an `--evaluate-test` flag. Argparse must reject it.

**Step 4: Verify GREEN and CLI help**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_bert_slowheat_replay_diagnostic.py -q
python3 -m experiments.bert_slowheat_replay_diagnostic --help
```

Expected: all tests PASS; help lists replay batch/memory flags and does not list `--evaluate-test`.

**Step 5: Document the follow-up immediately after the current mechanism diagnostic**

Add to `README.md` after the interpretation paragraph at lines 269-272:

```markdown
#### Replay interaction follow-up

After the two-task mechanism gate passes, compare replay and learned hard
protection as a predeclared 2x2 factorial:

```bash
CUDA_VISIBLE_DEVICES=2 python3 -m experiments.bert_slowheat_replay_diagnostic \
  --device cuda --batch-size 2 --replay-batch-size 2 \
  --replay-per-class 20 --seeds 11 22 33 \
  --telemetry --telemetry-every 10 \
  --output-dir results/bert_slowheat_replay_diagnostic
```

The four validation-only conditions are vanilla, replay, learned hard
FFN+attention protection, and learned hard FFN+attention protection with the
same frozen replay selector. The primary endpoint is final average validation
accuracy; the primary paired contrast is `hard_replay_minus_replay`.
```

Use an outer four-backtick fence while editing so the nested shell fence remains valid.

**Step 6: Commit**

```bash
git add \
  experiments/bert_slowheat_replay_diagnostic.py \
  tests/test_bert_slowheat_replay_diagnostic.py \
  README.md
git commit -m "docs: add SlowHeat replay diagnostic protocol"
```

---

## Task 9: Run all lightweight validation and regenerate only the old reports

**Objective:** Verify the implementation and update the two derived report files from the already completed artifacts without touching raw evidence.

**Files:**
- Derived outputs only: `results/bert_slowheat_diagnostic/diagnostic_summary.json`
- Derived outputs only: `results/bert_slowheat_diagnostic/diagnostic_table.md`

**Step 1: Run focused CPU tests**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_diagnostic.py \
  tests/test_bert_slowheat_replay_diagnostic.py \
  tests/test_split_clinc150.py -q
```

Expected: all selected tests PASS; no failures or errors.

**Step 2: Run optimizer/BERT mask regressions**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_optim.py tests/test_slow_heat_bert.py -q
```

Expected: all selected tests PASS.

**Step 3: Run the complete suite**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest -q
```

Expected: all tests PASS.

**Step 4: Run Ruff on touched Python files**

```bash
python3 -m ruff check \
  experiments/split_clinc150.py \
  experiments/bert_slowheat_diagnostic.py \
  experiments/bert_slowheat_replay_diagnostic.py \
  tests/test_split_clinc150.py \
  tests/test_bert_slowheat_diagnostic.py \
  tests/test_bert_slowheat_replay_diagnostic.py
```

Expected: `All checks passed!`

**Step 5: Reanalyze the completed diagnostic offline**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m experiments.bert_slowheat_diagnostic \
  --summarize-from results/bert_slowheat_diagnostic
```

Expected:
- exit code 0;
- no dataset download and no CUDA initialization;
- `diagnostic_summary.json` contains `final_average_accuracy`, `backward_transfer`, `tokens_per_second`, and `replay_memory_mib`;
- `diagnostic_table.md` displays soft drift in scientific notation and includes `Final average (%)`.

**Step 6: Validate report content programmatically**

```bash
python3 -c 'import json, pathlib; p=pathlib.Path("results/bert_slowheat_diagnostic"); s=json.loads((p/"diagnostic_summary.json").read_text()); assert s["conditions"]["slowheat_hard"]["final_average_accuracy"]["mean"] > s["conditions"]["vanilla"]["final_average_accuracy"]["mean"]; t=(p/"diagnostic_table.md").read_text(); assert "Final average (%)" in t and "e-04" in t; print("diagnostic reports verified")'
```

Expected: `diagnostic reports verified`.

**Step 7: Inspect scope before committing derived reports**

```bash
git diff --check
git status --short
git diff --stat
```

Expected: `git diff --check` emits nothing. Confirm manually that no raw result/checkpoint/telemetry artifact changed.

**Step 8: Commit the derived report refresh only if result files are tracked**

```bash
git ls-files --error-unmatch \
  results/bert_slowheat_diagnostic/diagnostic_summary.json \
  results/bert_slowheat_diagnostic/diagnostic_table.md
```

If exit code is 0:

```bash
git add \
  results/bert_slowheat_diagnostic/diagnostic_summary.json \
  results/bert_slowheat_diagnostic/diagnostic_table.md
git commit -m "docs: refresh SlowHeat diagnostic report"
```

If the files are ignored/untracked, do not force-add them; leave them as local derived artifacts.

---

## Task 10: Hand off the CUDA run and apply the predeclared gate

**Objective:** Produce the next evidence without allowing implementation automation to start an unapproved GPU workload.

**Files:** No code changes.

**Step 1: Give this exact command to the user; do not execute it automatically**

```bash
CUDA_VISIBLE_DEVICES=2 python3 -m experiments.bert_slowheat_replay_diagnostic \
  --device cuda --batch-size 2 --replay-batch-size 2 \
  --replay-per-class 20 --seeds 11 22 33 \
  --telemetry --telemetry-every 10 \
  --output-dir results/bert_slowheat_replay_diagnostic
```

Expected artifact layout:

```text
results/bert_slowheat_replay_diagnostic/
├── environment.json
├── diagnostic_summary.json
├── diagnostic_table.md
├── seed_11/
│   ├── vanilla/vanilla/{checkpoint.pt,results.json}
│   ├── replay/replay/{checkpoint.pt,results.json}
│   ├── slowheat_hard/slowheat_ffn_attention/{checkpoint.pt,results.json}
│   └── slowheat_hard_replay/slowheat_ffn_attention_replay/{checkpoint.pt,results.json}
├── seed_22/...
└── seed_33/...
```

**Step 2: Verify completion after the user runs it**

Use read-only analysis to assert:

- 12/12 `results.json` files exist;
- all JSON parses and contains no NaN/Inf;
- each telemetry stream ends with `method_end`, `run_end`, and `session_end`;
- all protocols use the same data/source hashes and task order;
- replay and hard+replay have identical replay-memory bytes within each seed;
- hard and hard+replay have exactly zero protected drift;
- raw paired differences are printed for every seed.

**Step 3: Apply this gate without post-hoc metric substitution**

Primary endpoint: validation `final_average_accuracy`.

Primary contrast: `slowheat_hard_replay - replay`.

Proceed to a new ten-task confirmatory plan only if all conditions hold:

1. the primary contrast is positive in every seed;
2. mean T2-acquisition loss of hard+replay versus replay is no worse than 2 percentage points;
3. protected RMS and max drift remain exactly zero;
4. hard+replay does not gain merely through a different replay memory, token schedule, task order, or initialization;
5. absolute peak allocated/reserved memory and elapsed time are reported alongside accuracy.

If the primary contrast reverses in any seed, report the result as unstable and do not start the ten-task sequence. If replay alone nearly saturates retention, reduce replay memory in a separately predeclared experiment rather than changing `replay_per_class` after seeing individual seeds. If hard+replay improves retention but harms acquisition beyond the threshold, tune plasticity budget on separate tuning seeds before confirmation.

---

## Tests / validation acceptance criteria

Implementation is complete only when:

1. The current report includes final average accuracy and BWT from raw `validation_metrics`.
2. Nonzero drift is rendered in scientific notation rather than as `0.00`.
3. Existing reports can be regenerated from raw files without dataset loading or training.
4. `slowheat_ffn_attention_replay` belongs to supported, SlowHeat, and replay method sets while retaining exactly FFN+attention coverage.
5. The tiny integration test proves replay memory is populated and hard-protected drift remains exactly zero.
6. The follow-up matrix is exactly the predeclared four-condition 2x2 factorial.
7. The follow-up is validation-only and has no `--evaluate-test` escape hatch.
8. The summary stores raw per-seed contrasts and declares final average accuracy as the sole primary endpoint.
9. Reports contain total peak allocated, total peak reserved, elapsed time, tokens, throughput, and replay memory.
10. Focused tests, optimizer/mask regressions, full pytest, Ruff, and `git diff --check` all pass.
11. No GPU benchmark, dataset download, dependency install, commit, or push occurs without the user’s explicit authorization at execution time.

## Risks, tradeoffs, and open questions

- **n=3 remains diagnostic:** Even if all paired signs agree, do not present the result as publication-grade uncertainty. The exact two-sided sign test with three non-ties cannot provide strong significance.
- **Replay may dominate:** Twenty memories per class can make the two-task problem easy. That is informative: if replay saturates retention, the next question is memory efficiency, not whether hard protection wins at that memory size.
- **Compute is intentionally unequal:** Replay processes extra examples. Compare accuracy jointly with tokens, throughput, elapsed time, and absolute memory; do not call one method computationally equivalent.
- **Protected topology remains partial:** FFN+attention bindings cover about 28% of trainable parameters, with about 21% of all trainable entries hard-protected in the completed diagnostic. Classifier, embeddings, residual paths, LayerNorm, and pooler remain unprotected.
- **Private helper reuse:** The follow-up imports underscored formatting/endpoint helpers from `bert_slowheat_diagnostic.py`. This is acceptable for two sibling experiment scripts and avoids a premature utility module. Extract a public reporting module only if a third runner needs the same API.
- **Dirty provenance:** The completed run records `git.dirty=true` plus `source_sha256`. Preserve both facts in any paper-facing report; do not imply the commit alone reproduces it.
- **No full-sequence run yet:** The present evidence establishes ranking signal and hard-mask correctness, not ten-task efficacy. A full sequence is conditional on the hard+replay versus replay gate above and should use held-out seeds/task orders in a separate plan.
- **Threshold choice:** The 2-point mean T2-acquisition tolerance is predeclared here to prevent post-hoc interpretation. Change it only before running the follow-up and record the rationale in this plan or a protocol manifest.
