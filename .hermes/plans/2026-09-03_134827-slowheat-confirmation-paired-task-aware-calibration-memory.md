# SlowHeat Confirmatory Follow-up Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task, with strict RED-GREEN-REFACTOR and a separate commit after each green slice.

**Goal:** Preserve the frozen Replay vs SlowHeat+Replay confirmation while adding read-only DER++ paired reporting, task-aware diagnostics, a matched Replay+calibration control, and honest per-method CPU/CUDA peak-memory measurements for new exploratory runs.

**Architecture:** Reuse `experiments/dualheat_pairs.py` and the existing per-seed `results.json` schema rather than creating a second paired-statistics implementation. Represent calibration as a declarative `MethodSpec` capability shared by Replay and SlowHeat+Replay. Instrument the complete per-method lifecycle with a small `PeakMemoryTracker`, keeping CPU process RSS and CUDA allocator peaks explicitly separate.

**Tech stack:** Python 3.10+, PyTorch, NumPy, pytest, JSON/CSV/Markdown artifacts, Linux `/proc` for dependency-free CPU RSS sampling.

---

## Current context / assumptions

- Repository: `/home/fachel/dual-heater-mlp-research`.
- Corrected train/eval commit: `d5b22adccb4ee80e2cee0cfa125a95570bea11f6`.
- Benchmarks and GPU training are not run on this machine. Commands labeled **execution machine only** are instructions for Fachel.
- Lightweight CPU tests are permitted during implementation, with CUDA hidden and thread counts limited.
- Frozen confirmation:
  - reference: `replay`;
  - candidate: `slowheat_replay_hidden_beta_30_budget_0.25`;
  - 20 frozen seeds;
  - sole primary endpoint: `final_average_accuracy`.
- Do not modify:
  - `configs/split_mnist_confirmation_preregistration.json`;
  - the method set or seeds in `experiments/confirmatory_split_mnist.py`;
  - `experiments/confirmatory_statistics.py:PRIMARY_ENDPOINT`.
- SlowHeat+DER++, task-aware results, classifier gap, calibration controls, runtime, and memory are secondary or exploratory.
- Per-seed artifacts already contain:
  - `metrics.final_average_accuracy`;
  - `metrics.average_forgetting`;
  - `task_aware_metrics.final_average_accuracy`;
  - `task_aware_metrics.average_forgetting`;
  - `classifier_gap`;
  - `accuracy_matrix`;
  - `task_aware_accuracy_matrix`.
- `classifier_gap` means task-aware final accuracy minus Class-IL final accuracy. For candidate-minus-reference differences, a negative gap difference means the candidate closes the classifier-interference gap.
- Task-aware evaluation supplies oracle task identity by restricting logits to `task.classes`; it is not the frozen Class-IL endpoint.
- `AGGREGATE_METRICS` and `experiments/dualheat_pairs.py:summarize_pair_results()` already calculate paired statistics for Class-IL, task-aware, classifier-gap, and cost metrics from per-seed records.
- Do not infer paired effects from marginal means or marginal confidence intervals in `aggregate.json`; reconstruct candidate-minus-reference values from matching `seed_<seed>/results.json` records.
- The registered path `results/split_mnist_protocol/slowheat_derpp_exploratory/` is absent in the current checkout. Equivalent historical raw data exist under `results/split_mnist_protocol/dualheat_pairs/` and `results/dualheat_pairs/split_mnist/`. Corrected post-fix outputs may exist only on the execution machine.
- Existing memory fields such as `replay_memory_bytes`, `stored_logits_bytes`, and `method_state_bytes` are persistent algorithmic-state sizes, not runtime peaks.
- Old artifacts cannot be assigned honest runtime peak values. Missing means unavailable, never zero.

## Acceptance criteria

1. The 20-seed confirmation is validated against its lock, exact seed set, corrected commit, and clean provenance before interpretation.
2. `experiments/dualheat_pairs.py` can produce either the historical four-pair report or a DER++-only report without requiring unrelated methods.
3. DER++-only output includes per-seed differences and paired statistics for Class-IL accuracy/forgetting, task-aware accuracy/forgetting, classifier gap, runtime, and available memory metrics.
4. Reports keep `final_average_accuracy` primary and label task-aware metrics as oracle-task-ID secondary diagnostics.
5. `replay_calibrated` and calibrated SlowHeat+Replay use the same validation-only calibration path and matched replay/training settings.
6. New runs store separate CPU sampled-RSS and CUDA allocator metrics with backend/availability metadata; legacy artifacts remain readable and show unavailable memory instead of zero.
7. The frozen confirmation files and preregistration remain unchanged.
8. Focused tests and the complete CPU suite pass with CUDA hidden.

---

## Task 1: Validate the completed frozen confirmation

**Objective:** Prove the confirmation completed under the corrected source before inspecting its outcome or changing benchmark code.

**Files read:**
- `results/protocol_post_eval_fix_d5b22ad/confirmation/preregistration.lock.json`
- `results/protocol_post_eval_fix_d5b22ad/confirmation/run_identity.json`
- `results/protocol_post_eval_fix_d5b22ad/confirmation/environment.json`
- `results/protocol_post_eval_fix_d5b22ad/confirmation/aggregate.json`
- `results/protocol_post_eval_fix_d5b22ad/confirmation/seed_*/results.json`

**Step 1: Wait for normal process completion**

On the **execution machine only**, wait until the confirmation process exits. Do not inspect a partially written aggregate and do not edit Python files in that checkout during the run.

**Step 2: Validate seed coverage and provenance**

```bash
cd ~/dual-heater-mlp-research
python -c '
import json
from pathlib import Path
root = Path("results/protocol_post_eval_fix_d5b22ad/confirmation")
lock = json.loads((root / "preregistration.lock.json").read_text())
environment = json.loads((root / "environment.json").read_text())
aggregate = json.loads((root / "aggregate.json").read_text())
expected = list(lock["confirmatory_seeds"])
completed = sorted(
    int(path.parent.name.removeprefix("seed_"))
    for path in root.glob("seed_*/results.json")
)
assert len(expected) == 20, len(expected)
assert len(completed) == len(set(completed)), "duplicate seed directories"
assert set(completed) == set(expected), (sorted(set(expected)-set(completed)), sorted(set(completed)-set(expected)))
assert aggregate["seeds"] == expected
assert set(aggregate["methods"]) == {"replay", "slowheat_replay_hidden_beta_30_budget_0.25"}
assert environment["git"]["commit"] == "d5b22adccb4ee80e2cee0cfa125a95570bea11f6"
assert environment["git"]["dirty"] is False
print("confirmation-valid: 20/20 seeds; corrected commit; clean provenance")
'
```

Expected: `confirmation-valid: 20/20 seeds; corrected commit; clean provenance`.

If any assertion fails, stop and record the discrepancy. Do not relabel the run as confirmatory.

**Step 3: Preserve the result tree**

Do not use `--fresh`, overwrite the directory, rerun inspected frozen seeds, or use the 20 confirmation seeds for tuning.

---

## Task 2: Parameterize the existing paired reporter for DER++-only extraction

**Objective:** Reuse the established paired-statistics implementation without requiring all four historical pairs.

**Files:**
- Modify: `experiments/dualheat_pairs.py`
- Modify: `tests/test_split_mnist.py` near existing pair-report tests

**Step 1: RED, add a DER++-only fixture test**

Add a fixture using only `derpp` and `slowheat_derpp_hidden_beta_30_budget_0.25`, with two seeds and all required metric blocks. Add this behavioral test:

```python
def test_summarize_slowheat_derpp_accepts_only_the_derpp_pair(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "report"
    _write_pair_fixture(
        source,
        methods=("derpp", SLOWHEAT_DERPP),
        seeds=(11, 22),
    )

    report = summarize_slowheat_derpp_results(source, output_dir=output)

    assert report["analysis_scope"] == "slowheat_derpp_only"
    assert report["primary_endpoint"] == "final_average_accuracy"
    assert len(report["pairs"]) == 1
    assert report["pairs"][0]["reference"] == "derpp"
    assert report["pairs"][0]["candidate"] == SLOWHEAT_DERPP
```

Use the existing pair fixture helper if one already exists; extend it instead of creating duplicate artifact-writing code.

**Step 2: Verify RED**

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_split_mnist.py::test_summarize_slowheat_derpp_accepts_only_the_derpp_pair -q
```

Expected: FAIL because `summarize_slowheat_derpp_results` does not exist or the reporter requires all four pairs.

**Step 3: GREEN, add a narrow pair constant and parameter**

In `experiments/dualheat_pairs.py`, define:

```python
DERPP_PAIR = MethodPair(
    "DER++",
    "derpp",
    SLOWHEAT_DERPP,
)
```

Change the existing API without changing the default historical behavior:

```python
def summarize_pair_results(
    source_dir: str | Path,
    *,
    output_dir: str | Path,
    pairs: Sequence[MethodPair] = METHOD_PAIRS,
) -> dict[str, Any]:
    ...
```

Add:

```python
def summarize_slowheat_derpp_results(
    source_dir: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    return summarize_pair_results(
        source_dir,
        output_dir=output_dir,
        pairs=(DERPP_PAIR,),
    )
```

Inside `summarize_pair_results`, derive required methods from `pairs`:

```python
required_methods = {
    method
    for pair in pairs
    for method in (pair.reference, pair.candidate)
}
```

Use `pairs` and `required_methods` for completeness validation, matched-cost validation, statistics, and output loops. Do not validate against the global `PAIRED_METHODS` when a subset is requested.

**Step 4: Make inference metadata scope-aware**

For one DER++ contrast, set:

```python
report["schema_version"] = 2
report["analysis_scope"] = (
    "historical_four_pairs" if tuple(pairs) == METHOD_PAIRS else "slowheat_derpp_only"
)
report["primary_endpoint"] = "final_average_accuracy"
report["secondary_endpoints"] = [
    "average_forgetting",
    "task_aware_final_accuracy",
    "task_aware_forgetting",
    "classifier_gap",
]
report["inference_regimes"] = {
    "final_average_accuracy": "class_incremental_no_task_id",
    "task_aware_final_accuracy": "oracle_task_identity_class_mask",
}
```

Preserve Holm correction over four primary contrasts for the default historical report. For DER++-only scope, record one planned exploratory contrast and do not claim correction across four tests; either omit `holm_adjusted_p` or state that Holm with one test equals the raw p-value.

**Step 5: Verify GREEN and regression**

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest \
  tests/test_split_mnist.py::test_summarize_slowheat_derpp_accepts_only_the_derpp_pair \
  tests/test_split_mnist.py -q
```

Expected: the focused test and all `tests/test_split_mnist.py` tests pass; the existing default report still contains four pairs.

**Step 6: Commit**

```bash
git add experiments/dualheat_pairs.py tests/test_split_mnist.py
git commit -m "feat: support focused DER++ paired reports"
```

---

## Task 3: Expose task-aware paired evidence in CSV and Markdown

**Objective:** Make already-computed task-aware diagnostics visible without changing their definitions.

**Files:**
- Modify: `experiments/dualheat_pairs.py`
- Modify: `tests/test_split_mnist.py`

**Step 1: RED, test paired values rather than marginal summaries**

Create two fixture seeds with intentionally different candidate-minus-reference values. Assert:

```python
def test_derpp_report_exposes_paired_task_aware_differences(tmp_path):
    report = _build_derpp_report_with_known_task_aware_values(tmp_path)
    metrics = report["pairs"][0]["metrics"]

    assert metrics["task_aware_final_accuracy"]["mean_difference"] == pytest.approx(0.03)
    assert metrics["task_aware_forgetting"]["mean_difference"] == pytest.approx(-0.04)
    assert metrics["classifier_gap"]["mean_difference"] == pytest.approx(-0.02)
```

Also assert `pair_summary.csv` contains:

```text
task_aware_accuracy_reference_percent
task_aware_accuracy_candidate_percent
task_aware_accuracy_delta_pp
task_aware_accuracy_ci95_t_low_pp
task_aware_accuracy_ci95_t_high_pp
task_aware_forgetting_delta_pp
classifier_gap_delta_pp
```

Assert Markdown has separate `Primary: Class-IL final average accuracy` and `Secondary diagnostics` sections and says task identity is supplied at inference.

**Step 2: Verify RED**

Run the new tests only. Expected: FAIL because CSV/Markdown currently omit these fields.

**Step 3: GREEN, extend serialization only**

Reuse the metric dictionaries already produced by `summarize_pair_results()`. Do not reimplement t-tests, bootstrap, sign counts, or `_result_metric()`.

Add the seven columns above to `pair_summary.csv`. In `pair_report.md`, render:

1. Primary Class-IL final average accuracy.
2. Secondary Class-IL forgetting.
3. Task-aware final accuracy and forgetting, explicitly labeled `oracle task identity supplied`.
4. Classifier gap.
5. Runtime and available memory diagnostics.

Keep full long-form values in `pair_differences.csv`.

**Step 4: GREEN, fail closed**

Add parameterized tests for:

- missing reference;
- missing candidate;
- duplicate seeds;
- mismatched seed sets;
- NaN metric;
- missing task-aware metric;
- partial result;
- mismatched controlled costs.

Validate all source artifacts before opening destination files. After a failure, assert no report files exist. Snapshot source bytes before a successful report and assert they are unchanged afterward.

**Step 5: Verify and commit**

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_split_mnist.py -q
```

Expected: all tests pass.

```bash
git add experiments/dualheat_pairs.py tests/test_split_mnist.py
git commit -m "feat: expose task-aware paired diagnostics"
```

---

## Task 4: Extract the existing SlowHeat+DER++ report read-only

**Objective:** Produce paired DER++ evidence from corrected per-seed artifacts without running training.

**Files read:** corrected DER++ result directory on the execution machine.

**Files created:** a new analysis directory outside the source result tree.

**Step 1: Locate the actual corrected source**

Check, in order:

```text
results/secondary_post_eval_fix_d5b22ad/slowheat_derpp_exploratory/
results/split_mnist_protocol/dualheat_pairs/
results/dualheat_pairs/split_mnist/
```

Use only artifacts produced after the train/eval correction for scientific claims. Historical directories may be used to test the reporter, not to replace corrected evidence.

**Step 2: Run the focused reporter, execution machine only**

Use the repository's existing module entry point for `experiments.dualheat_pairs`. If it has no subset CLI flag, add `--pair derpp` in the same TDD slice as Task 2, mapping it to `summarize_slowheat_derpp_results`.

```bash
mkdir -p analysis/slowheat_derpp_post_eval_fix_d5b22ad
python -m experiments.dualheat_pairs \
  --source results/secondary_post_eval_fix_d5b22ad/slowheat_derpp_exploratory \
  --output-dir analysis/slowheat_derpp_post_eval_fix_d5b22ad \
  --pair derpp
```

Expected files:

```text
pair_report.json
pair_report.md
pair_summary.csv
pair_differences.csv
```

Expected report: exactly one pair, `derpp` vs `slowheat_derpp_hidden_beta_30_budget_0.25`, with seed-matched statistics.

**Step 3: Evidence rule**

Report paired mean difference, Student-t interval/test, paired bootstrap interval, paired effect size, and sign counts. Positive is favorable for accuracy. Negative is favorable for forgetting, runtime, and memory. Label the entire contrast exploratory; do not promote it into the frozen confirmation.

---

## Task 5: Interpret task-aware results

**Objective:** Separate retained representations from class-head competition.

For Replay vs SlowHeat+Replay and DER++ vs SlowHeat+DER++, record per seed and paired aggregate:

- Class-IL final average accuracy;
- Class-IL peak-based average forgetting;
- task-aware final average accuracy;
- task-aware peak-based average forgetting;
- classifier gap.

Use this fixed interpretation table:

| Class-IL | Task-aware | Interpretation |
|---|---|---|
| improves | improves | evidence consistent with better retained representations |
| improves | unchanged | effect is more consistent with reduced global classifier competition |
| unchanged/worse | improves | representation retention improved, but classifier interference blocks Class-IL benefit |
| unchanged/worse | unchanged/worse | no supported retention benefit |

If task-aware and Class-IL conclusions disagree, inspect `accuracy_matrix` and `task_aware_accuracy_matrix` per seed. Confirm forgetting is peak-based over all old tasks and excludes the final task; do not substitute Task 0 retention.

Never make task-aware accuracy the confirmatory endpoint after seeing results.

---

## Task 6: Add calibration as a declarative method capability

**Objective:** Remove the exact-name calibration branch so Replay and SlowHeat+Replay can share one procedure.

**Files:**
- Modify: `experiments/method_specs.py`
- Modify: `experiments/split_mnist.py`
- Modify: `tests/test_split_mnist.py`

**Step 1: RED**

```python
def test_calibrated_replay_resolves_as_replay_without_slowheat():
    spec = _method_spec("replay_calibrated")
    assert spec is not None
    assert spec.replay
    assert spec.calibrated
    assert not spec.slowheat
```

Run the exact test; expected FAIL because the method/capability is absent.

**Step 2: GREEN**

Add to `MethodSpec`:

```python
calibrated: bool = False
```

Register explicitly in `_METHOD_SPECS`:

```python
"replay_calibrated": MethodSpec(replay=True, calibrated=True),
"slowheat_replay_hidden_beta_30_budget_0.25_calibrated": MethodSpec(
    slowheat=True,
    replay=True,
    calibrated=True,
    strength=30.0,
    budget=0.25,
    protect_output=False,
),
```

Add beside `_uses_replay()`:

```python
def _uses_calibration(method: str) -> bool:
    spec = _method_spec(method)
    return spec is not None and spec.calibrated
```

Replace the hard-coded calibration condition with:

```python
if _uses_calibration(method):
    calibration_started = time.perf_counter()
    logit_bias = _calibrate_old_class_bias(
        model, tasks, stage=stage, config=config
    )
    cost["head_calibration_seconds"] += time.perf_counter() - calibration_started
```

Do not alter `_calibrate_old_class_bias`; it already uses validation tensors.

**Step 3: Add checkpoint symmetry**

Add both calibrated methods to `STAGE_RESUMABLE_METHODS`:

```python
"replay_calibrated",
"slowheat_replay_hidden_beta_30_budget_0.25_calibrated",
```

The checkpoint already persists `logit_bias`. Increment checkpoint schema only if memory/resume fields change incompatibly.

**Step 4: Verify and commit**

Run the focused test, then all `tests/test_split_mnist.py`. Expected: all pass.

```bash
git add experiments/method_specs.py experiments/split_mnist.py tests/test_split_mnist.py
git commit -m "feat: add matched replay calibration capability"
```

---

## Task 7: Prove calibration-control symmetry

**Objective:** Verify the two calibrated methods invoke the same calibration path and Replay training is otherwise unchanged.

**Files:**
- Modify: `tests/test_split_mnist.py`

**Step 1: RED**

In a tiny deterministic run containing `replay`, `replay_calibrated`, and calibrated SlowHeat+Replay:

- monkeypatch `_calibrate_old_class_bias` to return a known bias;
- assert both calibrated methods call it once per completed stage;
- assert ordinary Replay never calls it;
- assert Replay and Replay-calibrated start byte-identically;
- assert their training losses, completed epochs, optimizer steps, current/replay examples, and replay/logit bytes are equal;
- allow evaluation matrices to differ after calibration.

Also test the domain-incremental branch where calibration returns zeros.

Expected RED: `replay_calibrated` is unsupported or never invokes calibration.

**Step 2: GREEN and commit**

Implement only missing behavior revealed by the test. Run the focused test and `tests/test_split_mnist.py`; expected all pass.

```bash
git add tests/test_split_mnist.py experiments/split_mnist.py
git commit -m "test: verify matched calibration behavior"
```

---

## Task 8: Register the matched calibration control as exploratory

**Objective:** Run the calibrated pair on identical seeds and schedules without changing confirmation.

**Files:**
- Modify: `experiments/split_mnist_suite.py`
- Modify: `tests/test_benchmark_runner.py`
- Modify: `tests/test_visual_generalization.py` if `ALL_VISUAL_METHODS` count is asserted

**Step 1: RED**

Assert `ABLATION_METHODS` includes adjacent explicit controls:

```python
assert "replay_calibrated" in ABLATION_METHODS
assert "slowheat_replay_hidden_beta_30_budget_0.25_calibrated" in ABLATION_METHODS
```

Assert `replay_calibrated` is included in `ALL_VISUAL_METHODS`; prefer explicit membership over only a magic count.

**Step 2: GREEN**

Define:

```python
REPLAY_CALIBRATED = "replay_calibrated"
CALIBRATED_CANDIDATE = "slowheat_replay_hidden_beta_30_budget_0.25_calibrated"
```

Place both in `ABLATION_METHODS` and `ALL_VISUAL_METHODS`.

The existing `paired_references` API compares every method against every reference. If retaining that API, add `REPLAY_CALIBRATED` and explicitly ignore unrelated generated contrasts in scientific reporting. Do not add the calibrated pair to the historical four-pair `METHOD_PAIRS` or frozen confirmation.

Preferred YAGNI-safe report: pass only the calibrated reference/candidate to the subset-capable reporter introduced in Task 2 rather than changing the historical report family.

**Step 3: Verify and commit**

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_benchmark_runner.py tests/test_visual_generalization.py tests/test_split_mnist.py -q
```

Expected: all focused tests pass.

```bash
git add experiments/split_mnist_suite.py tests/test_benchmark_runner.py tests/test_visual_generalization.py
git commit -m "exp: add matched replay calibration control"
```

---

## Task 9: Add a backend-explicit peak-memory tracker

**Objective:** Measure the complete method lifecycle without confusing CPU RSS, CUDA allocated memory, CUDA reserved memory, or static method state.

**Files:**
- Create: `experiments/peak_memory.py`
- Create: `tests/test_peak_memory.py`

**Metric contract:**

```text
peak_memory_available: bool
peak_memory_backend: "process_rss_sampled" | "cuda_allocator_allocated" | null
peak_memory_bytes: integer | null
peak_memory_baseline_bytes: integer | null
peak_memory_delta_bytes: integer | null
peak_cuda_reserved_bytes: integer | null
peak_memory_sampling_interval_seconds: float | null
```

All stored numeric values use bytes. Presentation converts to MiB. CPU and CUDA values are not pooled or directly compared as equivalent measurements.

**Step 1: RED, deterministic CPU tracker test**

Inject an RSS reader returning `[100, 140, 125]` and a controllable sampler stop. Assert baseline 100, peak 140, delta 40, backend `process_rss_sampled`, thread cleanup, and cleanup after an exception. Do not assert that a real tiny process grows.

**Step 2: RED, mocked CUDA tracker test**

Mock:

- `torch.cuda.synchronize`;
- `memory_allocated`;
- `reset_peak_memory_stats`;
- `max_memory_allocated`;
- `max_memory_reserved`.

Assert the configured device is passed, synchronization happens before reset and read, and output contains allocated and reserved values separately.

Expected RED: import fails because `experiments.peak_memory` is absent.

**Step 3: GREEN, implement the tracker**

Use this interface:

```python
class PeakMemoryTracker:
    def __init__(
        self,
        device: str,
        *,
        interval_seconds: float = 0.005,
        rss_reader: Callable[[], int] = process_rss_bytes,
    ) -> None: ...

    def __enter__(self) -> "PeakMemoryTracker": ...
    def snapshot(self) -> dict[str, int | float | str | bool | None]: ...
    def __exit__(self, exc_type, exc, traceback) -> None: ...
    def result(self) -> dict[str, int | float | str | bool | None]: ...
```

For CUDA entry:

```python
resolved = torch.device(device)
torch.cuda.synchronize(resolved)
baseline = int(torch.cuda.memory_allocated(resolved))
torch.cuda.reset_peak_memory_stats(resolved)
```

For CUDA snapshot/exit:

```python
torch.cuda.synchronize(resolved)
peak = int(torch.cuda.max_memory_allocated(resolved))
reserved = int(torch.cuda.max_memory_reserved(resolved))
```

For Linux CPU RSS, read `/proc/self/statm`, multiply resident pages by `os.sysconf("SC_PAGE_SIZE")`, and sample in a daemon thread until an event is set. Label it `process_rss_sampled`; it is not total system memory and may miss transients shorter than the sampling interval.

`__exit__` must stop and join the CPU sampler in `finally`, including exceptions.

**Step 4: Verify and commit**

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_peak_memory.py -q
```

Expected: all tracker tests pass.

```bash
git add experiments/peak_memory.py tests/test_peak_memory.py
git commit -m "feat: add CPU and CUDA peak memory tracker"
```

---

## Task 10: Integrate memory tracking into Split-MNIST results

**Objective:** Track model creation through final evaluation and preserve the largest observed segment across resume.

**Files:**
- Modify: `experiments/split_mnist.py`
- Modify: `tests/test_split_mnist.py`
- Modify: `tests/test_show_results.py`
- Modify: `show_results.py`

**Step 1: RED, tiny-run schema test**

Assert every newly executed method has the metric contract under `result["cost"]`; CPU values are numeric and backend is `process_rss_sampled`. Assert legacy fixture artifacts lacking these fields render as `N/A`, never `0 B`.

**Step 2: GREEN, instrument the full boundary**

Start `PeakMemoryTracker(config.device)` immediately inside the method loop, before model construction and `.to(device)`. Stop after cost/result finalization and before model deletion. This boundary includes model/optimizer construction, lazy Adam state, pre-evaluation, replay, backward, consolidation, calibration, and final evaluation.

At each stage checkpoint, call `tracker.snapshot()` and merge it with any prior checkpoint values by taking the maximum only when the backend matches. On resume, never compare CPU and CUDA backends. Preserve `None` for unavailable legacy values.

Do not add runtime peak fields to `experiments/dualheat_pairs.MATCHED_COSTS`; SlowHeat is expected to have different runtime memory.

**Step 3: GREEN, persistence and aggregation**

Add numeric available fields to `summary.csv`. Add optional metric paths to aggregation so a legacy missing value produces unavailable metadata rather than `_result_metric()` coercion or zero. Do not make old artifacts unreadable merely because peak memory was not measured.

Update `show_results.py` to print `N/A` when unavailable and show backend labels.

**Step 4: RED/GREEN, resume test**

Extend `test_multi_seed_resume_reuses_completed_matching_seeds`:

- first segment stores one peak;
- resumed segment stores another;
- final cost stores the maximum for the same backend;
- a legacy checkpoint has unavailable memory, not zero;
- a backend mismatch is rejected or marked incomparable.

**Step 5: Verify and commit**

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_peak_memory.py tests/test_split_mnist.py tests/test_show_results.py -q
```

Expected: all focused tests pass.

```bash
git add experiments/split_mnist.py show_results.py tests/test_split_mnist.py tests/test_show_results.py
git commit -m "feat: persist peak memory with legacy-safe metadata"
```

---

## Task 11: Verify the implementation without running benchmarks

**Step 1: Complete CPU suite**

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

Expected: exit code 0, no failures or errors.

**Step 2: Lint**

```bash
ruff check experiments tests run_all_tests.py show_results.py
```

Expected: `All checks passed!`.

**Step 3: Frozen-boundary check**

```bash
python -c '
from experiments.confirmatory_split_mnist import FROZEN_CONFIG
assert FROZEN_CONFIG.methods == (
    "replay",
    "slowheat_replay_hidden_beta_30_budget_0.25",
)
print("frozen-confirmation-unchanged")
'
```

Expected: `frozen-confirmation-unchanged`.

**Step 4: Dry-run only**

```bash
CUDA_VISIBLE_DEVICES='' python run_all_tests.py \
  --num-seeds 10 \
  --sections ablations slowheat-derpp \
  --device cuda \
  --no-download \
  --output-dir results/matched_controls_peak_memory \
  --dry-run
```

Expected: plan output only, no training/download, and calibrated reference/candidate both present.

---

## Task 12: Run only the new exploratory evidence on the execution machine

**Objective:** Measure the matched calibration pair and refreshed DER++ pair without touching confirmation.

After committing implementation, obtain the commit:

```bash
git rev-parse --short=12 HEAD
```

Use a new output root containing that commit. On the **execution machine only**:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python run_all_tests.py \
  --num-seeds 10 \
  --baseline-seeds \
    2101606466 1872839281 637029796 1357204345 1758462037 \
    748737880 1611510062 333205446 130501536 492612040 \
  --sections ablations slowheat-derpp \
  --device cuda \
  --no-download \
  --output-dir results/matched_controls_peak_memory_<NEW_COMMIT>
```

Expected:

- matched calibrated methods use identical 10 seeds;
- DER++ methods use identical 10 seeds;
- each new CUDA result has backend `cuda_allocator_allocated` and positive allocated peak;
- reserved peak is stored separately;
- confirmation directory is untouched.

This is post-hoc exploratory evidence. Reusing secondary seeds preserves pairing but does not create an independent confirmation.

For publication-grade CPU comparison, run each `(seed, method)` in an isolated subprocess. The in-process sampled RSS metric is descriptive because Python/PyTorch allocators retain pages across serial methods.

---

## Task 13: Report outcomes conservatively

**Files:**
- Modify after evidence is final: `docs/split_mnist_experiment_log.md`
- Modify after evidence is final: `article/manuscript.md`
- Modify after evidence is final: `README.md`

Report, in order:

1. Frozen primary paired final average accuracy.
2. Secondary peak-based forgetting with final accuracy alongside it.
3. Task-aware accuracy/forgetting and classifier gap, labeled oracle-task-ID diagnostics.
4. Exploratory SlowHeat+DER++ paired results.
5. Exploratory matched calibration result.
6. Paired runtime.
7. Runtime memory by backend:
   - CUDA allocated peak in bytes/MiB;
   - CUDA reserved peak separately;
   - sampled process RSS separately;
   - persistent `method_state_bytes`, replay bytes, and stored-logit bytes separately.
8. Negative/null outcomes and deferred work.

Decision language:

- forgetting improves but final accuracy does not: `stability-plasticity trade-off`, not efficacy;
- task-aware improves but Class-IL does not: `representation retention with unresolved classifier interference`;
- calibrated SlowHeat beats uncalibrated Replay only: effect is confounded;
- calibrated SlowHeat beats `replay_calibrated`: SlowHeat attribution becomes plausible but remains exploratory;
- paired DER++ accuracy and forgetting improve: exploratory additive evidence, not frozen confirmation.

Do not add state bytes and replay bytes and call the sum runtime peak. Do not compare CPU RSS numerically with CUDA allocator bytes as if they were the same metric.

Commit documentation separately:

```bash
git add README.md article/manuscript.md docs/split_mnist_experiment_log.md
git commit -m "docs: report confirmatory and exploratory SlowHeat evidence"
```

---

## Tests / validation summary

Run each new behavioral test once before production code and confirm it fails for the intended missing behavior. Then implement only enough to pass, rerun the focused test, rerun its file, and commit.

Final commands:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_peak_memory.py tests/test_split_mnist.py tests/test_benchmark_runner.py tests/test_visual_generalization.py tests/test_show_results.py -q

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider

ruff check experiments tests run_all_tests.py show_results.py
```

Expected: all tests pass, lint passes, and no benchmark/dataset artifacts are created locally.

## Risks, tradeoffs, and open questions

- Corrected DER++ artifacts may exist only on the execution machine. Do not substitute pre-fix artifacts for post-fix scientific evidence.
- A one-pair DER++ report must not retain wording that claims Holm correction over four contrasts.
- Task-aware inference has oracle task identity and can overstate deployable Class-IL performance.
- The existing broad `paired_references` API emits scientifically irrelevant cross-method contrasts. Ignore them or add an explicit pair API only if report consumers require it; do not over-refactor now.
- Sampled CPU RSS can miss short transients and is contaminated by allocator retention in serial execution. Isolated subprocesses are required for stronger CPU claims.
- CUDA allocated peak measures PyTorch live tensor allocation, not total board VRAM. CUDA reserved peak measures caching-allocator reservation and is reported separately.
- Memory from old artifacts is unavailable. Backfilling zero would create false evidence.
- Starting tracking before model construction may include allocator state from previous serial methods. Record method order; CUDA allocated is the primary GPU metric, while reserved memory is descriptive.
- Calibration must use validation data only. Any test-set selection of the bias invalidates the control.
- Do not change code while a confirmation run is active or after seeing its outcomes and then rerun the same frozen seeds.
- Do not mix unrelated existing README/manuscript edits into code commits.
