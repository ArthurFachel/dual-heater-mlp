# SlowHeat BERT Full-Coverage Implementation Plan

> **For Hermes:** Use subagent-driven-development to implement this plan task-by-task, with a spec review and a code-quality review after every commit.

**Goal:** Implement structured SlowHeat coverage for every trainable parameter family in Hugging Face BERT and expose controlled leave-one-family-out CLINC150 ablations.

**Architecture:** Preserve the existing functional units for FFN neurons and attention heads, then add hidden-dimension trackers at the embedding output, both residual junctions in each encoder block, and the pooler output. Build exactly one dynamic mask per parameter; when a matrix is controlled by both its producer and consumer, combine the two factors with elementwise `minimum`, so either protected endpoint can restrict the update. Keep all parameters trainable in the ablation suite, disable FastHeat, and vary only which SlowHeat factor families participate in masks.

**Tech stack:** Python 3.12, PyTorch, Hugging Face Transformers, pytest, ruff.

---

## Current context / assumptions

- Repository root: `/mnt/B-SSD/fachel/dual-heater-mlp`.
- Existing implementation:
  - `src/dual_heater/bert.py` instruments every encoder block for FFN and attention SlowHeat.
  - `src/dual_heater/transformer.py` provides `SlowHeatFFNTracker` and `SlowHeatAttentionTracker`.
  - `src/dual_heater/optim.py` requires exactly one `PlasticityMaskBinding` per parameter.
  - `experiments/split_clinc150.py` is the paired BERT/CLINC150 runner.
- No new dependency is required.
- This work targets `BertForSequenceClassification` only. Do not generalize to RoBERTa, SwiGLU, GQA, fused QKV, decoder-only LLMs, or LoRA full coverage in this change.
- CPU-only tiny-model tests are sufficient. Do not download datasets/models and do not run GPU training during implementation.
- “Full coverage” means every trainable BERT parameter has one optimizer mask. It does not mean one independent tracker per tensor.
- The scientific unit definitions are:
  - FFN: one unit per intermediate neuron.
  - Attention: one unit per attention head.
  - Embedding/residual/LayerNorm: one unit per hidden dimension.
  - Pooler: one unit per pooled hidden dimension.
  - Classifier: one unit per class logit.
- Extended variants use `freeze_unbound_parameters=False`. Removing a family therefore restores native AdamW updates for factors no longer covered; it does not freeze the removed component.
- Existing method identifiers and historical behavior must remain unchanged.
- Bump the persisted SlowHeat protocol schema from 2 to 3 because the configuration and state topology change. Reject schema-2 SlowHeat checkpoints instead of silently interpreting them as full coverage.

## Parameter graph and ownership

Use these nodes for layer `l`:

```text
embedding output E
  -> Q_l/K_l/V_l -> attention heads H_l -> attention dense -> attention residual A_l
  -> intermediate FFN F_l -> output dense -> block residual R_l
R_l becomes the input node for layer l+1
R_last -> pooler P -> classifier C
```

Each parameter receives at most one mask:

| Parameter | Row/output factor | Column/input factor |
|---|---|---|
| word/position/token-type embedding weights | none | E, when embeddings enabled |
| embedding LayerNorm weight/bias | E, when LayerNorm enabled | none |
| Q/K/V weight in layer l | H_l, when attention enabled | input residual, when residual enabled |
| Q/K/V bias | H_l, when attention enabled | none |
| attention output dense weight | A_l, when residual enabled | expanded H_l, when attention enabled |
| attention output dense bias | A_l, when residual enabled | none |
| attention output LayerNorm weight/bias | A_l, when LayerNorm enabled | none |
| intermediate dense weight | F_l, when FFN enabled | A_l, when residual enabled |
| intermediate dense bias | F_l, when FFN enabled | none |
| FFN output dense weight | R_l, when residual enabled | F_l, when FFN enabled |
| FFN output dense bias | R_l, when residual enabled | none |
| FFN output LayerNorm weight/bias | R_l, when LayerNorm enabled | none |
| pooler dense weight | P, when pooler enabled | R_last, when residual enabled |
| pooler dense bias | P, when pooler enabled | none |
| classifier weight | C, when classifier enabled | P, when pooler enabled |
| classifier bias | C, when classifier enabled | none |

For a weight with row factor `r` and column factor `c`, return:

```python
torch.minimum(r.reshape(-1, 1), c.reshape(1, -1))
```

This is the conservative endpoint rule already used elsewhere in the project: a weight is plastic only to the extent allowed by both functional endpoints.

## Methods to expose

Add these identifiers without changing old identifiers:

```python
BERT_FULL_COVERAGE_VARIANTS = (
    "vanilla",
    "slowheat_full_coverage",
    "slowheat_all_minus_embeddings",
    "slowheat_all_minus_layernorm",
    "slowheat_all_minus_residual",
    "slowheat_all_minus_attention",
    "slowheat_all_minus_ffn",
    "slowheat_all_minus_pooler",
    "slowheat_all_minus_classifier",
    "slowheat_ffn_attention",
)
```

All SlowHeat variants above use hierarchical capacity, no FastHeat, no replay, and all model parameters remain trainable. `slowheat_ffn_attention` is a clearly named alias of the existing unbound `slowheat` configuration; retain `slowheat` for historical artifacts.

---

## Step-by-step tasks

### Task 1: Lock the new configuration contract

**Objective:** Add explicit full-coverage switches and budgets while preserving old defaults.

**Files:**
- Modify: `tests/test_slow_heat_bert.py`
- Modify: `src/dual_heater/bert.py`

**Step 1: Write failing tests**

Append these tests to `tests/test_slow_heat_bert.py`:

```python
def test_extended_bert_config_defaults_preserve_historical_scope():
    config = BertSlowHeatConfig()

    assert config.track_ffn is True
    assert config.track_attention is True
    assert config.track_embeddings is False
    assert config.track_residual is False
    assert config.protect_layer_norm is False
    assert config.protect_pooler is False
    assert config.protect_classifier is False
    assert config.residual_plasticity_budget == 0.25
    assert config.pooler_plasticity_budget == 0.25


def test_extended_bert_config_validates_new_budgets():
    with pytest.raises(ValueError, match="residual_plasticity_budget"):
        BertSlowHeatConfig(residual_plasticity_budget=-0.1)
    with pytest.raises(ValueError, match="pooler_plasticity_budget"):
        BertSlowHeatConfig(pooler_plasticity_budget=1.1)
```

**Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_extended_bert_config_defaults_preserve_historical_scope tests/test_slow_heat_bert.py::test_extended_bert_config_validates_new_budgets -q
```

Expected: FAIL because the six new fields do not exist.

**Step 3: Implement minimally**

Extend `BertSlowHeatConfig` in `src/dual_heater/bert.py` with exactly:

```python
    residual_plasticity_budget: float = 0.25
    pooler_plasticity_budget: float = 0.25
    track_embeddings: bool = False
    track_residual: bool = False
    protect_layer_norm: bool = False
    protect_pooler: bool = False
```

Add both budgets to the existing `values` dictionary and validate them with:

```python
        if not 0.0 <= self.residual_plasticity_budget <= 1.0:
            raise ValueError("residual_plasticity_budget deve estar em [0, 1]")
        if not 0.0 <= self.pooler_plasticity_budget <= 1.0:
            raise ValueError("pooler_plasticity_budget deve estar em [0, 1]")
```

Validate all new switches explicitly:

```python
        switches = {
            "track_embeddings": self.track_embeddings,
            "track_residual": self.track_residual,
            "protect_layer_norm": self.protect_layer_norm,
            "protect_pooler": self.protect_pooler,
        }
        if any(not isinstance(value, bool) for value in switches.values()):
            raise TypeError("opções de cobertura BERT devem ser booleanas")
```

Update the “at least one family” guard so any new switch also makes the configuration valid.

**Step 4: Verify GREEN**

Run the command from Step 2.

Expected: `2 passed`.

**Step 5: Commit**

```bash
git add src/dual_heater/bert.py tests/test_slow_heat_bert.py
git commit -m "feat: define BERT full-coverage configuration"
```

### Task 2: Add reusable dynamic factor composition

**Objective:** Produce one mask for a parameter even when two functional endpoints control it.

**Files:**
- Modify: `tests/test_slow_heat_bert.py`
- Modify: `src/dual_heater/bert.py`

**Step 1: Write failing tests**

Add the helper import to the existing import list:

```python
from dual_heater.bert import _dynamic_matrix_mask
```

Append:

```python
def test_dynamic_matrix_mask_combines_row_and_column_factors_conservatively():
    rows = torch.tensor([1.0, 0.4])
    columns = torch.tensor([0.2, 0.8, 1.0])
    mask = _dynamic_matrix_mask(lambda: rows, lambda: columns)()

    torch.testing.assert_close(
        mask,
        torch.tensor([[0.2, 0.8, 1.0], [0.2, 0.4, 0.4]]),
    )


def test_dynamic_matrix_mask_requires_an_endpoint():
    with pytest.raises(ValueError, match="ao menos um endpoint"):
        _dynamic_matrix_mask(None, None)
```

**Step 2: Verify RED**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_dynamic_matrix_mask_combines_row_and_column_factors_conservatively tests/test_slow_heat_bert.py::test_dynamic_matrix_mask_requires_an_endpoint -q
```

Expected: collection ERROR because `_dynamic_matrix_mask` is missing.

**Step 3: Implement minimally**

Add `Callable` to imports from `collections.abc`, then add this complete helper near `_factor` in `src/dual_heater/bert.py`:

```python
FactorSource = Callable[[], Tensor]


def _dynamic_matrix_mask(
    row_source: FactorSource | None,
    column_source: FactorSource | None,
) -> FactorSource:
    """Build one dynamic matrix mask from zero, one, or two endpoint factors."""

    if row_source is None and column_source is None:
        raise ValueError("ao menos um endpoint deve fornecer uma máscara")

    def mask() -> Tensor:
        rows = row_source().reshape(-1, 1) if row_source is not None else None
        columns = (
            column_source().reshape(1, -1) if column_source is not None else None
        )
        if rows is None:
            assert columns is not None
            return columns
        if columns is None:
            return rows
        if rows.device != columns.device:
            columns = columns.to(rows.device)
        return torch.minimum(rows, columns)

    return mask
```

Do not add a generic multi-dimensional mask framework; BERT parameters here are vectors or matrices.

**Step 4: Verify GREEN and regression**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_dynamic_matrix_mask_combines_row_and_column_factors_conservatively tests/test_slow_heat_bert.py::test_dynamic_matrix_mask_requires_an_endpoint -q
python3 -m pytest tests/test_optim.py -q
```

Expected: `2 passed`; then all optimizer tests pass.

**Step 5: Commit**

```bash
git add src/dual_heater/bert.py tests/test_slow_heat_bert.py
git commit -m "feat: compose SlowHeat endpoint masks"
```

### Task 3: Instrument hidden-dimension and pooler states

**Objective:** Collect utility at embedding output, both residual junctions per encoder block, and pooler output without changing native forward values.

**Files:**
- Modify: `tests/test_slow_heat_bert.py`
- Modify: `src/dual_heater/bert.py`

**Step 1: Write failing test for topology and backward**

Append:

```python
def _full_coverage_config(**overrides):
    values = {
        "track_ffn": True,
        "track_attention": True,
        "track_embeddings": True,
        "track_residual": True,
        "protect_layer_norm": True,
        "protect_pooler": True,
        "protect_classifier": True,
        "capacity_scope": "hierarchical",
    }
    values.update(overrides)
    return BertSlowHeatConfig(**values)


def test_full_coverage_collects_embedding_two_residuals_per_layer_and_pooler():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(layers=2), _full_coverage_config()
    )
    model.train()

    _backward(
        model,
        torch.tensor([[2, 7, 9, 3]]),
        torch.ones(1, 4, dtype=torch.long),
    )

    residual = model.get_residual_trackers()
    assert len(residual) == 5  # embedding + attention/block output for two layers
    assert all(tracker.units == 8 for tracker in residual)
    assert all(tracker.task_step.item() == 1 for tracker in residual)
    assert model.pooler_tracker is not None
    assert model.pooler_tracker.units == 8
    assert model.pooler_tracker.task_step.item() == 1
```

**Step 2: Verify RED**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_full_coverage_collects_embedding_two_residuals_per_layer_and_pooler -q
```

Expected: FAIL because `get_residual_trackers` and `pooler_tracker` do not exist.

**Step 3: Implement state creation**

In `SlowHeatBertForSequenceClassification.__init__`, initialize before instrumentation:

```python
        self.residual_trackers = nn.ModuleList()
        self.pooler_tracker: SlowHeatFFNTracker | None = None
```

Add:

```python
    def _needs_residual_states(self) -> bool:
        config = self.slowheat_config
        return any(
            (
                config.track_embeddings,
                config.track_residual,
                config.protect_layer_norm,
                config.protect_pooler,
            )
        )

    def _new_residual_tracker(self) -> SlowHeatFFNTracker:
        return self._new_ffn_tracker(
            self.config.hidden_size,
            self.slowheat_config.residual_plasticity_budget,
        )

    def get_residual_trackers(self) -> list[SlowHeatFFNTracker]:
        return list(self.residual_trackers)
```

**Step 4: Install hooks**

In `_install_slowheat_instrumentation`, when `_needs_residual_states()` is true:

1. Create/reuse tracker index 0 and hook `self.bert.embeddings`.
2. For each encoder layer, create/reuse two trackers in order: attention residual, block residual.
3. Hook `layer.attention.output` and `layer.output`.
4. Each hook calls `state.observe(output, self._slowheat_validity_mask)` only when the mask is set.
5. When `protect_pooler=True`, create `pooler_tracker` with `hidden_size` and `pooler_plasticity_budget`, then hook `self.bert.pooler`; pooler output has shape `[B, H]`, so call `observe(output)` without a token mask.
6. If `self.bert.pooler is None` while `protect_pooler=True`, raise `RuntimeError("protect_pooler requer um pooler BERT")` during construction.

Use default arguments in every closure (`state=tracker`) to avoid late binding. Reuse existing ModuleList entries during `reinstall_slowheat_instrumentation`; do not append duplicates.

**Step 5: Verify GREEN and no-forward-change regression**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_full_coverage_collects_embedding_two_residuals_per_layer_and_pooler tests/test_slow_heat_bert.py::test_no_consolidation_slowheat_bert_matches_native_adamw -q
```

Expected: `2 passed`.

**Step 6: Commit**

```bash
git add src/dual_heater/bert.py tests/test_slow_heat_bert.py
git commit -m "feat: track BERT residual and pooler utility"
```

### Task 4: Make padding invariance cover all token-level trackers

**Objective:** Ensure padded tokens cannot change embedding or residual utility.

**Files:**
- Modify: `tests/test_slow_heat_bert.py`
- Modify only if needed: `src/dual_heater/bert.py`

**Step 1: Write failing/characterization test**

Append:

```python
def test_full_coverage_padding_does_not_change_residual_utility():
    torch.manual_seed(81)
    short = SlowHeatBertForSequenceClassification(
        _bert_config(), _full_coverage_config()
    )
    padded = SlowHeatBertForSequenceClassification(
        _bert_config(), _full_coverage_config()
    )
    padded.load_state_dict(deepcopy(short.state_dict()))
    short.train()
    padded.train()

    _backward(short, torch.tensor([[2, 7, 9, 3]]), torch.ones(1, 4, dtype=torch.long))
    _backward(
        padded,
        torch.tensor([[2, 7, 9, 3, 0, 0]]),
        torch.tensor([[1, 1, 1, 1, 0, 0]]),
    )

    for first, second in zip(
        short.get_residual_trackers(),
        padded.get_residual_trackers(),
        strict=True,
    ):
        torch.testing.assert_close(first.task_ema, second.task_ema, atol=1e-5, rtol=1e-5)
```

**Step 2: Verify RED or valid characterization**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_full_coverage_padding_does_not_change_residual_utility -q
```

Expected: PASS if Task 3 correctly forwarded the validity mask. If it fails, the failure must show differing `task_ema`, not a setup error.

**Step 3: Minimal correction if RED**

Ensure each token-level hook passes the same detached `[B,T]` `self._slowheat_validity_mask` to `SlowHeatFFNTracker.observe`. Do not mask pooler/classifier trackers because they are `[B,H]` and `[B,C]` example-level tensors.

**Step 4: Verify GREEN**

Run the command from Step 2; expected `1 passed`.

**Step 5: Commit**

```bash
git add src/dual_heater/bert.py tests/test_slow_heat_bert.py
git commit -m "test: enforce padding-safe BERT residual utility"
```

### Task 5: Replace BERT bindings with one graph-derived binding per parameter

**Objective:** Cover the complete BERT parameter graph and satisfy the optimizer’s no-duplicate-binding contract.

**Files:**
- Modify: `tests/test_slow_heat_bert.py`
- Modify: `src/dual_heater/bert.py`

**Step 1: Write failing full-coverage test**

Append:

```python
def test_full_coverage_binds_every_trainable_parameter_exactly_once():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(layers=2), _full_coverage_config()
    )
    bindings = model.mask_bindings()
    identifiers = [id(binding.parameter) for binding in bindings]

    assert len(identifiers) == len(set(identifiers))
    assert model.uncovered_trainable_parameters() == []
    assert {id(parameter) for parameter in model.parameters()} == set(identifiers)
```

Add a shape test:

```python
def test_full_coverage_masks_match_every_parameter_shape_by_broadcast():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(), _full_coverage_config()
    )
    for tracker in model.get_slow_states():
        tracker.slow_heat.copy_(
            torch.linspace(0.0, 1.0, tracker.slow_heat.numel())
        )

    for binding in model.mask_bindings():
        mask = binding.mask() if callable(binding.mask) else binding.mask
        assert torch.broadcast_shapes(mask.shape, binding.parameter.shape) == binding.parameter.shape
        assert torch.isfinite(mask).all()
        assert torch.all((0.0 <= mask) & (mask <= 1.0))
```

**Step 2: Verify RED**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_full_coverage_binds_every_trainable_parameter_exactly_once tests/test_slow_heat_bert.py::test_full_coverage_masks_match_every_parameter_shape_by_broadcast -q
```

Expected: FAIL with uncovered embedding, LayerNorm, residual-output, pooler, and/or classifier parameters.

**Step 3: Refactor `mask_bindings` around named factors**

In `src/dual_heater/bert.py`, keep `_factor` as the source of soft/hard semantics and add local helpers inside `mask_bindings`:

```python
        def source(state):
            return lambda: _factor(state, hard)

        def append(parameter, mask, kind):
            if parameter is None:
                return
            if any(binding.parameter is parameter for binding in bindings):
                raise RuntimeError(f"parâmetro BERT recebeu binding duplicado: {kind}")
            bindings.append(PlasticityMaskBinding(parameter, mask, kind))
```

Then rebuild the method according to the parameter table in this plan. Use these exact residual indices:

```python
embedding_state = self.residual_trackers[0]
attention_state = self.residual_trackers[1 + 2 * layer_index]
block_state = self.residual_trackers[2 + 2 * layer_index]
input_state = (
    embedding_state
    if layer_index == 0
    else self.residual_trackers[2 * layer_index]
)
```

Important implementation rules:

- Only request residual indices if `_needs_residual_states()` is true.
- Bind each embedding matrix with `source(embedding_state)().reshape(1, -1)` through a callable.
- Bind LayerNorm vectors directly with `source(state)`.
- Use `_dynamic_matrix_mask(row_source, column_source)` for every dense weight.
- For attention rows use a callable returning `tracker.expanded_head_scales(hard=hard)`.
- If both endpoint sources are `None`, do not bind that parameter.
- Biases only receive output/row factors; never input/column factors.
- Pooler input is the last block residual state.
- Classifier input is the pooler state.
- Stable `kind` values must encode the module path and endpoint families, for example `bert_layer_0_query_attention_rows_residual_columns`. This metadata is checkpoint identity; do not use generated object IDs.

The implementation should remain one method, plus the small mask helper from Task 2. Do not introduce a separate graph library.

**Step 4: Verify GREEN and optimizer registration**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_full_coverage_binds_every_trainable_parameter_exactly_once tests/test_slow_heat_bert.py::test_full_coverage_masks_match_every_parameter_shape_by_broadcast tests/test_optim.py -q
```

Expected: all selected tests pass.

**Step 5: Commit**

```bash
git add src/dual_heater/bert.py tests/test_slow_heat_bert.py
git commit -m "feat: bind every BERT parameter to functional SlowHeat"
```

### Task 6: Prove endpoint protection affects real optimizer updates

**Objective:** Verify that either endpoint can block an AdamW update, including weight decay.

**Files:**
- Modify: `tests/test_slow_heat_bert.py`

**Step 1: Write behavior test**

Append:

```python
def test_full_coverage_attention_weight_obeys_both_endpoint_masks():
    torch.manual_seed(91)
    model = SlowHeatBertForSequenceClassification(
        _bert_config(), _full_coverage_config()
    )
    attention = model.attention_trackers[0]
    input_residual = model.residual_trackers[0]
    attention.slow_heat.copy_(torch.tensor([1.0, 0.0]))
    input_residual.slow_heat.zero_()
    input_residual.slow_heat[0] = 1.0

    query = model.bert.encoder.layer[0].attention.self.query.weight
    before = query.detach().clone()
    optimizer = SlowHeatAdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    model.register_plasticity_masks(optimizer, hard=True)
    _backward(model, torch.tensor([[2, 7, 3]]), torch.ones(1, 3, dtype=torch.long))
    optimizer.step()

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    assert torch.equal(query[:head_dim], before[:head_dim])
    assert torch.equal(query[head_dim:, 0], before[head_dim:, 0])
    assert not torch.equal(query[head_dim:, 1:], before[head_dim:, 1:])
```

**Step 2: Verify RED**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_full_coverage_attention_weight_obeys_both_endpoint_masks -q
```

Expected: FAIL if the row/column composition or binding target is wrong.

**Step 3: Correct minimally**

Correct only the Q/K/V factorized binding. Do not modify optimizer semantics.

**Step 4: Verify GREEN**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_full_coverage_attention_weight_obeys_both_endpoint_masks tests/test_optim.py -q
```

Expected: all pass.

**Step 5: Commit**

```bash
git add tests/test_slow_heat_bert.py src/dual_heater/bert.py
git commit -m "test: verify BERT endpoint protection under AdamW"
```

### Task 7: Consolidate and persist all new scientific state

**Objective:** Include residual and pooler trackers in consolidation, capacity metrics, half-precision guarantees, and checkpoint identity.

**Files:**
- Modify: `tests/test_slow_heat_bert.py`
- Modify: `src/dual_heater/bert.py`

**Step 1: Write failing tests**

Append:

```python
def test_full_coverage_consolidates_every_tracker_family():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(), _full_coverage_config()
    )
    _backward(model, torch.tensor([[2, 7, 3]]), torch.ones(1, 3, dtype=torch.long))

    model.consolidate(strategy="max")

    states = model.get_slow_states()
    assert len(states) == 7  # FFN + attention + E/A/R + pooler + classifier
    assert all(state.consolidated_tasks.item() == 1 for state in states)
    assert all(state.task_step.item() == 0 for state in states)


def test_full_coverage_scientific_state_stays_fp32_after_half():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(), _full_coverage_config()
    ).half()

    assert all(state.slow_heat.dtype is torch.float32 for state in model.get_slow_states())
    assert all(state.task_step.dtype is torch.int64 for state in model.get_slow_states())
```

**Step 2: Verify RED**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_full_coverage_consolidates_every_tracker_family tests/test_slow_heat_bert.py::test_full_coverage_scientific_state_stays_fp32_after_half -q
```

Expected: FAIL because `get_slow_states` and non-local consolidation omit new trackers.

**Step 3: Implement state enumeration once**

Update `get_slow_states` in `src/dual_heater/bert.py` to return, in stable order:

```text
for each layer: FFN if enabled, attention if enabled
all residual trackers in E/A0/R0/A1/R1/... order
pooler tracker if present
classifier tracker if present
```

Update non-local `consolidate` families to:

```python
families = [
    list(self.ffn_trackers),
    list(self.attention_trackers),
    list(self.residual_trackers),
    [self.pooler_tracker] if self.pooler_tracker is not None else [],
    [self.classifier_tracker] if self.classifier_tracker is not None else [],
]
```

Remove the old separate classifier consolidation after the family loop to avoid double consolidation. `capacity_metrics()` must continue deriving directly from `get_slow_states()`.

**Step 4: Bump schema and update missing-key patterns**

Set:

```python
    slowheat_schema_version = 3
```

Add missing-key patterns for `residual_trackers.*` and `pooler_tracker.*`. Ensure `_slowheat_identity_payload()` already includes all config fields through `asdict`.

**Step 5: Verify GREEN and checkpoint tests**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_full_coverage_consolidates_every_tracker_family tests/test_slow_heat_bert.py::test_full_coverage_scientific_state_stays_fp32_after_half tests/test_slow_heat_bert.py -q
```

Expected: full BERT test file passes. If historical round-trip tests fail because they assert schema 2, update them to schema 3; do not weaken mismatch rejection.

**Step 6: Commit**

```bash
git add src/dual_heater/bert.py tests/test_slow_heat_bert.py
git commit -m "feat: persist full-coverage BERT SlowHeat state"
```

### Task 8: Define ablation configurations as data, not condition chains

**Objective:** Resolve all requested methods to explicit, auditable coverage switches.

**Files:**
- Modify: `tests/test_split_clinc150.py`
- Modify: `experiments/split_clinc150.py`

**Step 1: Write failing registry test**

In `tests/test_split_clinc150.py`, import `BERT_FULL_COVERAGE_VARIANTS`, then append:

```python
def test_full_coverage_variant_preset_contains_control_and_leave_one_out_methods():
    assert BERT_FULL_COVERAGE_VARIANTS == (
        "vanilla",
        "slowheat_full_coverage",
        "slowheat_all_minus_embeddings",
        "slowheat_all_minus_layernorm",
        "slowheat_all_minus_residual",
        "slowheat_all_minus_attention",
        "slowheat_all_minus_ffn",
        "slowheat_all_minus_pooler",
        "slowheat_all_minus_classifier",
        "slowheat_ffn_attention",
    )


def test_full_coverage_variants_change_exactly_the_named_family():
    config = SplitCLINC150Config(methods=BERT_FULL_COVERAGE_VARIANTS)
    full = clinc_module._slowheat_config(config, "slowheat_full_coverage")
    assert full.fast_heat is None
    assert full.freeze_unbound_parameters is False
    assert full.capacity_scope == "hierarchical"

    fields = {
        "embeddings": "track_embeddings",
        "layernorm": "protect_layer_norm",
        "residual": "track_residual",
        "attention": "track_attention",
        "ffn": "track_ffn",
        "pooler": "protect_pooler",
        "classifier": "protect_classifier",
    }
    for family, field in fields.items():
        candidate = clinc_module._slowheat_config(
            config, f"slowheat_all_minus_{family}"
        )
        for checked in fields.values():
            expected = False if checked == field else getattr(full, checked)
            assert getattr(candidate, checked) is expected

    baseline = clinc_module._slowheat_config(config, "slowheat_ffn_attention")
    assert baseline.track_ffn and baseline.track_attention
    assert not baseline.track_embeddings
    assert not baseline.track_residual
    assert not baseline.protect_layer_norm
    assert not baseline.protect_pooler
    assert not baseline.protect_classifier
```

**Step 2: Verify RED**

```bash
python3 -m pytest tests/test_split_clinc150.py::test_full_coverage_variant_preset_contains_control_and_leave_one_out_methods tests/test_split_clinc150.py::test_full_coverage_variants_change_exactly_the_named_family -q
```

Expected: collection ERROR because the constant and methods are missing.

**Step 3: Add the registry**

In `experiments/split_clinc150.py`, add the exact `BERT_FULL_COVERAGE_VARIANTS` tuple from this plan. Add all non-vanilla names to `SUPPORTED_METHODS` and `SLOWHEAT_METHODS`; do not add them to replay, LoRA, or FastHeat sets.

Add this data table near the constants:

```python
_FULL_COVERAGE_SWITCHES = {
    "track_ffn": True,
    "track_attention": True,
    "track_embeddings": True,
    "track_residual": True,
    "protect_layer_norm": True,
    "protect_pooler": True,
    "protect_classifier": True,
}

_FULL_COVERAGE_REMOVALS = {
    "slowheat_all_minus_embeddings": "track_embeddings",
    "slowheat_all_minus_layernorm": "protect_layer_norm",
    "slowheat_all_minus_residual": "track_residual",
    "slowheat_all_minus_attention": "track_attention",
    "slowheat_all_minus_ffn": "track_ffn",
    "slowheat_all_minus_pooler": "protect_pooler",
    "slowheat_all_minus_classifier": "protect_classifier",
}
```

Refactor `_slowheat_config` so:

- Existing method branches remain behaviorally identical.
- `slowheat_full_coverage` starts from `_FULL_COVERAGE_SWITCHES`.
- A leave-one-out method copies that dictionary and sets its mapped field to `False`.
- `slowheat_ffn_attention` resolves to the same switches as historical `slowheat`.
- New methods force `capacity_scope="hierarchical"`, `fast_heat=None`, and `freeze_unbound_parameters=False`.
- Pass `residual_plasticity_budget=config.residual_plasticity_budget` and `pooler_plasticity_budget=config.pooler_plasticity_budget` into `BertSlowHeatConfig`.

Do not infer a family by splitting arbitrary strings; use the explicit table so typos fail validation.

**Step 4: Verify GREEN**

```bash
python3 -m pytest tests/test_split_clinc150.py::test_full_coverage_variant_preset_contains_control_and_leave_one_out_methods tests/test_split_clinc150.py::test_full_coverage_variants_change_exactly_the_named_family tests/test_split_clinc150.py::test_bert_heat_variant_preset_is_a_matched_four_method_ablation -q
```

Expected: `3 passed`.

**Step 5: Commit**

```bash
git add experiments/split_clinc150.py tests/test_split_clinc150.py
git commit -m "feat: register BERT full-coverage ablations"
```

### Task 9: Add residual and pooler budgets to CLINC150 configuration

**Objective:** Persist and validate all capacity settings used by the new methods.

**Files:**
- Modify: `tests/test_split_clinc150.py`
- Modify: `experiments/split_clinc150.py`

**Step 1: Write failing test**

Append:

```python
def test_clinc_config_propagates_extended_capacity_budgets():
    config = SplitCLINC150Config(
        methods=("slowheat_full_coverage",),
        residual_plasticity_budget=0.5,
        pooler_plasticity_budget=0.75,
    )
    config.validate()
    resolved = clinc_module._slowheat_config(config, "slowheat_full_coverage")

    assert resolved.residual_plasticity_budget == 0.5
    assert resolved.pooler_plasticity_budget == 0.75
```

**Step 2: Verify RED**

```bash
python3 -m pytest tests/test_split_clinc150.py::test_clinc_config_propagates_extended_capacity_budgets -q
```

Expected: FAIL because `SplitCLINC150Config` lacks these fields.

**Step 3: Implement minimally**

Add to `SplitCLINC150Config`:

```python
    residual_plasticity_budget: float = 0.25
    pooler_plasticity_budget: float = 0.25
```

Include both when `validate()` constructs `BertSlowHeatConfig`. Do not add a calibration grid for them yet; the first study holds them equal to the existing default and reports this assumption.

**Step 4: Verify GREEN**

```bash
python3 -m pytest tests/test_split_clinc150.py::test_clinc_config_propagates_extended_capacity_budgets -q
```

Expected: `1 passed`.

**Step 5: Commit**

```bash
git add experiments/split_clinc150.py tests/test_split_clinc150.py
git commit -m "feat: configure residual and pooler capacity budgets"
```

### Task 10: Expose a dedicated CLI preset

**Objective:** Run the complete ablation matrix without replacing the existing heat-allocation preset.

**Files:**
- Modify: `tests/test_split_clinc150.py`
- Modify: `experiments/split_clinc150.py`

**Step 1: Write failing parser test**

Append:

```python
def test_cli_full_coverage_variants_is_exclusive_with_other_method_selectors():
    parser = clinc_module.build_parser()
    args = parser.parse_args(["--full-coverage-variants"])
    assert args.full_coverage_variants is True

    with pytest.raises(SystemExit):
        parser.parse_args(["--full-coverage-variants", "--heat-variants"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--full-coverage-variants", "--methods", "slowheat_full_coverage"]
        )
```

**Step 2: Verify RED**

```bash
python3 -m pytest tests/test_split_clinc150.py::test_cli_full_coverage_variants_is_exclusive_with_other_method_selectors -q
```

Expected: FAIL with unrecognized argument.

**Step 3: Implement minimally**

Add this argument to the existing mutually exclusive `method_group`:

```python
    method_group.add_argument(
        "--full-coverage-variants",
        action="store_true",
        help="executa vanilla, cobertura SlowHeat total e ablações leave-one-family-out",
    )
```

Resolve methods in `main()` in this precedence, which is safe because argparse enforces exclusivity:

```python
        methods=(
            BERT_FULL_COVERAGE_VARIANTS
            if args.full_coverage_variants
            else BERT_HEAT_VARIANTS
            if args.heat_variants
            else tuple(args.methods)
            if args.methods is not None
            else SplitCLINC150Config.methods
        ),
```

Do not change `--heat-variants` behavior.

**Step 4: Verify GREEN**

```bash
python3 -m pytest tests/test_split_clinc150.py::test_cli_full_coverage_variants_is_exclusive_with_other_method_selectors tests/test_split_clinc150.py::test_cli_evaluate_test_defaults_to_false -q
```

Expected: `2 passed`.

**Step 5: Commit**

```bash
git add experiments/split_clinc150.py tests/test_split_clinc150.py
git commit -m "feat: add full-coverage BERT ablation preset"
```

### Task 11: Record mask coverage so ablations cannot be misread

**Objective:** Report how many trainable parameters are masked in each variant without materializing expanded masks.

**Files:**
- Modify: `tests/test_slow_heat_bert.py`
- Modify: `src/dual_heater/bert.py`
- Modify: `tests/test_split_clinc150.py`
- Modify: `experiments/split_clinc150.py`

**Step 1: Write failing model-level test**

Append to `tests/test_slow_heat_bert.py`:

```python
def test_mask_coverage_counts_bound_parameter_elements_without_expansion():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(), _full_coverage_config()
    )
    summary = model.mask_coverage_summary()

    assert summary["trainable_parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert summary["masked_parameter_count"] == summary["trainable_parameter_count"]
    assert summary["masked_fraction"] == pytest.approx(1.0)
    assert summary["binding_count"] == len(model.mask_bindings())
```

**Step 2: Verify RED**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_mask_coverage_counts_bound_parameter_elements_without_expansion -q
```

Expected: FAIL because `mask_coverage_summary` is missing.

**Step 3: Implement model summary**

Add exactly:

```python
    def mask_coverage_summary(self) -> dict[str, int | float]:
        bindings = self.mask_bindings()
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        trainable_ids = {id(parameter) for parameter in trainable}
        masked = [
            binding.parameter
            for binding in bindings
            if id(binding.parameter) in trainable_ids
        ]
        trainable_count = sum(parameter.numel() for parameter in trainable)
        masked_count = sum(parameter.numel() for parameter in masked)
        return {
            "binding_count": len(bindings),
            "trainable_parameter_count": trainable_count,
            "masked_parameter_count": masked_count,
            "masked_fraction": (
                masked_count / trainable_count if trainable_count else 0.0
            ),
        }
```

This measures parameter coverage, not the current fraction of elements protected by heat. Name it accordingly.

**Step 4: Add runner result test first**

Append to `tests/test_split_clinc150.py` using the existing tiny-BERT patch helper:

```python
def test_tiny_full_coverage_run_reports_complete_mask_coverage(monkeypatch):
    _patch_tiny_bert(monkeypatch)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    config = SplitCLINC150Config(
        max_length=12,
        batch_size=30,
        replay_batch_size=5,
        replay_per_class=1,
        epochs_per_task=1,
        methods=("slowheat_full_coverage",),
    )

    result = run_split_clinc150(config, tasks)["slowheat_full_coverage"]

    assert result["mask_coverage"]["masked_fraction"] == pytest.approx(1.0)
```

Run:

```bash
python3 -m pytest tests/test_split_clinc150.py::test_tiny_full_coverage_run_reports_complete_mask_coverage -q
```

Expected: FAIL because the result omits `mask_coverage`.

**Step 5: Add runner output**

In the result dictionary in `experiments/split_clinc150.py`, set:

```python
            "mask_coverage": (
                slow_model.mask_coverage_summary()
                if slow_model is not None
                else None
            ),
```

Move the existing `_find_slowheat_model(model)` call before result construction and reuse that variable during cleanup; do not search the module tree twice.

**Step 6: Verify GREEN**

```bash
python3 -m pytest tests/test_slow_heat_bert.py::test_mask_coverage_counts_bound_parameter_elements_without_expansion tests/test_split_clinc150.py::test_tiny_full_coverage_run_reports_complete_mask_coverage -q
```

Expected: `2 passed`.

**Step 7: Commit**

```bash
git add src/dual_heater/bert.py experiments/split_clinc150.py tests/test_slow_heat_bert.py tests/test_split_clinc150.py
git commit -m "feat: report BERT SlowHeat mask coverage"
```

### Task 12: Extend telemetry without breaking its schema shape

**Objective:** Make residual, pooler, and classifier heat inspectable in experiment artifacts.

**Files:**
- Modify: `tests/test_live_telemetry.py`
- Modify: `experiments/live_telemetry.py`

**Step 1: Write failing test**

Use the tiny BERT configuration style already present in the test suite and add:

```python
def test_heat_snapshot_includes_extended_bert_tracker_families():
    transformers = pytest.importorskip("transformers")
    from dual_heater.bert import (
        BertSlowHeatConfig,
        SlowHeatBertForSequenceClassification,
    )

    config = transformers.BertConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=12,
        num_labels=4,
    )
    model = SlowHeatBertForSequenceClassification(
        config,
        BertSlowHeatConfig(
            track_embeddings=True,
            track_residual=True,
            protect_layer_norm=True,
            protect_pooler=True,
            protect_classifier=True,
        ),
    )
    snapshot = build_heat_snapshot(
        model,
        context={"method": "slowheat_full_coverage", "stage": 0},
        run_id="run",
        session_id="session",
        sequence=1,
    )

    assert len(snapshot["residual"]) == 3
    assert len(snapshot["pooler"]) == 1
    assert len(snapshot["classifier"]) == 1
```

Add missing imports (`pytest` and `build_heat_snapshot`) only if the file does not already contain them.

**Step 2: Verify RED**

```bash
python3 -m pytest tests/test_live_telemetry.py::test_heat_snapshot_includes_extended_bert_tracker_families -q
```

Expected: FAIL because the new keys are absent.

**Step 3: Implement minimally**

Initialize these keys in `build_heat_snapshot`:

```python
        "residual": [],
        "pooler": [],
        "classifier": [],
```

Populate them through `get_residual_trackers`, `pooler_tracker`, and `classifier_tracker`. Use `_tracker_snapshot`; represent singleton trackers as zero- or one-element lists so consumers can iterate uniformly. Update `available` to include all five SlowHeat families.

Bump `TELEMETRY_SCHEMA_VERSION` from 1 to 2 because the serialized snapshot shape changes. Do not alter event fields unrelated to heat snapshots.

**Step 4: Verify GREEN and telemetry regression**

```bash
python3 -m pytest tests/test_live_telemetry.py::test_heat_snapshot_includes_extended_bert_tracker_families tests/test_live_telemetry.py -q
```

Expected: complete telemetry test file passes.

**Step 5: Commit**

```bash
git add experiments/live_telemetry.py tests/test_live_telemetry.py
git commit -m "feat: expose full-coverage BERT heat telemetry"
```

### Task 13: Test each ablation’s actual mask difference

**Objective:** Ensure method names change mask ownership, not merely configuration metadata.

**Files:**
- Modify: `tests/test_split_clinc150.py`

**Step 1: Write parameter-family assertions**

Append a parametrized test that constructs a tiny model through `_build_model` or directly from `_slowheat_config`:

```python
@pytest.mark.parametrize(
    ("method", "uncovered_fragment"),
    [
        ("slowheat_all_minus_embeddings", "word_embeddings.weight"),
        ("slowheat_all_minus_layernorm", "LayerNorm.weight"),
        ("slowheat_all_minus_attention", "attention.self.query.bias"),
        ("slowheat_all_minus_ffn", "intermediate.dense.bias"),
        ("slowheat_all_minus_pooler", "pooler.dense.bias"),
        ("slowheat_all_minus_classifier", "classifier.bias"),
    ],
)
def test_leave_one_out_method_exposes_the_named_unmasked_family(
    method, uncovered_fragment
):
    model_config = transformers.BertConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=12,
        num_labels=4,
    )
    protocol = SplitCLINC150Config(methods=(method,))
    model = clinc_module.SlowHeatBertForSequenceClassification(
        model_config, clinc_module._slowheat_config(protocol, method)
    )

    assert any(
        uncovered_fragment in name
        for name in model.uncovered_trainable_parameters()
    )
```

Add a dedicated residual test because residual removal intentionally leaves parameters partially controlled by FFN/head endpoint factors:

```python
def test_minus_residual_removes_column_and_residual_row_factors():
    config = SplitCLINC150Config(methods=("slowheat_all_minus_residual",))
    model = SlowHeatBertForSequenceClassification(
        transformers.BertConfig(
            vocab_size=64,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=12,
            num_labels=4,
        ),
        clinc_module._slowheat_config(config, "slowheat_all_minus_residual"),
    )
    kinds = {binding.kind for binding in model.mask_bindings()}

    assert not any("residual_rows" in kind for kind in kinds)
    assert not any("residual_columns" in kind for kind in kinds)
    assert any("attention_rows" in kind for kind in kinds)
    assert any("ffn_rows" in kind for kind in kinds)
```

**Step 2: Verify RED**

```bash
python3 -m pytest tests/test_split_clinc150.py::test_leave_one_out_method_exposes_the_named_unmasked_family tests/test_split_clinc150.py::test_minus_residual_removes_column_and_residual_row_factors -q
```

Expected: FAIL for any wiring mismatch. A method test that passes before wiring is complete must be strengthened to assert a concrete parameter or stable `kind`.

**Step 3: Fix only resolution or binding errors**

Keep the method table and graph table as sources of truth. Do not add exceptions in tests for accidental coverage.

**Step 4: Verify GREEN**

Run the command from Step 2; expected all parameter cases plus residual test pass.

**Step 5: Commit**

```bash
git add tests/test_split_clinc150.py experiments/split_clinc150.py src/dual_heater/bert.py
git commit -m "test: verify BERT leave-one-family-out wiring"
```

### Task 14: Document semantics and execution protocol

**Objective:** Make the experimental claims and limitations explicit before any expensive run.

**Files:**
- Modify: `docs/mechanisms/functional_slowheat_transformers.md`
- Modify: `docs/audits/methods_catalog.md`
- Create: `docs/results/bert_full_coverage_ablation.md`

**Step 1: Add documentation**

`docs/results/bert_full_coverage_ablation.md` must include:

1. The parameter graph and table from this plan.
2. Exact definitions of all ten preset methods.
3. The distinction between “trainable” and “masked”.
4. Why endpoint factors use `minimum`.
5. Why padding is excluded.
6. Why FastHeat and replay are excluded from this first ablation.
7. Why the suite is not parameter-cost matched: removing a family changes `masked_parameter_count`; `mask_coverage` must accompany every comparison.
8. Metrics to inspect: FAA, forgetting, task-aware accuracy, macro-F1, classifier gap, trainable parameters, mask coverage, and peak memory.
9. The exploratory command:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m experiments.split_clinc150 \
  --full-coverage-variants \
  --device cpu \
  --seeds 0 \
  --epochs-per-task 1 \
  --batch-size 2 \
  --output-dir results/bert_full_coverage_smoke
```

State clearly that this command may load the configured remote model/dataset and is not part of automated validation. It must not be run without approval because dataset/model downloads may occur.

10. The intended confirmatory workflow: calibration on validation only, frozen manifest, disjoint seeds/orders, then test evaluation.

Update `docs/mechanisms/functional_slowheat_transformers.md` to link the new document and replace statements that embeddings/LayerNorm/pooler are always unbound with the new optional behavior. Keep residual-stream caveats: this implementation coordinates local graph endpoints but does not prove representation-basis invariance.

Update the BERT table in `docs/audits/methods_catalog.md` with the new preset and identifiers.

**Step 2: Verify references**

```bash
python3 - <<'PY'
from pathlib import Path
root = Path("docs")
for name in (
    "bert_full_coverage_ablation.md",
    "functional_slowheat_transformers.md",
    "methods_catalog.md",
):
    text = (root / name).read_text(encoding="utf-8")
    assert "slowheat_full_coverage" in text
print("documentation references verified")
PY
```

Expected: `documentation references verified`.

**Step 3: Commit**

```bash
git add docs/results/bert_full_coverage_ablation.md docs/mechanisms/functional_slowheat_transformers.md docs/audits/methods_catalog.md
git commit -m "docs: specify BERT full-coverage ablations"
```

### Task 15: Run the complete CPU quality gate

**Objective:** Verify code, tests, style, and the offline tiny-model runner before declaring implementation complete.

**Files:** None unless a failing test reveals a defect; any defect must receive a focused regression test before correction.

**Step 1: Run focused tests**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest tests/test_slow_heat_bert.py tests/test_split_clinc150.py tests/test_live_telemetry.py tests/test_optim.py -q
```

Expected: all selected tests pass, no CUDA allocation, no network access.

**Step 2: Run complete test suite**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest -q
```

Expected: all tests pass. Optional-dependency tests may be skipped only through their existing `pytest.importorskip`; there must be no new failures or warnings introduced by this feature.

**Step 3: Run lint**

```bash
python3 -m ruff check src/dual_heater/bert.py experiments/split_clinc150.py experiments/live_telemetry.py tests/test_slow_heat_bert.py tests/test_split_clinc150.py tests/test_live_telemetry.py
```

Expected: `All checks passed!`.

**Step 4: Inspect final diff**

```bash
git diff --check
git status --short
git log --oneline -15
```

Expected:

- `git diff --check` prints nothing.
- `git status --short` is empty after commits.
- The recent log contains the focused commits listed above.

**Step 5: Final commit only if validation required a tested fix**

```bash
git add <exact-tested-files>
git commit -m "fix: correct BERT full-coverage validation"
```

Do not create an empty “final” commit.

---

## Tests / validation summary

The implementation is acceptable only if all of these are demonstrated by real test output:

- Historical `BertSlowHeatConfig()` still enables only FFN and attention.
- Full coverage creates one embedding tracker, two residual trackers per encoder block, one pooler tracker, and one classifier tracker.
- Every token-level tracker excludes padding.
- Full coverage binds every trainable parameter exactly once.
- Every mask broadcasts to its target parameter and remains finite in `[0,1]`.
- A protected row or protected column independently blocks the complete AdamW update, including decoupled weight decay.
- Full coverage with no consolidation remains equivalent to native AdamW because every dynamic factor equals one.
- New scientific state stays FP32 when model parameters are FP16.
- Checkpoints persist and validate the schema-3 topology/configuration.
- Existing schema-2 or incompatible topology checkpoints are rejected clearly.
- Every leave-one-family-out method changes the intended wiring.
- Existing historical method IDs and `--heat-variants` behavior remain unchanged.
- `--full-coverage-variants` is mutually exclusive with other method selectors.
- Tiny offline CLINC150 tests complete without network or GPU.
- Results include `mask_coverage` so unequal coverage cannot be hidden.
- Telemetry includes FFN, attention, residual, pooler, and classifier states.
- Full test suite and ruff pass.

## Risks and tradeoffs

1. **Ablations are not cost matched.** All variants keep the same trainable parameter count, but removing a factor changes how many parameters are masked. Report `mask_coverage`; do not interpret a gain as proof that the removed biological/functional family is unnecessary without a later matched-effective-plasticity study.
2. **Residual dimensions are basis-dependent.** Hidden coordinates can rotate or be remixed across layers. Coordinated endpoint masking is structurally consistent for a fixed BERT parameterization, but it is not invariant to reparameterization.
3. **LayerNorm shares residual importance.** There is no separate LayerNorm tracker. The LayerNorm ablation changes only affine-parameter masking while retaining residual measurements. This is intentional and must be documented.
4. **Embedding masking is by hidden dimension, not token row.** It protects representation coordinates consistently across vocabulary, position, and token-type embeddings. Token-specific importance is deliberately out of scope.
5. **Attention removal is partial at the parameter level.** With residual tracking enabled, Q/K/V columns and attention output rows can still be masked by residual endpoints. The ablation removes head-specific importance, not every mask touching attention parameters.
6. **FFN removal is similarly partial.** Residual factors may still mask FFN input columns and output rows. The ablation removes intermediate-neuron importance.
7. **Pooler/classifier coupling.** In `minus_classifier`, classifier columns may still be constrained by pooler factors; in `minus_pooler`, pooler columns may still be constrained by final residual factors. Tests and docs must describe factor removal rather than claiming entire tensors are unmasked.
8. **Hook memory.** Added residual hooks retain detached activation storage until backward through the existing tracker mechanism. Measure peak memory before BERT-base; activation checkpointing remains unsupported.
9. **Checkpoint incompatibility.** Schema bump intentionally rejects old SlowHeat checkpoints. Native Hugging Face BERT checkpoints must continue loading with zeroed mechanism state.
10. **Runtime.** The ten-method preset is expensive. Automated validation must use tiny local configurations only; full CLINC150 runs require explicit approval for downloads and GPU use.
11. **LoRA.** Exact producer-only LoRA currently knows only FFN and attention producer targets. Do not claim new full-coverage methods support LoRA; reject or leave them outside `LORA_METHODS`.

## Open questions for the experiment phase

- Should confirmatory comparisons match effective plasticity by the number of masked parameter elements, mean dynamic mask value, or estimated update norm? The initial implementation reports coverage but does not tune it.
- Should residual capacity be one global family across embedding and all block junctions, or hierarchical with a guaranteed quota per junction? The proposed implementation uses the existing hierarchical allocator over all residual trackers.
- Should the pooler be included at all for models/configurations that do not use it? This plan fails clearly when `protect_pooler=True` and no pooler exists; support for pooler-free architectures is out of scope.
- Should a future classifier study protect old class rows only using label-conditioned utility rather than generic `|z*dL/dz|`? Keep the generic tracker for this implementation so the first comparison changes only structural coverage.
- After the smoke suite passes, choose calibration seeds and task orders before any test-set evaluation. Do not reuse the historical single-seed outputs as confirmatory evidence.
