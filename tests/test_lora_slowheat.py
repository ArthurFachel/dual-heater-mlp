"""Contract tests for the three SlowHeat-in-LoRA mechanisms.

Each mechanism claims a specific exactness property. These tests assert the
property by measuring drift in the quantity the mechanism says it protects,
after a real optimizer step, rather than by inspecting the mask.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, Qwen2ForSequenceClassification

from dual_heater.lora_slowheat import (
    LoRASlowHeatConfig,
    QwenLoRASlowHeat,
    build_lora_slowheat,
    rank_slice_bounds,
)
from dual_heater.optim import SlowHeatAdamW

RANK = 4
LABELS = 6


def _tiny_qwen() -> Qwen2ForSequenceClassification:
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    config.num_labels = LABELS
    config.num_hidden_layers = 2
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.pad_token_id = 0
    config.vocab_size = 128
    # Qwen's published config declares bfloat16. Pascal cards have no native
    # bf16 and PEFT's modules_to_save head is created in fp32, so an unforced
    # dtype makes the classifier head mismatch the body.
    config.torch_dtype = torch.float32
    torch.manual_seed(0)
    return Qwen2ForSequenceClassification(config)


def _build(method: str, **overrides):
    config = LoRASlowHeatConfig(method=method, rank=RANK, **overrides)
    model, instrumentation = build_lora_slowheat(_tiny_qwen(), config)
    return model, instrumentation


def _batch(batch_size: int = 2, length: int = 6):
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(1, 128, (batch_size, length), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -2:] = 0  # real padding, so the validity mask matters
    labels = torch.randint(0, LABELS, (batch_size,), generator=generator)
    return input_ids, attention_mask, labels


def _optimizer(model, instrumentation, lr: float = 0.1):
    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    instrumentation.register_plasticity_masks(optimizer)
    return optimizer


def _train_step(model, instrumentation, optimizer) -> None:
    model.train()
    instrumentation.train()
    input_ids, attention_mask, labels = _batch()
    with instrumentation.validity(attention_mask):
        loss = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _randomize_b(instrumentation: QwenLoRASlowHeat, scale: float = 0.2) -> None:
    """B starts at zero by LoRA convention; give it a dense, realistic value."""

    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for adapted in instrumentation.modules:
            adapted.lora_b.weight.copy_(
                torch.randn(
                    adapted.lora_b.weight.shape, generator=generator
                )
                * scale
            )


# ---------------------------------------------------------------------------
# schedule arithmetic
# ---------------------------------------------------------------------------


def test_rank_slices_partition_the_rank_exactly() -> None:
    for rank, tasks in ((8, 2), (8, 3), (5, 5), (7, 2)):
        covered: list[int] = []
        for index in range(tasks):
            start, end = rank_slice_bounds(rank, tasks, index)
            assert end > start, "toda tarefa deve receber ao menos uma dimensão"
            covered.extend(range(start, end))
        assert sorted(covered) == list(range(rank))


def test_slice_rejects_a_rank_smaller_than_the_task_count() -> None:
    with pytest.raises(ValueError):
        LoRASlowHeatConfig(method="slice", rank=2, task_count=3)


def test_slice_refuses_to_reuse_a_spent_schedule() -> None:
    _, instrumentation = _build("slice", task_count=2)
    instrumentation.begin_task(0)
    instrumentation.begin_task(1)
    with pytest.raises(RuntimeError):
        instrumentation.begin_task(2)


# ---------------------------------------------------------------------------
# binding hygiene
# ---------------------------------------------------------------------------


def test_vanilla_registers_no_masks() -> None:
    model, instrumentation = _build("vanilla")
    assert instrumentation.mask_bindings() == []
    assert instrumentation.effective_plasticity() == 1.0
    # And it must still be a working trainable LoRA.
    assert any(
        ".lora_A." in name and parameter.requires_grad
        for name, parameter in model.named_parameters()
    )


@pytest.mark.parametrize("method", ["exact", "rank", "leak", "slice"])
def test_every_masked_parameter_is_bound_exactly_once(method: str) -> None:
    _, instrumentation = _build(method, task_count=2)
    bindings = instrumentation.mask_bindings()
    identifiers = [id(binding.parameter) for binding in bindings]
    assert len(identifiers) == len(set(identifiers))
    assert all(binding.kind for binding in bindings)


@pytest.mark.parametrize("method", ["exact", "rank", "leak", "slice"])
def test_masks_broadcast_to_their_parameter_and_stay_in_the_unit_interval(
    method: str,
) -> None:
    _, instrumentation = _build(method, task_count=2)
    _randomize_b(instrumentation)
    for binding in instrumentation.mask_bindings():
        mask = binding.mask() if callable(binding.mask) else binding.mask
        assert (
            torch.broadcast_shapes(mask.shape, binding.parameter.shape)
            == binding.parameter.shape
        ), binding.kind
        assert torch.isfinite(mask).all(), binding.kind
        assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0, binding.kind


def test_exact_freezes_every_a_matrix() -> None:
    model, instrumentation = _build("exact")
    a_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".lora_A." in name
    ]
    assert a_parameters
    assert not any(parameter.requires_grad for parameter in a_parameters)
    # One binding per module: only B is trainable and masked.
    assert len(instrumentation.mask_bindings()) == len(instrumentation.modules)


# ---------------------------------------------------------------------------
# mechanism 1: rank-space protection
# ---------------------------------------------------------------------------


def test_rank_hard_mask_freezes_the_protected_component_exactly() -> None:
    """The claim: a hard-protected direction j leaves B[:, j] z_j untouched."""

    model, instrumentation = _build("rank", hard=True)
    _randomize_b(instrumentation)
    protected = 0

    for adapted in instrumentation.modules:
        with torch.no_grad():
            adapted.tracker.slow_heat.zero_()
            adapted.tracker.slow_heat[protected] = 1.0

    before = [
        torch.outer(
            adapted.lora_b.weight[:, protected], adapted.lora_a.weight[protected]
        ).clone()
        for adapted in instrumentation.modules
    ]
    optimizer = _optimizer(model, instrumentation)
    _train_step(model, instrumentation, optimizer)
    after = [
        torch.outer(
            adapted.lora_b.weight[:, protected], adapted.lora_a.weight[protected]
        )
        for adapted in instrumentation.modules
    ]

    for index, (old, new) in enumerate(zip(before, after, strict=True)):
        assert torch.equal(old, new), f"componente protegida {index} derivou"

    # The free directions must actually have moved, or the test is vacuous.
    moved = any(
        not torch.equal(
            adapted.lora_b.weight[:, protected + 1],
            torch.zeros_like(adapted.lora_b.weight[:, protected + 1]),
        )
        for adapted in instrumentation.modules
    )
    assert moved


def test_rank_tracker_lives_in_the_bottleneck_not_the_output() -> None:
    _, instrumentation = _build("rank")
    for adapted in instrumentation.modules:
        assert adapted.tracker.units == RANK
        assert adapted.tracker.units != adapted.lora_b.weight.shape[0]


def test_rank_importance_is_recorded_only_while_training() -> None:
    model, instrumentation = _build("rank")
    tracker = instrumentation.modules[0].tracker

    model.eval()
    instrumentation.eval()
    input_ids, attention_mask, labels = _batch()
    with instrumentation.validity(attention_mask):
        model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert int(tracker.task_step.item()) == 0

    optimizer = _optimizer(model, instrumentation)
    _train_step(model, instrumentation, optimizer)
    assert int(tracker.task_step.item()) > 0
    assert torch.isfinite(tracker.task_ema).all()


# ---------------------------------------------------------------------------
# mechanism 2: leak-aware protection with a trainable A
# ---------------------------------------------------------------------------


def test_leak_hard_mask_protects_an_output_row_exactly_with_a_trainable_a() -> None:
    """The claim ``DualHeatLoRALinear`` makes but does not keep.

    With A trainable, masking only rows of B leaves the protected output's
    delta free to move through the shared A. The leak bound must close that.
    """

    model, instrumentation = _build("leak", hard=True)
    _randomize_b(instrumentation)
    protected_output = 0

    for adapted in instrumentation.modules:
        with torch.no_grad():
            adapted.tracker.slow_heat.zero_()
            adapted.tracker.slow_heat[protected_output] = 1.0

    def delta_rows() -> list[torch.Tensor]:
        return [
            (adapted.lora_b.weight[protected_output] @ adapted.lora_a.weight).clone()
            for adapted in instrumentation.modules
        ]

    before = delta_rows()
    optimizer = _optimizer(model, instrumentation)
    _train_step(model, instrumentation, optimizer)
    after = delta_rows()

    for index, (old, new) in enumerate(zip(before, after, strict=True)):
        assert torch.equal(old, new), f"saída protegida {index} derivou"


def test_leak_without_the_a_bound_lets_the_protected_output_drift() -> None:
    """Mutation guard: drop the A binding and the protection must fail.

    This is what makes the previous test meaningful rather than vacuous.
    """

    model, instrumentation = _build("leak", hard=True)
    _randomize_b(instrumentation)
    protected_output = 0
    for adapted in instrumentation.modules:
        with torch.no_grad():
            adapted.tracker.slow_heat.zero_()
            adapted.tracker.slow_heat[protected_output] = 1.0

    b_only = [
        binding
        for binding in instrumentation.mask_bindings()
        if binding.kind.endswith("_B_output_rows")
    ]
    assert b_only
    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    optimizer.register_mask_bindings(b_only)

    before = [
        (adapted.lora_b.weight[protected_output] @ adapted.lora_a.weight).clone()
        for adapted in instrumentation.modules
    ]
    _train_step(model, instrumentation, optimizer)
    after = [
        adapted.lora_b.weight[protected_output] @ adapted.lora_a.weight
        for adapted in instrumentation.modules
    ]
    assert any(
        not torch.equal(old, new) for old, new in zip(before, after, strict=True)
    ), "sem o limite em A a proteção por saída deveria vazar"


def test_leak_bound_collapses_once_b_is_dense() -> None:
    """The mechanism's own falsifier, asserted rather than assumed.

    With a dense B and hard masks, every rank row reaches a protected output,
    so the bound drives all of A to zero plasticity: the method degenerates
    into "freeze A". This must be reported, not discovered later.
    """

    _, instrumentation = _build("leak", hard=True)
    _randomize_b(instrumentation)
    for adapted in instrumentation.modules:
        with torch.no_grad():
            adapted.tracker.slow_heat.zero_()
            adapted.tracker.slow_heat[0] = 1.0
    assert instrumentation.leak_collapse_fraction() == pytest.approx(1.0)

    # With no protected output the bound is inert and A stays fully plastic.
    for adapted in instrumentation.modules:
        with torch.no_grad():
            adapted.tracker.slow_heat.zero_()
    assert instrumentation.leak_collapse_fraction() == pytest.approx(0.0)


def test_leak_weighted_combination_retains_more_plasticity_than_min() -> None:
    results = {}
    for combination in ("min", "weighted"):
        _, instrumentation = _build(
            "leak", hard=False, leak_combination=combination
        )
        _randomize_b(instrumentation)
        for adapted in instrumentation.modules:
            with torch.no_grad():
                adapted.tracker.slow_heat.zero_()
                adapted.tracker.slow_heat[0] = 1.0
        results[combination] = instrumentation.effective_plasticity()
    assert results["weighted"] > results["min"]


# ---------------------------------------------------------------------------
# mechanism 3: rank slicing
# ---------------------------------------------------------------------------


def test_slice_freezes_a_past_task_slice_exactly() -> None:
    model, instrumentation = _build("slice", task_count=2)
    _randomize_b(instrumentation)
    instrumentation.begin_task(1)
    start, end = rank_slice_bounds(RANK, 2, 0)

    before = [
        (
            adapted.lora_a.weight[start:end].clone(),
            adapted.lora_b.weight[:, start:end].clone(),
        )
        for adapted in instrumentation.modules
    ]
    optimizer = _optimizer(model, instrumentation)
    _train_step(model, instrumentation, optimizer)

    for index, adapted in enumerate(instrumentation.modules):
        old_a, old_b = before[index]
        assert torch.equal(old_a, adapted.lora_a.weight[start:end])
        assert torch.equal(old_b, adapted.lora_b.weight[:, start:end])

    current_start, current_end = rank_slice_bounds(RANK, 2, 1)
    assert current_start == end
    moved = any(
        float(
            adapted.lora_b.weight[:, current_start:current_end].abs().sum()
        )
        > 0.0
        for adapted in instrumentation.modules
    )
    assert moved, "a fatia corrente deveria ser plástica"


def test_slice_leaves_the_first_task_fully_plastic() -> None:
    _, instrumentation = _build("slice", task_count=2)
    instrumentation.begin_task(0)
    assert instrumentation.effective_plasticity() == pytest.approx(1.0)
    instrumentation.begin_task(1)
    assert instrumentation.effective_plasticity() < 1.0


# ---------------------------------------------------------------------------
# shared lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["exact", "rank", "leak"])
def test_consolidation_respects_the_plasticity_budget(method: str) -> None:
    model, instrumentation = _build(method, plasticity_budget=0.5)
    optimizer = _optimizer(model, instrumentation)
    _train_step(model, instrumentation, optimizer)
    instrumentation.consolidate()

    for adapted in instrumentation.modules:
        tracker = adapted.tracker
        protected = float((tracker.slow_heat > 0.0).float().mean())
        assert protected <= 0.5 + 1e-9
        assert int(tracker.task_step.item()) == 0, "a próxima task começa do zero"


def test_padding_positions_do_not_contribute_to_importance() -> None:
    """Assert the validity mask on the tracker itself.

    Causal attention plus last-non-pad pooling already zeroes pad-position
    gradients, so an end-to-end padding test would pass even with the mask
    removed. The contract is therefore asserted where it is implemented.
    """

    _, instrumentation = _build("rank")
    tracker = instrumentation.modules[0].tracker
    tracker.train()

    hidden = torch.randn(2, 5, RANK, requires_grad=True)
    mask = torch.ones(2, 5)
    mask[:, -3:] = 0.0
    tracker.observe(hidden, mask)
    upstream = torch.randn(2, 5, RANK)
    hidden.backward(upstream)
    masked_ema = tracker.task_ema.clone()

    tracker.task_ema.zero_()
    tracker.task_step.zero_()
    trimmed = hidden.detach()[:, :2].clone().requires_grad_(True)
    tracker.observe(trimmed, torch.ones(2, 2))
    trimmed.backward(upstream[:, :2])

    assert torch.allclose(masked_ema, tracker.task_ema, atol=1e-6)


def test_the_hook_forwards_the_scoped_validity_mask_to_the_tracker() -> None:
    """Structural assertion at the call site, because the effect is invisible.

    Causal attention plus last-non-pad pooling already zeroes pad-position
    gradients, so deleting the mask from the hook changes no end-to-end number
    and survives every behavioural test. The contract is therefore asserted
    where it is wired: the hook must hand the scoped mask to ``observe``.
    """

    model, instrumentation = _build("rank")
    tracker = instrumentation.modules[0].tracker
    seen: list[torch.Tensor | None] = []
    original = tracker.observe

    def spy(hidden, validity_mask=None):
        seen.append(validity_mask)
        return original(hidden, validity_mask)

    tracker.observe = spy  # type: ignore[method-assign]
    model.train()
    instrumentation.train()
    input_ids, attention_mask, labels = _batch()
    with instrumentation.validity(attention_mask):
        model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert seen, "o hook não chamou observe()"
    assert all(mask is not None for mask in seen)
    assert all(torch.equal(mask, attention_mask) for mask in seen)


def test_the_validity_scope_is_restored_after_the_forward() -> None:
    _, instrumentation = _build("rank")
    _, attention_mask, _ = _batch()
    assert instrumentation._validity_mask is None
    with instrumentation.validity(attention_mask):
        assert instrumentation._validity_mask is not None
    assert instrumentation._validity_mask is None


# ---------------------------------------------------------------------------
# iso-plasticity controller and the lr_control falsifier
# ---------------------------------------------------------------------------


def test_lr_control_registers_no_mask_and_is_fully_plastic() -> None:
    """The falsifier must differ from a mechanism ONLY in how E is removed."""

    model, instrumentation = _build("lr_control")
    assert instrumentation.mask_bindings() == []
    assert instrumentation.effective_plasticity() == 1.0
    assert any(
        ".lora_A." in name and parameter.requires_grad
        for name, parameter in model.named_parameters()
    )


def _plasticity_floor(instrumentation) -> float:
    """E(beta -> infinity): the analytic floor a mask cannot go below."""

    saved = [adapted.tracker.slow_strength for adapted in instrumentation.modules]
    for adapted in instrumentation.modules:
        adapted.tracker.slow_strength = 1e6
    floor = instrumentation.effective_plasticity()
    for adapted, value in zip(instrumentation.modules, saved, strict=True):
        adapted.tracker.slow_strength = value
    return floor


def test_bottleneck_importance_is_zero_while_b_is_still_zero() -> None:
    """LoRA's default init makes the rank-space estimator start blind.

    With ``B = 0`` the gradient reaching the bottleneck is
    ``dL/dz = B^T dL/ddelta = 0``, so ``|z * dL/dz|`` is identically zero no
    matter what the data does. The rank mechanism therefore collects no
    evidence until ``B`` moves away from its initialization, and a single
    training step yields an all-zero importance vector. This is a property of
    the estimator, not a bug, and it must not be mistaken for "no important
    directions".
    """

    model, instrumentation = _build("rank")
    optimizer = _optimizer(model, instrumentation)
    _train_step(model, instrumentation, optimizer)
    tracker = instrumentation.modules[0].tracker
    assert float(tracker.task_ema.abs().sum()) == 0.0

    # Once B is non-zero the same estimator does see the bottleneck.
    _randomize_b(instrumentation)
    _train_step(model, instrumentation, optimizer)
    assert float(tracker.task_ema.abs().sum()) > 0.0


@pytest.mark.parametrize("method", ["exact", "rank", "leak"])
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.9])
def test_controller_hits_any_target_above_the_analytic_floor(
    method: str, fraction: float
) -> None:
    model, instrumentation = _build(method, plasticity_budget=0.5)
    _randomize_b(instrumentation)
    optimizer = _optimizer(model, instrumentation)
    _train_step(model, instrumentation, optimizer)
    instrumentation.consolidate()

    # Targets are chosen relative to the measured floor: E(beta) can never go
    # below (N - P) / N, so an absolute target is not portable across arms.
    floor = _plasticity_floor(instrumentation)
    target = floor + fraction * (1.0 - floor)

    solved = instrumentation.calibrate_to_target_plasticity(target)
    assert solved["solved"] == 1.0
    assert solved["achieved"] == pytest.approx(target, abs=1e-3)
    assert instrumentation.effective_plasticity() == pytest.approx(target, abs=1e-3)


@pytest.mark.parametrize("method", ["exact", "leak"])
def test_controller_reports_failure_below_the_analytic_floor(method: str) -> None:
    """A target under the floor must fail loudly, not silently approximate."""

    model, instrumentation = _build(method, plasticity_budget=0.5)
    _randomize_b(instrumentation)
    optimizer = _optimizer(model, instrumentation)
    _train_step(model, instrumentation, optimizer)
    instrumentation.consolidate()

    floor = _plasticity_floor(instrumentation)
    assert floor > 0.01, "o teste precisa de um piso positivo para ser válido"
    solved = instrumentation.calibrate_to_target_plasticity(floor * 0.5)
    assert solved["solved"] == 0.0


def test_controller_solves_the_slice_floor_not_the_strength() -> None:
    """slice has no beta: its only knob is the floor on frozen dimensions."""

    _, instrumentation = _build("slice", task_count=2)
    instrumentation.begin_task(1)
    assert instrumentation.effective_plasticity() == pytest.approx(0.5, abs=1e-6)

    solved = instrumentation.calibrate_to_target_plasticity(0.75)
    assert solved["solved"] == 1.0
    assert solved["achieved"] == pytest.approx(0.75, abs=1e-3)
    # Raising the floor above zero is exactly what trades the exactness away.
    assert solved["knob"] > 0.0


def test_a_zero_slice_floor_preserves_exact_freezing() -> None:
    """Default slice must stay exact; only calibration relaxes it."""

    model, instrumentation = _build("slice", task_count=2)
    _randomize_b(instrumentation)
    instrumentation.begin_task(1)
    start, end = rank_slice_bounds(RANK, 2, 0)
    before = [
        adapted.lora_b.weight[:, start:end].clone()
        for adapted in instrumentation.modules
    ]
    optimizer = _optimizer(model, instrumentation)
    _train_step(model, instrumentation, optimizer)
    for index, adapted in enumerate(instrumentation.modules):
        assert torch.equal(before[index], adapted.lora_b.weight[:, start:end])


def test_calibrating_arms_to_one_target_equalizes_measured_plasticity() -> None:
    """The point of the protocol: arms differ in HOW, not in HOW MUCH."""

    target = 0.95
    achieved = {}
    for method in ("exact", "rank", "leak"):
        model, instrumentation = _build(method, plasticity_budget=0.5)
        _randomize_b(instrumentation)
        optimizer = _optimizer(model, instrumentation)
        _train_step(model, instrumentation, optimizer)
        instrumentation.consolidate()
        solved = instrumentation.calibrate_to_target_plasticity(target)
        assert solved["solved"] == 1.0, method
        achieved[method] = instrumentation.effective_plasticity()
    for method, value in achieved.items():
        assert value == pytest.approx(target, abs=1e-3), method


def test_controller_rejects_an_out_of_range_target() -> None:
    _, instrumentation = _build("rank")
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            instrumentation.calibrate_to_target_plasticity(bad)


def test_removing_hooks_stops_all_observation() -> None:
    model, instrumentation = _build("rank")
    instrumentation.remove_hooks()
    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    _train_step(model, instrumentation, optimizer)
    assert all(
        int(adapted.tracker.task_step.item()) == 0
        for adapted in instrumentation.modules
    )
