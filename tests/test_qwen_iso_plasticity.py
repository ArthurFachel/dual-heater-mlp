"""CPU tests for the iso-plasticity ablation runner.

Every model is a tiny randomly-initialized Qwen2 (hidden 8, intermediate 12),
so the file runs in seconds and never downloads a checkpoint. The five
assertions of section H of `goals/protocol_iso_plasticity.md` are pinned here,
each with the mutation it is supposed to catch.
"""

import pytest
import torch

pytest.importorskip("transformers")

import transformers

from dual_heater.optim import SlowHeatAdamW
from dual_heater.qwen import (
    QwenSlowHeatConfig,
    SlowHeatQwen2ForSequenceClassification,
)
from experiments.capacity_calibration import (
    effective_plasticity,
    protected_heat_scoped,
)
from experiments.qwen_iso_plasticity import (
    ArmSpec,
    build_arms,
    _endpoints,
    layer_importances,
    measured_plasticity,
    permute_slowheat_heat,
    protocol_hash,
    reachable_iso_points,
    register_reduced_lr_arm,
    resolve_beta,
)


def _qwen_config(*, layers: int = 2, labels: int = 4, intermediate: int = 12):
    return transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_labels=labels,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
    )


def _model(*, budget=0.25, scope="local", layers=2, strength=0.0):
    return SlowHeatQwen2ForSequenceClassification(
        _qwen_config(layers=layers),
        QwenSlowHeatConfig(
            slow_strength=strength,
            ffn_plasticity_budget=budget,
            capacity_scope=scope,
            freeze_unbound_parameters=True,
        ),
    )


def _tokens():
    return torch.tensor([[2, 7, 9, 3]]), torch.ones(1, 4, dtype=torch.long)


def _train_and_consolidate(model, *, label=1):
    model.train()
    input_ids, attention_mask = _tokens()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=torch.tensor([label]),
    )
    output.loss.backward()
    model.zero_grad(set_to_none=True)
    model.consolidate(strategy="max")


def _seed_distinct_importance(model, *, seed=5):
    """Give each tracker a non-degenerate, distinct importance vector.

    A uniform vector would make every budget produce the same heat and let a
    scope bug pass unnoticed.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for index, tracker in enumerate(model.get_ffn_trackers()):
            values = torch.rand(tracker.units, generator=generator) * (10.0**index)
            tracker.task_ema.copy_(values)
            tracker.task_step.fill_(1)
    model.consolidate(strategy="max")


# ---------------------------------------------------------------------------
# arm construction (assertion 5: nothing is chosen after the fact)
# ---------------------------------------------------------------------------


def test_every_declared_budget_becomes_an_arm_plus_three_controls():
    arms = build_arms(
        target_plasticity=0.75,
        budgets=[0.05, 0.10, 0.25, 0.50],
        permuted_budget=0.25,
    )

    kinds = [arm.kind for arm in arms]
    assert kinds.count("iso") == 4
    assert kinds.count("permuted") == 1
    assert kinds.count("reduced_lr") == 1
    assert kinds.count("vanilla") == 1
    assert len(arms) == 7
    assert all(arm.target_plasticity == 0.75 for arm in arms)


def test_permuted_budget_must_be_a_declared_budget():
    """A control at an undeclared budget would not be matched to any arm."""

    with pytest.raises(ValueError, match="permuted_budget"):
        build_arms(
            target_plasticity=0.75,
            budgets=[0.05, 0.25],
            permuted_budget=0.40,
        )


def test_masked_arm_requires_a_budget_and_unmasked_forbids_one():
    with pytest.raises(ValueError, match="exige budget"):
        ArmSpec(name="bad", kind="iso", target_plasticity=0.75)
    with pytest.raises(ValueError, match="não pode ter budget"):
        ArmSpec(
            name="bad", kind="reduced_lr", target_plasticity=0.75, budget=0.25
        )


def test_unreachable_budget_is_discarded_not_saturated():
    """`E*` below a budget's floor has no solution; beta must not be clamped."""

    torch.manual_seed(2)
    importances = [torch.rand(200) + 0.01 for _ in range(3)]

    points, discarded = reachable_iso_points(
        importances,
        target_plasticity=0.50,
        budgets=[0.25, 0.50, 0.75],
        scope="local",
    )

    solved = {point.budget for point in points}
    dropped = {entry["budget"] for entry in discarded}
    # A budget of 0.50 leaves half the units fully plastic, so E >= 0.50 for
    # every beta and the target sits exactly on the floor. 0.75 is strictly
    # above the target and cannot be reached either.
    assert 0.75 in dropped
    assert 0.25 in solved
    assert solved.isdisjoint(dropped)
    for entry in discarded:
        assert entry["floor"] >= entry["target_plasticity"] - 1e-9


def test_reachable_points_hit_the_target_and_spread_the_protected_count():
    torch.manual_seed(4)
    importances = [torch.rand(256) for _ in range(3)]

    points, _ = reachable_iso_points(
        importances,
        target_plasticity=0.75,
        budgets=[0.05, 0.10, 0.25, 0.50],
        scope="local",
    )

    assert len(points) == 4
    for point in points:
        assert point.achieved_plasticity == pytest.approx(0.75, abs=1e-6)
    counts = [point.protected_units for point in points]
    strengths = [point.slow_strength for point in points]
    assert counts == sorted(counts, reverse=True)
    assert strengths == sorted(strengths)
    # The contrast is the whole point of the family.
    assert max(counts) / min(counts) > 1.5


# ---------------------------------------------------------------------------
# assertion 1: every arm reaches E* within 1e-6, under the declared scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.75, 0.50])
@pytest.mark.parametrize("budget", [0.05, 0.25])
def test_resolved_beta_makes_the_model_reach_the_target(target, budget):
    """Measured on the model's own heat, not recomputed from importance.

    `measured_plasticity` reads `slow_heat` and `slow_strength` as the
    optimizer sees them, so a mismatch between the arithmetic used to solve
    beta and the arithmetic the model applies fails here instead of passing.
    """

    model = _model(budget=budget)
    _seed_distinct_importance(model)
    arm = ArmSpec(
        name="iso", kind="iso", target_plasticity=target, budget=budget
    )

    beta = resolve_beta(model, arm=arm, scope="local")

    assert beta is not None
    for tracker in model.get_ffn_trackers():
        tracker.slow_strength = beta
    assert measured_plasticity(model) == pytest.approx(target, abs=1e-6)


def test_solving_under_the_wrong_scope_misses_the_target():
    """Capacity scope is not cosmetic: it changes the beta that reaches E*.

    This is the observable that makes the "ignore capacity_scope" mutation
    detectable. The model consolidates under `local`; solving under `global`
    normalizes across layers with very different magnitudes and yields a beta
    that does not reach the target on the model's real heat.
    """

    model = _model(budget=0.25, scope="local")
    _seed_distinct_importance(model)
    importances = layer_importances(model)

    correct = solve_under(importances, 0.25, 0.75, "local")
    wrong = solve_under(importances, 0.25, 0.75, "global")

    for tracker in model.get_ffn_trackers():
        tracker.slow_strength = wrong
    assert measured_plasticity(model) != pytest.approx(0.75, abs=1e-6)

    for tracker in model.get_ffn_trackers():
        tracker.slow_strength = correct
    assert measured_plasticity(model) == pytest.approx(0.75, abs=1e-6)


def solve_under(importances, budget, target, scope):
    from experiments.capacity_calibration import (
        solve_strength_for_plasticity_scoped,
    )

    strength = solve_strength_for_plasticity_scoped(
        importances, budget, target, scope=scope
    )
    assert strength is not None
    return strength


def test_measured_plasticity_rejects_divergent_strengths():
    """A per-layer beta would make the reported E meaningless."""

    model = _model()
    _seed_distinct_importance(model)
    trackers = model.get_ffn_trackers()
    trackers[0].slow_strength = 3.0
    trackers[1].slow_strength = 9.0

    with pytest.raises(RuntimeError, match="slow_strength divergiu"):
        measured_plasticity(model)


def test_an_arm_applying_the_wrong_target_is_visible_in_the_measurement():
    model = _model(budget=0.25)
    _seed_distinct_importance(model)
    right = ArmSpec(name="a", kind="iso", target_plasticity=0.75, budget=0.25)
    wrong = ArmSpec(name="b", kind="iso", target_plasticity=0.50, budget=0.25)

    beta_wrong = resolve_beta(model, arm=wrong, scope="local")
    for tracker in model.get_ffn_trackers():
        tracker.slow_strength = beta_wrong

    assert measured_plasticity(model) == pytest.approx(0.50, abs=1e-6)
    assert measured_plasticity(model) != pytest.approx(
        right.target_plasticity, abs=1e-6
    )


# ---------------------------------------------------------------------------
# assertion 2: the permuted control preserves the heat distribution and E
# ---------------------------------------------------------------------------


def test_permutation_preserves_the_sorted_heat_and_the_protected_count():
    model = _model(budget=0.25, layers=2)
    _seed_distinct_importance(model)
    before = [
        tracker.slow_heat.detach().clone() for tracker in model.get_slow_states()
    ]

    reordered = permute_slowheat_heat(model, seed=7)

    assert reordered is True
    for original, tracker in zip(before, model.get_slow_states(), strict=True):
        after = tracker.slow_heat
        torch.testing.assert_close(
            torch.sort(after).values, torch.sort(original).values
        )
        assert int(torch.count_nonzero(after > 0.0)) == int(
            torch.count_nonzero(original > 0.0)
        )
        # A permutation that left everything in place would not be a control.
        assert not torch.equal(after, original)


def test_permutation_leaves_effective_plasticity_identical():
    model = _model(budget=0.25)
    _seed_distinct_importance(model)
    for tracker in model.get_ffn_trackers():
        tracker.slow_strength = 12.5
    before = measured_plasticity(model)

    permute_slowheat_heat(model, seed=11)

    assert measured_plasticity(model) == pytest.approx(before, abs=1e-9)


def test_hard_protection_is_not_an_iso_e_control():
    """The mistake the protocol calls out, pinned as a test.

    Zeroing the heat and setting the same number of random units to 1.0 keeps
    the protected COUNT but changes the distribution, so E moves. That is why
    `permute_slowheat_heat` shuffles the vector instead.
    """

    model = _model(budget=0.25)
    _seed_distinct_importance(model)
    for tracker in model.get_ffn_trackers():
        tracker.slow_strength = 12.5
    before = measured_plasticity(model)

    generator = torch.Generator(device="cpu").manual_seed(3)
    with torch.no_grad():
        for tracker in model.get_slow_states():
            protected = int(torch.count_nonzero(tracker.slow_heat).item())
            tracker.slow_heat.zero_()
            indices = torch.randperm(
                tracker.slow_heat.numel(), generator=generator
            )[:protected]
            tracker.slow_heat[indices] = 1.0

    assert measured_plasticity(model) != pytest.approx(before, abs=1e-6)


def test_permutation_is_a_no_op_on_a_fully_plastic_model():
    """Before the first consolidation there is nothing to permute."""

    model = _model()

    assert permute_slowheat_heat(model, seed=1) is False
    assert all(
        torch.count_nonzero(tracker.slow_heat) == 0
        for tracker in model.get_slow_states()
    )


# ---------------------------------------------------------------------------
# assertion 3: the reduced-LR control registers no mask at all
# ---------------------------------------------------------------------------


def test_reduced_lr_arm_accepts_an_optimizer_with_no_masks():
    model = _model()
    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )

    register_reduced_lr_arm(model, optimizer)  # must not raise

    assert optimizer._plasticity_masks == {}


def test_reduced_lr_arm_rejects_an_optimizer_with_masks():
    """Registering a mask here would silently turn the control into an arm."""

    model = _model()
    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )
    model.register_plasticity_masks(optimizer, hard=False)

    with pytest.raises(RuntimeError, match="LR reduzido não pode registrar"):
        register_reduced_lr_arm(model, optimizer)


def test_reduced_lr_scaling_is_the_target_times_the_base_rate():
    """The control has to sit at the same effective plasticity by construction."""

    base = 1e-5
    for target in (0.75, 0.50):
        arm = ArmSpec(
            name="reduced_lr", kind="reduced_lr", target_plasticity=target
        )
        assert base * arm.target_plasticity == pytest.approx(base * target)


# ---------------------------------------------------------------------------
# assertion 4: protected drift is exactly zero under a hard mask
# ---------------------------------------------------------------------------


def test_protected_parameters_do_not_move_under_a_hard_mask():
    from experiments.split_clinc150 import (
        capture_parameter_drift_reference,
        summarize_parameter_drift,
    )

    torch.manual_seed(31)
    model = _model(budget=0.25)
    _seed_distinct_importance(model)
    model.train()

    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.05,
        weight_decay=0.1,
    )
    model.register_plasticity_masks(optimizer, hard=True)
    reference = capture_parameter_drift_reference(model.mask_bindings(hard=True))

    input_ids, attention_mask = _tokens()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=torch.tensor([1]),
    )
    output.loss.backward()
    optimizer.step()

    drift = summarize_parameter_drift(reference)
    assert drift["protected_count"] > 0
    assert drift["protected_max_abs"] == 0.0
    # Without plastic movement the assertion above would hold vacuously.
    assert drift["plastic_count"] > 0
    assert drift["plastic_max_abs"] > 0.0


def test_mask_axis_is_rows_for_producers_and_columns_for_the_consumer():
    """Inverting the axis would protect the wrong weights while drift stays 0.

    Drift alone cannot catch an inverted axis when the matrix is square, so the
    binding shapes are asserted structurally. The tiny model here is
    deliberately non-square (hidden 8, intermediate 12).
    """

    model = _model()
    tracker = model.get_ffn_trackers()[0]
    tracker.slow_heat.copy_(torch.linspace(0.0, 1.0, tracker.units))

    by_kind = {binding.kind: binding for binding in model.mask_bindings()}
    gate = next(b for k, b in by_kind.items() if k.endswith("_gate_producer_rows"))
    down = next(
        b for k, b in by_kind.items() if k.endswith("_down_consumer_columns")
    )

    assert gate.mask().shape == (12, 1)
    assert down.mask().shape == (1, 12)


# ---------------------------------------------------------------------------
# assertion 5: the manifest records everything needed to audit the run
# ---------------------------------------------------------------------------


def test_protocol_hash_is_stable_and_order_independent():
    first = {"model": "qwen", "targets": [0.75, 0.50], "seed": 0}
    second = {"seed": 0, "targets": [0.75, 0.50], "model": "qwen"}

    assert protocol_hash(first) == protocol_hash(second)


def test_protocol_hash_changes_when_any_declared_value_changes():
    base = {"model": "qwen", "targets": [0.75], "capacity_scope": "local"}
    changed = {"model": "qwen", "targets": [0.75], "capacity_scope": "global"}

    assert protocol_hash(base) != protocol_hash(changed)


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------


def test_forgetting_excludes_the_task_that_was_just_learned():
    """A freshly learned task has had no chance to be forgotten.

    matrix[0] = [0.9, None]; matrix[1] = [0.4, 0.8].
    Forgetting is read over task 0 only: 0.9 - 0.4 = 0.5. Including task 1
    would average in a structural zero and halve the reported effect.
    """

    matrix = [[0.9, None], [0.4, 0.8]]

    endpoints = _endpoints(matrix)

    assert endpoints["mean_forgetting"] == pytest.approx(0.5)
    assert endpoints["retention_first_task"] == pytest.approx(0.4)
    assert endpoints["acquisition_last_task"] == pytest.approx(0.8)
    assert endpoints["final_average_accuracy"] == pytest.approx(0.6)
    assert endpoints["backward_transfer"] == pytest.approx(-0.5)


def test_endpoints_over_three_tasks_take_the_best_ever_seen():
    matrix = [
        [0.90, None, None],
        [0.70, 0.80, None],
        [0.50, 0.60, 0.85],
    ]

    endpoints = _endpoints(matrix)

    # task 0: 0.90 - 0.50 = 0.40 ; task 1: 0.80 - 0.60 = 0.20
    assert endpoints["mean_forgetting"] == pytest.approx(0.30)
    assert endpoints["final_average_accuracy"] == pytest.approx(0.65)


def test_endpoints_of_a_perfectly_stable_run_report_zero_forgetting():
    matrix = [[0.8, None], [0.8, 0.8]]

    endpoints = _endpoints(matrix)

    assert endpoints["mean_forgetting"] == pytest.approx(0.0)
    assert endpoints["backward_transfer"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# end-to-end mechanism smoke on the tiny model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Class-IL evaluation: seen-class logit masking
# ---------------------------------------------------------------------------


def _fake_task(domain, classes, *, labels):
    from experiments.split_clinc150 import CLINC150Task, TokenizedTextSplit

    count = len(labels)
    split = TokenizedTextSplit(
        torch.tensor([[2, 7, 9, 3]] * count, dtype=torch.long),
        torch.ones(count, 4, dtype=torch.long),
        torch.zeros(count, 4, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.arange(count, dtype=torch.long),
    )
    return CLINC150Task(
        domain=domain, classes=tuple(classes), train=split,
        validation=split, test=None,
    )


def _pin_logits(model, values):
    """Force the classification head to emit a fixed logit vector."""

    with torch.no_grad():
        model.score.weight.zero_()
        # A zero weight makes logits identically zero; add the desired profile
        # as a bias-like constant through a hook instead of touching weights.
    handle = model.score.register_forward_hook(
        lambda _m, _i, output: torch.zeros_like(output)
        + torch.tensor(values, dtype=output.dtype, device=output.device)
    )
    return handle


def test_evaluation_masks_classes_no_task_has_introduced_yet():
    """Without the mask, an unseen class can win and accuracy collapses.

    Labels 0-1 belong to task 0 and 2-3 to task 1. At stage 0 only 0-1 have
    been seen, so a head that most strongly favours label 3 must still be
    forced to choose between 0 and 1. Dropping `seen_classes` would let label 3
    win every example and drive the stage-0 accuracy to zero.
    """

    from experiments.qwen_iso_plasticity import _evaluate_seen, _seen_classes

    model = _model(layers=1)
    tasks = [
        _fake_task("a", (0, 1), labels=[0, 0, 1]),
        _fake_task("b", (2, 3), labels=[2, 3]),
    ]
    # Unseen label 3 is the argmax; among the seen labels, 0 wins.
    handle = _pin_logits(model, [2.0, 1.0, 0.5, 9.0])
    try:
        assert _seen_classes(tasks, 0) == (0, 1)
        assert _seen_classes(tasks, 1) == (0, 1, 2, 3)

        stage0 = _evaluate_seen(
            model, tasks, stage=0, batch_size=2, device=torch.device("cpu")
        )
    finally:
        handle.remove()

    assert len(stage0) == 1
    # Two of three examples are label 0, which is the argmax over seen classes.
    assert stage0[0]["class_il_accuracy"] == pytest.approx(2 / 3)


def test_evaluation_covers_every_task_seen_so_far():
    from experiments.qwen_iso_plasticity import _evaluate_seen

    model = _model(layers=1)
    tasks = [
        _fake_task("a", (0, 1), labels=[0, 1]),
        _fake_task("b", (2, 3), labels=[2, 3]),
    ]

    results = _evaluate_seen(
        model, tasks, stage=1, batch_size=2, device=torch.device("cpu")
    )

    assert [entry["task"] for entry in results] == [0, 1]
    assert [entry["domain"] for entry in results] == ["a", "b"]
    assert all(0.0 <= entry["class_il_accuracy"] <= 1.0 for entry in results)


def test_evaluation_lets_a_later_task_class_beat_an_earlier_task():
    """Class-IL, not Task-IL: an old task must compete with newer classes.

    This is the distinction stage 0 cannot show, because there `seen_classes`
    and the task's own classes are the same set. At stage 1, evaluating task 0
    must range over labels 0-3; restricting it to task 0's own labels 0-1 would
    hide exactly the cross-task confusion Class-IL exists to measure and turn
    the endpoint into Task-IL accuracy.
    """

    from experiments.qwen_iso_plasticity import _evaluate_seen

    model = _model(layers=1)
    tasks = [
        _fake_task("a", (0, 1), labels=[0, 0]),
        _fake_task("b", (2, 3), labels=[2, 2]),
    ]
    # Label 2 belongs to task 1 and is SEEN at stage 1. It outranks label 0.
    handle = _pin_logits(model, [2.0, 1.0, 9.0, 0.5])
    try:
        results = _evaluate_seen(
            model, tasks, stage=1, batch_size=2, device=torch.device("cpu")
        )
    finally:
        handle.remove()

    first, second = results
    # Task 0's examples are all label 0 but lose to the newer label 2.
    assert first["class_il_accuracy"] == pytest.approx(0.0)
    # Task-IL accuracy, restricted to task 0's own classes, still gets them.
    assert first["task_il_accuracy"] == pytest.approx(1.0)
    # Task 1's own examples are label 2 and win outright.
    assert second["class_il_accuracy"] == pytest.approx(1.0)


def test_evaluation_restores_training_mode():
    """The BERT runner bug: eval mode leaked and importance stopped accruing."""

    from experiments.qwen_iso_plasticity import _evaluate_seen

    model = _model(layers=1)
    tasks = [_fake_task("a", (0, 1), labels=[0, 1])]
    model.train()

    _evaluate_seen(
        model, tasks, stage=0, batch_size=2, device=torch.device("cpu")
    )

    assert model.training is True


# ---------------------------------------------------------------------------
# memory release between arms
# ---------------------------------------------------------------------------


def test_release_drops_the_model_so_the_next_arm_can_allocate():
    """Regression: arms leaked and the second one hit CUDA OOM on an 11 GB card.

    The SlowHeat forward hooks close over the model and the mask bindings close
    over the trackers and the optimizer, so a finished arm sits in a reference
    cycle. A weakref shows whether the cycle was actually broken; `del` alone
    leaves the object alive until an arbitrary later collection.
    """

    import weakref

    from experiments.qwen_iso_plasticity import _release

    model = _model(layers=2)
    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )
    model.register_plasticity_masks(optimizer, hard=False)
    # Take a real step so optimizer state exists and holds parameter refs.
    input_ids, attention_mask = _tokens()
    model.train()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=torch.tensor([1]),
    )
    output.loss.backward()
    optimizer.step()

    witness = weakref.ref(model)
    assert witness() is not None

    del output
    _release(model, optimizer, torch.device("cpu"))
    del model, optimizer

    assert witness() is None


def test_release_tolerates_a_half_built_arm():
    """The finally block runs even when the model failed to build."""

    from experiments.qwen_iso_plasticity import _release

    _release(None, None, torch.device("cpu"))  # must not raise


def test_two_boundaries_keep_every_arm_on_target_with_per_boundary_betas():
    """The A2 policy: beta is re-resolved and E stays pinned at each boundary."""

    torch.manual_seed(41)
    model = _model(budget=0.25, layers=2)
    arm = ArmSpec(name="iso", kind="iso", target_plasticity=0.75, budget=0.25)
    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    model.register_plasticity_masks(optimizer, hard=False)

    betas = []
    for label in (1, 2):
        model.train()
        input_ids, attention_mask = _tokens()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.tensor([label]),
        )
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.consolidate(strategy="max")
        beta = resolve_beta(model, arm=arm, scope="local")
        assert beta is not None
        for tracker in model.get_ffn_trackers():
            tracker.slow_strength = beta
        betas.append(beta)
        assert measured_plasticity(model) == pytest.approx(0.75, abs=1e-6)

    assert len(betas) == 2
    assert all(value > 0.0 for value in betas)


def test_the_real_heat_matches_the_scoped_arithmetic_used_to_solve_beta():
    """If these diverged, every solved beta would target the wrong vector."""

    model = _model(budget=0.25, layers=2)
    _seed_distinct_importance(model)

    recomputed = protected_heat_scoped(
        layer_importances(model), 0.25, scope="local"
    )
    real = torch.cat(
        [tracker.slow_heat.detach().float().cpu() for tracker in model.get_ffn_trackers()]
    )

    torch.testing.assert_close(real, recomputed, atol=1e-6, rtol=1e-6)
    assert effective_plasticity(real, 10.0) == pytest.approx(
        effective_plasticity(recomputed, 10.0), abs=1e-9
    )
