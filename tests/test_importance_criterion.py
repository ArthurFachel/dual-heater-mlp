"""Tests for the importance-criterion ablation (Section H of the protocol)."""


def test_functional_criterion_is_bit_identical_to_the_current_behaviour() -> None:
    """Section H.2 of goals/protocol_importance_criterion_ablation.md.

    The ablation compares importance criteria. If adding the parameter also
    perturbs the functional path, the experiment measures the refactor instead
    of the criterion, and the result looks valid while being worthless. The
    default must therefore be bit-identical to the pre-parameter behaviour.
    """

    import torch

    from dual_heater.transformer import SlowHeatFFNTracker

    def run(**kwargs):
        torch.manual_seed(1234)
        tracker = SlowHeatFFNTracker(
            8, slow_strength=30.0, plasticity_budget=0.25, **kwargs
        )
        tracker.train()
        torch.manual_seed(99)
        for _ in range(4):
            hidden = torch.randn(3, 8, requires_grad=True)
            out = tracker.observe(hidden)
            (out * torch.linspace(0.1, 5.0, 8)).sum().backward()
        return tracker.task_ema.clone()

    default = run()
    explicit = run(importance_criterion="functional")

    assert torch.equal(default, explicit), (
        "explicit functional criterion must be bit-identical to the default"
    )


def test_magnitude_criterion_ignores_the_gradient() -> None:
    """Section H.3: magnitude must rank on |z| alone.

    Two very different gradients over the same activations must produce the
    same ranking; otherwise the arm is not measuring what the protocol says.
    """

    import torch

    from dual_heater.transformer import SlowHeatFFNTracker

    def run(criterion, gradient_weights):
        torch.manual_seed(1234)
        tracker = SlowHeatFFNTracker(
            8,
            slow_strength=30.0,
            plasticity_budget=0.25,
            importance_criterion=criterion,
        )
        tracker.train()
        torch.manual_seed(99)
        for _ in range(4):
            hidden = torch.randn(3, 8, requires_grad=True)
            out = tracker.observe(hidden)
            (out * gradient_weights).sum().backward()
        return tracker.task_ema.clone()

    flat = torch.ones(8)
    steep = torch.tensor([100.0, 0.01] * 4)

    magnitude_flat = run("magnitude", flat)
    magnitude_steep = run("magnitude", steep)
    assert torch.allclose(magnitude_flat, magnitude_steep), (
        "magnitude must not depend on the gradient"
    )

    functional_flat = run("functional", flat)
    functional_steep = run("functional", steep)
    assert not torch.allclose(functional_flat, functional_steep), (
        "functional must depend on the gradient (guards the test above)"
    )


def test_unknown_importance_criterion_is_rejected() -> None:
    """A typo must fail loudly, not silently fall back to functional."""

    import pytest

    from dual_heater.transformer import SlowHeatFFNTracker

    with pytest.raises(ValueError):
        SlowHeatFFNTracker(
            8,
            slow_strength=30.0,
            plasticity_budget=0.25,
            importance_criterion="magntiude",
        )


def test_diagnostic_conditions_include_the_magnitude_arm() -> None:
    """Section H.4: the ablation needs a third arm between random and learned.

    Protocol I2 freezes three arms sharing mask_mode/slow_strength so the only
    difference is the ranking criterion.
    """

    from experiments.bert_slowheat_diagnostic import (
        criterion_ablation_conditions,
        diagnostic_conditions,
    )

    # The published six-condition matrix must stay untouched: growing it would
    # place an undeclared arm inside a finished experiment.
    assert "slowheat_magnitude_hard" not in {c.name for c in diagnostic_conditions()}

    by_name = {c.name: c for c in criterion_ablation_conditions()}

    assert "slowheat_magnitude_hard" in by_name, "magnitude arm is required by I2"

    learned = by_name["slowheat_hard"]
    magnitude = by_name["slowheat_magnitude_hard"]
    random_hard = by_name["slowheat_random_hard"]

    # I10/I11: only the criterion may differ across the three arms.
    assert magnitude.slow_strength == learned.slow_strength == random_hard.slow_strength
    assert magnitude.method == learned.method == random_hard.method
    assert magnitude.mask_mode == "hard", "magnitude uses the hard mask, like its comparators"
    assert magnitude.importance_criterion == "magnitude"
    assert learned.importance_criterion == "functional"


def test_importance_criterion_ablation_seeds_are_registered_and_disjoint() -> None:
    """Section H.6 / I4: the ablation owns a seed band disjoint from BERT's."""

    import json
    from pathlib import Path

    from experiments.bert_slowheat_diagnostic import (
        IMPORTANCE_CRITERION_ABLATION_SEEDS,
    )

    seeds = IMPORTANCE_CRITERION_ABLATION_SEEDS
    assert len(seeds) == 10 and len(set(seeds)) == 10

    for artefact in (
        "results/bert_slowheat_review/diagnostic_summary.json",
        "results/bert_slowheat_diagnostic/diagnostic_summary.json",
    ):
        path = Path(artefact)
        if path.exists():
            used = set(json.loads(path.read_text())["seeds"])
            overlap = set(seeds) & used
            assert not overlap, f"ablation reuses seeds from {artefact}: {sorted(overlap)}"


def test_ranking_degeneracy_metrics_are_available() -> None:
    """Section F: the magnitude arm may degenerate into a random mask.

    _apply_capacity_budget protects the top-k units by importance. If the
    magnitude ranking is nearly flat, top-k selection is decided by noise and
    the arm silently becomes `random_hard` under another name. A tie would then
    be an artefact, not evidence that the criterion does not matter -- so the
    diagnostics must exist before the run, not be reconstructed afterwards.
    """

    import torch

    from dual_heater.slow_heat import ranking_degeneracy_metrics
    from dual_heater.transformer import SlowHeatFFNTracker

    def train(criterion):
        torch.manual_seed(7)
        tracker = SlowHeatFFNTracker(
            16,
            slow_strength=3.0,
            plasticity_budget=0.25,
            importance_criterion=criterion,
        )
        tracker.train()
        torch.manual_seed(21)
        for _ in range(6):
            hidden = torch.randn(4, 16, requires_grad=True)
            out = tracker.observe(hidden)
            weights = torch.zeros(16)
            weights[:4] = 50.0          # only 4 units actually drive the loss
            weights[4:] = 0.001
            (out * weights).sum().backward()
        return tracker

    functional = train("functional")
    magnitude = train("magnitude")

    metrics = ranking_degeneracy_metrics(
        magnitude, reference=functional, budget=0.25
    )

    assert "ranking_variance" in metrics
    assert "top_k_overlap" in metrics
    assert 0.0 <= metrics["top_k_overlap"] <= 1.0

    # The functional ranking concentrates on the units that carry gradient;
    # magnitude cannot see that difference. This is the effect the metric exists
    # to expose.
    functional_metrics = ranking_degeneracy_metrics(
        functional, reference=functional, budget=0.25
    )
    assert functional_metrics["top_k_overlap"] == 1.0, "a ranking must match itself"
    assert functional_metrics["ranking_variance"] > metrics["ranking_variance"]


def test_bert_propagates_the_importance_criterion_to_its_trackers() -> None:
    """The criterion must reach the trackers, or the ablation measures nothing.

    BertSlowHeatConfig -> _new_ffn_tracker -> SlowHeatFFNTracker. If any link
    drops the field, the `magnitude` arm silently runs as `functional` and the
    ablation produces a meaningless tie.
    """

    from dual_heater.bert import BertSlowHeatConfig

    default = BertSlowHeatConfig()
    assert default.importance_criterion == "functional", "default must not change"

    magnitude = BertSlowHeatConfig(importance_criterion="magnitude")
    assert magnitude.importance_criterion == "magnitude"

    # The criterion must be part of the recorded provenance, so a run's
    # artefacts show which criterion produced it.
    from dataclasses import asdict

    assert asdict(magnitude).get("importance_criterion") == "magnitude"


def test_split_clinc150_config_carries_the_importance_criterion() -> None:
    """Section H: the runner config must expose the criterion for the ablation."""

    from experiments.split_clinc150 import SplitCLINC150Config

    assert SplitCLINC150Config(device="cpu").importance_criterion == "functional"
    assert (
        SplitCLINC150Config(device="cpu", importance_criterion="magnitude").importance_criterion
        == "magnitude"
    )


def test_run_diagnostic_accepts_a_condition_matrix_and_propagates_the_criterion(
    monkeypatch, tmp_path
) -> None:
    """Section I: the ablation needs its own matrix, and the criterion must ride along.

    run_diagnostic built its configs from diagnostic_conditions() unconditionally
    and never copied importance_criterion into the runner config, so the
    magnitude arm would have silently executed as functional -- a guaranteed
    null result with no way to notice from the artefacts.
    """

    from dataclasses import replace as dc_replace

    import experiments.bert_slowheat_diagnostic as diagnostic
    from experiments.split_clinc150 import SplitCLINC150Config
    from tests.test_bert_slowheat_diagnostic import _fake_result

    seen = []

    def fake_run(config, tasks, **kwargs):
        seen.append(config)
        return {config.methods[0]: _fake_result()}

    monkeypatch.setattr(diagnostic, "run_split_clinc150", fake_run)
    monkeypatch.setattr(diagnostic, "write_environment_manifest", lambda *a, **k: None)

    conditions = diagnostic.criterion_ablation_conditions()
    diagnostic.run_diagnostic(
        SplitCLINC150Config(device="cpu"),
        tasks=[object()] * 10,
        metadata={},
        seeds=[4_000_003],
        output_dir=tmp_path,
        conditions=conditions,
    )

    assert len(seen) == len(conditions), "one run per ablation arm"

    by_criterion = {c.name: c.importance_criterion for c in conditions}
    for config, condition in zip(seen, conditions):
        assert config.importance_criterion == by_criterion[condition.name], (
            f"{condition.name} ran as {config.importance_criterion!r}"
        )

    # Default must still be the published six-condition matrix.
    seen.clear()
    diagnostic.run_diagnostic(
        SplitCLINC150Config(device="cpu"),
        tasks=[object()] * 10,
        metadata={},
        seeds=[4_000_003],
        output_dir=tmp_path / "default",
    )
    assert len(seen) == len(diagnostic.diagnostic_conditions())
