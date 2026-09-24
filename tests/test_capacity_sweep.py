"""Tests for the capacity (width) sweep and its primary slope test.

These cover the invariants that make the sweep interpretable rather than just
runnable: that only the width varies, that the shared rung matches the
hard-versus-soft suite's seeds, that the slope machinery is numerically right,
and that the analyzer refuses to fit a slope across rungs whose configs drift.

The statistics are validated against published t-table critical values and a
closed-form OLS, because a silently wrong p-value here would be reported as a
scientific result.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from experiments.analyze_capacity_sweep import (
    _betainc,
    _exact_sign_flip_p,
    _one_sided_t,
    _slope,
    check_integrity,
    fit_slopes,
)
from experiments.capacity_sweep import (
    METHOD_PAIRS,
    PAIRED_METHODS,
    REPLICATION_WIDTH,
    SWEEP_SEEDS,
    WIDTHS,
    sweep_config,
    sweep_protocol,
)
from experiments.confirmatory_split_mnist import CONFIRMATORY_SEEDS
from experiments.hard_vs_soft import PAIRED_METHODS as HARD_VS_SOFT_METHODS
from experiments.hard_vs_soft import suite_config

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Design invariants
# --------------------------------------------------------------------------


def test_only_the_width_differs_between_rungs() -> None:
    """The whole claim rests on this: every other config field must be equal."""
    configs = {width: sweep_config("split_cifar10", width) for width in WIDTHS}
    baseline = configs[WIDTHS[0]]
    for width, config in configs.items():
        assert tuple(config.hidden_dims) == width
        # Compare everything except the one field under study.
        assert replace(config, hidden_dims=baseline.hidden_dims) == baseline, width


def test_the_ladder_doubles_and_keeps_a_fixed_layer_ratio() -> None:
    """A 2:1 ratio at every rung keeps depth and shape out of the contrast."""
    for width in WIDTHS:
        assert len(width) == 2
        assert width[0] == 2 * width[1], width
    for smaller, larger in pairwise(WIDTHS):
        assert larger[0] == 2 * smaller[0], (smaller, larger)


def test_the_sweep_shares_seeds_with_the_hard_versus_soft_suite() -> None:
    """The shared rung is only a replication control if the seeds match."""
    existing = ROOT / "results" / "hard_vs_soft" / "split_cifar10_mlp"
    if not (existing / "multi_seed_config.json").is_file():
        pytest.skip("artefato hard_vs_soft ausente")
    recorded = json.loads((existing / "multi_seed_config.json").read_text())["seeds"]
    assert list(SWEEP_SEEDS) == recorded


def test_the_shared_rung_matches_the_hard_versus_soft_config() -> None:
    """At 1024x512 the sweep must reproduce the other suite's configuration.

    If any field drifts, the rung stops being a replication and the comparison
    in the results document becomes meaningless.
    """
    mine = sweep_config("split_cifar10", REPLICATION_WIDTH)
    theirs = suite_config("split_cifar10", "mlp")
    assert tuple(mine.hidden_dims) == tuple(theirs.hidden_dims)
    assert mine == theirs


def test_the_sweep_uses_the_same_contrasts_as_hard_versus_soft() -> None:
    assert set(PAIRED_METHODS) == set(HARD_VS_SOFT_METHODS)
    assert METHOD_PAIRS[0].label == "Hard vs Soft"


def test_replication_width_is_on_the_ladder() -> None:
    assert REPLICATION_WIDTH in WIDTHS


def test_config_refuses_a_cnn_dataset() -> None:
    with pytest.raises(ValueError, match="dataset não suportado"):
        sweep_config("split_cifar10_cnn", (256, 128))


def test_config_refuses_a_degenerate_width() -> None:
    with pytest.raises(ValueError, match="dimensões positivas"):
        sweep_config("split_cifar10", (0,))


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------


def test_protocol_refuses_confirmatory_seeds() -> None:
    config = sweep_config("split_cifar10", (256, 128))
    reserved = [int(next(iter(CONFIRMATORY_SEEDS)))]
    with pytest.raises(ValueError, match="reservadas"):
        sweep_protocol(config, reserved, dataset="split_cifar10", width=(256, 128))


def test_protocol_refuses_a_width_that_disagrees_with_the_config() -> None:
    """A protocol claiming one width while the config trains another would
    silently mislabel the entire rung."""
    config = sweep_config("split_cifar10", (256, 128))
    with pytest.raises(ValueError, match="diverge da largura declarada"):
        sweep_protocol(config, [11, 12], dataset="split_cifar10", width=(512, 256))


def test_protocol_declares_the_primary_test_and_a_falsifier() -> None:
    config = sweep_config("split_cifar10", (256, 128))
    protocol = sweep_protocol(
        config, [11, 12], dataset="split_cifar10", width=(256, 128)
    )
    assert protocol["status"] == "exploratory_frozen_before_execution"
    assert protocol["primary_test"]["alternative"] == "b > 0 (hard gains on soft as capacity grows)"
    joined = " ".join(protocol["declared_expectations"]).lower()
    assert "null slope falsifies" in joined
    assert "hidden_dims only" in protocol["varied"]
    json.dumps(protocol)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("statistic", "df", "expected"),
    [
        # Published one-sided critical values.
        (1.8331, 9, 0.05),
        (2.2622, 9, 0.025),
        (3.2498, 9, 0.005),
        (1.7291, 19, 0.05),
        (2.1318, 4, 0.05),
    ],
)
def test_t_distribution_matches_published_critical_values(
    statistic: float, df: int, expected: float
) -> None:
    x = df / (df + statistic * statistic)
    one_sided = _betainc(df / 2.0, 0.5, x) / 2.0
    assert one_sided == pytest.approx(expected, abs=1e-4)


def test_slope_matches_the_closed_form() -> None:
    """A perfect line must recover its own slope exactly."""
    xs = [7.0, 8.0, 9.0, 10.0, 11.0]
    for true_slope in (-2.5, 0.0, 0.75, 3.0):
        ys = [1.3 + true_slope * x for x in xs]
        assert _slope(xs, ys) == pytest.approx(true_slope, abs=1e-9)


def test_slope_refuses_widths_without_spread() -> None:
    with pytest.raises(ValueError, match="precisam variar"):
        _slope([9.0, 9.0, 9.0], [1.0, 2.0, 3.0])


def test_exact_sign_flip_reaches_its_floor_only_when_every_seed_agrees() -> None:
    assert _exact_sign_flip_p([1.0] * 10) == pytest.approx(1 / 1024)
    assert _exact_sign_flip_p([-1.0] * 10) == pytest.approx(1.0)


def test_a_null_slope_is_not_significant() -> None:
    """Flat differences must not produce a capacity claim."""
    xs = [7.0, 8.0, 9.0, 10.0]
    rungs = _synthetic_rungs(xs, slope=0.0, noise=0.0)
    result = fit_slopes(rungs)
    assert result["student_t_one_sided"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert result["student_t_one_sided"]["one_sided_p"] >= 0.5


def test_a_strong_positive_slope_is_detected() -> None:
    xs = [7.0, 8.0, 9.0, 10.0]
    rungs = _synthetic_rungs(xs, slope=0.5, noise=0.0)
    result = fit_slopes(rungs)
    assert result["student_t_one_sided"]["mean"] == pytest.approx(0.5, abs=1e-9)
    assert result["positive_slopes"] == 10
    assert result["exact_sign_flip_p_greater"] == pytest.approx(1 / 1024)


def test_a_negative_slope_does_not_read_as_support() -> None:
    """Hard doing *worse* with more capacity must not pass the one-sided test."""
    xs = [7.0, 8.0, 9.0, 10.0]
    rungs = _synthetic_rungs(xs, slope=-0.5, noise=0.0)
    result = fit_slopes(rungs)
    assert result["student_t_one_sided"]["one_sided_p"] > 0.95
    assert result["negative_slopes"] == 10


def test_one_sided_t_agrees_with_the_manual_formula() -> None:
    values = [0.1, 0.2, 0.15, 0.3, 0.05, 0.22, 0.18, 0.25, 0.12, 0.2]
    result = _one_sided_t(values)
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    expected = mean / (math.sqrt(variance) / math.sqrt(len(values)))
    assert result["statistic"] == pytest.approx(expected, rel=1e-12)


def test_fit_slopes_requires_at_least_three_rungs() -> None:
    """Two points always fit a line perfectly; a trend claim needs more."""
    rungs = _synthetic_rungs([7.0, 8.0], slope=1.0, noise=0.0)
    with pytest.raises(SystemExit, match="ao menos 3 degraus"):
        fit_slopes(rungs)


# --------------------------------------------------------------------------
# Integrity gate
# --------------------------------------------------------------------------


def _synthetic_rungs(
    xs: list[float], *, slope: float, noise: float
) -> dict[tuple[int, ...], dict]:
    """Rungs whose per-seed differences follow a known slope in log2(width)."""
    rungs = {}
    for index, x in enumerate(xs):
        width = (int(2**x), int(2 ** (x - 1)))
        rungs[width] = {
            "protocol": {"config": {"epochs_per_task": 10}, "width": list(width)},
            "differences": {
                seed: slope * x + noise * ((seed % 3) - 1) + 0.01 * seed
                for seed in range(10)
            },
        }
        assert index >= 0
    return rungs


def test_integrity_gate_rejects_config_drift_between_rungs() -> None:
    """If anything but the width changed, the slope is not a capacity effect."""
    rungs = _synthetic_rungs([7.0, 8.0, 9.0], slope=0.1, noise=0.0)
    drifted = max(rungs)
    rungs[drifted]["protocol"]["config"]["epochs_per_task"] = 20
    with pytest.raises(SystemExit, match="configuração divergente"):
        check_integrity(rungs)


def test_integrity_gate_accepts_rungs_that_differ_only_in_width() -> None:
    rungs = _synthetic_rungs([7.0, 8.0, 9.0], slope=0.1, noise=0.0)
    summary = check_integrity(rungs)
    assert summary["only_width_varies"] is True
    assert len(summary["widths"]) == 3


def test_integrity_gate_rejects_mismatched_seeds() -> None:
    """Unequal seed sets break the pairing the whole analysis depends on."""
    rungs = _synthetic_rungs([7.0, 8.0, 9.0], slope=0.1, noise=0.0)
    victim = max(rungs)
    rungs[victim]["differences"].pop(0)
    with pytest.raises(SystemExit, match="seeds divergem"):
        check_integrity(rungs)


def test_hidden_dims_is_exempt_from_the_drift_check() -> None:
    """The gate must not flag the very field the sweep is varying."""
    rungs = _synthetic_rungs([7.0, 8.0, 9.0], slope=0.1, noise=0.0)
    for width, rung in rungs.items():
        rung["protocol"]["config"]["hidden_dims"] = list(width)
    assert check_integrity(rungs)["only_width_varies"] is True
