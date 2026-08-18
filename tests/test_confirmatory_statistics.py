import pytest

from experiments.confirmatory_split_mnist import (
    CONFIRMATORY_SEEDS,
    DECLARED_EXPLORATORY_SEEDS,
    FROZEN_CONFIG,
    validate_preregistration,
)
from experiments.confirmatory_statistics import paired_confirmatory_summary


def test_paired_summary_reports_student_bootstrap_and_signs():
    summary = paired_confirmatory_summary(
        [0.01, 0.02, -0.01, 0.0], bootstrap_resamples=100, bootstrap_seed=3
    )

    assert summary["student_t"]["degrees_of_freedom"] == 3
    assert summary["paired_bootstrap"]["resamples"] == 100
    assert summary["signs"] == {
        "positive": 2,
        "negative": 1,
        "ties": 1,
        "exact_two_sided_p": 1.0,
    }


def test_confirmatory_protocol_is_frozen_and_nonoverlapping():
    validate_preregistration()

    assert len(CONFIRMATORY_SEEDS) == len(set(CONFIRMATORY_SEEDS)) == 20
    assert not set(CONFIRMATORY_SEEDS) & set(DECLARED_EXPLORATORY_SEEDS)
    assert FROZEN_CONFIG.epochs_per_task == 10
    assert FROZEN_CONFIG.replay_per_class == 20
    assert FROZEN_CONFIG.methods == (
        "replay",
        "slowheat_replay_hidden_beta_30_budget_0.25",
    )


def test_paired_summary_rejects_unpaired_singleton():
    with pytest.raises(ValueError):
        paired_confirmatory_summary([0.1])
