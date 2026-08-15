import numpy as np
import pytest

from dual_heater.metrics import compute_cl_metrics


def test_metrics_report_zero_forgetting_when_old_tasks_do_not_drop():
    matrix = np.array(
        [
            [0.8, np.nan],
            [0.8, 0.9],
        ]
    )

    metrics = compute_cl_metrics(matrix)

    assert metrics.final_average_accuracy == pytest.approx(np.mean(matrix[-1]))
    assert metrics.average_forgetting == pytest.approx(0.0)
    assert metrics.backward_transfer == pytest.approx(0.0)
    assert metrics.per_task_forgetting == pytest.approx([0.0])


def test_metrics_average_forgetting_uses_every_old_task():
    matrix = np.array(
        [
            [0.9, np.nan, np.nan],
            [0.9, 0.8, np.nan],
            [0.9, 0.5, 0.85],
        ]
    )

    metrics = compute_cl_metrics(matrix)

    expected = ((0.9 - 0.9) + (0.8 - 0.5)) / 2
    assert metrics.average_forgetting == pytest.approx(expected)
    assert metrics.per_task_forgetting == pytest.approx([0.0, 0.3])


def test_metrics_forgetting_uses_best_historical_score_after_recovery():
    matrix = np.array(
        [
            [0.8, np.nan, np.nan],
            [0.6, 0.9, np.nan],
            [0.75, 0.9, 0.85],
        ]
    )

    metrics = compute_cl_metrics(matrix)

    assert metrics.per_task_forgetting[0] == pytest.approx(0.8 - 0.75)


def test_metrics_do_not_confuse_task_zero_retention_with_average_forgetting():
    matrix = np.array(
        [
            [1.0, np.nan, np.nan],
            [0.5, 1.0, np.nan],
            [0.5, 0.0, 1.0],
        ]
    )

    metrics = compute_cl_metrics(matrix)

    task_zero_drop = matrix[0, 0] - matrix[-1, 0]
    assert task_zero_drop == pytest.approx(0.5)
    assert metrics.average_forgetting == pytest.approx((0.5 + 1.0) / 2)


def test_forward_transfer_uses_pretraining_scores_minus_random_baselines():
    matrix = np.array(
        [
            [0.8, np.nan, np.nan],
            [0.7, 0.9, np.nan],
            [0.6, 0.8, 0.85],
        ]
    )
    pretraining_scores = np.array([0.1, 0.4, 0.5])
    random_baselines = np.array([0.1, 0.2, 0.3])

    metrics = compute_cl_metrics(
        matrix,
        pretrain_scores=pretraining_scores,
        baseline_scores=random_baselines,
    )

    assert metrics.forward_transfer == pytest.approx(
        np.mean(pretraining_scores[1:] - random_baselines[1:])
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.ones((2, 3)),
        np.array([[0.8, np.nan], [0.7, np.nan]]),
    ],
)
def test_metrics_reject_invalid_result_matrices(matrix):
    with pytest.raises(ValueError):
        compute_cl_metrics(matrix)


@pytest.mark.parametrize("invalid_score", [-0.1, 1.1])
def test_metrics_reject_scores_outside_probability_range(invalid_score):
    matrix = np.array([[invalid_score]])

    with pytest.raises(ValueError):
        compute_cl_metrics(matrix)
