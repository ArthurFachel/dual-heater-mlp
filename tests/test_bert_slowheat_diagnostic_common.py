import pytest

pytest.importorskip("torch")

from experiments.bert_slowheat_diagnostic_common import (  # noqa: E402
    condition_endpoints,
    format_mean_std,
    format_mean_std_scientific,
    summarize_values,
)


def _result():
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
                "protected_rms": 8.0e-4,
                "protected_max_abs": 1.0e-3,
                "plastic_count": 8,
                "plastic_rms": 2.0e-3,
                "plastic_max_abs": 3.0e-3,
            },
        ],
        "elapsed_seconds": 10.0,
        "tokens_processed": 1_000,
        "replay_memory_bytes": 2**20,
        "peak_memory": {
            "peak_memory_bytes": 2 * 2**20,
            "peak_cuda_reserved_bytes": 3 * 2**20,
        },
    }


def test_common_helpers_preserve_existing_endpoint_and_format_contract():
    endpoints = condition_endpoints(_result())

    assert endpoints["final_average_accuracy"] == pytest.approx(0.75)
    assert endpoints["task1_retention"] == pytest.approx(0.6)
    assert endpoints["tokens_per_second"] == pytest.approx(100.0)
    assert endpoints["replay_memory_mib"] == pytest.approx(1.0)
    assert summarize_values([1.0, 3.0]) == {
        "n": 2,
        "mean": 2.0,
        "sample_std": pytest.approx(2**0.5),
    }
    assert format_mean_std(
        {"n": 1, "mean": 0.75, "sample_std": 0.0}, 100.0
    ) == "75.00 ± 0.00"
    assert format_mean_std_scientific(
        {"n": 1, "mean": 8.0e-4, "sample_std": 0.0}
    ) == "8.000e-04 ± 0.000e+00"
