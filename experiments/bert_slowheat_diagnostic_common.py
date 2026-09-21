"""Shared endpoint aggregation and formatting for BERT diagnostics."""

from __future__ import annotations

from statistics import mean, stdev
from typing import Any


def summarize_values(
    values: list[float | None],
) -> dict[str, float | int | None]:
    finite = [value for value in values if value is not None]
    return {
        "n": len(finite),
        "mean": mean(finite) if finite else None,
        "sample_std": stdev(finite) if len(finite) > 1 else (0.0 if finite else None),
    }


def condition_endpoints(result: dict[str, Any]) -> dict[str, float | None]:
    matrix = result["validation_accuracy_matrix"]
    drift = result["parameter_drift_history"][1]
    peak_reserved = result["peak_memory"]["peak_cuda_reserved_bytes"]
    validation_metrics = result["validation_metrics"]
    elapsed_seconds = float(result["elapsed_seconds"])
    tokens_processed = float(result["tokens_processed"])
    if elapsed_seconds <= 0.0:
        raise ValueError("elapsed_seconds deve ser positivo")
    return {
        "task1_acquisition": float(matrix[0][0]),
        "task1_retention": float(matrix[1][0]),
        "task1_forgetting": float(matrix[0][0] - matrix[1][0]),
        "task2_acquisition": float(matrix[1][1]),
        "final_average_accuracy": float(
            validation_metrics["final_average_accuracy"]
        ),
        "backward_transfer": float(validation_metrics["backward_transfer"]),
        "protected_rms_drift": (
            None if drift["protected_rms"] is None else float(drift["protected_rms"])
        ),
        "protected_max_abs_drift": (
            None
            if drift["protected_max_abs"] is None
            else float(drift["protected_max_abs"])
        ),
        "protected_count": float(drift["protected_count"]),
        "plastic_count": float(drift["plastic_count"]),
        "plastic_rms_drift": (
            None if drift["plastic_rms"] is None else float(drift["plastic_rms"])
        ),
        "plastic_max_abs_drift": (
            None
            if drift["plastic_max_abs"] is None
            else float(drift["plastic_max_abs"])
        ),
        "elapsed_seconds": elapsed_seconds,
        "tokens_processed": tokens_processed,
        "tokens_per_second": tokens_processed / elapsed_seconds,
        "replay_memory_mib": float(result["replay_memory_bytes"]) / 2**20,
        "peak_memory_mib": float(result["peak_memory"]["peak_memory_bytes"]) / 2**20,
        "peak_reserved_mib": (
            None if peak_reserved is None else float(peak_reserved) / 2**20
        ),
    }


def format_mean_std(
    summary: dict[str, float | int | None],
    scale: float = 1.0,
) -> str:
    mean_value = summary["mean"]
    std_value = summary["sample_std"]
    if mean_value is None or std_value is None:
        return "—"
    return (
        f"{float(mean_value) * scale:.2f} ± "
        f"{float(std_value) * scale:.2f}"
    )


def format_mean_std_scientific(
    summary: dict[str, float | int | None],
) -> str:
    mean_value = summary["mean"]
    std_value = summary["sample_std"]
    if mean_value is None or std_value is None:
        return "—"
    return f"{float(mean_value):.3e} ± {float(std_value):.3e}"
