"""Standard metrics for class-incremental continual-learning experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CLMetrics:
    """Metrics derived from a stage-by-task accuracy matrix.

    ``accuracy_matrix[t, k]`` is the accuracy on task ``k`` after training
    through task ``t``. Entries above the diagonal may be NaN because those
    tasks have not been trained yet.
    """

    final_average_accuracy: float
    average_forgetting: float
    backward_transfer: float
    forward_transfer: float | None
    per_task_forgetting: tuple[float, ...]


def _as_score_vector(
    scores: Sequence[float] | NDArray[np.floating],
    *,
    name: str,
    task_count: int,
) -> NDArray[np.float64]:
    vector = np.asarray(scores, dtype=np.float64)
    if vector.shape != (task_count,):
        raise ValueError(f"{name} deve ter shape ({task_count},)")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} deve conter apenas valores finitos")
    if np.any(vector < 0.0) or np.any(vector > 1.0):
        raise ValueError(f"{name} deve conter accuracies no intervalo [0, 1]")
    return vector


def compute_cl_metrics(
    accuracy_matrix: Sequence[Sequence[float]] | NDArray[np.floating],
    *,
    pretrain_scores: Sequence[float] | NDArray[np.floating] | None = None,
    baseline_scores: Sequence[float] | NDArray[np.floating] | None = None,
) -> CLMetrics:
    """Compute accuracy, forgetting, BWT and optional standard FWT.

    Forgetting for an old task is its best score from the stage where it was
    learned through the final stage, minus its final score. The average excludes
    the final task because it has no later task over which to forget.

    Standard forward transfer requires two separately measured vectors:
    performance immediately before each task is trained (``pretrain_scores``)
    and the corresponding random-initialization or chance baselines
    (``baseline_scores``). Task zero is excluded from FWT.
    """

    matrix = np.asarray(accuracy_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("accuracy_matrix deve ser uma matriz quadrada 2D")
    task_count = matrix.shape[0]
    if task_count == 0:
        raise ValueError("accuracy_matrix não pode ser vazia")

    observed = matrix[np.tril_indices(task_count)]
    if not np.isfinite(observed).all():
        raise ValueError("a diagonal e o triângulo inferior devem ser finitos")
    if np.any(observed < 0.0) or np.any(observed > 1.0):
        raise ValueError("as accuracies observadas devem estar no intervalo [0, 1]")

    old_task_count = task_count - 1
    forgetting = tuple(
        float(np.max(matrix[task:, task]) - matrix[-1, task])
        for task in range(old_task_count)
    )
    average_forgetting = float(np.mean(forgetting)) if forgetting else 0.0
    backward_transfer = (
        float(np.mean(matrix[-1, :old_task_count] - np.diag(matrix)[:old_task_count]))
        if old_task_count
        else 0.0
    )

    if (pretrain_scores is None) != (baseline_scores is None):
        raise ValueError(
            "pretrain_scores e baseline_scores devem ser fornecidos juntos"
        )

    forward_transfer: float | None = None
    if pretrain_scores is not None and baseline_scores is not None:
        pretrain = _as_score_vector(
            pretrain_scores,
            name="pretrain_scores",
            task_count=task_count,
        )
        baseline = _as_score_vector(
            baseline_scores,
            name="baseline_scores",
            task_count=task_count,
        )
        forward_transfer = (
            float(np.mean(pretrain[1:] - baseline[1:]))
            if old_task_count
            else 0.0
        )

    return CLMetrics(
        final_average_accuracy=float(np.mean(matrix[-1])),
        average_forgetting=average_forgetting,
        backward_transfer=backward_transfer,
        forward_transfer=forward_transfer,
        per_task_forgetting=forgetting,
    )
