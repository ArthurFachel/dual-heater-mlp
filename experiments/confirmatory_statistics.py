"""Paired confirmatory statistics with an explicitly declared primary endpoint."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

PRIMARY_ENDPOINT = "final_average_accuracy"


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction used by the regularized incomplete beta."""

    maximum_iterations = 200
    epsilon = 3e-14
    minimum = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x / (
            (qam + even) * (a + even)
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + even) * (qap + even)
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise RuntimeError("fração contínua beta não convergiu")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_two_sided_p(statistic: float, degrees_of_freedom: int) -> float:
    if statistic == 0.0:
        return 1.0
    x = degrees_of_freedom / (degrees_of_freedom + statistic * statistic)
    return _regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)


def _student_t_critical_95(degrees_of_freedom: int) -> float:
    low = 0.0
    high = 2.0
    while _student_t_two_sided_p(high, degrees_of_freedom) > 0.05:
        high *= 2.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _student_t_two_sided_p(midpoint, degrees_of_freedom) > 0.05:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def paired_confirmatory_summary(
    differences: list[float],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_815,
) -> dict[str, Any]:
    """Summarize pre-paired candidate-minus-reference differences.

    Exact zeroes are reported as ties and ignored only by the exact sign test.
    The bootstrap is the percentile interval of the paired mean; resampling is
    over pairs rather than over marginal method results.
    """

    if len(differences) < 2:
        raise ValueError("ao menos duas diferenças pareadas são necessárias")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples deve ser >= 1")
    values = np.asarray(differences, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("diferenças devem ser finitas")

    count = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    standard_error = std / math.sqrt(count)
    critical = _student_t_critical_95(count - 1)
    if standard_error == 0.0:
        t_statistic = None if mean != 0.0 else 0.0
        p_value = 0.0 if mean != 0.0 else 1.0
    else:
        t_statistic = mean / standard_error
        p_value = _student_t_two_sided_p(abs(t_statistic), count - 1)

    generator = np.random.default_rng(bootstrap_seed)
    indices = generator.integers(0, count, size=(bootstrap_resamples, count))
    bootstrap_means = values[indices].mean(axis=1)
    bootstrap_low, bootstrap_high = np.quantile(
        bootstrap_means, [0.025, 0.975]
    )

    positives = int(np.count_nonzero(values > 0.0))
    negatives = int(np.count_nonzero(values < 0.0))
    ties = count - positives - negatives
    nonzero = positives + negatives
    if nonzero == 0:
        sign_p = None
    else:
        tail = min(positives, negatives)
        sign_p = min(
            1.0,
            2.0
            * sum(math.comb(nonzero, index) for index in range(tail + 1))
            / 2**nonzero,
        )

    return {
        "n_pairs": count,
        "mean_difference": mean,
        "std_difference": std,
        "student_t": {
            "statistic": t_statistic,
            "degrees_of_freedom": count - 1,
            "two_sided_p": p_value,
            "ci95": [mean - critical * standard_error, mean + critical * standard_error],
        },
        "paired_bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "ci95_percentile": [float(bootstrap_low), float(bootstrap_high)],
        },
        "signs": {
            "positive": positives,
            "negative": negatives,
            "ties": ties,
            "exact_two_sided_p": sign_p,
        },
    }
