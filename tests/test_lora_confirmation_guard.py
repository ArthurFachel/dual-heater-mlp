"""Guard-rail tests for the frozen LoRA confirmation.

A pre-registration only has force if the code refuses to deviate from it. Each
test here pins one way the protocol could be silently violated.
"""

from __future__ import annotations

import subprocess
import sys

from experiments.confirmatory_split_mnist import CONFIRMATORY_SEEDS
from experiments.qwen_lora_sweep import (
    DEFAULT_SEEDS,
    LORA_CONFIRMATORY_SEEDS,
    PREREGISTERED_ARMS,
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "experiments.qwen_lora_sweep", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_three_seed_bands_are_disjoint() -> None:
    """Two protocols must never be able to consume each other's seeds."""

    lora = set(LORA_CONFIRMATORY_SEEDS)
    mnist = set(CONFIRMATORY_SEEDS)
    exploratory = set(DEFAULT_SEEDS)

    assert lora & mnist == set()
    assert lora & exploratory == set()
    assert mnist & exploratory == set()
    assert len(lora) == len(LORA_CONFIRMATORY_SEEDS) == 10


def test_exploration_cannot_burn_the_confirmatory_seeds() -> None:
    """The whole point of reserving seeds: an exploratory run must refuse."""

    result = _run(
        "--output", "/tmp/should_not_exist",
        "--seeds", str(LORA_CONFIRMATORY_SEEDS[0]),
        "--arms", "vanilla",
    )
    assert result.returncode != 0
    assert "reservadas" in result.stderr or "reservadas" in result.stdout


def test_confirmatory_refuses_a_different_arm_set() -> None:
    """Adding an arm would change the Holm family size after the fact."""

    result = _run(
        "--output", "/tmp/should_not_exist",
        "--seeds", *[str(s) for s in LORA_CONFIRMATORY_SEEDS],
        "--arms", "vanilla", "exact", "lr_control", "rank",
        "--target-plasticity", "0.85",
        "--confirmatory",
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "braços pré-registrados" in combined


def test_confirmatory_refuses_a_subset_of_the_seeds() -> None:
    """Stopping early on a subset would be optional stopping."""

    result = _run(
        "--output", "/tmp/should_not_exist",
        "--seeds", *[str(s) for s in LORA_CONFIRMATORY_SEEDS[:5]],
        "--arms", *PREREGISTERED_ARMS,
        "--target-plasticity", "0.85",
        "--confirmatory",
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "seeds pré-registradas" in combined


def test_confirmatory_requires_the_declared_plasticity_target() -> None:
    """Running the confirmation unpaired would answer a different question."""

    result = _run(
        "--output", "/tmp/should_not_exist",
        "--seeds", *[str(s) for s in LORA_CONFIRMATORY_SEEDS],
        "--arms", *PREREGISTERED_ARMS,
        "--confirmatory",
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "target-plasticity" in combined


def test_the_preregistered_arm_set_matches_the_protocol_document() -> None:
    """The frozen document and the code must not drift apart."""

    assert set(PREREGISTERED_ARMS) == {"vanilla", "exact", "lr_control"}
