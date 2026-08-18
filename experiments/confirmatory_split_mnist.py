"""Immutable preregistration and entry point for independent confirmation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.confirmatory_statistics import PRIMARY_ENDPOINT
from experiments.split_mnist import (
    SplitMNISTConfig,
    config_payload,
    run_split_mnist_multi_seed,
)

CANDIDATE = "slowheat_replay_hidden_beta_30_budget_0.25"
REFERENCE = "replay"
PREREGISTERED_AT = "2026-08-15"

# Chosen and committed before any confirmatory execution. The deliberately
# distant range avoids every seed explicitly present in the repository and the
# commonly used exploratory 11*k sequence through 220.
CONFIRMATORY_SEEDS = (
    104_729,
    130_363,
    155_921,
    181_081,
    206_369,
    231_701,
    257_053,
    282_377,
    307_723,
    333_017,
    358_373,
    383_729,
    409_063,
    434_399,
    459_749,
    485_071,
    510_403,
    535_751,
    561_097,
    586_429,
)
DECLARED_EXPLORATORY_SEEDS = tuple(11 * index for index in range(1, 21))

FROZEN_CONFIG = SplitMNISTConfig(
    hidden_dims=(256, 128),
    batch_size=128,
    epochs_per_task=10,
    train_per_class=1_000,
    validation_per_class=200,
    test_per_class=500,
    learning_rate=1e-3,
    weight_decay=1e-4,
    slow_strength=30.0,
    plasticity_budget=0.25,
    optimizer_state_policy="follow_update",
    replay_per_class=20,
    replay_batch_size=64,
    methods=(REFERENCE, CANDIDATE),
)


def preregistration_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "preregistered_at": PREREGISTERED_AT,
        "status": "frozen_before_execution",
        "primary_endpoint": PRIMARY_ENDPOINT,
        "candidate": CANDIDATE,
        "reference": REFERENCE,
        "difference_direction": "candidate_minus_reference",
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "declared_exploratory_seeds": list(DECLARED_EXPLORATORY_SEEDS),
        "config": config_payload(FROZEN_CONFIG),
        "analysis": {
            "student_t": "paired, two-sided, 95% CI",
            "bootstrap": "paired percentile CI, 10000 resamples",
            "signs": "positive/negative/tie counts and exact two-sided sign test",
            "multiplicity": (
                "final_average_accuracy is the sole primary endpoint; all other "
                "metrics and baselines are secondary/exploratory"
            ),
        },
        "immutability_rule": (
            "No hyperparameter, seed, endpoint or analysis choice may be changed "
            "after inspecting confirmatory outcomes."
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_preregistration() -> None:
    FROZEN_CONFIG.validate()
    if len(CONFIRMATORY_SEEDS) != 20 or len(set(CONFIRMATORY_SEEDS)) != 20:
        raise RuntimeError("a confirmação requer exatamente 20 seeds únicas")
    overlap = set(CONFIRMATORY_SEEDS) & set(DECLARED_EXPLORATORY_SEEDS)
    if overlap:
        raise RuntimeError(f"seeds confirmatórias sobrepõem exploração: {overlap}")
    if FROZEN_CONFIG.methods != (REFERENCE, CANDIDATE):
        raise RuntimeError("métodos confirmatórios foram alterados")
    if (
        FROZEN_CONFIG.epochs_per_task != 10
        or FROZEN_CONFIG.replay_per_class != 20
        or FROZEN_CONFIG.slow_strength != 30.0
        or FROZEN_CONFIG.plasticity_budget != 0.25
    ):
        raise RuntimeError("hiperparâmetros confirmatórios foram alterados")


def run_confirmation(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    """Run or safely resume the frozen paired experiment.

    An existing identical lock is accepted only in resume mode. Completed
    seeds with matching saved configurations are reused; a different lock or
    per-seed configuration fails closed.
    """

    validate_preregistration()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = preregistration_manifest()
    lock_path = output_path / "preregistration.lock.json"
    if lock_path.exists():
        with lock_path.open(encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        serialized_manifest = json.loads(json.dumps(manifest))
        if existing_manifest != serialized_manifest:
            raise RuntimeError(
                "preregistration.lock.json difere do protocolo atual; "
                "use outro diretório e não sobrescreva o lock"
            )
        if not resume:
            raise FileExistsError(
                f"pré-registro já existe em {lock_path}; use resume=True para "
                "reutilizar seeds concluídas"
            )
    else:
        with lock_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    return run_split_mnist_multi_seed(
        replace(FROZEN_CONFIG, device=device),
        seeds=list(CONFIRMATORY_SEEDS),
        data_dir=data_dir,
        output_dir=output_path,
        download=download,
        verbose=verbose,
        paired_references=(REFERENCE,),
        resume=resume,
    )
