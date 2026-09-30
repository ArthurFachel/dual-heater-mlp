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
PREREGISTRATION_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "split_mnist_confirmation_preregistration.json"
)

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

#: Seeds frozen for the SlowHeat+DER++ confirmation (goals/protocol_derpp_confirmation.md).
#: Disjoint from CONFIRMATORY_SEEDS, from the exploratory multiples of 11, from the LoRA
#: confirmatory band (700001+) and from QB-2's band (1000003+).
#: Exploratory runs must never touch these.
DERPP_CONFIRMATORY_SEEDS: tuple[int, ...] = (
    2_000_003, 2_025_011, 2_050_017, 2_075_037, 2_100_043,
    2_125_059, 2_150_061, 2_175_087, 2_200_089, 2_225_097,
    2_250_101, 2_275_103, 2_300_119, 2_325_127, 2_350_133,
    2_375_149, 2_400_151, 2_425_163, 2_450_169, 2_475_179,
)

#: Seeds frozen for the penalty re-evaluation pilot
#: (goals/protocol_penalty_reevaluation.md, section G). Twelve seeds, chosen so
#: the design's significance floor (m * 2 / 2^n = 6 * 2 / 2^12 = 0.0029) clears
#: 0.05 with room for two dissenting seeds. Disjoint from every band already
#: spent; pinned by tests/test_penalty_reevaluation_seeds.py.
PENALTY_REEVALUATION_SEEDS: tuple[int, ...] = (
    7_000_003, 7_025_011, 7_050_017, 7_075_037, 7_100_043,
    7_125_059, 7_150_061, 7_175_087, 7_200_089, 7_225_097,
    7_250_101, 7_275_103,
)

#: Banda do protocolo L2 (`goals/protocol_sgd_plasticity.md` §D.1): mede se
#: `E > 1` persiste sob SGD puro, ou se é artefato de otimizador adaptativo.
#: Disjunta de todas as bandas acima, verificado em
#: `tests/test_sgd_plasticity.py`.
SGD_PLASTICITY_SEEDS: tuple[int, ...] = (
    8_000_011, 8_025_013, 8_050_021, 8_075_027, 8_100_037,
    8_125_043, 8_150_053, 8_175_059, 8_200_063, 8_225_069,
    8_250_077, 8_275_081,
)

#: Calibration seed for section I: cost and wiring only, deliberately OUTSIDE
#: the band above so its endpoints can never be pooled with the confirmatory
#: ones. Its n=1 endpoints carry no statistical standing.
PENALTY_REEVALUATION_CALIBRATION_SEED: int = 7_999_991

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
    with PREREGISTRATION_SOURCE.open(encoding="utf-8") as handle:
        committed = json.load(handle)
    frozen_fields = committed.get("frozen", {})
    expected_frozen = {
        field: getattr(FROZEN_CONFIG, field)
        for field in frozen_fields
    }
    expected_frozen["hidden_dims"] = list(FROZEN_CONFIG.hidden_dims)
    expected_identity = {
        "candidate": CANDIDATE,
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "difference_direction": "candidate_minus_reference",
        "frozen": expected_frozen,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "preregistered_at": PREREGISTERED_AT,
        "reference": REFERENCE,
        "rule": "No changes after confirmatory outcomes are inspected.",
        "status": "frozen_before_execution",
    }
    if committed != expected_identity:
        raise RuntimeError(
            "o pré-registro executável diverge de "
            "configs/split_mnist_confirmation_preregistration.json"
        )
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
