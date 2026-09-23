"""Hard versus soft protection across MLP and CNN hosts.

Motivation: the Transformer results (BERT/CLINC150) use *hard* protection --
the consolidated units are frozen outright -- while every MLP and CNN result
in the project uses *soft* protection, ``1 / (1 + beta * h)`` with beta = 30.
That leaves the regime of protection confounded with the architecture, so the
manuscript cannot say whether hard protection was necessary on Transformers or
merely convenient.  This suite removes the confound by running both regimes on
the non-Transformer hosts under one paired protocol.

Design decisions that matter for interpretation:

* The soft comparator is ``slowheat_beta_30_budget_0.25``, NOT the
  ``..._hidden_...`` variant used by :mod:`experiments.dualheat_pairs`.  Hard
  freezing protects the output layer, so the iso-scope soft arm must protect it
  too; otherwise the contrast would mix protection regime with protection
  scope.
* Both arms share the same capacity budget (0.25).  ``hard_freeze`` binarizes
  exactly the set the budget selects, so the two regimes protect the same units
  and differ only in how hard they are held.
* Replay arms are included because the BERT finding that mattered was hard
  protection *losing* to replay.  ``hard_freeze_replay`` is the direct analogue.

The primary endpoint, contrasts, seeds and multiplicity rule are frozen to a
``hard_vs_soft_protocol.json`` file before the first training step.  Re-running
with a different protocol in the same output directory is refused.

This module never touches the frozen Split-MNIST confirmation and refuses any
seed reserved for it.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from experiments.artifacts import write_json_atomic
from experiments.confirmatory_split_mnist import CONFIRMATORY_SEEDS
from experiments.confirmatory_statistics import PRIMARY_ENDPOINT
from experiments.dualheat_pairs import MethodPair, _holm_adjust, summarize_pair_results
from experiments.provenance import relative_path
from experiments.split_mnist import (
    SplitMNISTConfig,
    config_payload,
    replay_selection_is_method_independent,
    run_split_mnist_multi_seed,
)
from experiments.split_mnist_suite import baseline_config
from experiments.visual_generalization import (
    generalization_configs,
    load_permuted_mnist,
    load_split_cifar10,
    load_split_cifar100,
)

HARD = "hard_freeze"
HARD_REPLAY = "hard_freeze_replay"
# Iso-scope soft comparators: these protect the output layer, exactly like the
# hard arms.  The `_hidden_` variants used elsewhere in the project do not.
SOFT = "slowheat_beta_30_budget_0.25"
SOFT_REPLAY = "slowheat_replay_beta_30_budget_0.25"

#: The contrast that answers the professor's question comes first.  Holm is
#: applied across exactly these four accuracy contrasts within each dataset.
METHOD_PAIRS = (
    MethodPair("Hard vs Soft", SOFT, HARD),
    MethodPair("Hard vs Soft (replay)", SOFT_REPLAY, HARD_REPLAY),
    MethodPair("Hard vs Replay", "replay", HARD_REPLAY),
    MethodPair("Hard vs Vanilla", "vanilla", HARD),
)

PAIRED_METHODS = tuple(
    dict.fromkeys(
        method for pair in METHOD_PAIRS for method in (pair.reference, pair.candidate)
    )
)

DATASETS = ("split_mnist", "permuted_mnist", "split_cifar10", "split_cifar100")
BACKBONES = ("mlp", "cnn")

#: Which concrete config from :func:`generalization_configs` backs each
#: (dataset, backbone) pair.  ``None`` means the combination has no versioned
#: config and is refused rather than silently downgraded to an MLP -- running a
#: CNN-labelled directory on an MLP would invalidate the whole comparison.
_CONFIG_KEYS: dict[tuple[str, str], str | None] = {
    ("split_mnist", "mlp"): "__baseline__",
    ("split_mnist", "cnn"): None,
    ("permuted_mnist", "mlp"): "permuted_mnist",
    ("permuted_mnist", "cnn"): None,
    ("split_cifar10", "mlp"): "split_cifar10",
    ("split_cifar10", "cnn"): "split_cifar10_cnn",
    ("split_cifar100", "mlp"): "split_cifar100",
    ("split_cifar100", "cnn"): None,
}


@dataclass(frozen=True)
class SuiteTarget:
    dataset: str
    backbone: str

    @property
    def name(self) -> str:
        return f"{self.dataset}_{self.backbone}"


def suite_config(
    dataset: str, backbone: str, device: str = "cpu"
) -> SplitMNISTConfig:
    if dataset not in DATASETS:
        raise ValueError(f"dataset não suportado: {dataset}")
    if backbone not in BACKBONES:
        raise ValueError(f"backbone não suportado: {backbone}")
    key = _CONFIG_KEYS[(dataset, backbone)]
    if key is None:
        raise ValueError(
            f"não existe config versionado para {dataset} com backbone "
            f"{backbone}; combinações disponíveis: "
            + ", ".join(
                f"{d}/{b}" for (d, b), v in _CONFIG_KEYS.items() if v is not None
            )
        )
    base = (
        baseline_config(device=device)
        if key == "__baseline__"
        else generalization_configs(device)[key]
    )
    if base.backbone != backbone:
        raise ValueError(
            f"config {key} tem backbone {base.backbone}, esperado {backbone}"
        )
    return replace(
        base,
        methods=PAIRED_METHODS,
        slow_strength=30.0,
        plasticity_budget=0.25,
        optimizer_state_policy="follow_update",
    )


def _validate_seeds(seeds: list[int]) -> None:
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds deve conter inteiros não negativos")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds não pode conter duplicatas")
    reserved = set(seeds).intersection(CONFIRMATORY_SEEDS)
    if reserved:
        raise ValueError(
            f"seeds reservadas à confirmação congelada não podem entrar: {sorted(reserved)}"
        )


def suite_protocol(
    config: SplitMNISTConfig, seeds: list[int], *, dataset: str, backbone: str
) -> dict[str, Any]:
    config.validate()
    _validate_seeds(seeds)
    if config.methods != PAIRED_METHODS:
        raise ValueError("a suíte requer exatamente os métodos declarados")
    if config.optimizer_state_policy != "follow_update":
        raise ValueError("a suíte requer optimizer_state_policy=follow_update")
    return {
        "schema_version": 1,
        "status": "exploratory_frozen_before_execution",
        "question": (
            "Hard protection is used on the Transformer hosts and soft "
            "protection on MLP/CNN. Does the regime, not the architecture, "
            "explain the difference?"
        ),
        "dataset": dataset,
        "backbone": backbone,
        "pairs": [asdict(pair) for pair in METHOD_PAIRS],
        "primary_contrast": METHOD_PAIRS[0].label,
        "seeds": seeds,
        "config": json.loads(json.dumps(config_payload(config))),
        "component_settings": {
            "hard": "binary freeze of the consolidated set (slow_heat > 0)",
            "soft": "1 / (1 + 30 * slow_heat)",
            "shared_capacity_budget": 0.25,
            "scope": "output layer protected in BOTH arms (iso-scope)",
            "consolidation": "max_at_known_task_boundaries",
            "optimizer_state_policy": "follow_update",
        },
        "primary_endpoint": PRIMARY_ENDPOINT,
        "difference_direction": "candidate_minus_reference",
        "multiplicity": "Holm over the four accuracy contrasts within this dataset",
        "fairness": "paired initialization, data, batches, replay indices and epochs",
        "declared_expectations": [
            (
                "BERT showed hard > soft on accuracy and hard < replay. If the "
                "regime explains the Transformer result, hard should beat soft "
                "here too."
            ),
            (
                "A null or reversed result is informative and will be reported: "
                "it would mean the Transformer finding is architecture-specific."
            ),
        ],
        "limitations": [
            "Fixed defaults, not individually tuned baselines; no universal benefit claim.",
            "No task ID at inference; known task boundaries at consolidation.",
            "Exploratory: this suite is not a preregistered confirmation.",
        ],
    }


def run_target(
    *,
    dataset: str,
    backbone: str,
    seeds: list[int],
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    download: bool = True,
    verbose: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    config = suite_config(dataset, backbone, device)
    if not replay_selection_is_method_independent(config.replay_selection):
        raise ValueError(
            "a suíte requer replay_selection='first'; seleção adaptativa "
            "pertence ao tratamento experimental"
        )
    loaders = {
        "split_mnist": None,
        "permuted_mnist": load_permuted_mnist,
        "split_cifar10": load_split_cifar10,
        "split_cifar100": load_split_cifar100,
    }
    protocol = suite_protocol(config, seeds, dataset=dataset, backbone=backbone)
    output = Path(output_dir)
    lock = output / "hard_vs_soft_protocol.json"
    if lock.exists():
        if not resume or json.loads(lock.read_text(encoding="utf-8")) != protocol:
            raise ValueError(
                "protocolo diferente ou resume desativado; use outro output_dir"
            )
    elif output.exists() and any(output.iterdir()):
        raise ValueError("output_dir deve estar vazio para iniciar a suíte")
    else:
        write_json_atomic(lock, protocol)
    run_split_mnist_multi_seed(
        config,
        seeds=seeds,
        data_dir=data_dir,
        output_dir=output,
        download=download,
        verbose=verbose,
        paired_references=(),
        task_loader=loaders[dataset],
        resume=resume,
    )
    report = summarize_pair_results(output, output_dir=output, pairs=METHOD_PAIRS)
    # summarize_pair_results only applies Holm for its own four-pair family.
    if len(seeds) >= 2:
        adjusted = _holm_adjust(
            [
                pair["metrics"][PRIMARY_ENDPOINT]["student_t"]["two_sided_p"]
                for pair in report["pairs"]
            ]
        )
        for pair, value in zip(report["pairs"], adjusted, strict=True):
            pair["metrics"][PRIMARY_ENDPOINT]["holm_adjusted_p"] = value
        report["multiplicity"] = (
            "Holm over the four hard-vs-soft accuracy contrasts within this dataset"
        )
        write_json_atomic(output / "pair_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["split_mnist"])
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=["mlp"])
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/hard_vs_soft"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not 1 <= args.num_seeds <= 2**31:
        parser.error("--num-seeds deve estar entre 1 e 2**31")
    if args.seeds is not None and len(args.seeds) != args.num_seeds:
        parser.error("--seeds deve conter exatamente --num-seeds valores")
    seeds = args.seeds
    if seeds is None:
        # Fixed draw seed: the same call always yields the same seeds, and they
        # are recorded in the protocol file before any training starts.
        seeds = random.Random(20_260_923).sample(range(2**31), args.num_seeds)

    targets = [
        SuiteTarget(dataset, backbone)
        for dataset in dict.fromkeys(args.datasets)
        for backbone in dict.fromkeys(args.backbones)
    ]
    if args.dry_run:
        payload = {
            target.name: suite_protocol(
                suite_config(target.dataset, target.backbone, args.device),
                seeds,
                dataset=target.dataset,
                backbone=target.backbone,
            )
            for target in targets
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    for target in targets:
        output = args.output_dir / target.name
        run_target(
            dataset=target.dataset,
            backbone=target.backbone,
            seeds=seeds,
            data_dir=args.data_dir,
            output_dir=output,
            device=args.device,
            download=not args.no_download,
        )
        print(f"Relatório: {relative_path(output / 'pair_report.md', base=Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
