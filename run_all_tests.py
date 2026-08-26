#!/usr/bin/env python3
"""Run the complete experiment protocol from the confirmatory notebook.

The default invocation executes every notebook section serially without
running pytest. Completed seeds are reused safely on subsequent invocations
when their saved configuration matches.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.confirmatory_split_mnist import (
    CANDIDATE,
    CONFIRMATORY_SEEDS,
    FROZEN_CONFIG,
    preregistration_manifest,
    run_confirmation,
    validate_preregistration,
)
from experiments.dualheat_pairs import (
    METHOD_PAIRS,
    PAIRED_METHODS,
    pair_protocol,
    paired_config,
    run_dualheat_pairs,
)
from experiments.multi_seed import run_multi_seed
from experiments.provenance import relative_path
from experiments.split_mnist_suite import (
    ABLATION_METHODS,
    ALL_BASELINES,
    ALL_VISUAL_METHODS,
    CAPACITY_ARCHITECTURES,
    REPLAY_MEMORY_SIZES,
    SLOWHEAT_DERPP_METHODS,
    run_ablation_matrix,
    run_all_baselines,
    run_all_visual_methods,
    run_capacity_generalization,
    run_equal_example_budget,
    run_slowheat_derpp_test,
)
from experiments.synthetic_cl import SYNTHETIC_METHODS, load_config
from experiments.visual_generalization import (
    generalization_configs,
    run_visual_generalization,
)

SEED_GENERATOR_SEED = 20_260_819
MAX_GENERATED_SEED = 2**31 - 1
SECTION_NAMES = (
    "dualheat-pairs",
    "synthetic-all-methods",
    "split-mnist-all-methods",
    "confirmation",
    "all-baselines",
    "equal-examples",
    "ablations",
    "slowheat-derpp",
    "split-mnist-generalization",
    "permuted-mnist",
    "split-cifar10",
    "split-cifar10-cnn",
    "split-cifar10-cnn-sweep",
    "split-cifar100",
)
DEFAULT_SECTION_NAMES = tuple(
    name
    for name in SECTION_NAMES
    if name
    not in {
        "dualheat-pairs",
        "synthetic-all-methods",
        "split-mnist-all-methods",
        "split-cifar10-cnn",
        "split-cifar10-cnn-sweep",
    }
)
ALL_DATASET_METHOD_SECTIONS = (
    "synthetic-all-methods",
    "split-mnist-all-methods",
    "permuted-mnist",
    "split-cifar10",
    "split-cifar100",
)
SECTION_OUTPUT_DIRS = {
    "dualheat-pairs": "dualheat_pairs",
    "synthetic-all-methods": "synthetic_all_methods",
    "split-mnist-all-methods": "split_mnist_all_methods",
    "confirmation": "confirmation",
    "all-baselines": "all_baselines_equal_epochs",
    "equal-examples": "all_baselines_equal_examples",
    "ablations": "ablations",
    "slowheat-derpp": "slowheat_derpp_exploratory",
    "split-mnist-generalization": "split_mnist_generalization",
    "permuted-mnist": "permuted_mnist",
    "split-cifar10": "split_cifar10",
    "split-cifar10-cnn": "split_cifar10_cnn",
    "split-cifar10-cnn-sweep": "split_cifar10_cnn_sweep",
    "split-cifar100": "split_cifar100",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="diretório de datasets (padrão: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/split_mnist_protocol"),
        help="raiz dos resultados do protocolo",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="dispositivo aceito pelo PyTorch, por exemplo cpu ou cuda",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        required=True,
        help="quantidade de seeds aleatórias para as análises secundárias",
    )
    parser.add_argument(
        "--baseline-seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "seeds das análises secundárias; quando fornecidas, a quantidade "
            "deve coincidir com --num-seeds; a confirmação permanece congelada"
        ),
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=SECTION_NAMES,
        default=list(DEFAULT_SECTION_NAMES),
        help="subconjunto de seções; por padrão executa todas",
    )
    parser.add_argument(
        "--all-datasets-all-methods",
        action="store_true",
        help=(
            "executar todos os métodos compatíveis em Synthetic, Split-MNIST, "
            "Permuted-MNIST, Split-CIFAR-10 e Split-CIFAR-100"
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="não baixar datasets; exige os dados já disponíveis",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="não reutilizar seeds concluídas (requer diretórios de saída novos)",
    )
    parser.add_argument(
        "--run-unit-tests",
        action="store_true",
        help="executar pytest como pré-validação opcional dos benchmarks",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="reduzir mensagens dos runners experimentais",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validar e imprimir o plano sem executar testes, downloads ou treino",
    )
    args = parser.parse_args(raw_argv)
    if args.all_datasets_all_methods:
        if "--sections" in raw_argv:
            parser.error(
                "--all-datasets-all-methods não pode ser combinado com --sections"
            )
        args.sections = list(ALL_DATASET_METHOD_SECTIONS)
    custom_baseline_seeds = args.baseline_seeds is not None
    if args.num_seeds <= 0:
        parser.error("--num-seeds deve ser maior que zero")
    if custom_baseline_seeds:
        if len(set(args.baseline_seeds)) != len(args.baseline_seeds):
            parser.error("--baseline-seeds não pode conter duplicatas")
        if len(args.baseline_seeds) != args.num_seeds:
            parser.error(
                "a quantidade de valores em --baseline-seeds deve coincidir "
                "com --num-seeds"
            )
    elif args.num_seeds > MAX_GENERATED_SEED + 1:
        parser.error(
            f"--num-seeds não pode exceder {MAX_GENERATED_SEED + 1}"
        )
    else:
        generator = random.Random(SEED_GENERATOR_SEED)
        args.baseline_seeds = generator.sample(
            range(MAX_GENERATED_SEED + 1), args.num_seeds
        )
    return args


def _project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _portable_project_path(path: Path) -> str:
    """Keep metadata relative to the project, including external directories."""
    return relative_path(_project_path(path), base=PROJECT_ROOT)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    temporary.replace(path)


def _methods_by_section(device: str) -> dict[str, list[str]]:
    visual_configs = generalization_configs(device)
    return {
        "dualheat-pairs": list(PAIRED_METHODS),
        "synthetic-all-methods": list(SYNTHETIC_METHODS),
        "split-mnist-all-methods": list(ALL_VISUAL_METHODS),
        "confirmation": list(FROZEN_CONFIG.methods),
        "all-baselines": list(ALL_BASELINES),
        "equal-examples": [
            method
            for method in ALL_BASELINES
            if method not in {"replay_more_epochs", "replay_early_stopping"}
        ],
        "ablations": list(ABLATION_METHODS),
        "slowheat-derpp": list(SLOWHEAT_DERPP_METHODS),
        "split-mnist-generalization": list(SLOWHEAT_DERPP_METHODS),
        "permuted-mnist": list(visual_configs["permuted_mnist"].methods),
        "split-cifar10": list(visual_configs["split_cifar10"].methods),
        "split-cifar10-cnn": list(visual_configs["split_cifar10_cnn"].methods),
        "split-cifar10-cnn-sweep": list(
            visual_configs["split_cifar10_cnn_sweep"].methods
        ),
        "split-cifar100": list(visual_configs["split_cifar100"].methods),
    }


def build_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _project_path(args.output_dir)
    methods = _methods_by_section(args.device)
    sections: dict[str, Any] = {}
    for name in args.sections:
        details: dict[str, Any] = {
            "status": "pending",
            "methods": methods[name],
            "seeds": (
                list(CONFIRMATORY_SEEDS)
                if name == "confirmation"
                else list(args.baseline_seeds)
            ),
            "output_dir": _portable_project_path(
                output_dir / SECTION_OUTPUT_DIRS[name]
            ),
        }
        if name == "dualheat-pairs":
            details["protocol"] = pair_protocol(
                paired_config(device=args.device), list(args.baseline_seeds)
            )
            details["pairs"] = [
                {"reference": pair.reference, "candidate": pair.candidate}
                for pair in METHOD_PAIRS
            ]
            details["analysis_status"] = "exploratory_not_independent_confirmation"
        elif name == "ablations":
            details["replay_memory_per_class"] = list(REPLAY_MEMORY_SIZES)
        elif name == "split-mnist-generalization":
            details["architectures"] = [
                list(dims) for dims in CAPACITY_ARCHITECTURES.values()
            ]
        elif name == "synthetic-all-methods":
            config = load_config(PROJECT_ROOT / "configs/synthetic_ablation_pilot.json")
            details["protocol"] = {
                "scenario": "class_incremental",
                "task_count": config.task_count,
                "classes_per_task": config.classes_per_task,
                "class_count": config.class_count,
                "inference_task_id": False,
                "device": "cpu",
            }
        elif name in {
            "split-cifar10",
            "split-cifar10-cnn",
            "split-cifar10-cnn-sweep",
            "split-cifar100",
        }:
            config = generalization_configs(args.device)[name.replace("-", "_")]
            details["protocol"] = {
                "scenario": config.scenario,
                "task_count": config.task_count,
                "classes_per_task": config.classes_per_task,
                "class_count": len(config.class_order),
                "inference_task_id": False,
            }
            if config.backbone == "cnn":
                details["protocol"].update(
                    {
                        "backbone": "cnn",
                        "image_shape": list(config.image_shape or ()),
                        "channels": list(config.cnn_channels),
                        "pooled_size": list(config.cnn_pooled_size),
                        "epochs_per_task": config.epochs_per_task,
                    }
                )
        sections[name] = details
    return {
        "status": "planned",
        "created_at": _now(),
        "project_root": ".",
        "data_dir": _portable_project_path(args.data_dir),
        "output_dir": _portable_project_path(output_dir),
        "device": args.device,
        "download": not args.no_download,
        "resume": not args.fresh,
        "unit_tests": {
            "status": "pending" if args.run_unit_tests else "skipped",
            "command": [
                Path(sys.executable).name,
                "-m",
                "pytest",
                "tests",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
        },
        "preregistration": preregistration_manifest(),
        "sections": sections,
    }


def _run_unit_tests(command: list[str]) -> int:
    env = os.environ.copy()
    project_paths = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if existing_pythonpath := env.get("PYTHONPATH"):
        project_paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(project_paths)
    # Metadata records a portable executable name; execution keeps this interpreter.
    return subprocess.call([sys.executable, *command[1:]], cwd=PROJECT_ROOT, env=env)


def _run_section(
    name: str,
    args: argparse.Namespace,
    *,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if name == "synthetic-all-methods":
        config = load_config(PROJECT_ROOT / "configs/synthetic_ablation_pilot.json")
        return run_multi_seed(
            config,
            seeds=list(args.baseline_seeds),
            output_dir=output_dir / SECTION_OUTPUT_DIRS[name],
        )

    common = {
        "seeds": list(args.baseline_seeds),
        "data_dir": data_dir,
        "output_dir": output_dir / SECTION_OUTPUT_DIRS[name],
        "device": args.device,
        "download": not args.no_download,
        "verbose": not args.quiet,
        "resume": not args.fresh,
    }
    if name == "confirmation":
        confirmation_common = dict(common)
        confirmation_common.pop("seeds")
        return run_confirmation(**confirmation_common)
    if name == "dualheat-pairs":
        return run_dualheat_pairs(**common)
    if name == "split-mnist-all-methods":
        return run_all_visual_methods(**common)
    if name == "all-baselines":
        return run_all_baselines(**common)
    if name == "equal-examples":
        return run_equal_example_budget(**common)
    if name == "ablations":
        return run_ablation_matrix(**common)
    if name == "slowheat-derpp":
        return run_slowheat_derpp_test(**common)
    if name == "split-mnist-generalization":
        return run_capacity_generalization(**common)
    if name == "permuted-mnist":
        return run_visual_generalization("permuted_mnist", **common)
    if name == "split-cifar10":
        return run_visual_generalization("split_cifar10", **common)
    if name == "split-cifar10-cnn":
        return run_visual_generalization("split_cifar10_cnn", **common)
    if name == "split-cifar10-cnn-sweep":
        return run_visual_generalization("split_cifar10_cnn_sweep", **common)
    if name == "split-cifar100":
        return run_visual_generalization("split_cifar100", **common)
    raise ValueError(f"seção desconhecida: {name}")


def _save_primary_result(output_dir: Path) -> Path:
    aggregate_path = output_dir / "confirmation" / "aggregate.json"
    with aggregate_path.open(encoding="utf-8") as handle:
        aggregate = json.load(handle)
    primary = aggregate["paired_differences_vs_replay"][CANDIDATE][
        "final_average_accuracy"
    ]["confirmatory"]
    destination = output_dir / "primary_result.json"
    _write_json(destination, primary)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_preregistration()
    plan = build_run_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0

    data_dir = _project_path(args.data_dir)
    output_dir = _project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "benchmark_index.json"
    plan["status"] = "running"
    plan["started_at"] = _now()
    _write_json(index_path, plan)

    if args.run_unit_tests:
        print("[pré-validação] executando pytest", flush=True)
        plan["unit_tests"]["status"] = "running"
        _write_json(index_path, plan)
        return_code = _run_unit_tests(plan["unit_tests"]["command"])
        plan["unit_tests"]["finished_at"] = _now()
        plan["unit_tests"]["return_code"] = return_code
        plan["unit_tests"]["status"] = "completed" if return_code == 0 else "failed"
        if return_code != 0:
            plan["status"] = "failed"
            plan["finished_at"] = _now()
            _write_json(index_path, plan)
            return return_code
        _write_json(index_path, plan)

    for index, name in enumerate(args.sections, start=1):
        section = plan["sections"][name]
        print(
            f"[benchmark {index}/{len(args.sections)}] {name}",
            flush=True,
        )
        section["status"] = "running"
        section["started_at"] = _now()
        _write_json(index_path, plan)
        try:
            _run_section(
                name,
                args,
                data_dir=data_dir,
                output_dir=output_dir,
            )
        except BaseException as error:
            section["status"] = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            )
            section["error"] = f"{type(error).__name__}: {error}"
            section["finished_at"] = _now()
            plan["status"] = section["status"]
            plan["finished_at"] = _now()
            _write_json(index_path, plan)
            raise
        section["status"] = "completed"
        section["finished_at"] = _now()
        _write_json(index_path, plan)

    if "confirmation" in args.sections:
        primary_path = _save_primary_result(output_dir)
        plan["primary_result"] = _portable_project_path(primary_path)
    plan["status"] = "completed"
    plan["finished_at"] = _now()
    _write_json(index_path, plan)
    print(
        f"Protocolo concluído. Índice: {_portable_project_path(index_path)}",
        flush=True,
    )
    if "primary_result" in plan:
        print(f"Endpoint primário: {plan['primary_result']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
