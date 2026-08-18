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
from experiments.split_mnist_suite import (
    ABLATION_METHODS,
    ALL_BASELINES,
    CLASS_ORDERS,
    SLOWHEAT_DERPP_METHODS,
    run_ablation_matrix,
    run_all_baselines,
    run_equal_example_budget,
    run_order_and_capacity_generalization,
    run_slowheat_derpp_test,
)
from experiments.visual_generalization import (
    CORE50_RUNS,
    generalization_configs,
    run_visual_generalization,
)

BASELINE_SEEDS = (311, 617, 919, 1223, 1523, 1823, 2129, 2423, 2729, 3037)
SECTION_NAMES = (
    "confirmation",
    "all-baselines",
    "equal-examples",
    "ablations",
    "slowheat-derpp",
    "split-mnist-generalization",
    "permuted-mnist",
    "core50",
)
SECTION_OUTPUT_DIRS = {
    "confirmation": "confirmation",
    "all-baselines": "all_baselines_equal_epochs",
    "equal-examples": "all_baselines_equal_examples",
    "ablations": "ablations",
    "slowheat-derpp": "slowheat_derpp_exploratory",
    "split-mnist-generalization": "split_mnist_generalization",
    "permuted-mnist": "permuted_mnist",
    "core50": "core50_nc",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="diretório de datasets (padrão: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "split_mnist_protocol",
        help="raiz dos resultados do protocolo",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="dispositivo aceito pelo PyTorch, por exemplo cpu ou cuda",
    )
    parser.add_argument(
        "--baseline-seeds",
        nargs="+",
        type=int,
        default=list(BASELINE_SEEDS),
        help="seeds das análises secundárias; a confirmação permanece congelada",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=SECTION_NAMES,
        default=list(SECTION_NAMES),
        help="subconjunto de seções; por padrão executa todas",
    )
    parser.add_argument(
        "--core50-dir",
        type=Path,
        help=(
            "raiz local do CORe50 128x128 com s1/..s11/ e os filelists "
            "oficiais NC_inc"
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="não baixar MNIST; exige o dataset já disponível",
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
    args = parser.parse_args(argv)
    if len(set(args.baseline_seeds)) != len(args.baseline_seeds):
        parser.error("--baseline-seeds não pode conter duplicatas")
    return args


def _project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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
        "core50": list(visual_configs["core50"].methods),
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
                else (
                    list(CORE50_RUNS)
                    if name == "core50"
                    else list(args.baseline_seeds)
                )
            ),
            "output_dir": str(output_dir / SECTION_OUTPUT_DIRS[name]),
        }
        if name == "ablations":
            details["replay_memory_per_class"] = [5, 10, 20, 50, 100]
        elif name == "split-mnist-generalization":
            details["class_orders"] = [list(order) for order in CLASS_ORDERS]
            details["architectures"] = [
                [256, 128],
                [512, 256],
                [512, 512, 256],
            ]
        elif name == "core50":
            core50_config = generalization_configs(args.device)["core50"]
            details["protocol"] = {
                "scenario": "new_classes_class_incremental",
                "official_runs": list(CORE50_RUNS),
                "task_count": core50_config.task_count,
                "task_class_counts": list(core50_config.task_class_counts or ()),
                "class_count": len(core50_config.class_order),
                "inference_task_id": False,
                "primary_evaluation": "class_il_seen_classes",
                "secondary_evaluation": "task_il_diagnostic",
                "paired_references": ["replay", "derpp"],
            }
        sections[name] = details
    return {
        "status": "planned",
        "created_at": _now(),
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(_project_path(args.data_dir)),
        "output_dir": str(output_dir),
        "device": args.device,
        "download": not args.no_download,
        "resume": not args.fresh,
        "unit_tests": {
            "status": "pending" if args.run_unit_tests else "skipped",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                str(PROJECT_ROOT / "tests"),
                "-q",
                "-p",
                "no:cacheprovider",
            ],
        },
        "preregistration": preregistration_manifest(),
        "sections": sections,
        "core50_dir": (
            str(_project_path(args.core50_dir))
            if args.core50_dir is not None
            else None
        ),
    }


def _run_unit_tests(command: list[str]) -> int:
    env = os.environ.copy()
    project_paths = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if existing_pythonpath := env.get("PYTHONPATH"):
        project_paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(project_paths)
    return subprocess.call(command, cwd=PROJECT_ROOT, env=env)


def _run_section(
    name: str,
    args: argparse.Namespace,
    *,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
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
    if name == "all-baselines":
        return run_all_baselines(**common)
    if name == "equal-examples":
        return run_equal_example_budget(**common)
    if name == "ablations":
        return run_ablation_matrix(**common)
    if name == "slowheat-derpp":
        return run_slowheat_derpp_test(**common)
    if name == "split-mnist-generalization":
        return run_order_and_capacity_generalization(**common)
    if name == "permuted-mnist":
        return run_visual_generalization("permuted_mnist", **common)
    if name == "core50":
        if args.core50_dir is None:
            raise ValueError("--core50-dir não foi informado")
        core50_common = dict(common)
        core50_common["seeds"] = list(CORE50_RUNS)
        core50_common["data_dir"] = _project_path(args.core50_dir)
        core50_common["download"] = False
        return run_visual_generalization("core50", **core50_common)
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
        if name == "core50" and args.core50_dir is None:
            section["status"] = "skipped"
            section["reason"] = "--core50-dir não foi informado"
            section["finished_at"] = _now()
            print(
                "[CORe50] ignorado: informe --core50-dir para executá-lo",
                flush=True,
            )
            _write_json(index_path, plan)
            continue

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
        plan["primary_result"] = str(primary_path)
    plan["status"] = "completed"
    plan["finished_at"] = _now()
    _write_json(index_path, plan)
    print(f"Protocolo concluído. Índice: {index_path}", flush=True)
    if "primary_result" in plan:
        print(f"Endpoint primário: {plan['primary_result']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
