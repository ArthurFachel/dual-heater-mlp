"""Show accuracy and forgetting for replay-cache experiments."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_ORDER = ("first", "loss", "representative", "hybrid")
LEARNERS = {
    "replay": "Replay",
    "slowheat_replay_hidden_beta_30_budget_0.25": "SlowHeat+Replay",
}
METRICS = ("final_average_accuracy", "average_forgetting")


class ResultsError(RuntimeError):
    """Raised when result artifacts cannot be interpreted safely."""


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    ci95_half_width: float


@dataclass(frozen=True)
class ResultRow:
    benchmark: str
    cache: str
    learner_key: str
    seeds: int
    accuracy: MetricSummary
    forgetting: MetricSummary
    partial: bool = False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ResultsError(f"não foi possível ler JSON válido em {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ResultsError(f"o JSON deve conter um objeto: {path}")
    return payload


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultsError(f"valor numérico inválido em {context}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ResultsError(f"valor não finito em {context}")
    return converted


def _summary_from_payload(payload: Any, *, context: str) -> MetricSummary:
    if not isinstance(payload, dict):
        raise ResultsError(f"resumo inválido em {context}")
    mean = _finite_number(payload.get("mean"), context=f"{context}.mean")
    half_width = _finite_number(
        payload.get("ci95_normal_half_width", 0.0),
        context=f"{context}.ci95_normal_half_width",
    )
    if half_width < 0.0:
        raise ResultsError(f"IC95 negativo em {context}")
    return MetricSummary(mean, half_width)


def _summary_from_values(values: list[float]) -> MetricSummary:
    if not values:
        raise ResultsError("não há seeds para agregar")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return MetricSummary(mean, 0.0)
    standard_deviation = statistics.stdev(values)
    return MetricSummary(mean, 1.96 * standard_deviation / math.sqrt(len(values)))


def _validate_metric_range(summary: MetricSummary, *, context: str) -> None:
    if not 0.0 <= summary.mean <= 1.0:
        raise ResultsError(f"métrica fora de [0, 1] em {context}")


def _rows_from_sweep_report(path: Path) -> list[ResultRow]:
    report = _read_json(path)
    summaries = report.get("summaries")
    if not isinstance(summaries, dict):
        raise ResultsError(f"sweep_report.json não contém summaries: {path}")
    seeds_payload = report.get("seeds", [])
    seed_count = len(seeds_payload) if isinstance(seeds_payload, list) else 0
    rows: list[ResultRow] = []
    for benchmark, methods in summaries.items():
        if not isinstance(benchmark, str) or not isinstance(methods, dict):
            raise ResultsError(f"estrutura de benchmark inválida em {path}")
        for learner_key in LEARNERS:
            selectors = methods.get(learner_key)
            if not isinstance(selectors, dict):
                continue
            for cache in CACHE_ORDER:
                metrics = selectors.get(cache)
                if not isinstance(metrics, dict):
                    continue
                accuracy = _summary_from_payload(
                    metrics.get(METRICS[0]),
                    context=f"{benchmark}/{cache}/{learner_key}/{METRICS[0]}",
                )
                forgetting = _summary_from_payload(
                    metrics.get(METRICS[1]),
                    context=f"{benchmark}/{cache}/{learner_key}/{METRICS[1]}",
                )
                _validate_metric_range(accuracy, context=f"{benchmark}/accuracy")
                _validate_metric_range(forgetting, context=f"{benchmark}/forgetting")
                metric_values = metrics[METRICS[0]].get("values", [])
                row_seed_count = (
                    len(metric_values) if isinstance(metric_values, list) else seed_count
                )
                rows.append(
                    ResultRow(
                        benchmark=benchmark,
                        cache=cache,
                        learner_key=learner_key,
                        seeds=row_seed_count or seed_count,
                        accuracy=accuracy,
                        forgetting=forgetting,
                    )
                )
    return rows


def _infer_run_context(run_dir: Path) -> tuple[str, str]:
    if run_dir.name in CACHE_ORDER:
        return run_dir.parent.name, run_dir.name

    cache = "first"
    config_candidates = [
        run_dir / "multi_seed_config.json",
        run_dir / "config.json",
    ]
    config_candidates.extend(sorted(run_dir.glob("seed_*/config.json"))[:1])
    for config_path in config_candidates:
        if not config_path.is_file():
            continue
        config = _read_json(config_path)
        base = config.get("base_config", config)
        if isinstance(base, dict):
            configured = base.get("replay_selection", "first")
            if configured not in CACHE_ORDER:
                raise ResultsError(
                    f"replay_selection desconhecido em {config_path}: {configured}"
                )
            cache = configured
            break
    return run_dir.name, cache


def _seed_count_from_run(run_dir: Path, aggregate: dict[str, Any] | None) -> int:
    if aggregate is not None and isinstance(aggregate.get("seeds"), list):
        return len(aggregate["seeds"])
    manifest_path = run_dir / "multi_seed_config.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if isinstance(manifest.get("seeds"), list):
            return len(manifest["seeds"])
    return len(tuple(run_dir.glob("seed_*/results.json")))


def _rows_from_aggregate(run_dir: Path, path: Path) -> list[ResultRow]:
    aggregate = _read_json(path)
    methods = aggregate.get("methods")
    if not isinstance(methods, dict):
        raise ResultsError(f"aggregate.json não contém methods: {path}")
    benchmark, cache = _infer_run_context(run_dir)
    seed_count = _seed_count_from_run(run_dir, aggregate)
    rows = []
    for learner_key in LEARNERS:
        metrics = methods.get(learner_key)
        if not isinstance(metrics, dict):
            continue
        accuracy = _summary_from_payload(
            metrics.get(METRICS[0]), context=f"{path}:{learner_key}/{METRICS[0]}"
        )
        forgetting = _summary_from_payload(
            metrics.get(METRICS[1]), context=f"{path}:{learner_key}/{METRICS[1]}"
        )
        _validate_metric_range(accuracy, context=f"{path}:accuracy")
        _validate_metric_range(forgetting, context=f"{path}:forgetting")
        rows.append(
            ResultRow(
                benchmark=benchmark,
                cache=cache,
                learner_key=learner_key,
                seeds=seed_count,
                accuracy=accuracy,
                forgetting=forgetting,
            )
        )
    return rows


def _rows_from_partial_seeds(run_dir: Path) -> list[ResultRow]:
    seed_paths = sorted(run_dir.glob("seed_*/results.json"))
    if not seed_paths:
        return []
    benchmark, cache = _infer_run_context(run_dir)
    values: dict[str, dict[str, list[float]]] = {
        learner: {metric: [] for metric in METRICS} for learner in LEARNERS
    }
    for seed_path in seed_paths:
        payload = _read_json(seed_path)
        for learner_key in LEARNERS:
            result = payload.get(learner_key)
            if not isinstance(result, dict):
                continue
            metrics = result.get("metrics")
            if not isinstance(metrics, dict):
                raise ResultsError(f"metrics ausente para {learner_key} em {seed_path}")
            for metric in METRICS:
                values[learner_key][metric].append(
                    _finite_number(
                        metrics.get(metric), context=f"{seed_path}:{learner_key}/{metric}"
                    )
                )

    rows = []
    for learner_key, metrics in values.items():
        if not metrics[METRICS[0]]:
            continue
        if len(metrics[METRICS[0]]) != len(metrics[METRICS[1]]):
            raise ResultsError(f"métricas desalinhadas para {learner_key} em {run_dir}")
        accuracy = _summary_from_values(metrics[METRICS[0]])
        forgetting = _summary_from_values(metrics[METRICS[1]])
        _validate_metric_range(accuracy, context=f"{run_dir}:accuracy")
        _validate_metric_range(forgetting, context=f"{run_dir}:forgetting")
        rows.append(
            ResultRow(
                benchmark=benchmark,
                cache=cache,
                learner_key=learner_key,
                seeds=len(metrics[METRICS[0]]),
                accuracy=accuracy,
                forgetting=forgetting,
                partial=True,
            )
        )
    return rows


def _candidate_run_dirs(root: Path) -> list[Path]:
    candidates: set[Path] = set()
    if (root / "aggregate.json").is_file() or any(root.glob("seed_*/results.json")):
        candidates.add(root)
    candidates.update(path.parent for path in root.rglob("aggregate.json"))
    candidates.update(path.parent.parent for path in root.rglob("seed_*/results.json"))
    return sorted(
        (path for path in candidates if path.name != "no_memory"),
        key=lambda item: item.as_posix(),
    )


def discover_rows(path: str | Path) -> list[ResultRow]:
    root = Path(path)
    if not root.exists():
        raise ResultsError(f"path não existe: {root}")
    if root.is_file():
        if root.name != "sweep_report.json":
            raise ResultsError("quando RESULTS_PATH é arquivo, use sweep_report.json")
        return _rows_from_sweep_report(root)
    sweep_report = root / "sweep_report.json"
    if sweep_report.is_file():
        return _rows_from_sweep_report(sweep_report)

    rows: list[ResultRow] = []
    for run_dir in _candidate_run_dirs(root):
        aggregate_path = run_dir / "aggregate.json"
        if aggregate_path.is_file():
            rows.extend(_rows_from_aggregate(run_dir, aggregate_path))
        else:
            rows.extend(_rows_from_partial_seeds(run_dir))
    return rows


def filter_rows(
    rows: list[ResultRow], *, benchmark: str | None, cache: str | None
) -> list[ResultRow]:
    available_benchmarks = sorted({row.benchmark for row in rows})
    available_caches = [name for name in CACHE_ORDER if any(row.cache == name for row in rows)]
    if benchmark is not None and benchmark not in available_benchmarks:
        available = ", ".join(available_benchmarks) or "nenhum"
        raise ResultsError(f"benchmark desconhecido: {benchmark}; disponíveis: {available}")
    if cache is not None and cache not in available_caches:
        available = ", ".join(available_caches) or "nenhum"
        raise ResultsError(f"cache sem resultados: {cache}; disponíveis: {available}")
    selected = [
        row
        for row in rows
        if (benchmark is None or row.benchmark == benchmark)
        and (cache is None or row.cache == cache)
    ]
    if not selected:
        raise ResultsError("nenhum resultado de Replay ou SlowHeat+Replay foi encontrado")
    learner_order = {learner: index for index, learner in enumerate(LEARNERS)}
    cache_order = {name: index for index, name in enumerate(CACHE_ORDER)}
    return sorted(
        selected,
        key=lambda row: (
            row.benchmark,
            cache_order[row.cache],
            learner_order[row.learner_key],
        ),
    )


def select_extreme_rows(
    rows: list[ResultRow], *, metric: str | None, direction: str | None
) -> list[ResultRow]:
    """Select one cache per benchmark/learner using a metric mean."""
    if (metric is None) != (direction is None):
        raise ResultsError(
            "use --accuracy ou --forget junto com --high ou --low"
        )
    if metric is None:
        return rows

    grouped: dict[tuple[str, str], list[ResultRow]] = {}
    for row in rows:
        grouped.setdefault((row.benchmark, row.learner_key), []).append(row)

    selected: list[ResultRow] = []
    for candidates in grouped.values():
        def metric_mean(row: ResultRow) -> float:
            summary = row.accuracy if metric == "accuracy" else row.forgetting
            return summary.mean

        choose = max if direction == "high" else min
        selected.append(choose(candidates, key=metric_mean))

    selected_ids = {id(row) for row in selected}
    return [row for row in rows if id(row) in selected_ids]


def _format_metric(summary: MetricSummary) -> str:
    mean = 100.0 * summary.mean
    half_width = 100.0 * summary.ci95_half_width
    return f"{mean:.2f} [{mean - half_width:.2f}, {mean + half_width:.2f}]"


def format_table(rows: list[ResultRow]) -> str:
    headers = (
        "Benchmark",
        "Cache",
        "Learner",
        "Seeds",
        "Accuracy % (IC95%)",
        "Forgetting % (IC95%)",
    )
    body = [
        (
            row.benchmark,
            row.cache,
            LEARNERS[row.learner_key],
            str(row.seeds),
            _format_metric(row.accuracy),
            _format_metric(row.forgetting),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in body))
        for index in range(len(headers)) 
    ]

    def render(values: tuple[str, ...]) -> str:
        return " | ".join(
            value.rjust(widths[index]) if index == 3 else value.ljust(widths[index])
            for index, value in enumerate(values)
            
        )

    separator = " + ".join("-" * width for width in widths)
    output = [render(headers), separator]
    previous_cache = None

    for result, rendered_row in zip(rows, body):
        if previous_cache is not None and result.cache != previous_cache:
            output.append(separator)

        output.append(render(rendered_row))
        previous_cache = result.cache

    return "\n".join(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_path",
        type=Path,
        metavar="RESULTS_PATH",
        help="raiz do sweep, pasta de benchmark/cache ou sweep_report.json",
    )
    parser.add_argument("--benchmark", help="filtrar pelo nome exato do benchmark")
    parser.add_argument(
        "--cache",
        choices=CACHE_ORDER,
        help="filtrar por first, loss, representative ou hybrid",
    )
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument(
        "--accuracy",
        action="store_true",
        help="selecionar os caches pela acurácia média",
    )
    metric_group.add_argument(
        "--forget",
        action="store_true",
        help="selecionar os caches pelo forgetting médio",
    )
    direction_group = parser.add_mutually_exclusive_group()
    direction_group.add_argument(
        "--high",
        action="store_true",
        help="mostrar o maior valor da métrica escolhida",
    )
    direction_group.add_argument(
        "--low",
        action="store_true",
        help="mostrar o menor valor da métrica escolhida",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = select_extreme_rows(
            filter_rows(
                discover_rows(args.results_path),
                benchmark=args.benchmark,
                cache=args.cache,
            ),
            metric=(
                "accuracy" if args.accuracy else "forgetting" if args.forget else None
            ),
            direction="high" if args.high else "low" if args.low else None,
        )
    except ResultsError as error:
        print(f"erro: {error}", file=sys.stderr)
        return 2
    if any(row.partial for row in rows):
        print(
            "aviso: existem resultados parciais; o IC95% usa apenas as seeds concluídas.",
            file=sys.stderr,
        )
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
