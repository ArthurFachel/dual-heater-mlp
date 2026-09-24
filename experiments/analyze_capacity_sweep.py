"""Aggregate the width sweep and run its declared primary test.

The sweep's primary question is not "is hard better than soft at width w?"
but "does the Hard-minus-Soft difference grow with width?". That is a slope,
not a contrast, so this script fits one slope per seed and tests the ten
slopes -- a single primary test on paired data, no multiplicity correction
needed for it.

Why per-seed slopes rather than a regression on per-width means: seeds differ
substantially in absolute accuracy, and fitting one line through the pooled
means would let that between-seed spread masquerade as trend noise. Each seed
supplies its own line, so seed-level variation cancels exactly the way the
paired difference already cancels it within a width.

The sign test is exact. With 10 seeds there are 2^10 = 1024 sign assignments,
so the null distribution is enumerated rather than approximated.

Usage:
    PYTHONPATH=. python experiments/analyze_capacity_sweep.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from itertools import product
from pathlib import Path
from typing import Any

PRIMARY = "final_average_accuracy"
PRIMARY_CONTRAST = "Hard vs Soft"


def _slope(xs: list[float], ys: list[float]) -> float:
    """OLS slope of ys on xs. Raises when xs has no spread."""
    mean_x, mean_y = st.fmean(xs), st.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0.0:
        raise ValueError("as larguras precisam variar para estimar uma inclinação")
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator


def _exact_sign_flip_p(values: list[float], *, alternative: str = "greater") -> float:
    """Exact sign-flip test: enumerate all 2^n sign assignments under the null.

    The null is that the width has no effect, under which flipping the sign of
    any seed's slope is equally likely. Capped at 2^20 enumerations.
    """
    n = len(values)
    if n == 0:
        raise ValueError("valores não podem ser vazios")
    if n > 20:
        raise ValueError(f"enumeração exata inviável para n={n}")
    observed = st.fmean(values)
    total = 0
    extreme = 0
    for signs in product((1.0, -1.0), repeat=n):
        candidate = st.fmean([s * v for s, v in zip(signs, values, strict=True)])
        total += 1
        if alternative == "greater":
            extreme += candidate >= observed
        elif alternative == "less":
            extreme += candidate <= observed
        else:
            extreme += abs(candidate) >= abs(observed)
    return extreme / total


def _one_sided_t(values: list[float]) -> dict[str, float]:
    """One-sided paired t against H0: mean = 0, H1: mean > 0."""
    n = len(values)
    if n < 2:
        raise ValueError("o teste t requer ao menos 2 observações")
    mean = st.fmean(values)
    sd = st.stdev(values)
    if sd == 0.0:
        return {
            "mean": mean,
            "std": 0.0,
            "statistic": math.inf if mean > 0 else -math.inf,
            "degrees_of_freedom": n - 1,
            "one_sided_p": 0.0 if mean > 0 else 1.0,
        }
    statistic = mean / (sd / math.sqrt(n))
    # Student-t survival via the regularized incomplete beta, from math only.
    df = n - 1
    x = df / (df + statistic * statistic)
    two_sided = _betainc(df / 2.0, 0.5, x)
    one_sided = two_sided / 2.0 if statistic > 0 else 1.0 - two_sided / 2.0
    return {
        "mean": mean,
        "std": sd,
        "statistic": statistic,
        "degrees_of_freedom": df,
        "one_sided_p": one_sided,
    }


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) by continued fraction."""
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
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def _beta_cf(a: float, b: float, x: float, iterations: int = 300) -> float:
    tiny = 1e-30
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    result = d
    for m in range(1, iterations + 1):
        numerator = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        d = 1.0 + numerator * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + numerator / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        result *= d * c
        numerator = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + numerator * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + numerator / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return result


def _per_seed_differences(report_dir: Path, contrast: str) -> dict[int, float]:
    """Recompute per-seed hard-minus-soft from the seed files, not the summary.

    The aggregate carries only the mean; the slope needs each seed's value, so
    this reads seed_*/results.json directly.
    """
    manifest = json.loads((report_dir / "multi_seed_config.json").read_text())
    report = json.loads((report_dir / "pair_report.json").read_text())
    pair = next(p for p in report["pairs"] if p["label"] == contrast)
    reference, candidate = pair["reference"], pair["candidate"]
    out = {}
    for seed in manifest["seeds"]:
        results = json.loads((report_dir / f"seed_{seed}" / "results.json").read_text())
        out[seed] = (
            results[candidate][PRIMARY] - results[reference][PRIMARY]
        ) * 100.0
    return out


def collect(root: Path, dataset: str) -> dict[str, Any]:
    """Gather every width present for one dataset."""
    rungs = {}
    for directory in sorted(root.glob(f"{dataset}_w*")):
        if not (directory / "pair_report.json").is_file():
            continue
        protocol = json.loads((directory / "capacity_sweep_protocol.json").read_text())
        width = tuple(protocol["width"])
        rungs[width] = {
            "directory": directory,
            "protocol": protocol,
            "differences": _per_seed_differences(directory, PRIMARY_CONTRAST),
        }
    if not rungs:
        raise SystemExit(f"nenhum degrau encontrado para {dataset} em {root}")
    return rungs


def check_integrity(rungs: dict[tuple[int, ...], dict]) -> dict[str, Any]:
    """Refuse to fit a slope across rungs that did not hold everything else fixed.

    The whole point of the sweep is that only the width moved; if any other
    config field differs between rungs, the slope is not a capacity effect and
    must not be reported as one.
    """
    ignored = {"hidden_dims", "seed", "device"}
    baseline_width = min(rungs)
    baseline = rungs[baseline_width]["protocol"]["config"]
    drift = {}
    for width, rung in rungs.items():
        config = rung["protocol"]["config"]
        for key, value in config.items():
            if key in ignored:
                continue
            if baseline.get(key) != value:
                drift.setdefault(key, []).append(
                    {"width": list(width), "value": value}
                )
    seed_sets = {tuple(sorted(rung["differences"])) for rung in rungs.values()}
    if len(seed_sets) != 1:
        raise SystemExit("as seeds divergem entre degraus; o pareamento é inválido")
    if drift:
        raise SystemExit(
            "configuração divergente entre degraus (só a largura pode variar): "
            + json.dumps(drift, indent=2, sort_keys=True)
        )
    return {
        "widths": [list(w) for w in sorted(rungs)],
        "seeds": sorted(next(iter(seed_sets))),
        "only_width_varies": True,
    }


def fit_slopes(rungs: dict[tuple[int, ...], dict]) -> dict[str, Any]:
    """One slope per seed of (hard - soft) against log2(first hidden width)."""
    widths = sorted(rungs)
    if len(widths) < 3:
        raise SystemExit(
            f"a inclinação exige ao menos 3 degraus; encontrados {len(widths)}"
        )
    xs = [math.log2(width[0]) for width in widths]
    seeds = sorted(rungs[widths[0]]["differences"])
    slopes = {}
    for seed in seeds:
        ys = [rungs[width]["differences"][seed] for width in widths]
        slopes[seed] = _slope(xs, ys)
    values = [slopes[seed] for seed in seeds]
    positive = sum(1 for v in values if v > 0)
    return {
        "per_seed_slope": {str(k): v for k, v in slopes.items()},
        "n_seeds": len(values),
        "positive_slopes": positive,
        "negative_slopes": len(values) - positive,
        "student_t_one_sided": _one_sided_t(values),
        "exact_sign_flip_p_greater": _exact_sign_flip_p(values, alternative="greater"),
        "units": "accuracy points per doubling of the first hidden width",
    }


def replication_check(rungs: dict[tuple[int, ...], dict], dataset: str, root: Path) -> dict[str, Any]:
    """Compare the shared rung against the existing hard-versus-soft aggregate."""
    shared = (1024, 512)
    if shared not in rungs:
        return {"available": False, "reason": "o degrau compartilhado não foi executado"}
    existing = root.parent / "hard_vs_soft" / f"{dataset}_mlp" / "pair_report.json"
    if not existing.is_file():
        return {"available": False, "reason": f"ausente: {existing}"}
    report = json.loads(existing.read_text())
    pair = next(p for p in report["pairs"] if p["label"] == PRIMARY_CONTRAST)
    previous = pair["metrics"][PRIMARY]["mean_difference"] * 100.0
    current = st.fmean(list(rungs[shared]["differences"].values()))
    return {
        "available": True,
        "hard_vs_soft_suite": previous,
        "this_sweep": current,
        "absolute_difference": abs(current - previous),
        "artifact": str(existing),
    }


def render(dataset: str, rungs: dict, integrity: dict, slopes: dict, replication: dict) -> str:
    lines = [f"\n{'=' * 72}", f"  {dataset}", f"{'=' * 72}"]
    lines.append("\nIntegridade: só a largura varia entre degraus. OK")
    lines.append(f"Seeds: {len(integrity['seeds'])} | degraus: {len(rungs)}")
    lines.append("\nDiferença Hard - Soft por largura (pontos de acurácia):")
    lines.append(f"  {'largura':>14}  {'média':>8}  {'dp':>7}  sinais")
    for width in sorted(rungs):
        values = list(rungs[width]["differences"].values())
        positive = sum(1 for v in values if v > 0)
        label = "x".join(str(w) for w in width)
        deviation = st.stdev(values) if len(values) > 1 else 0.0
        lines.append(
            f"  {label:>14}  {st.fmean(values):+8.3f}  {deviation:7.3f}  "
            f"{positive}+/{len(values) - positive}-"
        )
    t = slopes["student_t_one_sided"]
    lines.append("\nTESTE PRIMÁRIO: inclinação por seed vs log2(largura)")
    lines.append(f"  inclinação média : {t['mean']:+.4f} pontos por dobra de largura")
    lines.append(f"  desvio-padrão    : {t['std']:.4f}")
    lines.append(f"  t({t['degrees_of_freedom']})          : {t['statistic']:+.3f}")
    lines.append(f"  p unilateral (t) : {t['one_sided_p']:.4g}")
    lines.append(f"  p sinal exato    : {slopes['exact_sign_flip_p_greater']:.4g}")
    lines.append(
        f"  inclinações      : {slopes['positive_slopes']}+/"
        f"{slopes['negative_slopes']}-"
    )
    verdict = (
        "a leitura de capacidade SOBREVIVE"
        if t["one_sided_p"] < 0.05
        else "a leitura de capacidade NAO tem suporte"
    )
    lines.append(f"  veredito         : {verdict}")
    if replication["available"]:
        lines.append("\nControle de replicação no degrau 1024x512:")
        lines.append(f"  suíte hard_vs_soft : {replication['hard_vs_soft_suite']:+.3f}")
        lines.append(f"  este sweep         : {replication['this_sweep']:+.3f}")
        lines.append(f"  diferença absoluta : {replication['absolute_difference']:.3f}")
    else:
        lines.append(f"\nControle de replicação indisponível: {replication['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/capacity_sweep"))
    parser.add_argument(
        "--datasets", nargs="+", default=["split_cifar10", "split_cifar100"]
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    payload = {}
    for dataset in args.datasets:
        rungs = collect(args.root, dataset)
        integrity = check_integrity(rungs)
        slopes = fit_slopes(rungs)
        replication = replication_check(rungs, dataset, args.root)
        print(render(dataset, rungs, integrity, slopes, replication))
        payload[dataset] = {
            "integrity": integrity,
            "per_width": {
                "x".join(str(w) for w in width): {
                    "mean_difference": st.fmean(list(rung["differences"].values())),
                    "per_seed": {str(k): v for k, v in rung["differences"].items()},
                }
                for width, rung in sorted(rungs.items())
            },
            "primary_test": slopes,
            "replication_control": replication,
        }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nJSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
