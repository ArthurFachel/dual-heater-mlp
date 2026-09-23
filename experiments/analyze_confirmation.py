"""Aggregate the 10-seed confirmation run into the tables used for reporting.

Reads the manifests written by ``qwen_iso_plasticity.py`` and emits:

  1. an integrity check (one protocol hash, one task fingerprint, declared
     configuration) -- if this fails, nothing below is trustworthy;
  2. per-arm endpoints averaged over seeds;
  3. paired per-seed differences with an exact sign-flip test and Holm
     correction over the whole family of comparisons.

Pairing matters: the between-seed spread (~0.04 FAA) is larger than most of
the effects being measured, so an unpaired comparison would drown the signal
in seed variance. Every arm sees the same seed, the same task order, and the
same data, so the per-seed difference cancels that variance.

The sign-flip test is exact, not asymptotic: with 10 seeds there are only
2^10 = 1024 sign assignments, so the null distribution is enumerated in full
rather than approximated. The null is that protecting units has no effect,
under which the sign of each per-seed difference is arbitrary.

Usage:
    PYTHONPATH=. python experiments/analyze_confirmation.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

ENDPOINTS = [
    ("final_average_accuracy", "FAA"),
    ("retention_first_task", "retencao t0"),
    ("mean_forgetting", "forgetting"),
    ("acquisition_last_task", "aquisicao t_ult"),
]


def load_manifests(root: Path, seeds: list[int], prefix: str) -> dict[int, dict]:
    out = {}
    for seed in seeds:
        path = root / f"{prefix}{seed}" / "manifest.json"
        if not path.exists():
            raise SystemExit(f"manifesto ausente: {path}")
        out[seed] = json.loads(path.read_text())
    return out


def check_integrity(manifests: dict[int, dict]) -> dict[str, Any]:
    """Refuse to aggregate runs that did not come from one declared protocol."""

    hashes = {m["protocol_hash"] for m in manifests.values()}
    prints = {m["task_fingerprint"] for m in manifests.values()}
    steps = {m["protocol"].get("steps_per_task") for m in manifests.values()}
    ntasks = {len(m["protocol"].get("tasks", [])) for m in manifests.values()}
    precision = {m["protocol"].get("precision") for m in manifests.values()}
    scopes = {m.get("capacity_scope") for m in manifests.values()}

    report = {
        "seeds": sorted(manifests),
        "protocol_hash_unico": len(hashes) == 1,
        "protocol_hash": next(iter(hashes)) if len(hashes) == 1 else sorted(hashes),
        "task_fingerprint_unico": len(prints) == 1,
        "steps_per_task": sorted(steps),
        "tarefas": sorted(ntasks),
        "precision": sorted(x for x in precision if x is not None),
        "capacity_scope": sorted(x for x in scopes if x is not None),
    }
    report["ok"] = (
        report["protocol_hash_unico"]
        and report["task_fingerprint_unico"]
        and len(steps) == 1
        and len(ntasks) == 1
        and precision == {"fp32"}
    )
    return report


def arm_endpoint(manifest: dict, family: str, arm: str, key: str) -> float | None:
    for record in manifest["families"][family]["arms"]:
        if record["arm"]["name"] == arm and not record.get("discarded"):
            return record["endpoints"][key]
    return None


def arm_names(manifest: dict, family: str) -> list[str]:
    return [r["arm"]["name"] for r in manifest["families"][family]["arms"]]


def sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided sign-flip p-value. Enumerates all 2^n assignments."""

    n = len(diffs)
    if n == 0:
        return float("nan")
    observed = abs(st.mean(diffs))
    hits = 0
    for mask in range(1 << n):
        flipped = [d if mask >> i & 1 else -d for i, d in enumerate(diffs)]
        if abs(st.mean(flipped)) >= observed - 1e-12:
            hits += 1
    return hits / (1 << n)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved.

    Controls the family-wise error rate. With 24 comparisons an uncorrected
    0.05 threshold would be expected to produce false positives by chance
    alone, so the corrected column is the one to read.
    """

    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/qwen_iso_plasticity")
    parser.add_argument("--prefix", default="confirm120_seed")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(10, 20))
    )
    args = parser.parse_args()

    manifests = load_manifests(Path(args.root), args.seeds, args.prefix)
    seeds = sorted(manifests)

    print("=" * 78)
    print("INTEGRIDADE")
    print("=" * 78)
    report = check_integrity(manifests)
    for key, value in report.items():
        print(f"  {key}: {value}")
    if not report["ok"]:
        raise SystemExit("\nINTEGRIDADE FALHOU -- nao agregue estes manifestos.")

    # ---- per-arm endpoints ------------------------------------------------
    for family in ("0.75", "0.5"):
        print()
        print("=" * 78)
        print(f"E* = {family}   ({len(seeds)} seeds, endpoints medios)")
        print("=" * 78)
        header = "  %-18s" % "braco" + "".join("%-20s" % n for _, n in ENDPOINTS)
        print(header)
        for name in arm_names(manifests[seeds[0]], family):
            cells = []
            missing = False
            for key, _ in ENDPOINTS:
                vals = [arm_endpoint(manifests[s], family, name, key) for s in seeds]
                vals = [v for v in vals if v is not None]
                if not vals:
                    missing = True
                    break
                cells.append("%.4f+-%.4f" % (st.mean(vals), st.stdev(vals)))
            if missing:
                print("  %-18s (descartado em todas as seeds)" % name)
                continue
            print("  %-18s" % name + "".join("%-20s" % c for c in cells))

    # ---- paired comparisons ----------------------------------------------
    comparisons = []
    for family in ("0.75", "0.5"):
        iso = "iso_b0.25"
        hard = "hard_b0.75" if family == "0.75" else "hard_b0.5"
        pairs = [
            ("iso - permutado", iso, "permuted_b0.25"),
            ("hard - iso", hard, iso),
            ("hard - vanilla", hard, "vanilla"),
            ("iso - vanilla", iso, "vanilla"),
        ]
        for label, a, b in pairs:
            for key, kname in ENDPOINTS[:3]:
                da = [arm_endpoint(manifests[s], family, a, key) for s in seeds]
                db = [arm_endpoint(manifests[s], family, b, key) for s in seeds]
                if any(x is None for x in da) or any(x is None for x in db):
                    continue
                diffs = [x - y for x, y in zip(da, db)]
                comparisons.append(
                    {
                        "family": family,
                        "label": label,
                        "endpoint": kname,
                        "mean": st.mean(diffs),
                        "sd": st.stdev(diffs),
                        "wins": sum(1 for d in diffs if d > 0),
                        "n": len(diffs),
                        "p": sign_flip_p(diffs),
                    }
                )

    adjusted = holm([c["p"] for c in comparisons])
    for comparison, value in zip(comparisons, adjusted):
        comparison["p_holm"] = value

    print()
    print("=" * 78)
    print(f"DIFERENCAS PAREADAS POR SEED (n={len(seeds)}, sign-flip exato)")
    print(f"Holm sobre as {len(comparisons)} comparacoes. * = p_holm < 0.05")
    print("=" * 78)
    print(
        "  %-6s %-18s %-12s %-18s %-8s %-9s %-9s"
        % ("E*", "comparacao", "endpoint", "diferenca", "seeds", "p", "p_holm")
    )
    for c in comparisons:
        print(
            "  %-6s %-18s %-12s %+.4f+-%.4f  %2d/%-5d %-9.4f %-9.4f%s"
            % (
                c["family"],
                c["label"],
                c["endpoint"],
                c["mean"],
                c["sd"],
                c["wins"],
                c["n"],
                c["p"],
                c["p_holm"],
                " *" if c["p_holm"] < 0.05 else "",
            )
        )

    survivors = [c for c in comparisons if c["p_holm"] < 0.05]
    print(f"\n  sobrevivem a Holm: {len(survivors)}/{len(comparisons)}")
    for c in survivors:
        print(
            "    E*=%s %s %s (%+.4f, %d/%d)"
            % (c["family"], c["label"], c["endpoint"], c["mean"], c["wins"], c["n"])
        )


if __name__ == "__main__":
    main()
