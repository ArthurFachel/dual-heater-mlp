"""Diagnose exact-zero accuracy cells in the TRACE pilot run.

Context
-------
In the seed-1234 pilot (``results/trace_full/``) several accuracy cells of the
matrix are exactly ``0.000``. The FOMC case was traced to a formatting
artefact: every generation was a correct letter with a leading space, which
``caculate_accuracy`` (raw ``==``, official quirk 4) scores as zero.

This script generalises that check to every TRACE task whose primary metric is
accuracy, because a metric that silently reports 0.000 for a model answering
correctly invalidates the primary endpoint wherever it fires.

Task-specific scoring, mirroring ``dual_heater.trace_metrics``
--------------------------------------------------------------
* ``C-STANCE``, ``FOMC``: targets are the single characters ``A``/``B``/``C``.
  Compared with raw ``==``.
* ``ScienceQA``: ``resolve_scienceqa`` takes ``item[0]``, so accuracy compares
  the FIRST CHARACTER of generation and target. A leading space makes the
  first character ``" "`` and the example is a guaranteed miss.
* ``NumGLUE-cm``, ``NumGLUE-ds``: targets are numeric strings of varying
  length (``"3"``, ``"10"``, ``"18"``). Raw ``==``, so trailing prose or a
  leading space is a miss even when the number is right.

Recovery rules are reported side by side so the gap between "model was wrong"
and "metric could not see the answer" is explicit and quantified.

NOTHING here changes the preregistered metric. ``protocol_exact_decomposition``
§D7 forbids amending a frozen protocol after seeing results; these numbers
exist so the decision can be made and committed deliberately.

Usage (CPU only, no GPU, no model load)::

    PYTHONPATH=src:. python scripts/diagnose_accuracy_zeros.py \
        --run-dir results/trace_full --data-root data/TRACE
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

ARMS = ("vanilla", "frozen_a_control", "exact")

#: Tasks whose TRACE primary metric is accuracy, with their stage index.
ACCURACY_TASKS = {
    "C-STANCE": 0,
    "FOMC": 1,
    "ScienceQA": 4,
    "NumGLUE-cm": 5,
    "NumGLUE-ds": 6,
}

LETTER_TASKS = {"C-STANCE", "FOMC"}
NUMERIC_TASKS = {"NumGLUE-cm", "NumGLUE-ds"}

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _load_predictions(run_dir: Path, arm: str) -> dict[str, list[str]] | None:
    """Return the arm's ``{stage}:{task} -> generations`` map, or None."""

    direct = run_dir / arm / arm / "predictions.json"
    candidates = [direct] if direct.exists() else sorted(
        run_dir.glob(f"{arm}/**/predictions.json")
    )
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "predictions" in payload:
            payload = payload["predictions"]
        if isinstance(payload, dict) and payload:
            return payload
    return None


def _rules(task: str) -> dict[str, Callable[[str, str], bool]]:
    """Matching rules for one task, from strictest (official) to loosest."""

    if task == "ScienceQA":
        # Official: accuracy compares item[0] of generation and target.
        return {
            "official (first char)": lambda p, t: (p[0] if p else "") == (t[0] if t else ""),
            "+ strip then first char": lambda p, t: (
                p.strip()[0] if p.strip() else ""
            ) == (t[0] if t else ""),
            "+ upper": lambda p, t: (
                p.strip()[:1].upper()
            ) == (t[:1].upper() if t else ""),
        }
    if task in NUMERIC_TASKS:
        return {
            "official (raw ==)": lambda p, t: p == t,
            "+ strip": lambda p, t: p.strip() == t.strip(),
            "+ first number": lambda p, t: _first_number(p) == _first_number(t),
        }
    return {
        "official (raw ==)": lambda p, t: p == t,
        "+ strip": lambda p, t: p.strip() == t,
        "+ strip + upper": lambda p, t: p.strip().upper() == t.upper(),
        "+ first letter": lambda p, t: bool(p.strip()) and p.strip()[0].upper() == t.upper(),
    }


def _first_number(text: str) -> str | None:
    match = _NUMBER.search(text)
    return match.group(0) if match else None


def _classify(prediction: str, task: str, valid: set[str]) -> str:
    """Bucket one generation by HOW it misses the official rule."""

    if prediction == "":
        return "empty"
    if prediction in valid:
        return "exact-match-form"
    stripped = prediction.strip()
    if stripped in valid:
        return "right-but-whitespace"
    if stripped.upper() in {v.upper() for v in valid}:
        return "right-but-case"
    if task in NUMERIC_TASKS and _first_number(prediction) is not None:
        return "number-plus-text"
    if task not in NUMERIC_TASKS and stripped[:1].upper() in {
        v[:1].upper() for v in valid if v
    }:
        return "right-prefix-plus-text"
    if len(prediction) > 60:
        return "long-text"
    return "other"


def diagnose(run_dir: Path, data_root: Path) -> int:
    missing: list[str] = []
    findings: list[tuple[str, str, int, float, float, str]] = []

    for arm in ARMS:
        predictions_by_key = _load_predictions(run_dir, arm)
        if predictions_by_key is None:
            missing.append(arm)
            continue

        print("=" * 78)
        print(f"ARM: {arm}")
        print("=" * 78)

        for task, first_stage in ACCURACY_TASKS.items():
            test = json.loads((data_root / task / "test.json").read_text())
            targets = [row["answer"] for row in test]
            valid = set(targets)
            rules = _rules(task)

            print(f"\n--- {task} (n={len(targets)}, {len(valid)} distinct targets)")
            for stage in range(first_stage, 8):
                key = f"{stage}:{task}"
                predictions = predictions_by_key.get(key)
                if predictions is None:
                    continue

                buckets = Counter(_classify(p, task, valid) for p in predictions)
                scores = {
                    name: sum(1 for p, t in zip(predictions, targets) if rule(p, t))
                    / len(targets)
                    for name, rule in rules.items()
                }
                official = next(iter(scores.values()))
                best = max(scores.values())

                flag = ""
                if official == 0.0 and best > 0.05:
                    flag = "  <<< METRIC BLIND"
                    findings.append((arm, task, stage, official, best, "zero-but-recoverable"))
                elif best - official > 0.10:
                    flag = "  <<< large gap"
                    findings.append((arm, task, stage, official, best, "large-gap"))

                top = ", ".join(
                    f"{name} {100 * count / len(predictions):.0f}%"
                    for name, count in buckets.most_common(3)
                )
                scored = "  ".join(f"{name}={value:.4f}" for name, value in scores.items())
                print(f"  stage {stage}: {scored}{flag}")
                print(f"           forms: {top}")
                if flag:
                    for sample in predictions[:2]:
                        print(f"           sample: {sample[:80]!r}")

        print()

    print("=" * 78)
    print("SUMMARY: cells where the official metric could not see the answer")
    print("=" * 78)
    if not findings:
        print("none")
    else:
        print("| arm | task | stage | official | recoverable | kind |")
        print("|---|---|---|---|---|---|")
        for arm, task, stage, official, best, kind in findings:
            print(f"| {arm} | {task} | {stage} | {official:.4f} | {best:.4f} | {kind} |")

    if missing:
        print(f"\nNO generations on disk for: {', '.join(missing)}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/trace_full"))
    parser.add_argument("--data-root", type=Path, default=Path("data/TRACE"))
    args = parser.parse_args(argv)
    return diagnose(args.run_dir, args.data_root)


if __name__ == "__main__":
    sys.exit(main())
