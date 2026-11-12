"""Diagnose the exact-zero FOMC cells in the TRACE pilot run.

Context
-------
In the seed-1234 pilot (``results/trace_full/``) the FOMC cell of the accuracy
matrix is exactly ``0.000`` for five consecutive stages in both arms that
freeze ``A`` (``frozen_a_control``, ``exact``), while ``vanilla`` oscillates.

FOMC targets are the single characters ``A``, ``B`` or ``C`` (496 test
examples, frequencies .260/.246/.494) and ``caculate_accuracy`` is raw ``==``
with no strip and no case folding (``trace_metrics.py``, official quirk 4).

Any prediction distribution supported on ``{A, B, C}`` and independent of the
input scores at least ``min freq = 0.246``. Scoring exactly 0.000 therefore
requires the generations to fall OUTSIDE ``{A, B, C}``. This script reads the
stored generations and reports WHICH way they fall outside, so the team can
decide whether the zero is a formatting artefact (knowledge retained, metric
blind) or a genuine collapse (knowledge lost).

The distinction matters because it moves the primary endpoint: FOMC is one of
the two tasks where ``exact - frozen_a_control`` goes the wrong way.

Usage (CPU only, no GPU, no model load)::

    PYTHONPATH=src:. python scripts/diagnose_fomc_zero.py \
        --run-dir results/trace_full --data-root data/TRACE

Requires the run to have finished, because ``trace_cl`` only writes
``predictions`` into the final manifest (see ``run_trace_cl``). While the run
is in flight the generations live in memory and this script cannot help.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

VALID = ("A", "B", "C")
ARMS = ("vanilla", "frozen_a_control", "exact")


def _load_predictions(run_dir: Path, arm: str) -> dict[str, list[str]] | None:
    """Return the arm's ``{stage}:{task} -> generations`` map, or None.

    ``trace_cl`` writes a standalone ``predictions.json`` next to the manifest
    once the run completes; while the run is in flight the generations live
    only in memory and nothing can be inferred.
    """

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


def _classify(prediction: str) -> str:
    """Bucket one generation by HOW it misses raw string equality."""

    if prediction == "":
        return "empty"
    if prediction in VALID:
        return "exact-letter"
    stripped = prediction.strip()
    if stripped in VALID:
        return "letter-with-whitespace"
    if stripped.upper() in VALID:
        return "letter-wrong-case"
    if len(stripped) > 1 and stripped[0].upper() in VALID:
        return "letter-plus-trailing-text"
    if len(prediction) > 40:
        return "long-text"
    return "other"


def _recovered_accuracy(predictions: list[str], targets: list[str]) -> dict[str, float]:
    """Accuracy under progressively more forgiving matching rules."""

    pairs = list(zip(predictions, targets))
    total = len(pairs)
    if total == 0:
        return {}

    def rate(fn) -> float:
        return sum(1 for p, t in pairs if fn(p, t)) / total

    return {
        "official (raw ==)": rate(lambda p, t: p == t),
        "+ strip": rate(lambda p, t: p.strip() == t),
        "+ strip + upper": rate(lambda p, t: p.strip().upper() == t),
        "+ first letter": rate(
            lambda p, t: bool(p.strip()) and p.strip()[0].upper() == t
        ),
    }


def diagnose(run_dir: Path, data_root: Path) -> int:
    test = json.loads((data_root / "FOMC" / "test.json").read_text())
    targets = [row["answer"] for row in test]
    freqs = Counter(targets)
    floor = min(freqs.values()) / len(targets)

    print(f"FOMC test: n={len(targets)} targets={dict(freqs)}")
    print(f"floor for any in-format predictor: {floor:.4f}")
    print("an exact 0.0000 therefore proves the generations left {A, B, C}\n")

    missing = []
    for arm in ARMS:
        predictions_by_key = _load_predictions(run_dir, arm)
        if predictions_by_key is None:
            missing.append(arm)
            continue

        print(f"=== {arm} " + "=" * 50)
        for key in sorted(predictions_by_key, key=lambda k: int(k.split(":")[0])):
            stage, task = key.split(":", 1)
            if task != "FOMC":
                continue
            predictions = predictions_by_key[key]
            buckets = Counter(_classify(p) for p in predictions)

            print(f"  stage {stage}:")
            for name, count in buckets.most_common():
                print(f"    {name:28s} {count:4d}  ({100 * count / len(predictions):5.1f}%)")
            for rule, value in _recovered_accuracy(predictions, targets).items():
                print(f"    acc {rule:22s} {value:.4f}")
            for sample in predictions[:3]:
                print(f"    sample: {sample[:90]!r}")
            print()

    if missing:
        print(f"NO generations on disk for: {', '.join(missing)}")
        print("trace_cl writes `predictions` only in the final manifest, so this")
        print("script needs those arms to have finished. Nothing was inferred")
        print("for them: absence of data is not a result.")
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
