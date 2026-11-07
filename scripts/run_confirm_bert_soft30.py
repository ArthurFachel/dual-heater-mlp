#!/usr/bin/env python3
"""CONFIRMATORY (F12): BERT CLINC150 soft-30 SlowHeat vs uniform_soft on 5 domain pairs.

Protocol frozen in docs/protocols/confirmatory_bert_soft30.md before any run. Same
conditions and config as scripts/run_domain_pairs.py (F5) with a fresh seed band
(17.000.029 + 8.011k, k=0..19). Do not change anything here after the protocol commit.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from experiments.bert_slowheat_diagnostic import DiagnosticCondition, run_diagnostic
from experiments.split_clinc150 import SplitCLINC150Config, load_clinc150_tasks

SEEDS = tuple(17_000_029 + 8_011 * k for k in range(20))
PAIRS = [('banking', 'credit_cards'), ('auto_and_commute', 'travel'), ('kitchen_and_dining', 'home'), ('small_talk', 'meta'), ('utility', 'work')]
CONDITIONS = (
    DiagnosticCondition("vanilla", "vanilla", 3.0, "soft"),  # summary needs it; reference only
    DiagnosticCondition("slowheat_magnitude_soft_b30", "slowheat_ffn_attention", 30.0, "soft",
                        importance_criterion="magnitude"),
    DiagnosticCondition("slowheat_uniform_soft_b30", "slowheat_ffn_attention", 30.0, "uniform_soft",
                        importance_criterion="magnitude"),
)
BUDGET = 0.5


def reorder(tasks, metadata, first: str, second: str):
    by_name = {t.domain: t for t in tasks}
    if first not in by_name or second not in by_name or first == second:
        raise SystemExit(f"bad pair {first} -> {second}; domains: {sorted(by_name)}")
    rest = [t for t in tasks if t.domain not in (first, second)]
    ordered = [by_name[first], by_name[second], *rest]
    meta = dict(metadata)
    meta["domain_order"] = [t.domain for t in ordered]
    meta["f12_pair"] = [first, second]
    meta["f12_protocol"] = "docs/protocols/confirmatory_bert_soft30.md"
    return ordered, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pair", nargs=2, required=True, metavar=("T1", "T2"))
    p.add_argument("--output-dir", default="results/confirm_bert_soft30")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), help="smoke only")
    p.add_argument("--epochs-per-task", type=int, default=None, help="smoke only")
    a = p.parse_args()
    if tuple(a.pair) not in PAIRS:
        raise SystemExit(f"pair not in protocol: {a.pair}")
    base = SplitCLINC150Config(device="cpu", evaluate_test=False)
    if a.epochs_per_task is not None:
        base = replace(base, epochs_per_task=a.epochs_per_task)
    tasks, metadata = load_clinc150_tasks(base, include_test=False)
    tasks, metadata = reorder(tasks, metadata, *a.pair)
    config = replace(base, ffn_plasticity_budget=BUDGET, attention_plasticity_budget=BUDGET,
                     domain_order=tuple(metadata["domain_order"]))
    out = Path(a.output_dir) / f"{a.pair[0]}__{a.pair[1]}" / f"budget_{BUDGET:g}"
    t0 = time.time()
    run_diagnostic(config, tasks, metadata=metadata, seeds=a.seeds, output_dir=out,
                   conditions=CONDITIONS, resume=True)
    print(f"pair {a.pair[0]} -> {a.pair[1]} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
