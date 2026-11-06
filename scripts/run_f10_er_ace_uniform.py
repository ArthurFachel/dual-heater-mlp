#!/usr/bin/env python3
"""EXPLORATORY (F10, 2026-11-06): the F4 plasticity-matched control on ER-ACE.

Same frozen config and the same 20 seeds as F4 (scripts/run_f4_uniform.py). Arms:
  er_ace, slowheat_er_ace_hidden_beta_30_budget_0.25, uniform_er_ace_hidden_beta_30_budget_0.25
The uniform_* arm mirror their SlowHeat arm field by field; every covered entry gets
one shared factor equal to the mean of that arm's own SlowHeat mask (no selection).

Pre-declared reading (fila.md F10, written before any run): for B = ER-ACE,
(SlowHeat+B) - (B+uniform), paired by seed, final average accuracy. Mean > 0 with
paired 95% CI excluding 0 = selection adds to B; CI including 0 = it does not.
Also reported: SlowHeat+B - B and B+uniform - B. CPU only, all arms on the same
device. NOT pre-registered.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from experiments.confirmatory_split_mnist import FROZEN_CONFIG
from experiments.split_mnist import run_split_mnist_multi_seed

ARMS = (
    "er_ace",
    "slowheat_er_ace_hidden_beta_30_budget_0.25",
    "uniform_er_ace_hidden_beta_30_budget_0.25",
)
REFERENCES = ("er_ace",)
SEEDS = [104729, 130363, 155921, 181081, 206369, 231701, 257053, 282377, 307723, 333017,
         358373, 383729, 409063, 434399, 459749, 485071, 510403, 535751, 561097, 586429]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--output-dir", default="results/f10_er_ace_uniform")
    p.add_argument("--data-dir", default="/mnt/B-SSD/fachel/dual-heater-mlp/data")
    p.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch: wiring check only")
    a = p.parse_args()
    config = replace(FROZEN_CONFIG, methods=ARMS, device="cpu")
    if a.smoke:
        config = replace(config, train_per_class=20, validation_per_class=5,
                         test_per_class=5, epochs_per_task=1)
    run_split_mnist_multi_seed(config, seeds=a.seeds, data_dir=a.data_dir,
                               output_dir=a.output_dir, download=False, verbose=True,
                               paired_references=REFERENCES, resume=True)


if __name__ == "__main__":
    main()
