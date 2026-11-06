#!/usr/bin/env python3
"""EXPLORATORY (F8, 2026-11-06): Permuted-MNIST hard protection against a
plasticity-matched uniform control.

The hard_vs_soft suite reported hard_freeze - vanilla = +16.99 pp FAA (10/10) on
permuted_mnist_mlp, with no matched-plasticity arm (audit item 4). Same config as
that suite (experiments.hard_vs_soft.suite_config) and the same 10 seeds. Arms:
vanilla, hard_freeze, uniform_hard_freeze (hard_freeze with every covered entry,
output layer included, set to the plastic fraction of its own binary mask).

Pre-declared reading (fila.md F8, written before any run): hard - uniform, paired by
seed, final average accuracy. Mean > 0 with paired 95% CI excluding 0 and >= 8/10 in
favour = selection pays; CI including 0 = the gain is plasticity. Also reported:
uniform - vanilla. CPU only. NOT pre-registered.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from experiments.hard_vs_soft import suite_config
from experiments.split_mnist import run_split_mnist_multi_seed
from experiments.visual_generalization import load_permuted_mnist

ARMS = ("vanilla", "hard_freeze", "uniform_hard_freeze")
SEEDS = [1483544251, 1059155855, 1258205799, 1589997122, 1055802198,
         1809614999, 2119980623, 874968385, 116993065, 140912400]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--output-dir", default="results/f8_permuted_uniform_hard")
    p.add_argument("--data-dir", default="/mnt/B-SSD/fachel/dual-heater-mlp/data")
    p.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch: wiring check only")
    a = p.parse_args()
    config = replace(suite_config("permuted_mnist", "mlp", "cpu"), methods=ARMS)
    if a.smoke:
        config = replace(config, train_per_class=20, validation_per_class=5,
                         test_per_class=5, epochs_per_task=1)
    run_split_mnist_multi_seed(config, seeds=a.seeds, data_dir=a.data_dir,
                               output_dir=a.output_dir, download=False, verbose=True,
                               paired_references=("vanilla",), task_loader=load_permuted_mnist,
                               resume=True)


if __name__ == "__main__":
    main()
