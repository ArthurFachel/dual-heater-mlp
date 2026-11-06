#!/usr/bin/env python3
"""EXPLORATORY (F4, 2026-11-05): the confirmatory Split-MNIST result against a
plasticity-matched control.

Same frozen config as experiments/confirmatory_split_mnist.py (FROZEN_CONFIG) and the
same 20 seeds as results/protocol_post_eval_fix_d5b22ad/confirmation, plus one arm:
uniform_replay_hidden_beta_30_budget_0.25 (Replay + one shared lr factor equal to the
mean of its own SlowHeat mask over the covered surface; no unit selection).

Pre-declared reading (written before any run): primary contrast
(SlowHeat+Replay) - (Replay+uniform), paired by seed, final average accuracy.
Mean > 0 with paired 95% CI excluding 0 = selection pays beyond plasticity.
CI including 0 = the +0.87 pp is explained by reduced plasticity.
Also reported: (SlowHeat+Replay) - Replay (should reproduce ~+0.87 pp) and
(Replay+uniform) - Replay. CPU only. NOT pre-registered.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from experiments.confirmatory_split_mnist import CANDIDATE, FROZEN_CONFIG, REFERENCE
from experiments.split_mnist import run_split_mnist_multi_seed

UNIFORM = "uniform_replay_hidden_beta_30_budget_0.25"
SEEDS = [104729, 130363, 155921, 181081, 206369, 231701, 257053, 282377, 307723, 333017,
         358373, 383729, 409063, 434399, 459749, 485071, 510403, 535751, 561097, 586429]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--output-dir", default="results/f4_split_mnist_uniform")
    p.add_argument("--data-dir", default="/mnt/B-SSD/fachel/dual-heater-mlp/data")
    a = p.parse_args()
    config = replace(FROZEN_CONFIG, methods=(REFERENCE, CANDIDATE, UNIFORM), device="cpu")
    run_split_mnist_multi_seed(config, seeds=a.seeds, data_dir=a.data_dir,
                               output_dir=a.output_dir, download=False, verbose=True,
                               paired_references=(REFERENCE,), resume=True)


if __name__ == "__main__":
    main()
