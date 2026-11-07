#!/usr/bin/env python3
"""CONFIRMATORY (2026-11-07): SlowHeat vs plasticity-matched uniform on 4 baselines.
Protocol frozen in docs/protocols/confirmatory_uniform_control.md (read it first)."""
from __future__ import annotations

import argparse
from dataclasses import replace

from experiments.confirmatory_split_mnist import FROZEN_CONFIG
from experiments.split_mnist import run_split_mnist_multi_seed

ARMS = (
    "replay", "slowheat_replay_hidden_beta_30_budget_0.25", "uniform_replay_hidden_beta_30_budget_0.25",
    "derpp", "slowheat_derpp_hidden_beta_30_budget_0.25", "uniform_derpp_hidden_beta_30_budget_0.25",
    "replay_calibrated", "slowheat_replay_hidden_beta_30_budget_0.25_calibrated",
    "uniform_replay_hidden_beta_30_budget_0.25_calibrated",
    "er_ace", "slowheat_er_ace_hidden_beta_30_budget_0.25", "uniform_er_ace_hidden_beta_30_budget_0.25",
)
REFERENCES = ("replay", "derpp", "replay_calibrated", "er_ace")
SEEDS = [16000057, 16007976, 16015895, 16023814, 16031733, 16039652, 16047571, 16055490, 16063409, 16071328, 16079247, 16087166, 16095085, 16103004, 16110923, 16118842, 16126761, 16134680, 16142599, 16150518]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--output-dir", default="results/confirm_uniform_control")
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
