#!/usr/bin/env python3
"""CONFIRMATORY T3c: SlowHeat vs same-shape random units with a FIXED random order (MLP).

Protocol frozen in docs/protocols/confirmatory_random_fixed_mlp.md (read it first).
--setting split: Split-MNIST, FROZEN_CONFIG, 4 baselines x {B, SlowHeat+B, B+uniform,
B+random_fixed}, 16 arms in one process per seed.
--setting permuted: Permuted-MNIST, hard_vs_soft suite config, vanilla / hard_freeze /
uniform_hard_freeze / random_fixed_hard_freeze.
CPU only; every arm of a seed shares process and device.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from experiments.split_mnist import run_split_mnist_multi_seed

BASES = {
    "replay": "replay_hidden_beta_30_budget_0.25",
    "derpp": "derpp_hidden_beta_30_budget_0.25",
    "replay_calibrated": "replay_hidden_beta_30_budget_0.25_calibrated",
    "er_ace": "er_ace_hidden_beta_30_budget_0.25",
}
SPLIT_ARMS = tuple(
    arm
    for base, tail in BASES.items()
    for arm in (base, f"slowheat_{tail}", f"uniform_{tail}", f"random_fixed_{tail}")
)
PERMUTED_ARMS = ("vanilla", "hard_freeze", "uniform_hard_freeze", "random_fixed_hard_freeze")
SPLIT_SEEDS = [19000033 + 8209 * k for k in range(20)]
PERMUTED_SEEDS = [19500017 + 8317 * k for k in range(10)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=("split", "permuted"), required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--data-dir", default="/mnt/B-SSD/fachel/dual-heater-mlp/data")
    p.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch: wiring check only")
    a = p.parse_args()
    if a.setting == "split":
        from experiments.confirmatory_split_mnist import FROZEN_CONFIG

        config = replace(FROZEN_CONFIG, methods=SPLIT_ARMS, device="cpu")
        seeds, refs, loader = SPLIT_SEEDS, tuple(BASES), None
    else:
        from experiments.hard_vs_soft import suite_config
        from experiments.visual_generalization import load_permuted_mnist

        config = replace(suite_config("permuted_mnist", "mlp", "cpu"), methods=PERMUTED_ARMS)
        seeds, refs, loader = PERMUTED_SEEDS, ("vanilla",), load_permuted_mnist
    if a.smoke:
        config = replace(config, train_per_class=20, validation_per_class=5,
                         test_per_class=5, epochs_per_task=1)
    out = a.output_dir or f"results/t3c_random_fixed_mlp/{a.setting}"
    kwargs = dict(seeds=a.seeds or seeds, data_dir=a.data_dir, output_dir=out,
                  download=False, verbose=True, paired_references=refs, resume=True)
    if loader is not None:
        kwargs["task_loader"] = loader
    run_split_mnist_multi_seed(config, **kwargs)


if __name__ == "__main__":
    main()
