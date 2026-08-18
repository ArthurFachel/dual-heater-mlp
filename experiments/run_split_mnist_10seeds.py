"""Run the expanded ten-seed Split-MNIST continual-learning benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.split_mnist import SplitMNISTConfig, run_split_mnist_multi_seed


BENCHMARK_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/split_mnist_10seeds"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-per-class", type=int, default=1_000)
    parser.add_argument("--test-per-class", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SplitMNISTConfig(
        hidden_dims=(256, 128),
        batch_size=128,
        epochs_per_task=args.epochs,
        train_per_class=args.train_per_class,
        validation_per_class=200,
        test_per_class=args.test_per_class,
        learning_rate=1e-3,
        weight_decay=1e-4,
        slow_strength=30.0,
        plasticity_budget=0.25,
        optimizer_state_policy="follow_update",
        replay_per_class=20,
        replay_batch_size=64,
        distillation_strength=1.0,
        distillation_temperature=2.0,
        methods=(
            "vanilla",
            "slowheat_beta_10",
            "slowheat_beta_30",
            "slowheat_beta_100",
            "hard_freeze",
            "replay",
            "distillation",
            "slowheat_replay",
            "slowheat_distillation",
        ),
        device=args.device,
    )
    aggregate = run_split_mnist_multi_seed(
        config,
        seeds=list(BENCHMARK_SEEDS),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        download=True,
        verbose=True,
    )
    print(f"Resultados agregados: {args.output_dir / 'aggregate.csv'}")
    best = max(
        aggregate["methods"],
        key=lambda method: aggregate["methods"][method]["final_average_accuracy"][
            "mean"
        ],
    )
    print(f"Maior acurácia final média: {best}")


if __name__ == "__main__":
    main()
