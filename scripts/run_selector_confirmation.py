"""Run the pre-registered replay-selector inversion confirmation.

Protocol (frozen, commit 7807e62):
    goals/protocol_replay_selector_confirmation.md

R1 dataset   : split_cifar100
R2 selectors : first (structural reference), loss, representative
R4 seeds     : REPLAY_SELECTOR_CONFIRMATORY_SEEDS (20, band 3_000_003+)

The confirmatory family (R8) is 2: the slowheat_derpp - derpp contrast under
`loss` and under `representative`. Everything else the runner computes is
exploratory and is reported as such.
"""

from pathlib import Path

from experiments.replay_selection_sweep import (
    REPLAY_SELECTOR_CONFIRMATORY_SEEDS,
    run_replay_selection_sweep,
)

OUTPUT = Path("results/replay_selector_confirmation")


def main() -> None:
    report = run_replay_selection_sweep(
        seeds=list(REPLAY_SELECTOR_CONFIRMATORY_SEEDS),
        data_dir="data",
        output_dir=OUTPUT,
        device="cpu",
        download=False,
        verbose=True,
        resume=True,
        datasets=("split_cifar100",),
        selectors=("first", "loss", "representative"),
    )
    print("status:", report.get("status"))
    print("learner runs:", report.get("learner_run_count"))


if __name__ == "__main__":
    main()
