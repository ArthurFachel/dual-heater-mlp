#!/usr/bin/env bash
# Launch the ten confirmatory seeds of the criterion ablation, fully detached.
#
# Detached on purpose: a session-tracked background run was killed by SIGTERM
# (agent_close) at 34 min earlier today, losing work in progress.
#
# GPU 0 is a 1080 Ti; the Titan Xp is busy with the selector confirmation.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/criterion_ablation
setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  .venv/bin/python scripts/run_criterion_ablation.py \
  > results/criterion_ablation/run.log 2>&1 < /dev/null &
disown || true
echo "launched, log: results/criterion_ablation/run.log"
