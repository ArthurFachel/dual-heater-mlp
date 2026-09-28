#!/usr/bin/env bash
# Launch the ten confirmatory seeds of the `exact` decomposition, detached.
#
# Detached on purpose: a session-tracked background run was killed by SIGTERM
# (agent_close) at 34 min earlier today, losing work in progress.
#
# GPU 0 is a free 1080 Ti; the Titan Xp is busy with the selector confirmation.
# Measured cost: 21.6 min/seed, so about 3.6 h for ten.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/exact_decomposition
setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  .venv/bin/python scripts/run_exact_decomposition.py \
  > results/exact_decomposition/run.log 2>&1 < /dev/null &
disown || true
echo "launched, log: results/exact_decomposition/run.log"
