#!/usr/bin/env bash
# Launch the criterion-ablation calibration seed fully detached.
#
# Detached on purpose: a session-tracked background process was killed by
# SIGTERM (agent_close) at 34 min earlier today, losing a run in progress.
#
# GPU 0 is a 1080 Ti; the Titan Xp is busy with the selector confirmation.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/criterion_ablation_calibration
setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  .venv/bin/python scripts/run_criterion_calibration.py \
  > results/criterion_ablation_calibration/run.log 2>&1 < /dev/null &
disown || true
echo "launched, log: results/criterion_ablation_calibration/run.log"
