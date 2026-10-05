#!/usr/bin/env bash
# Launch the long-sequence (T=5) criterion calibration seed, fully detached.
#
# Detached on purpose: a session-tracked background run was killed by SIGTERM
# (agent_close) at 34 min on 2026-09-28, losing work in progress.
#
# CUDA_DEVICE_ORDER=PCI_BUS_ID is required: without it the CUDA index does not
# match nvidia-smi's on this host (verified by UUID).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/long_sequence_criterion_calibration
setsid nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. \
  .venv/bin/python scripts/run_long_sequence_calibration.py \
  > results/long_sequence_criterion_calibration/run.log 2>&1 < /dev/null &
disown || true
echo "launched, log: results/long_sequence_criterion_calibration/run.log"
echo "verifique PPID=1 com: ps -o pid,ppid,cmd -C python | grep long_sequence"
