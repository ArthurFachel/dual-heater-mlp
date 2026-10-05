#!/usr/bin/env bash
# Launch the ten confirmatory seeds of the long-sequence (T=5) criterion
# extension, fully detached.
#
# Do NOT run this before the calibration seed has reported a projection within
# the 4 h budget of Section I of goals/protocol_long_sequence_criterion.md.
#
# CUDA_DEVICE_ORDER=PCI_BUS_ID is required: without it the CUDA index does not
# match nvidia-smi's on this host (verified by UUID).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f results/long_sequence_criterion_calibration/calibration.json ]]; then
  echo "recusado: a calibração do §I não rodou." >&2
  echo "rode scripts/launch_long_sequence_calibration.sh primeiro." >&2
  exit 1
fi

mkdir -p results/long_sequence_criterion
setsid nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. \
  .venv/bin/python scripts/run_long_sequence_criterion.py \
  > results/long_sequence_criterion/run.log 2>&1 < /dev/null &
disown || true
echo "launched, log: results/long_sequence_criterion/run.log"
echo "verifique PPID=1 com: ps -o pid,ppid,cmd -C python | grep long_sequence"
