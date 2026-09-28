#!/usr/bin/env bash
# Calibration seed for the `exact` decomposition (protocol D8).
#
# Measures real cost before committing to the ten pre-registered seeds.
# Estimating by analogy already failed twice in this project (4.7x error), so
# the protocol requires a measured number.
#
# Hyperparameters are copied verbatim from the original confirmation manifest
# (results/qwen_lora_confirmation/seed_700001/manifest.json) so the new arm is
# measured under the same conditions as the result it decomposes.
#
# Three arms only: vanilla, frozen_a_control, exact. The other mechanisms are
# not part of this decomposition.
#
# GPU 0 is a free 1080 Ti; the Titan Xp is busy with the selector confirmation.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/exact_decomposition_calibration
setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  .venv/bin/python experiments/qwen_lora_slowheat.py \
    --output results/exact_decomposition_calibration \
    --arms vanilla frozen_a_control exact \
    --seed 5000003 \
    --tasks 10 \
    --rank 16 \
    --alpha 16.0 \
    --train-per-class 50 \
    --eval-per-class 20 \
    --epochs-per-task 3 \
    --batch-size 8 \
    --max-length 48 \
    --slow-strength 3.0 \
    --plasticity-budget 0.5 \
    --target-plasticity 0.85 \
    --device cuda:0 \
  > results/exact_decomposition_calibration/run.log 2>&1 < /dev/null &
disown || true
echo "launched, log: results/exact_decomposition_calibration/run.log"
