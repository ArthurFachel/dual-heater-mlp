#!/usr/bin/env bash
# Phase 3b confirmatory (goals/protocol_3b_confirmatory.md): one queue worker per GPU.
#   setsid nohup bash scripts/confirm_3b.sh > results/3b_confirm/launcher.log 2>&1 < /dev/null &
set -u
cd /mnt/B-SSD/fachel/dh-qwen-f14
ROOT=results/3b_confirm
mkdir -p $ROOT
echo "=== protocol commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start $(date -Is)"
for g in 0 1 2; do
  /mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python scripts/confirm_3b_queue.py --gpu $g > $ROOT/worker_gpu$g.log 2>&1 &
done
wait
echo "CONFIRM DONE $(date -Is)"
