#!/usr/bin/env bash
# Confirmation run: 10 declared seeds x 10 CLINC150 domains, 120 steps per task.
#
# Seeds {10..19} and the domain order were declared in the frozen protocol
# before any accuracy was seen (section A3 / section D). Nothing here chooses a
# hyperparameter.
#
# Three GPUs, one seed resident per GPU at a time, queued: seeds 10,13,16,19 on
# GPU 0; 11,14,17 on GPU 1; 12,15,18 on GPU 2. A finished seed writes its
# manifest before the next one starts, so an interrupted run loses at most the
# seed in flight and can be resumed by re-running this script (completed seeds
# are skipped).
set -u
cd "$(dirname "$0")"

DOMAINS="banking credit_cards kitchen_and_dining home auto_and_commute travel utility work small_talk meta"
STEPS=120
mkdir -p results/run_logs/confirm120

run_queue() {
  local gpu=$1; shift
  for seed in "$@"; do
    local out="results/qwen_iso_plasticity/confirm120_seed${seed}/manifest.json"
    if [ -f "$out" ]; then
      echo "[gpu$gpu] seed$seed already done, skipping"
      continue
    fi
    echo "[gpu$gpu] seed$seed starting $(date +%H:%M:%S)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu \
    HF_HOME=.hf-cache PYTHONPATH=. \
    .venv/bin/python experiments/qwen_iso_plasticity.py \
      --seed "$seed" \
      --steps-per-task $STEPS \
      --batch-size 2 \
      --device cuda \
      --domains $DOMAINS \
      --output "$out" \
      > "results/run_logs/confirm120/seed${seed}.log" 2>&1
    echo "[gpu$gpu] seed$seed exit=$? $(date +%H:%M:%S)"
  done
}

run_queue 0 10 13 16 19 &
run_queue 1 11 14 17 &
run_queue 2 12 15 18 &
wait

echo "=== CONFIRMATION COMPLETE $(date) ==="
for s in 10 11 12 13 14 15 16 17 18 19; do
  f="results/qwen_iso_plasticity/confirm120_seed${s}/manifest.json"
  if [ -f "$f" ]; then echo "seed$s OK"; else echo "seed$s MISSING"; fi
done
