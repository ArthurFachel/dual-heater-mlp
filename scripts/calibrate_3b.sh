#!/usr/bin/env bash
# 3b.2 CALIBRATION (goals/calibracao_3b.md, criterion declared before the run). One process per GPU.
#   setsid nohup bash scripts/calibrate_3b.sh > results/3b_calibration/launcher.log 2>&1 < /dev/null &
set -u
cd /mnt/B-SSD/fachel/dh-qwen-f14
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.:src HF_HOME=/mnt/B-SSD/fachel/dual-heater-mlp/.hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
P=/mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python
ROOT=results/3b_calibration
mkdir -p $ROOT
run() {
  local gpu=$1 ep=$2
  echo "=== ep$ep gpu=$gpu start $(date -Is)"
  CUDA_VISIBLE_DEVICES=$gpu $P experiments/superni_cl.py --output $ROOT/ep$ep --order 1 --seed 950 \
    --epochs-per-task $ep --batch-size 2 --eval-batch-size 8 --precision fp32 --device cuda:0 \
    --acquisition-margin 0.05 --min-vanilla-forgetting 0.05 \
    --arms vanilla exact frozen_a_control > $ROOT/ep$ep.log 2>&1
  echo "=== ep$ep gpu=$gpu exit=$? $(date -Is)"
}
run 0 1 &
run 1 3 &
run 2 5 &
wait
echo "CALIBRATION DONE $(date -Is)"
