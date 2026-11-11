#!/usr/bin/env bash
# 3b.2 CALIBRATION, part 2 (goals/calibracao_3b.md): the two lr_control arms at the chosen 1 epoch/task.
# Scale = measured surface_plasticity of the arm each control falsifies, read from results/3b_calibration/ep1/manifest.json:
#   exact 0.5598747924522117 -> GPU 0 ; frozen_a_control (LoRA-FA) 0.6148148148148148 -> GPU 1
#   setsid nohup bash scripts/lr_control_3b.sh > results/3b_calibration/lr_launcher.log 2>&1 < /dev/null &
set -u
cd /mnt/B-SSD/fachel/dh-qwen-f14
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.:src HF_HOME=/mnt/B-SSD/fachel/dual-heater-mlp/.hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
P=/mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python
ROOT=results/3b_calibration
run() {
  local gpu=$1 name=$2 scale=$3
  echo "=== ${name} gpu=$gpu scale=$scale start $(date -Is)"
  CUDA_VISIBLE_DEVICES=$gpu $P experiments/superni_cl.py --output "$ROOT/ep1_${name}" --order 1 --seed 950 \
    --epochs-per-task 1 --batch-size 2 --eval-batch-size 8 --precision fp32 --device cuda:0 \
    --acquisition-margin 0.05 --min-vanilla-forgetting 0.05 \
    --arms lr_control --learning-rate-scale "$scale" > "$ROOT/ep1_${name}.log" 2>&1
  echo "=== ${name} gpu=$gpu exit=$? $(date -Is)"
}
run 0 lrc_exact 0.5598747924522117 &
run 1 lrc_lorafa 0.6148148148148148 &
wait
echo "LR_CONTROL DONE $(date -Is)"
