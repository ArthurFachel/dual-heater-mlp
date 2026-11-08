#!/usr/bin/env bash
# F14 CONFIRMATORY (docs/protocols/confirmatory_qwen_hard.md, frozen before run). One process per seed.
#   GPUS="0 1 2" nohup setsid bash scripts/f14_confirm_qwen_hard.sh > results/f14_confirm_qwen_hard.log 2>&1 < /dev/null &
set -u
cd /mnt/B-SSD/fachel/dh-qwen-f14
GPUS=${GPUS:?set GPUS}
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.:src HF_HOME=/mnt/B-SSD/fachel/dual-heater-mlp/.hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
P=/mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python
ROOT=${ROOT:-results/f14_confirm_qwen_hard}
DOMAINS="banking credit_cards kitchen_and_dining home auto_and_commute travel utility work small_talk meta"
mkdir -p $ROOT
lane() {
  local gpu=$1
  export CUDA_VISIBLE_DEVICES=$gpu
  for s in ${SEEDS:-200 201 202 203 204 205 206 207 208 209}; do
    local dir=$ROOT/seed$s
    [ -f "$dir/manifest.json" ] && continue
    mkdir -p "$dir"; mkdir "$dir/.claim" 2>/dev/null || continue
    echo "=== seed$s gpu=$gpu start $(date -Is)"
    $P experiments/qwen_iso_plasticity.py --domains $DOMAINS --steps-per-task ${STEPS:-120} \
      --learning-rate 1e-5 --batch-size 2 --max-length 64 --target-plasticity 0.75 \
      --arms hard_b0.75 reduced_lr vanilla --seed $s --device cuda --output "$dir/manifest.json" > "$dir/stdout.log" 2>&1
    echo "=== seed$s gpu=$gpu exit=$? $(date -Is)"
  done
  echo "LANE gpu=$gpu FREE $(date -Is)"
}
for g in $GPUS; do lane "$g" & done
wait
echo "F14 DONE $(date -Is)"
