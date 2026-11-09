#!/usr/bin/env bash
# PILOT, EXPLORATORY (goals/piloto_tempo_da_mascara.md). Mask-update limit and head warmup. One process per seed.
#   GPUS="0 1 2" nohup setsid bash scripts/pilot_mask_timing.sh > results/pilot_mask_timing.log 2>&1 < /dev/null &
set -u
cd /mnt/B-SSD/fachel/dh-qwen-f14
GPUS=${GPUS:?set GPUS}
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.:src HF_HOME=/mnt/B-SSD/fachel/dual-heater-mlp/.hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
P=/mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python
ROOT=${ROOT:-results/pilot_mask_timing}
DOMAINS="banking credit_cards kitchen_and_dining home auto_and_commute travel utility work small_talk meta"
mkdir -p $ROOT
lane() {
  local gpu=$1
  export CUDA_VISIBLE_DEVICES=$gpu
  for s in ${SEEDS:-500 501 502 503 504 505 506 507 508 509}; do
    local dir=$ROOT/seed$s
    [ -f "$dir/manifest.json" ] && continue
    mkdir -p "$dir"; mkdir "$dir/.claim" 2>/dev/null || continue
    echo "=== seed$s gpu=$gpu start $(date -Is)"
    $P experiments/qwen_iso_plasticity.py --domains $DOMAINS --steps-per-task ${STEPS:-120} \
      --learning-rate 1e-5 --batch-size 2 --max-length 64 --target-plasticity 0.75 --pilot-freeze-after 1 3 --pilot-head-warmup 20 \
      --arms hard_b0.75 reduced_lr hard_freeze1_b0.75 hard_freeze3_b0.75 hard_warmup20_b0.75 reduced_lr_warmup20 --seed $s --device ${DEVICE:-cuda} --output "$dir/manifest.json" > "$dir/stdout.log" 2>&1
    echo "=== seed$s gpu=$gpu exit=$? $(date -Is)"
  done
  echo "LANE gpu=$gpu FREE $(date -Is)"
}
for g in $GPUS; do lane "$g" & done
wait
echo "PILOT DONE $(date -Is)"
