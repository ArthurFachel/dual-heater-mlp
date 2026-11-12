#!/usr/bin/env bash
# F17 CONFIRMATORY (docs/protocols/confirmatory_qwen_hard_random_fixed_f17.md, frozen before run). One process per seed.
#   GPUS="0 1 2" nohup setsid bash scripts/f16_confirm_qwen_hard_random_fixed.sh > results/f17_confirm_qwen_hard_random_fixed.log 2>&1 < /dev/null &
set -u
cd /mnt/B-SSD/fachel/dh-f17
GPUS=${GPUS:?set GPUS}
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.:src HF_HOME=/mnt/B-SSD/fachel/dual-heater-mlp/.hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
P=/mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python
ROOT=${ROOT:-results/f17_confirm_qwen_hard_random_fixed}
DOMAINS="banking credit_cards kitchen_and_dining home auto_and_commute travel utility work small_talk meta"
mkdir -p $ROOT
lane() {
  local gpu=$1
  export CUDA_VISIBLE_DEVICES=$gpu
  for s in ${SEEDS:-600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619}; do
    local dir=$ROOT/seed$s
    [ -f "$dir/manifest.json" ] && continue
    mkdir -p "$dir"; mkdir "$dir/.claim" 2>/dev/null || continue
    echo "=== seed$s gpu=$gpu start $(date -Is)"
    $P experiments/qwen_iso_plasticity.py --domains $DOMAINS --steps-per-task ${STEPS:-120} \
      --learning-rate 1e-5 --batch-size 2 --max-length 64 --target-plasticity 0.75 --hard-permuted-control --hard-random-fixed-control \
      --arms hard_b0.75 hard_random_fixed_b0.75 hard_permuted_b0.75 --seed $s --device ${DEVICE:-cuda} --output "$dir/manifest.json" > "$dir/stdout.log" 2>&1
    echo "=== seed$s gpu=$gpu exit=$? $(date -Is)"
  done
  echo "LANE gpu=$gpu FREE $(date -Is)"
}
for g in $GPUS; do lane "$g" & done
wait
echo "F17 DONE $(date -Is)"
