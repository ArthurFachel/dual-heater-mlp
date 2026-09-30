#!/usr/bin/env bash
# Lança a passada 1 do CIFAR-100 em 3 GPUs, um shard por placa.
#
# `goals/protocol_penalty_reevaluation.md` §C.1 e §D.
#
# CUDA_DEVICE_ORDER=PCI_BUS_ID é obrigatório: por padrão o CUDA enumera as
# placas por FASTEST_FIRST, e não pela ordem que o nvidia-smi mostra. Sem isso
# `CUDA_VISIBLE_DEVICES=2` cai na GPU física 1 (verificado por UUID nesta
# máquina), o que não quebra o sharding — valores distintos continuam indo para
# placas distintas — mas torna impossível dizer qual placa rodou o quê.
#
# setsid + stdin de /dev/null + disown: uma run já morreu por SIGTERM aos 34 min
# por não estar destacada. Verificar PPID=1 depois de lançar.
set -euo pipefail

cd /mnt/B-SSD/fachel/dual-heater-mlp

SHARDS=3
LOG_DIR=results/penalty_pass1_cifar
mkdir -p "$LOG_DIR"

for shard in $(seq 0 $((SHARDS - 1))); do
  setsid nohup env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$shard" \
    OMP_NUM_THREADS=4 \
    PYTHONPATH=. \
    .venv/bin/python scripts/run_penalty_pass1_cifar.py \
      --shard "$shard" --shards "$SHARDS" \
    < /dev/null > "$LOG_DIR/shard_${shard}.log" 2>&1 &
  disown
  echo "shard $shard lançado na GPU física $shard"
done

sleep 5
echo
echo "=== verificação de destacamento (exige PPID=1) ==="
ps -eo pid,ppid,sess,cmd | grep run_penalty_pass1_cifar | grep -v grep
echo
echo "=== placas em uso ==="
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
