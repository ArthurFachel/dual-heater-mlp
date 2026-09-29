#!/usr/bin/env bash
# Lança as dez seeds confirmatórias do protocolo de plasticidade pareada,
# destacado, uma seed inteira por GPU.
#
# Destacado de propósito: uma run rastreada pela sessão foi morta por SIGTERM
# (agent_close) aos 34 min em 28/09, perdendo trabalho em andamento. setsid põe
# cada shard na própria sessão, fora do ciclo de vida do agente.
#
# Sharding por SEED, não por braço: os três braços de uma seed ficam na mesma
# GPU, compartilhando fluxo de dados, ordenação e contagem de tokens, que é o
# que mantém o teste de sinal pareado válido.
#
# Custo medido nos manifestos da decomposição, mesmo cenário:
# ~1040 s/seed (348 + ~356 + 335). Dez seeds em 3 GPUs = 4 ondas, ~70 min.
#
# Retomada: cada shard pula as seeds que já têm summary.md, então uma
# interrupção custa no máximo a seed em andamento.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results/plasticity_matched

GPUS=(0 1 2)
SHARDS=${#GPUS[@]}

for shard in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard]}"
  log="results/plasticity_matched/run_shard${shard}.log"
  setsid nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. \
    .venv/bin/python scripts/run_plasticity_matched.py \
      --shard "$shard" --shards "$SHARDS" \
    > "$log" 2>&1 < /dev/null &
  disown || true
  echo "shard $shard -> GPU $gpu, log: $log"
done

echo "lançado. acompanhar: tail -f results/plasticity_matched/run_shard*.log"
