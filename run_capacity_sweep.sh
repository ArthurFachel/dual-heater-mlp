#!/usr/bin/env bash
# Width sweep queue: does capacity alone decide whether hard protection helps?
#
# Ten independent targets (2 datasets x 5 widths), each training 10 seeds x 6
# methods. Targets are independent, so they run in parallel; seeds within a
# target stay sequential because the runner pairs them.
#
# Thread budget: each worker gets OMP_NUM_THREADS=3. With 10 workers that is 30
# of 32 cores, leaving the machine usable. Do not raise this without checking
# nproc -- oversubscribing makes every worker slower, not faster.
#
# Resumable: re-running skips seeds already on disk. Killing it is safe.
set -uo pipefail
cd "$(dirname "$0")"

LOG_DIR="results/run_logs/capacity_sweep"
OUT_DIR="results/capacity_sweep"
THREADS="${THREADS:-3}"
mkdir -p "$LOG_DIR"

# Refuse to start against a dirty tree: the hard-versus-soft suite is the only
# aggregate in this repository with a clean-tree provenance, and this sweep is
# meant to meet the same bar.
if [ -n "$(git status --porcelain)" ]; then
  echo "ERRO: árvore Git suja. Faça commit antes de rodar." >&2
  git status --short >&2
  exit 1
fi
COMMIT="$(git rev-parse HEAD)"
echo "commit: $COMMIT" | tee "$LOG_DIR/queue.log"

# Widest rungs first: they dominate the critical path, so starting them last
# would leave cores idle at the end.
WIDTHS="2048x1024 1024x512 512x256 256x128 128x64"
DATASETS="split_cifar100 split_cifar10"

pids=()
names=()
for dataset in $DATASETS; do
  for width in $WIDTHS; do
    name="${dataset}_w${width}"
    echo "[$(date +%H:%M:%S)] iniciando $name" | tee -a "$LOG_DIR/queue.log"
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      PYTHONPATH=. .venv/bin/python -m experiments.capacity_sweep \
        --datasets "$dataset" \
        --widths "$width" \
        --output-dir "$OUT_DIR" \
        --device cpu \
        --no-download \
      > "$LOG_DIR/${name}.log" 2>&1 &
    pids+=($!)
    names+=("$name")
  done
done

echo "[$(date +%H:%M:%S)] ${#pids[@]} alvos em execução" | tee -a "$LOG_DIR/queue.log"

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[$(date +%H:%M:%S)] OK   ${names[$i]}" | tee -a "$LOG_DIR/queue.log"
  else
    echo "[$(date +%H:%M:%S)] FALHOU ${names[$i]} (ver $LOG_DIR/${names[$i]}.log)" \
      | tee -a "$LOG_DIR/queue.log"
    failed=$((failed + 1))
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "=== $failed alvo(s) falharam; análise NÃO executada ===" | tee -a "$LOG_DIR/queue.log"
  exit 1
fi

echo "[$(date +%H:%M:%S)] === fila concluída; analisando ===" | tee -a "$LOG_DIR/queue.log"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python experiments/analyze_capacity_sweep.py \
  --root "$OUT_DIR" \
  --json-out "$OUT_DIR/capacity_sweep_analysis.json" \
  2>&1 | tee -a "$LOG_DIR/queue.log"
