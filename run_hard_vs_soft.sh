#!/usr/bin/env bash
# Hard vs soft protection on MLP and CNN hosts -- queued behind the Qwen run.
#
# WHY THIS EXISTS
# The Transformer results use hard protection (the consolidated units are
# frozen); every MLP/CNN result uses soft protection, 1/(1+30*h). That confounds
# the protection regime with the architecture. This suite runs both regimes on
# the non-Transformer hosts so the manuscript can separate the two.
#
# WHAT IT WAITS FOR
# The Qwen confirmation owns all three GPUs. This script blocks until no
# qwen_iso_plasticity.py process remains, then runs on CPU only
# (CUDA_VISIBLE_DEVICES=''), so it cannot disturb a late Qwen seed even if one
# is restarted. Nothing here reads or writes any Qwen artifact.
#
# SAFETY
# * Refuses to start on a dirty Git tree: these results are meant to be citable,
#   and 89 of 93 existing aggregates are already unciteable for that reason.
# * The 10 seeds are drawn from a fixed generator and written to
#   hard_vs_soft_protocol.json before the first training step; the runner
#   refuses seeds reserved for the frozen confirmation.
# * Resumable: a completed (dataset, backbone) target is skipped on re-run.
#
# USAGE
#   ./run_hard_vs_soft.sh                  # wait for Qwen, then run
#   ./run_hard_vs_soft.sh --no-wait        # run immediately
#   ./run_hard_vs_soft.sh --allow-dirty    # override the clean-tree check
set -uo pipefail
cd "$(dirname "$0")"

WAIT_FOR_QWEN=1
ALLOW_DIRTY=0
for arg in "$@"; do
  case "$arg" in
    --no-wait) WAIT_FOR_QWEN=0 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    *) echo "argumento desconhecido: $arg" >&2; exit 2 ;;
  esac
done

OUT_ROOT="results/hard_vs_soft"
LOG_DIR="results/run_logs/hard_vs_soft"
mkdir -p "$LOG_DIR"

# Count only real interpreter processes running the Qwen runner. A plain
# `pgrep -f qwen_iso_plasticity` also matches any shell whose command line
# happens to contain the string -- including this script when invoked as
# `bash run_hard_vs_soft.sh ...` -- which would make the wait loop never end.
qwen_count() {
  pgrep -af "experiments/qwen_iso_plasticity\.py" 2>/dev/null \
    | awk -v self="$$" '$1 != self && $2 ~ /python/ { n++ } END { print n+0 }'
}

# (dataset, backbone) targets. split_cifar10/cnn is the only versioned CNN
# config; the runner refuses the combinations that have none.
TARGETS=(
  "split_mnist mlp"
  "permuted_mnist mlp"
  "split_cifar10 mlp"
  "split_cifar100 mlp"
  "split_cifar10 cnn"
)

echo "=== hard vs soft: fila iniciada $(date '+%F %T') ==="

if [ "$WAIT_FOR_QWEN" -eq 1 ]; then
  echo "[fila] aguardando o término da confirmação Qwen..."
  while [ "$(qwen_count)" -gt 0 ]; do
    echo "[fila] $(date '+%F %T') Qwen ainda ativo ($(qwen_count) processo(s)); nova checagem em 5 min"
    sleep 300
  done
  echo "[fila] Qwen concluído em $(date '+%F %T'); aguardando 60 s para a GPU liberar"
  sleep 60
fi

if [ "$ALLOW_DIRTY" -eq 0 ] && [ -n "$(git status --porcelain)" ]; then
  echo "[ERRO] árvore Git suja. Estes resultados nascem não citáveis." >&2
  echo "       Commite as mudanças ou use --allow-dirty conscientemente." >&2
  git status --short >&2
  exit 1
fi

COMMIT=$(git rev-parse --short HEAD)
echo "[fila] commit: $COMMIT | árvore: $([ -z "$(git status --porcelain)" ] && echo limpa || echo SUJA)"
echo "[fila] $(nproc) CPUs disponíveis; execução em CPU (GPU não é usada)"

failures=0
for target in "${TARGETS[@]}"; do
  set -- $target
  dataset=$1; backbone=$2
  name="${dataset}_${backbone}"
  out="${OUT_ROOT}/${name}"
  log="${LOG_DIR}/${name}.log"

  if [ -f "${out}/pair_report.json" ]; then
    echo "[${name}] já concluído, pulando"
    continue
  fi

  echo "[${name}] iniciando $(date '+%F %T') -> ${log}"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH=.:src \
  .venv/bin/python -m experiments.hard_vs_soft \
    --datasets "$dataset" \
    --backbones "$backbone" \
    --num-seeds 10 \
    --data-dir data \
    --output-dir "$OUT_ROOT" \
    --device cpu \
    --no-download \
    > "$log" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "[${name}] OK $(date '+%F %T')"
  else
    echo "[${name}] FALHOU (exit=$status) -- veja $log"
    tail -5 "$log" | sed 's/^/    /'
    failures=$((failures + 1))
  fi
done

echo
echo "=== hard vs soft: fila concluída $(date '+%F %T') ==="
for target in "${TARGETS[@]}"; do
  set -- $target
  name="${1}_${2}"
  if [ -f "${OUT_ROOT}/${name}/pair_report.json" ]; then
    echo "  ${name}: OK"
  else
    echo "  ${name}: AUSENTE"
  fi
done
[ $failures -eq 0 ] || echo "${failures} alvo(s) falharam"
exit $failures
