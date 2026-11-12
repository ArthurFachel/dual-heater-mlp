#!/usr/bin/env bash
# Add ONE arm of the full TRACE run to the existing tmux session `trace-full`.
#
# Same command, seed (1234), data and output layout as launch_trace_full.sh,
# so the new arm is paired with the running ones in data order and sampling
# seeds. Refuses to start if the arm already has a window, a live process or
# any output: two writers on one output dir is how a stale run silently wins.
#
# Usage: scripts/launch_trace_arm.sh <arm> <nvidia-smi gpu index>
set -euo pipefail
cd "$(dirname "$0")/.."

ARM=${1:?arm}
GPU=${2:?gpu index (nvidia-smi numbering)}
SESSION=trace-full
OUT=results/trace_full

tmux has-session -t "$SESSION" 2>/dev/null || { echo "sessão $SESSION não existe" >&2; exit 1; }
if tmux list-windows -t "$SESSION" -F '#W' | grep -qx "$ARM"; then
  echo "janela $ARM já existe" >&2; exit 1
fi
if pgrep -f "^.venv/bin/python -m experiments.trace_cl.*--arms $ARM " >/dev/null; then
  echo "processo de $ARM já está vivo" >&2; exit 1
fi
if [ -e "$OUT/$ARM" ] || [ -e "$OUT/$ARM.log" ]; then
  echo "$OUT/$ARM já existe; use outro output" >&2; exit 1
fi
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$GPU" '$1==g{print $2}')
[ "${busy:-99999}" -lt 500 ] || { echo "GPU $GPU ocupada (${busy} MiB)" >&2; exit 1; }

mkdir -p "$OUT/launch_$ARM"
git rev-parse HEAD > "$OUT/launch_$ARM/git_head.txt"
git status --porcelain > "$OUT/launch_$ARM/git_status_at_launch.txt"
git diff > "$OUT/launch_$ARM/git_diff_at_launch.patch"

CMD="cd $(pwd) && env CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:. \
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -m experiments.trace_cl --data-root data/TRACE \
--output $OUT/$ARM --arms $ARM --seed 1234 --device cuda:0 \
> $OUT/$ARM.log 2>&1; echo EXIT=\$? >> $OUT/$ARM.log"
tmux new-window -t "$SESSION" -n "$ARM" "bash -c '$CMD'"

sleep 60
pid=$(pgrep -f "^.venv/bin/python -m experiments.trace_cl.*--arms $ARM " | head -1)
bus=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | awk -F', ' -v p="$pid" '$1==p{print $2}')
want=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
echo "$ARM pid=$pid bus=$bus esperado=$want"
[ -n "$bus" ] && [ "$bus" = "$want" ] || { echo "GPU ERRADA para $ARM" >&2; exit 2; }
