#!/usr/bin/env bash
# Full TRACE run, LoRA vanilla vs LoRA SlowHeat `exact`. EXPLORATORY.
#
# Not preregistered (roadmap 3b.3 still open): one seed, no endpoint gate.
# Numbers from this run are cost, dispersion and feasibility evidence only.
#
# Official protocol (8 tasks, official LoRA epochs 5,3,7,5,3,5,5,7, 5000-row
# release, eval on every past task after each task, temperature 0.1,
# max_new_tokens 512), with deviations R1-R10 of experiments/trace_cl.py.
# Both arms share seed 1234 (TRACE's own seed) so the data order and sampling
# seeds are paired.
#
# One arm per GPU, each in its own tmux window of session `trace-full`:
#   vanilla -> GPU 0 (GTX 1080 Ti 11 GB), measured worst-case reserve 10.80 GB
#   exact   -> GPU 2 (Titan Xp 12 GB),    measured worst-case reserve 10.85 GB
#
# Progress after every task: results/trace_full/<arm>/<arm>/progress.json
# Logs: results/trace_full/<arm>.log
set -euo pipefail
cd "$(dirname "$0")/.."

SESSION=trace-full
OUT=results/trace_full
mkdir -p "$OUT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "sessão tmux '$SESSION' já existe; não relanço (evita duas runs no mesmo output)" >&2
  exit 1
fi

# PCI_BUS_ID makes CUDA_VISIBLE_DEVICES match nvidia-smi indices. Without it
# CUDA orders FASTEST_FIRST and index 0 is the Titan Xp, which silently swapped
# the two arms' cards on the first launch.
COMMON="CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:."

launch() {
  local arm=$1 gpu=$2
  local cmd="cd $(pwd) && env $COMMON CUDA_VISIBLE_DEVICES=$gpu \
.venv/bin/python -m experiments.trace_cl --data-root data/TRACE \
--output $OUT/$arm --arms $arm --seed 1234 --device cuda:0 \
> $OUT/$arm.log 2>&1; echo EXIT=\$? >> $OUT/$arm.log"
  if [ "$arm" = vanilla ]; then
    tmux new-session -d -s "$SESSION" -n "$arm" "bash -c '$cmd'"
  else
    tmux new-window -t "$SESSION" -n "$arm" "bash -c '$cmd'"
  fi
}

git rev-parse HEAD > "$OUT/git_head.txt"
git status --porcelain > "$OUT/git_status_at_launch.txt"
git diff > "$OUT/git_diff_at_launch.patch"

launch vanilla 0
launch exact 2
tmux ls | grep "$SESSION"
sleep 60
# Fail loudly if a card is not the one declared above.
for pair in "vanilla:0" "exact:2"; do
  arm=${pair%%:*}; gpu=${pair##*:}
  pid=$(pgrep -f "^.venv/bin/python -m experiments.trace_cl.*--arms $arm " | head -1)
  bus=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | awk -F', ' -v p="$pid" '$1==p{print $2}')
  want=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v g="$gpu" '$1==g{print $2}')
  echo "$arm pid=$pid bus=$bus esperado=$want"
  [ -n "$bus" ] && [ "$bus" = "$want" ] || { echo "GPU ERRADA para $arm" >&2; exit 2; }
done
