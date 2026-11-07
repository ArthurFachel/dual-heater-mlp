#!/bin/bash
# F7 + F8 (fila.md, 06/11): CPU only, nice 10, 6 threads each.
cd /mnt/B-SSD/fachel/dh-uniform-ctrl
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONPATH=.:src
PY=/mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python
mkdir -p results/logs
nohup setsid nice -n 10 $PY scripts/run_f7_baselines_uniform.py > results/logs/f7.log 2>&1 < /dev/null &
echo f7 $!
nohup setsid nice -n 10 $PY scripts/run_f8_permuted_uniform_hard.py > results/logs/f8.log 2>&1 < /dev/null &
echo f8 $!
