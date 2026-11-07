#!/bin/bash
cd /mnt/B-SSD/fachel/dh-uniform-ctrl
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONPATH=.:src
mkdir -p results/logs
nohup setsid nice -n 10 /mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python scripts/run_f10_er_ace_uniform.py > results/logs/f10.log 2>&1 < /dev/null &
echo f10 $!
