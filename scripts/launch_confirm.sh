#!/bin/bash
cd /mnt/B-SSD/fachel/dh-uniform-ctrl
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=.:src
nohup setsid nice -n 10 /mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python scripts/run_confirm_uniform.py > results/logs/confirm_uniform.log 2>&1 < /dev/null &
echo confirm $!
