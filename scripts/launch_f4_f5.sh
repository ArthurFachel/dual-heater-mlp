#!/usr/bin/env bash
# F4 + F5 on the Lab CPU (GPUs busy with F2/F3). ~12 cores total, nice 10.
P=/mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python
export CUDA_VISIBLE_DEVICES=""

cd /mnt/B-SSD/fachel/dh-uniform-ctrl && rm -rf .store_smoke_f4
( export PYTHONPATH=.:src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  nohup setsid nice -n 10 $P scripts/run_f4_uniform.py > results/f4_split_mnist_uniform.log 2>&1 < /dev/null & )

cd /mnt/B-SSD/fachel/dh-conc2x2
mkdir -p results/domain_pairs
for pair in "kitchen_and_dining home" "auto_and_commute travel" "utility work" "small_talk meta"; do
  name=${pair// /__}
  ( export PYTHONPATH=.:src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HOME=$PWD/.hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
    nohup setsid nice -n 10 $P scripts/run_domain_pairs.py --pair $pair > results/domain_pairs/$name.log 2>&1 < /dev/null & )
done
sleep 5
ps -u fachel -o pid,args | grep -E "run_f4_uniform|run_domain_pairs" | grep -v grep
echo "launched $(date -Is)"
