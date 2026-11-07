#!/usr/bin/env bash
# F12 confirmatory BERT soft-30 on the Lab CPU: 5 pairs in parallel, 4 threads each, nice 10.
set -e
cd /mnt/B-SSD/fachel/dh-conc2x2
P=/mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python
OUT=results/confirm_bert_soft30
mkdir -p $OUT
for pair in "banking credit_cards" "auto_and_commute travel" "kitchen_and_dining home" "small_talk meta" "utility work"; do
  name=${pair// /__}
  ( export CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      HF_HOME=$PWD/.hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
    nohup setsid nice -n 10 $P scripts/run_confirm_bert_soft30.py --pair $pair > $OUT/${name}.log 2>&1 < /dev/null & )
done
sleep 5
ps -u fachel -o pid,args | grep run_confirm_bert_soft30 | grep -v grep
echo "launched $(date -Is)"
