#!/usr/bin/env bash
# Waits for F13 (NCP) to finish, runs its frozen analysis, then launches F14 on GPUs 0 1 2.
#   nohup setsid bash scripts/chain_f13_f14.sh > results/chain_f13_f14.log 2>&1 < /dev/null &
NCP=/mnt/B-SSD/fachel/ncp
echo "chain start $(date -Is)"
until grep -q "F13 DONE" $NCP/runs/logs/f13_confirm_ffn_soft30.log 2>/dev/null; do sleep 60; done
echo "F13 finished $(date -Is)"
( cd $NCP && /home/fachel/novoenv/bin/python scripts/analyze_f13.py > runs/f13_confirm_ffn_soft30/analysis_preregistered.txt 2>&1 )
echo "F13 analysis written $(date -Is)"
cd /mnt/B-SSD/fachel/dh-qwen-f14
GPUS="0 1 2" bash scripts/f14_confirm_qwen_hard.sh > results/f14_confirm_qwen_hard.log 2>&1
echo "F14 finished $(date -Is)"
/home/fachel/novoenv/bin/python scripts/analyze_f14.py > results/f14_confirm_qwen_hard/analysis_preregistered.txt 2>&1
echo "F14 analysis written $(date -Is)"
