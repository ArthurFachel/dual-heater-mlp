#!/bin/bash
# T3c launcher: two detached CPU processes (split, permuted). Protocol docs/protocols/confirmatory_random_fixed_mlp.md
cd /mnt/B-SSD/fachel/dh-uniform-ctrl || exit 1
PY=/mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python
OUT=results/t3c_random_fixed_mlp
mkdir -p "$OUT"
COMMIT=$(git rev-parse HEAD)
echo "launch $(date -Iseconds) commit $COMMIT protocol docs/protocols/confirmatory_random_fixed_mlp.md" >> "$OUT/launcher.log"
for S in split permuted; do
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONPATH=.:src \
    nohup setsid bash -c "nice -n 10 $PY scripts/run_t3c_random_mlp.py --setting $S > $OUT/${S}.log 2>&1; echo \"${S} DONE exit=\$? \$(date -Iseconds)\" >> $OUT/launcher.log" \
    < /dev/null > /dev/null 2>&1 &
done
sleep 2
ps -eo pid,ppid,etime,args | grep run_t3c_random_mlp | grep -v grep
