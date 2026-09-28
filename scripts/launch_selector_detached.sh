#!/usr/bin/env bash
# Launch the selector confirmation fully detached from the agent session.
#
# The first GPU attempt was killed by SIGTERM (agent_close) at 34 min because it
# was a session-tracked background process. setsid + nohup put the run in its own
# session so it survives the agent's process lifecycle.
#
# resume=True in the runner picks up the seeds already on disk.
set -euo pipefail

cd /mnt/B-SSD/fachel/dual-heater-mlp
LOG=results/replay_selector_confirmation/run.log

setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    .venv/bin/python scripts/run_selector_confirmation.py \
    >> "$LOG" 2>&1 < /dev/null &

disown || true
echo "launched, log: $LOG"
