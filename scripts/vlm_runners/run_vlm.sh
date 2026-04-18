#!/bin/bash
# run_vlm.sh — Run dual hypothesis VLM runner inside tmux
# Usage:
#   bash run_vlm.sh internvl     # single model
#   bash run_vlm.sh all          # all 6 models sequentially
#
# To reattach: tmux attach -t vlm
# To detach:   Ctrl+B then D

MODEL=${1:-internvl}
SESSION="vlm_${MODEL}"

echo "Starting $MODEL in tmux session: $SESSION"
echo "Reattach with: tmux attach -t $SESSION"
echo ""

tmux kill-session -t "$SESSION" 2>/dev/null

tmux new-session -d -s "$SESSION" \
    "cd /home/tbvl_akshay && python3 -u dual_vlm_runner.py --model $MODEL 2>&1 | tee ${MODEL}_run.log"

echo "Running! Use 'tmux attach -t $SESSION' to watch."
