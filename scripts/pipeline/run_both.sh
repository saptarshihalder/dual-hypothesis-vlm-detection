#!/bin/bash
export HF_TOKEN=<HF_TOKEN_REDACTED>
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
cd /NAS_DISK/Saptarshi_data/

TOTAL=42764
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Waiting for Phi-4 to finish before starting Gemma ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

START_TIME=$(date +%s)
FIRST_COUNT=$(python3 -c "import json; d=json.load(open('/NAS_DISK/Saptarshi_data/results/phi4_results.json')); print(len(d['results']))" 2>/dev/null || echo "0")

while pgrep -f "dual_vlm_runner.py --model phi4" > /dev/null; do
    DONE=$(python3 -c "import json; d=json.load(open('/NAS_DISK/Saptarshi_data/results/phi4_results.json')); print(len(d['results']))" 2>/dev/null || echo "0")
    PCT=$((DONE * 100 / TOTAL))
    FILLED=$((PCT / 2))
    EMPTY=$((50 - FILLED))
    BAR=$(printf '█%.0s' $(seq 1 $FILLED 2>/dev/null) ; printf '░%.0s' $(seq 1 $EMPTY 2>/dev/null))
    
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    PROGRESS=$((DONE - FIRST_COUNT))
    if [ "$PROGRESS" -gt 0 ] && [ "$ELAPSED" -gt 0 ]; then
        REMAINING=$((TOTAL - DONE))
        RATE_PER_SEC=$(echo "scale=4; $PROGRESS / $ELAPSED" | bc 2>/dev/null || echo "0")
        if [ "$(echo "$RATE_PER_SEC > 0" | bc 2>/dev/null)" = "1" ]; then
            ETA_SEC=$(echo "scale=0; $REMAINING / $RATE_PER_SEC" | bc 2>/dev/null || echo "0")
            ETA_H=$((ETA_SEC / 3600))
            ETA_M=$(( (ETA_SEC % 3600) / 60 ))
            ETA_STR="${ETA_H}h ${ETA_M}m"
        else
            ETA_STR="calculating..."
        fi
    else
        ETA_STR="calculating..."
    fi

    printf "\r  Phi-4 [${BAR}] ${PCT}%%  ${DONE}/${TOTAL}  ETA: ${ETA_STR}    "
    sleep 120
done

echo ""
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✓ Phi-4 COMPLETE! Starting Gemma-3 now..."
echo "  $(date)"
echo "═══════════════════════════════════════════════════════"
echo ""

~/miniforge3/envs/phi4/bin/python -u dual_vlm_runner.py --model gemma3 2>&1 | tee gemma3_fresh.log

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✓ Gemma-3 COMPLETE! All 6 VLMs done!"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════"
