#!/usr/bin/env bash
# BASSE evaluation with layer selection on layers 21-27 (Lu et al., EMNLP 2025).
#
# Each layer in [21..27] is tested independently on the validation split.
# The single (layer, sigma) pair with the best val Spearman is then used
# for the test evaluation — matching the procedure from the paper.
#
# Usage:
#   bash run_llama3b_basse.sh [GPU_ID]
#
# GPU_ID defaults to 0.
#
# Prerequisites: same parallel data layout as run_aya_basse.sh.
# Outputs are written under results/basse_gptscore_llama3b/.

set -uo pipefail

MODEL="meta-llama/Llama-3.2-3B"
GPU="${1:-0}"
MAX_PAIRS=500
RESULTS_DIR="results/basse_gptscore_llama3b"
DATA_ROOT="./data/ncwm_flores"
LAYERS="21 22 23 24 25 26 27"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " BASSE evaluation — Llama-3.2-3B"
echo " Model: $MODEL"
echo " GPU:   $GPU"
echo "============================================"

for LANG in es eu; do
    PARALLEL_DIR="${DATA_ROOT}/en-${LANG}"

    if [[ ! -d "$PARALLEL_DIR" ]]; then
        echo ""
        echo "WARNING: parallel data not found for lang='${LANG}' at ${PARALLEL_DIR}."
        echo "         Skipping ${LANG}."
        continue
    fi

    for METHOD in vector; do
        for PLANG in en tgt; do
            OUT="${RESULTS_DIR}/llama3b_${LANG}_${METHOD}_${PLANG}.json"
            echo ""
            echo "--- lang=${LANG}  method=${METHOD}  prompting=${PLANG} ---"
            if python eval_basse_gptscore.py \
                --model          "$MODEL" \
                --lang           "$LANG" \
                --parallel-data  "$PARALLEL_DIR" \
                --method         "$METHOD" \
                --prompting-lang "$PLANG" \
                --max-pairs      "$MAX_PAIRS" \
                --gpu            "$GPU" \
                --layers         $LAYERS \
                --select-layer \
                --pairwise \
                --out            "$OUT"; then
                echo "Saved → $OUT"
            else
                echo "ERROR: run failed (lang=${LANG} method=${METHOD} prompting=${PLANG})" >&2
            fi
        done
    done
done

echo ""
echo "All done.  Results in ${RESULTS_DIR}/"
