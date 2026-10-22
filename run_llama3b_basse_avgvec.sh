#!/usr/bin/env bash
# BASSE GPTScore evaluation with language-averaged steering vector — Llama-3.2-3B.
#
# Fixed hyperparameters: layer=27, sigma=-2.5, English prompts.
# Vectors are averaged across seahorse source languages (de es ru tr vi).
# Full dataset used (val + test combined, no split).
#
# Usage:
#   bash run_llama3b_basse_avgvec.sh [GPU_ID]

set -uo pipefail

MODEL="meta-llama/Llama-3.2-3B"
GPU="${1:-0}"
LANGS="es eu"
MAX_PAIRS=500
DATA_ROOT="./data/ncwm_flores"
RESULTS_DIR="results/basse_avgvec_gptscore_llama3b"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " BASSE avgvec eval — Llama-3.2-3B"
echo " Model: $MODEL"
echo " GPU:   $GPU"
echo " Layer: 27  Sigma: -2.5  Prompt: EN"
echo "============================================"

for LANG in $LANGS; do
    OUT="${RESULTS_DIR}/llama3b_${LANG}_avgvec_en.json"
    echo ""
    echo "--- lang=${LANG} ---"
    if python eval_basse_avgvec_gptscore.py \
        --model      "$MODEL" \
        --lang       "$LANG" \
        --data-root  "$DATA_ROOT" \
        --max-pairs  "$MAX_PAIRS" \
        --gpu        "$GPU" \
        --out        "$OUT"; then
        echo "Saved -> $OUT"
    else
        echo "ERROR: run failed (lang=${LANG})" >&2
    fi
done

echo ""
echo "All done.  Results in ${RESULTS_DIR}/"
