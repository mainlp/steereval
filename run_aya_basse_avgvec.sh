#!/usr/bin/env bash
# BASSE GPTScore evaluation with language-averaged steering vector — aya-expanse-8b.
#
# Fixed hyperparameters: layer=31, sigma=-2.5, English prompts.
# Vectors are averaged across seahorse source languages (de es ru tr vi).
# Full dataset used (val + test combined, no split).
#
# Usage:
#   bash run_aya_basse_avgvec.sh [GPU_ID]

set -uo pipefail

MODEL="CohereForAI/aya-expanse-8b"
GPU="${1:-0}"
LANGS="es eu"
MAX_PAIRS=500
DATA_ROOT="./data/ncwm_flores"
RESULTS_DIR="results/basse_avgvec_gptscore_aya"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " BASSE avgvec eval — aya-expanse-8b"
echo " Model: $MODEL"
echo " GPU:   $GPU"
echo " Layer: 31  Sigma: -2.5  Prompt: EN"
echo "============================================"

for LANG in $LANGS; do
    OUT="${RESULTS_DIR}/aya_${LANG}_avgvec_en.json"
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
