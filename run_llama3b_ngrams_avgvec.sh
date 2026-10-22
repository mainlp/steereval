#!/usr/bin/env bash
# Beyond-Ngrams GPTScore evaluation with language-averaged steering vector — Llama-3.2-3B.
#
# Fixed hyperparameters: layer=27, sigma=-2.5, English prompts.
# Vectors are averaged across seahorse source languages (de es ru tr vi).
# The whole dataset is used as test (no val split).
#
# Usage:
#   bash run_llama3b_ngrams_avgvec.sh [GPU_ID]

set -uo pipefail

MODEL="meta-llama/Llama-3.2-3B"
GPU="${1:-0}"
LANGS="arabic chinese hebrew japanese spanish turkish ukrainian yoruba"
MAX_PAIRS=500
DATA_DIR="./data/ngrams"
DATA_ROOT="./data/ncwm_flores"
RESULTS_DIR="results/ngrams_avgvec_gptscore_llama3b"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " Beyond-Ngrams avgvec eval — Llama-3.2-3B"
echo " Model: $MODEL"
echo " GPU:   $GPU"
echo " Layer: 27  Sigma: -2.5  Prompt: EN"
echo "============================================"

for LANG in $LANGS; do
    OUT="${RESULTS_DIR}/llama3b_${LANG}_avgvec_en.json"
    echo ""
    echo "--- lang=${LANG} ---"
    if python eval_ngrams_avgvec_gptscore.py \
        --model      "$MODEL" \
        --lang       "$LANG" \
        --data-dir   "$DATA_DIR" \
        --data-root  "$DATA_ROOT" \
        --max-pairs  "$MAX_PAIRS" \
        --gpu        "$GPU" \
        --out        "$OUT"; then
        echo "Saved → $OUT"
    else
        echo "ERROR: run failed (lang=${LANG})" >&2
    fi
done

echo ""
echo "All done.  Results in ${RESULTS_DIR}/"
