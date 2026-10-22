#!/usr/bin/env bash
# Beyond-Ngrams GPTScore evaluation with language-averaged steering vector — aya-expanse-8b.
#
# Fixed hyperparameters: layer=31, sigma=-2.5, English prompts.
# Vectors are averaged across seahorse source languages (de es ru tr vi).
# The whole dataset is used as test (no val split).
#
# Usage:
#   bash run_aya_ngrams_avgvec.sh [GPU_ID]

set -uo pipefail

MODEL="CohereForAI/aya-expanse-8b"
GPU="${1:-0}"
LANGS="arabic chinese hebrew japanese spanish turkish ukrainian yoruba"
MAX_PAIRS=500
DATA_DIR="./data/ngrams"
DATA_ROOT="./data/ncwm_flores"
RESULTS_DIR="results/ngrams_avgvec_gptscore_aya"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " Beyond-Ngrams avgvec eval — aya-expanse-8b"
echo " Model: $MODEL"
echo " GPU:   $GPU"
echo " Layer: 31  Sigma: -2.5  Prompt: EN"
echo "============================================"

for LANG in $LANGS; do
    OUT="${RESULTS_DIR}/aya_${LANG}_avgvec_en.json"
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
