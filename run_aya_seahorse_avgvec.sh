#!/usr/bin/env bash
# SEAHORSE GPTScore evaluation with language-averaged steering vector — aya-expanse-8b.
#
# Fixed hyperparameters: layer=31, sigma_en=-2.5, sigma_tgt=-2.5.
# Vectors are averaged across all five source languages (de es ru tr vi).
#
# Usage:
#   bash run_aya_seahorse_avgvec.sh [GPU_ID]

set -uo pipefail

MODEL="CohereForAI/aya-expanse-8b"
GPU="${1:-0}"
shift || true
PLANGS="${*:-en tgt}"
LANGS="de es ru tr vi"
MAX_PAIRS=500
SEAHORSE_DIR="./data/seahorse"
DATA_ROOT="./data/ncwm_flores"
RESULTS_DIR="results/seahorse_avgvec_gptscore_aya"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " SEAHORSE avgvec evaluation — aya-expanse-8b"
echo " Model: $MODEL"
echo " GPU:   $GPU"
echo " Layer: 31  sigma_en=-2.5  sigma_tgt=-2.5"
echo "============================================"

if [[ ! -f "${SEAHORSE_DIR}/validation.tsv" ]]; then
    echo "ERROR: SEAHORSE data not found at ${SEAHORSE_DIR}."
    exit 1
fi

for LANG in $LANGS; do
    for PLANG in $PLANGS; do
        OUT="${RESULTS_DIR}/aya_${LANG}_avgvec_${PLANG}.json"
        echo ""
        echo "--- lang=${LANG}  prompting=${PLANG} ---"
        if python eval_seahorse_avgvec_gptscore.py \
            --model          "$MODEL" \
            --lang           "$LANG" \
            --prompting-lang "$PLANG" \
            --data-dir       "$SEAHORSE_DIR" \
            --data-root      "$DATA_ROOT" \
            --max-pairs      "$MAX_PAIRS" \
            --gpu            "$GPU" \
            --out            "$OUT"; then
            echo "Saved -> $OUT"
        else
            echo "ERROR: run failed (lang=${LANG} prompting=${PLANG})" >&2
        fi
    done
done

echo ""
echo "All done.  Results in ${RESULTS_DIR}/"
