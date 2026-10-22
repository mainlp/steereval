#!/usr/bin/env bash
# SEAHORSE GPTScore evaluation with language-averaged steering vector — aya-expanse-32b.
#
# Hyperparameters: layer=39 (final layer, 0-indexed; 40 total), sigma=-2.5.
# Same sigma as aya-expanse-8b EN-prompt runs. EN prompting only.
# Vectors are averaged across all five source languages (de es ru tr vi).
#
# Note: steering vectors are computed fresh for this model (different
# representations from aya-expanse-8b) and cached under cache/seahorse/.
#
# VERIFY layer count before running:
#   python -c "from transformers import AutoConfig; \
#              c = AutoConfig.from_pretrained('CohereForAI/aya-expanse-32b'); \
#              print('num_hidden_layers:', c.num_hidden_layers)"
# Expected: 40 (final 0-indexed layer = 39).
#
# Usage:
#   bash run_aya32b_seahorse_avgvec.sh [GPU_ID]

set -uo pipefail

MODEL="CohereForAI/aya-expanse-32b"
GPU="${1:-0}"
LAYER=39
SIGMA=-2.5
LANGS="de es ru tr vi"
MAX_PAIRS=500
SEAHORSE_DIR="./data/seahorse"
DATA_ROOT="./data/ncwm_flores"
RESULTS_DIR="results/seahorse_avgvec_gptscore_aya32b"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " SEAHORSE avgvec evaluation — aya-expanse-32b"
echo " Model:  $MODEL"
echo " GPU:    $GPU"
echo " Layer:  $LAYER  sigma=$SIGMA  Prompt: EN"
echo "============================================"

if [[ ! -f "${SEAHORSE_DIR}/validation.tsv" ]]; then
    echo "ERROR: SEAHORSE data not found at ${SEAHORSE_DIR}."
    exit 1
fi

for LANG in $LANGS; do
    OUT="${RESULTS_DIR}/aya32b_${LANG}_avgvec_en.json"
    echo ""
    echo "--- lang=${LANG} ---"
    if python eval_seahorse_avgvec_gptscore.py \
            --model          "$MODEL" \
            --lang           "$LANG" \
            --prompting-lang en \
            --data-dir       "$SEAHORSE_DIR" \
            --data-root      "$DATA_ROOT" \
            --max-pairs      "$MAX_PAIRS" \
            --layer          "$LAYER" \
            --sigma          "$SIGMA" \
            --gpu            "$GPU" \
            --out            "$OUT"; then
        echo "Saved -> $OUT"
    else
        echo "ERROR: run failed (lang=${LANG})" >&2
    fi
done

echo ""
echo "All done.  Results in ${RESULTS_DIR}/"
