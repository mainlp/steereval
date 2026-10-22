#!/usr/bin/env bash
# SEAHORSE G-eval evaluation with layer selection on layers 21-27 (Lu et al., EMNLP 2025).
#
# Each layer in [21..27] is swept individually on SEAHORSE's validation split.
# The best (layer, sigma) pair is used for the test split evaluation.
#
# Usage:
#   bash run_llama3b_seahorse_geval.sh [GPU_ID]
#
# Prerequisites
# -------------
# • SEAHORSE TSV files (validation.tsv, test.tsv) extracted to $SEAHORSE_DIR.
#   Download from: https://storage.googleapis.com/seahorse-public/seahorse_data.zip
# • Parallel data for each language at $DATA_ROOT/en-{lang}/ (train.en + train.{lang}).
#   English is run as baseline only (no steering needed).
#   Languages without parallel data are skipped automatically.
#
# Outputs are written under results/seahorse_geval_llama3b/.

set -uo pipefail

MODEL="meta-llama/Llama-3.2-3B"
GPU="${1:-0}"
shift || true
LANGS="${*:-de es ru tr vi}"
MAX_PAIRS=500
SEAHORSE_DIR="./data/seahorse"
DATA_ROOT="./data/ncwm_flores"
RESULTS_DIR="results/seahorse_geval_llama3b"
LAYERS="21 22 23 24 25 26 27"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " SEAHORSE G-eval evaluation — Llama-3.2-3B"
echo " Model: $MODEL"
echo " GPU:   $GPU"
echo "============================================"

if [[ ! -f "${SEAHORSE_DIR}/validation.tsv" ]]; then
    echo "ERROR: SEAHORSE data not found at ${SEAHORSE_DIR}."
    echo "       Download from https://storage.googleapis.com/seahorse-public/seahorse_data.zip"
    exit 1
fi

# ── Non-English languages: vector + map steering ────────────────────────────
for LANG in $LANGS; do
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
            if python eval_seahorse_geval.py \
                --model          "$MODEL" \
                --lang           "$LANG" \
                --data-dir       "$SEAHORSE_DIR" \
                --parallel-data  "$PARALLEL_DIR" \
                --method         "$METHOD" \
                --prompting-lang "$PLANG" \
                --max-pairs      "$MAX_PAIRS" \
                --layers         $LAYERS \
                --select-layer \
                --gpu            "$GPU" \
                --out            "$OUT"; then
                echo "Saved → $OUT"
            else
                echo "ERROR: run failed (lang=${LANG} method=${METHOD} prompting=${PLANG})" >&2
            fi
        done
    done
done

# ── English: baseline only (no cross-lingual steering) ──────────────────────
echo ""
echo "--- lang=en  method=baseline ---"
OUT="${RESULTS_DIR}/llama3b_en_baseline.json"
if python eval_seahorse_geval.py \
    --model          "$MODEL" \
    --lang           "en" \
    --data-dir       "$SEAHORSE_DIR" \
    --method         "baseline" \
    --prompting-lang "en" \
    --gpu            "$GPU" \
    --out            "$OUT"; then
    echo "Saved → $OUT"
else
    echo "ERROR: run failed (lang=en method=baseline)" >&2
fi

echo ""
echo "All done.  Results in ${RESULTS_DIR}/"
