#!/usr/bin/env bash
# Baseline evaluation on English SEAHORSE (lang=en, sigma=0, no steering).
# Runs both GPTScore and G-eval for Llama-3.2-3B and aya-expanse-8b.
#
# Usage:
#   bash run_seahorse_en_baseline.sh [GPU_ID]

set -uo pipefail

GPU="${1:-0}"
SEAHORSE_DIR="./data/seahorse"
RESULTS_DIR="results/seahorse_en_baseline"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " SEAHORSE English baseline (no steering)"
echo " GPU: $GPU"
echo "============================================"

if [[ ! -f "${SEAHORSE_DIR}/validation.tsv" ]]; then
    echo "ERROR: SEAHORSE data not found at ${SEAHORSE_DIR}."
    exit 1
fi

declare -A MODELS
MODELS["llama3b"]="meta-llama/Llama-3.2-3B"
MODELS["aya"]="CohereForAI/aya-expanse-8b"

for MODEL_KEY in llama3b aya; do
    MODEL="${MODELS[$MODEL_KEY]}"

    echo ""
    echo "=== GPTScore  model=${MODEL_KEY} ==="
    OUT="${RESULTS_DIR}/${MODEL_KEY}_en_baseline_gptscore.json"
    if python eval_seahorse_gptscore.py \
            --model          "$MODEL" \
            --lang           en \
            --method         baseline \
            --prompting-lang en \
            --data-dir       "$SEAHORSE_DIR" \
            --gpu            "$GPU" \
            --out            "$OUT"; then
        echo "Saved -> $OUT"
    else
        echo "ERROR: GPTScore baseline failed (model=${MODEL_KEY})" >&2
    fi

    echo ""
    echo "=== G-eval  model=${MODEL_KEY} ==="
    OUT="${RESULTS_DIR}/${MODEL_KEY}_en_baseline_geval.json"
    if python eval_seahorse_geval.py \
            --model          "$MODEL" \
            --lang           en \
            --method         baseline \
            --prompting-lang en \
            --data-dir       "$SEAHORSE_DIR" \
            --gpu            "$GPU" \
            --out            "$OUT"; then
        echo "Saved -> $OUT"
    else
        echo "ERROR: G-eval baseline failed (model=${MODEL_KEY})" >&2
    fi
done

echo ""
echo "All done.  Results in ${RESULTS_DIR}/"
