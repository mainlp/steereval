#!/usr/bin/env bash
# Run HiTZ/BASSE experiments with aya-expanse-8b (GPTScore metric).
#
# Usage:
#   bash run_aya_basse.sh [GPU_ID]
#
# GPU_ID defaults to 0.  Pass a different index to run on another GPU, e.g.:
#   bash run_aya_basse.sh 1
#
# Prerequisites
# -------------
# • Parallel data for Spanish must be present at ./data/ncwm_old/en-es/
#   (train.en + train.es).
# • For Basque, place parallel data at ./data/ncwm_old/en-eu/
#   (train.en + train.eu).  Download Basque–English sentence pairs from
#   OPUS (https://opus.nlpl.eu/) or FLORES-200 (eus_Latn) and convert to
#   that directory layout.  Basque is skipped automatically when absent.
#
# Outputs are written under results/basse_gptscore/.

# -u: error on undefined variables  -o pipefail: propagate pipe errors
# -e is intentionally omitted so a single failing run does not abort the rest.
set -uo pipefail

MODEL="CohereForAI/aya-expanse-8b"
GPU="${1:-0}"
MAX_PAIRS=500
RESULTS_DIR="results/basse_gptscore"
DATA_ROOT="./data/ncwm_flores"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " BASSE evaluation — aya-expanse-8b"
echo " GPU: $GPU"
echo "============================================"

for LANG in eu es; do
    PARALLEL_DIR="${DATA_ROOT}/en-${LANG}"

    if [[ ! -d "$PARALLEL_DIR" ]]; then
        echo ""
        echo "WARNING: parallel data not found for lang='${LANG}' at ${PARALLEL_DIR}."
        echo "         Skipping ${LANG}."
        continue
    fi

    for METHOD in baseline vector map; do
        for PLANG in en tgt; do
            OUT="${RESULTS_DIR}/aya_${LANG}_${METHOD}_${PLANG}.json"
            echo ""
            echo "--- lang=${LANG}  method=${METHOD}  prompting=${PLANG} ---"
            if python eval_basse_gptscore.py \
                --model  "$MODEL" \
                --lang   "$LANG" \
                --parallel-data "$PARALLEL_DIR" \
                --method "$METHOD" \
                --prompting-lang "$PLANG" \
                --max-pairs "$MAX_PAIRS" \
                --gpu    "$GPU" \
                --pairwise \
                --out    "$OUT"; then
                echo "Saved → $OUT"
            else
                echo "ERROR: run failed (lang=${LANG} method=${METHOD} prompting=${PLANG})" >&2
            fi
        done
    done
done

echo ""
echo "All done.  Results in ${RESULTS_DIR}/"
