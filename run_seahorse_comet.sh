#!/usr/bin/env bash
# SEAHORSE COMET evaluation with cross-lingual pooling-space intervention.
#
# Unbabel/wmt22-comet-da is steered in its pooling space using a mean-diff
# vector (or linear map) learned from parallel sentence pairs.  Sigma is
# swept on SEAHORSE's validation split; best sigma is used for test.
#
# Usage:
#   bash run_seahorse_comet.sh [GPU_ID]
#
# Prerequisites
# -------------
# • SEAHORSE TSV files (validation.tsv, test.tsv) at $SEAHORSE_DIR.
#   Download from: https://storage.googleapis.com/seahorse-public/seahorse_data.zip
# • Parallel data for each language at $DATA_ROOT/en-{lang}/ (train.en + train.{lang}).
#   Languages without parallel data are skipped automatically.
#
# Outputs are written under results/seahorse_comet/.

set -uo pipefail

GPU="${1:-0}"
shift || true
LANGS="${*:-de es ru tr vi}"
MAX_PAIRS=500
SEAHORSE_DIR="./data/seahorse"
DATA_ROOT="./data/ncwm_flores"
RESULTS_DIR="results/seahorse_comet"
AVGVEC_DIR="results/seahorse_avgvec_comet"

mkdir -p "$RESULTS_DIR" "$AVGVEC_DIR"

echo "============================================"
echo " SEAHORSE COMET evaluation"
echo " GPU: $GPU"
echo "============================================"

if [[ ! -f "${SEAHORSE_DIR}/validation.tsv" ]]; then
    echo "ERROR: SEAHORSE data not found at ${SEAHORSE_DIR}."
    echo "       Download from https://storage.googleapis.com/seahorse-public/seahorse_data.zip"
    exit 1
fi

# ── Non-English languages: vector + map steering ─────────────────────────────
for LANG in $LANGS; do
    PARALLEL_DIR="${DATA_ROOT}/en-${LANG}"

    if [[ ! -d "$PARALLEL_DIR" ]]; then
        echo ""
        echo "WARNING: parallel data not found for lang='${LANG}' at ${PARALLEL_DIR}."
        echo "         Skipping ${LANG}."
        continue
    fi

    for METHOD in vector; do
        OUT="${RESULTS_DIR}/${LANG}_${METHOD}.json"
        echo ""
        echo "--- lang=${LANG}  method=${METHOD} ---"
        if python eval_seahorse_comet.py \
            --lang           "$LANG" \
            --data-dir       "$SEAHORSE_DIR" \
            --parallel-data  "$PARALLEL_DIR" \
            --method         "$METHOD" \
            --max-pairs      "$MAX_PAIRS" \
            --gpu            "$GPU" \
            --out            "$OUT"; then
            echo "Saved → $OUT"
        else
            echo "ERROR: run failed (lang=${LANG} method=${METHOD})" >&2
        fi
    done
done

# ── Avgvec: language-averaged zero-shot steering ─────────────────────────────
for LANG in $LANGS; do
    OUT="${AVGVEC_DIR}/${LANG}_avgvec.json"
    echo ""
    echo "--- lang=${LANG}  method=avgvec ---"
    if python eval_seahorse_avgvec_comet.py \
        --lang       "$LANG" \
        --data-dir   "$SEAHORSE_DIR" \
        --data-root  "$DATA_ROOT" \
        --max-pairs  "$MAX_PAIRS" \
        --gpu        "$GPU" \
        --out        "$OUT"; then
        echo "Saved → $OUT"
    else
        echo "ERROR: run failed (lang=${LANG} method=avgvec)" >&2
    fi
done

# ── English: baseline only (no cross-lingual steering needed) ────────────────
echo ""
echo "--- lang=en  method=baseline ---"
OUT="${RESULTS_DIR}/en_baseline.json"
if python eval_seahorse_comet.py \
    --lang           "en" \
    --data-dir       "$SEAHORSE_DIR" \
    --method         "baseline" \
    --gpu            "$GPU" \
    --out            "$OUT"; then
    echo "Saved → $OUT"
else
    echo "ERROR: run failed (lang=en method=baseline)" >&2
fi

echo ""
echo "All done.  Results in ${RESULTS_DIR}/  avgvec in ${AVGVEC_DIR}/"
