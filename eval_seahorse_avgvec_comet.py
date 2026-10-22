"""COMET SEAHORSE evaluation with a language-averaged steering vector (avgvec).

Loads cached COMET pooling-space steering vectors for all five source languages,
averages them, and evaluates with a fixed sigma — testing whether a universal
COMET vector generalises across languages without per-language tuning.

Usage
-----
python eval_seahorse_avgvec_comet.py \\
    --lang      de \\
    --data-dir  ./data/seahorse \\
    --data-root ./data/ncwm_flores \\
    --sigma     -2.5 \\
    --gpu       0 \\
    --out       results/seahorse_avgvec_comet/de_avgvec.json
"""

import argparse
import json
import os

import numpy as np
import torch

from constants import MAX_PAIRS
from load_seahorse import SEAHORSE_LANG2NAME, load_seahorse
from eval_seahorse_comet import (
    compute_comet_params,
    eval_metrics,
    load_references,
    _mean_auc,
    _mean_pearson,
)

SOURCE_LANGS = ["de", "es", "ru", "tr", "vi"]
DEFAULT_SIGMA = -2.5


def build_averaged_comet_params(data_root: str, max_pairs: int) -> np.ndarray:
    """Average per-language COMET vector params across all source languages."""
    vecs = []
    for lang in SOURCE_LANGS:
        parallel_dir = os.path.join(data_root, f"en-{lang}")
        params = compute_comet_params(parallel_dir, lang, "vector", max_pairs)
        vecs.append(np.asarray(params, dtype=np.float32))
    avg = np.mean(vecs, axis=0)
    print(f"Averaged {len(vecs)} language COMET vectors.", flush=True)
    return avg


def main():
    parser = argparse.ArgumentParser(
        description="COMET SEAHORSE evaluation with language-averaged steering vector"
    )
    parser.add_argument("--lang", choices=["de", "es", "ru", "tr", "vi"], required=True,
                        help="SEAHORSE evaluation language")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with SEAHORSE validation.tsv / test.tsv")
    parser.add_argument("--data-root", default="./data/ncwm_flores",
                        help="Root for parallel data (en-{lang}/ subdirs)")
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help=f"Fixed steering sigma (default: {DEFAULT_SIGMA})")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_grad_enabled(False)

    lang_name = SEAHORSE_LANG2NAME[args.lang]
    print(f"Language:    {lang_name} ({args.lang})", flush=True)
    print(f"Fixed sigma: {args.sigma:+.2f}", flush=True)

    # ── Build averaged vector ─────────────────────────────────────────────────
    print("\nLoading / computing source language COMET vectors...", flush=True)
    steer_params = build_averaged_comet_params(args.data_root, args.max_pairs)

    # ── Data ─────────────────────────────────────────────────────────────────
    _, test_rows = load_seahorse(args.data_dir, lang=args.lang)
    if not test_rows:
        raise RuntimeError(f"No usable test rows for lang='{args.lang}'.")

    print("Loading SEAHORSE references…", flush=True)
    references = load_references(args.data_dir)
    print(f"  Loaded {len(references)} references", flush=True)

    # ── Baseline (sigma=0) ────────────────────────────────────────────────────
    print("\n=== Test baseline (sigma=0.0) ===", flush=True)
    base_metrics, base_samples = eval_metrics(test_rows, references, None, 0.0, "vector")
    base_mean_auc = _mean_auc(base_metrics)
    base_mean_pe  = _mean_pearson(base_metrics)
    for dim, (pe, auc) in sorted(base_metrics.items()):
        print(f"  [{dim}] Pearson={pe:.4f}  AUC={auc:.4f}")
    print(f"  mean_AUC={base_mean_auc:.4f}  mean_Pearson={base_mean_pe:.4f}")

    # ── Steered (fixed sigma) ─────────────────────────────────────────────────
    print(f"\n=== Test averaged vector (sigma={args.sigma:+.2f}) ===", flush=True)
    test_metrics, test_samples = eval_metrics(test_rows, references, steer_params, args.sigma, "vector")
    test_mean_auc = _mean_auc(test_metrics)
    test_mean_pe  = _mean_pearson(test_metrics)
    for dim, (pe, auc) in sorted(test_metrics.items()):
        print(f"  [{dim}] Pearson={pe:.4f}  AUC={auc:.4f}")
    print(f"  mean_AUC={test_mean_auc:.4f}  mean_Pearson={test_mean_pe:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "lang":          args.lang,
        "method":        "avgvec",
        "max_pairs":     args.max_pairs,
        "source_langs":  SOURCE_LANGS,
        "best_sigma":    float(args.sigma),
        "test_baseline": {
            "sigma":        0.0,
            "per_dim":      {d: {"pearson": pe, "auc": auc}
                             for d, (pe, auc) in base_metrics.items()},
            "mean_auc":     base_mean_auc,
            "mean_pearson": base_mean_pe,
        },
        "test_best_sigma": {
            "sigma":        float(args.sigma),
            "per_dim":      {d: {"pearson": pe, "auc": auc}
                             for d, (pe, auc) in test_metrics.items()},
            "mean_auc":     test_mean_auc,
            "mean_pearson": test_mean_pe,
        },
        "test_samples": {
            "baseline":   base_samples,
            "best_sigma": test_samples,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
