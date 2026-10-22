#!/usr/bin/env python3
"""Patch an existing SEAHORSE result JSON with per-sample test scores.

Reads the stored optimal hyperparameters (intervention_layers, best_sigma) from
a result JSON, re-runs inference on the test set at sigma=0 and best_sigma, and
writes a 'test_samples' field back into the same JSON.  This avoids redoing the
validation sweep — only the two test-set forward passes are run.

Supports all five eval types:
  seahorse_gptscore_*     → GPTScore with per-language vector
  seahorse_geval_*        → G-eval with per-language vector
  seahorse_avgvec_gptscore_* → GPTScore with language-averaged vector
  seahorse_comet          → COMET with per-language vector
  seahorse_avgvec_comet   → COMET with language-averaged vector

Usage
-----
python rescore_seahorse.py \\
    --json-path results/seahorse_gptscore_llama3b/llama3b_de_vector_en.json \\
    --data-dir  ./data/seahorse \\
    --data-root ./data/ncwm_flores \\
    --gpu       0

# Batch: process every JSON in a directory
python rescore_seahorse.py \\
    --json-dir  results/seahorse_gptscore_llama3b \\
    --data-dir  ./data/seahorse \\
    --data-root ./data/ncwm_flores
"""

import argparse
import json
import os
import sys
from typing import List, Optional


def _infer_type(json_path: str) -> str:
    dirname = os.path.basename(os.path.dirname(os.path.abspath(json_path)))
    if "geval" in dirname:
        return "geval"
    if "avgvec_gptscore" in dirname:
        return "avgvec_gptscore"
    if "gptscore" in dirname:
        return "gptscore"
    if "avgvec_comet" in dirname:
        return "avgvec_comet"
    if "comet" in dirname:
        return "comet"
    raise ValueError(f"Cannot infer eval type from directory name: {dirname!r}")


def _patch_llm(json_path: str, eval_type: str,
               data_dir: str, data_root: str, gpu: str, max_pairs: int):
    """Patch a GPTScore or G-eval result JSON."""
    import torch
    torch.set_grad_enabled(False)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    with open(json_path) as f:
        result = json.load(f)

    model_name      = result["model"]
    lang            = result["lang"]
    method          = result["method"]
    prompting_lang  = result.get("prompting_lang", "en")
    layers          = result["intervention_layers"]
    best_sigma      = float(result["best_sigma"])

    from load_seahorse import load_seahorse, SEAHORSE_LANG2NAME
    from model_and_tok import ModelAndTokenizer

    lang_name = SEAHORSE_LANG2NAME[lang]
    _, test_rows = load_seahorse(data_dir, lang=lang)
    if not test_rows:
        raise RuntimeError(f"No test rows for lang={lang!r}")

    print(f"Loading model {model_name!r}…", flush=True)
    mt = ModelAndTokenizer(model_name)

    if eval_type == "geval":
        from eval_seahorse_geval import (
            build_cache, eval_metrics, compute_steer_params
        )
    else:
        from eval_seahorse_gptscore import (
            build_cache, eval_metrics, compute_steer_params
        )

    if method == "baseline":
        steer_params = None
    else:
        parallel_dir = os.path.join(data_root, f"en-{lang}")
        print(f"Computing steer params for lang={lang!r}…", flush=True)
        steer_params = compute_steer_params(mt, parallel_dir, lang, method, max_pairs)

    print("Building test cache…", flush=True)
    test_cache = build_cache(test_rows, lang_name, prompting_lang, mt, layers)

    print("Scoring baseline (sigma=0)…", flush=True)
    _, base_samples = eval_metrics(
        test_rows, None, 0.0, layers, mt, lang_name, prompting_lang, test_cache
    )

    print(f"Scoring steered (sigma={best_sigma:+.2f})…", flush=True)
    _, steer_samples = eval_metrics(
        test_rows, steer_params, best_sigma, layers, mt, lang_name, prompting_lang, test_cache
    )

    result["test_samples"] = {"baseline": base_samples, "best_sigma": steer_samples}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Patched → {json_path}", flush=True)


def _patch_avgvec_gptscore(json_path: str,
                            data_dir: str, data_root: str, gpu: str, max_pairs: int):
    """Patch a GPTScore avgvec result JSON."""
    import torch
    torch.set_grad_enabled(False)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    with open(json_path) as f:
        result = json.load(f)

    model_name     = result["model"]
    lang           = result["lang"]
    prompting_lang = result.get("prompting_lang", "en")
    layers         = result["intervention_layers"]
    best_sigma     = float(result["best_sigma"])
    fixed_layer    = layers[0]

    from load_seahorse import load_seahorse, SEAHORSE_LANG2NAME
    from model_and_tok import ModelAndTokenizer
    from eval_seahorse_avgvec_gptscore import (
        build_cache, eval_metrics, build_averaged_steer_params
    )

    lang_name = SEAHORSE_LANG2NAME[lang]
    _, test_rows = load_seahorse(data_dir, lang=lang)
    if not test_rows:
        raise RuntimeError(f"No test rows for lang={lang!r}")

    print(f"Building averaged steer params for {model_name!r}…", flush=True)
    steer_params = build_averaged_steer_params(model_name, data_root, max_pairs, fixed_layer, gpu)

    print(f"Loading model {model_name!r} for evaluation…", flush=True)
    mt = ModelAndTokenizer(model_name)

    print("Building test cache…", flush=True)
    test_cache = build_cache(test_rows, lang_name, prompting_lang, mt, layers)

    print("Scoring baseline (sigma=0)…", flush=True)
    _, base_samples = eval_metrics(
        test_rows, None, 0.0, layers, mt, test_cache
    )

    print(f"Scoring steered (sigma={best_sigma:+.2f})…", flush=True)
    _, steer_samples = eval_metrics(
        test_rows, steer_params, best_sigma, layers, mt, test_cache
    )

    result["test_samples"] = {"baseline": base_samples, "best_sigma": steer_samples}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Patched → {json_path}", flush=True)


def _patch_comet(json_path: str,
                 data_dir: str, data_root: str, gpu: str, max_pairs: int):
    """Patch a COMET (vector or avgvec) result JSON."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    with open(json_path) as f:
        result = json.load(f)

    lang       = result["lang"]
    method     = result["method"]
    best_sigma = float(result["best_sigma"])
    is_avgvec  = (method == "avgvec")

    from load_seahorse import load_seahorse
    from eval_seahorse_comet import eval_metrics, load_references, compute_comet_params

    _, test_rows = load_seahorse(data_dir, lang=lang)
    references   = load_references(data_dir)
    if not test_rows:
        raise RuntimeError(f"No test rows for lang={lang!r}")

    if is_avgvec:
        from eval_seahorse_avgvec_comet import build_averaged_comet_params
        print("Building averaged COMET params…", flush=True)
        steer_params = build_averaged_comet_params(data_root, max_pairs)
        comet_method = "vector"
    else:
        parallel_dir = os.path.join(data_root, f"en-{lang}")
        print(f"Computing COMET params for lang={lang!r}…", flush=True)
        steer_params = compute_comet_params(parallel_dir, lang, method, max_pairs)
        comet_method = method

    print("Scoring baseline (sigma=0)…", flush=True)
    _, base_samples = eval_metrics(test_rows, references, None, 0.0, comet_method)

    print(f"Scoring steered (sigma={best_sigma:+.2f})…", flush=True)
    _, steer_samples = eval_metrics(
        test_rows, references, steer_params, best_sigma, comet_method
    )

    result["test_samples"] = {"baseline": base_samples, "best_sigma": steer_samples}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Patched → {json_path}", flush=True)


def patch(json_path: str, data_dir: str, data_root: str, gpu: str, max_pairs: int):
    with open(json_path) as f:
        result = json.load(f)
    if "test_samples" in result:
        print(f"Already patched, skipping: {json_path}", flush=True)
        return

    eval_type = _infer_type(json_path)
    print(f"\n{'='*60}", flush=True)
    print(f"Patching [{eval_type}]: {json_path}", flush=True)
    print(f"{'='*60}", flush=True)

    if eval_type in ("gptscore", "geval"):
        _patch_llm(json_path, eval_type, data_dir, data_root, gpu, max_pairs)
    elif eval_type == "avgvec_gptscore":
        _patch_avgvec_gptscore(json_path, data_dir, data_root, gpu, max_pairs)
    elif eval_type in ("comet", "avgvec_comet"):
        _patch_comet(json_path, data_dir, data_root, gpu, max_pairs)
    else:
        raise ValueError(f"Unknown eval type: {eval_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill test_samples into SEAHORSE result JSONs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json-path",  help="Single result JSON to patch")
    group.add_argument("--json-dir",   help="Directory of result JSONs to patch")
    group.add_argument("--json-paths", nargs="+", metavar="PATH",
                       help="Multiple result JSONs to patch")
    parser.add_argument("--data-dir",  default="./data/seahorse",
                        help="SEAHORSE data directory (validation.tsv / test.tsv)")
    parser.add_argument("--data-root", default="./data/ncwm_flores",
                        help="Parallel data root (contains en-{lang}/ subdirs)")
    parser.add_argument("--gpu",       default="0")
    parser.add_argument("--max-pairs", type=int, default=500)
    args = parser.parse_args()

    if args.json_path:
        paths = [args.json_path]
    elif args.json_paths:
        paths = args.json_paths
    else:
        paths = sorted(
            os.path.join(args.json_dir, f)
            for f in os.listdir(args.json_dir)
            if f.endswith(".json")
        )

    errors = []
    for p in paths:
        try:
            patch(p, args.data_dir, args.data_root, args.gpu, args.max_pairs)
        except Exception as e:
            print(f"ERROR patching {p}: {e}", file=sys.stderr)
            errors.append((p, e))

    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for p, e in errors:
            print(f"  {p}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
