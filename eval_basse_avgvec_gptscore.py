"""GPTScore evaluation on HiTZ/BASSE with a language-averaged steering vector.

Loads cached XX->EN steering vectors for the five seahorse source languages
(de es ru tr vi), averages them at a fixed layer, and evaluates on the full
BASSE dataset (val + test combined, no split).

Fixed hyperparameters per model
--------------------------------
  Llama-3.2-3B      layer=27  sigma=-2.5
  aya-expanse-8b    layer=31  sigma=-2.5

Prompting language is always English.

Usage
-----
python eval_basse_avgvec_gptscore.py \\
    --model  meta-llama/Llama-3.2-3B \\
    --lang   es \\
    --gpu    0 \\
    --out    results/basse_avgvec_gptscore_llama3b/llama3b_es_avgvec_en.json
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from constants import BASSE_DIMS, BASSE_LANG2NAME, MAX_LENGTH
from load_basse import load_basse
from model_and_tok import ModelAndTokenizer
from utils import (
    apply_steering,
    conditional_avg_logprob_with_patch,
    get_residual_last_token,
    get_layer_name,
    get_residual_mean_over_tokens,
    safe_corr,
    tokenize_input,
)

# ── fixed hyperparameters ─────────────────────────────────────────────────────

SOURCE_LANGS = ["de", "es", "ru", "tr", "vi"]

FIXED_LAYER: Dict[str, int] = {
    "meta-llama/Llama-3.2-3B":   27,
    "CohereForAI/aya-expanse-8b": 31,
}
FIXED_SIGMA = -2.5

# ── prompt templates (same as eval_basse_gptscore.py) ────────────────────────

_EN_PROMPTS: Dict[str, str] = {
    "coherence":    "Generate a coherent summary for the following text in {lang}: {article}\nSummary:",
    "consistency":  "Generate a factually consistent summary for the following text in {lang}: {article}\nSummary:",
    "fluency":      "Generate a fluent summary for the following text in {lang}: {article}\nSummary:",
    "relevance":    "Generate a relevant summary for the following text in {lang}: {article}\nSummary:",
    "5W1H":         "Generate a who/what/when/where/why/how summary for the following text in {lang}: {article}\nSummary:",
}


# ── vector loading ────────────────────────────────────────────────────────────

def _vec_cache_path(lang: str, model_name: str, max_pairs: int) -> str:
    safe = model_name.replace("/", "_")
    return os.path.join("cache", "seahorse", f"{lang}_vector_{max_pairs}_{safe}_residual.npy")


def _load_or_compute(lang: str, model_name: str, data_root: str,
                     max_pairs: int, mt=None) -> Optional[List[np.ndarray]]:
    path = _vec_cache_path(lang, model_name, max_pairs)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        print(f"  [{lang}] loaded from {path}", flush=True)
        return list(data)
    if mt is None:
        return None
    parallel_dir = os.path.join(data_root, f"en-{lang}")
    if not os.path.isdir(parallel_dir):
        print(f"  [{lang}] parallel data not found at {parallel_dir}, skipping.", flush=True)
        return None

    en_path = os.path.join(parallel_dir, "train.en")
    xx_path = os.path.join(parallel_dir, f"train.{lang}")
    with open(en_path, encoding="utf-8") as f:
        en_lines = [l.strip() for l in f]
    with open(xx_path, encoding="utf-8") as f:
        xx_lines = [l.strip() for l in f]

    EN_reps, XX_reps = [], []
    i, cap = 0, max_pairs
    while len(EN_reps) < cap and i < min(len(en_lines), len(xx_lines)):
        if not en_lines[i] or not xx_lines[i]:
            cap += 1
            i += 1
            continue
        for text, store in ((en_lines[i], EN_reps), (xx_lines[i], XX_reps)):
            enc = mt.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            rep = get_residual_mean_over_tokens(
                mt.model, enc["input_ids"].to(mt.model.device),
                mt.num_layers, mt.model_name,
                enc.get("attention_mask", None).to(mt.model.device),
            )
            store.append(rep)
        i += 1
        if i % 100 == 0:
            print(f"    encoded {len(EN_reps)} pairs...", flush=True)

    EN = np.stack(EN_reps)
    XX = np.stack(XX_reps)
    params = [EN[:, l, :].mean(0) - XX[:, l, :].mean(0) for l in range(mt.num_layers)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.array(params, dtype=object), allow_pickle=True)
    print(f"  [{lang}] saved -> {path}", flush=True)
    return params


def build_averaged_steer_params(
    model_name: str, data_root: str, max_pairs: int, fixed_layer: int, gpu: str
) -> List[np.ndarray]:
    """Average the XX->EN steering vector at fixed_layer across all source languages."""
    per_lang = {l: _load_or_compute(l, model_name, data_root, max_pairs) for l in SOURCE_LANGS}
    missing = [l for l in SOURCE_LANGS if per_lang[l] is None]

    if missing:
        import torch as _torch
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"  Loading model to compute vectors for: {missing}...", flush=True)
        mt = ModelAndTokenizer(model_name)
        _torch.set_grad_enabled(False)
        for lang in missing:
            per_lang[lang] = _load_or_compute(lang, model_name, data_root, max_pairs, mt)
        del mt
        _torch.cuda.empty_cache()

    available = [v for v in per_lang.values() if v is not None]
    if not available:
        raise RuntimeError("No steering vectors could be loaded or computed.")

    layer_vecs = [np.asarray(v[fixed_layer], dtype=np.float32) for v in available]
    avg_vec = np.mean(layer_vecs, axis=0)
    print(f"  Averaged {len(layer_vecs)} language vectors at layer {fixed_layer}.", flush=True)

    n_layers = len(available[0])
    dummy = np.zeros_like(avg_vec)
    steer_params: List[np.ndarray] = [dummy] * n_layers
    steer_params[fixed_layer] = avg_vec
    return steer_params


# ── prompt cache and evaluation ───────────────────────────────────────────────

def build_cache(rows, lang_name, mt, intervention_layers):
    cache = {}
    for row in rows:
        key = (row["article"], row["dimension"])
        if key in cache:
            continue
        prompt = _EN_PROMPTS[row["dimension"]].format(lang=lang_name, article=row["article"])
        input_ids, attention_mask = tokenize_input(prompt, mt)
        residual_last = get_residual_last_token(
            mt.model, input_ids, mt.num_layers, mt.model_name, attention_mask
        )
        ntoks = input_ids.shape[1]
        states = [(ntoks - 1, get_layer_name(l, mt.model_name)) for l in intervention_layers]
        cache[key] = (input_ids, residual_last, states)
    return cache


def eval_metrics(rows, steer_params, sigma, intervention_layers,
                 mt, cache) -> Dict[str, Tuple[float, float, float]]:
    """Returns {dim: (spearman, kendall, pearson)}."""
    by_dim = defaultdict(lambda: ([], []))
    for row in rows:
        key = (row["article"], row["dimension"])
        prompt_ids, residual_last, states = cache[key]
        steered = apply_steering(residual_last, steer_params, sigma, intervention_layers)
        score = conditional_avg_logprob_with_patch(mt, prompt_ids, row["candidate"], steered, states)
        by_dim[row["dimension"]][0].append(score)
        by_dim[row["dimension"]][1].append(row["human_score"])
    return {dim: safe_corr(scores, labels)
            for dim, (scores, labels) in by_dim.items()}


def _mean_metric(dim_metrics, idx):
    vals = [v[idx] for v in dim_metrics.values() if not np.isnan(v[idx])]
    return float(np.mean(vals)) if vals else float("nan")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPTScore on BASSE with language-averaged steering vector"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang", choices=list(BASSE_LANG2NAME), required=True,
                        help="BASSE language: es or eu")
    parser.add_argument("--data-root", default="./data/ncwm_flores",
                        help="Root for parallel data used to build steering vectors")
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.model not in FIXED_LAYER:
        raise ValueError(
            f"No fixed layer configured for '{args.model}'. "
            f"Known: {list(FIXED_LAYER)}"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_grad_enabled(False)

    fixed_layer = FIXED_LAYER[args.model]
    lang_name   = BASSE_LANG2NAME[args.lang]
    print(f"Language:     {lang_name} ({args.lang})", flush=True)
    print(f"Fixed layer:  {fixed_layer}", flush=True)
    print(f"Fixed sigma:  {FIXED_SIGMA}", flush=True)

    # ── Averaged vector ──────────────────────────────────────────────────────
    print("\nLoading / computing source language vectors...", flush=True)
    steer_params = build_averaged_steer_params(
        args.model, args.data_root, args.max_pairs, fixed_layer, args.gpu
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print("\nLoading model for evaluation...", flush=True)
    mt = ModelAndTokenizer(args.model)
    intervention_layers = [fixed_layer]

    # ── Data: full dataset (val + test combined) ─────────────────────────────
    val_rows, test_rows = load_basse(args.lang, val_frac=0.5)
    rows = val_rows + test_rows
    if not rows:
        raise RuntimeError(f"No rows loaded for lang='{args.lang}'.")
    print(f"Total rows: {len(rows)}", flush=True)

    print(f"\nBuilding prompt cache ({len(rows)} rows)...", flush=True)
    cache = build_cache(rows, lang_name, mt, intervention_layers)

    # ── Baseline ─────────────────────────────────────────────────────────────
    print("\n=== Baseline (sigma=0.0) ===", flush=True)
    base_metrics = eval_metrics(rows, None, 0.0, intervention_layers, mt, cache)
    for dim, (sp, kd, pe) in base_metrics.items():
        print(f"  [{dim}] Spearman={sp:.4f}  Kendall={kd:.4f}  Pearson={pe:.4f}")
    base_mean_sp = _mean_metric(base_metrics, 0)
    base_mean_pe = _mean_metric(base_metrics, 2)
    print(f"  mean_Spearman={base_mean_sp:.4f}  mean_Pearson={base_mean_pe:.4f}")

    # ── Steered ──────────────────────────────────────────────────────────────
    print(f"\n=== Averaged vector (sigma={FIXED_SIGMA:+.2f}) ===", flush=True)
    test_metrics = eval_metrics(rows, steer_params, FIXED_SIGMA, intervention_layers, mt, cache)
    for dim, (sp, kd, pe) in test_metrics.items():
        print(f"  [{dim}] Spearman={sp:.4f}  Kendall={kd:.4f}  Pearson={pe:.4f}")
    test_mean_sp = _mean_metric(test_metrics, 0)
    test_mean_pe = _mean_metric(test_metrics, 2)
    print(f"  mean_Spearman={test_mean_sp:.4f}  mean_Pearson={test_mean_pe:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "model":               args.model,
        "lang":                args.lang,
        "method":              "avgvec",
        "prompting_lang":      "en",
        "source_langs":        SOURCE_LANGS,
        "intervention_layers": intervention_layers,
        "best_sigma":          float(FIXED_SIGMA),
        "test_baseline": {
            "sigma":          0.0,
            "per_dim":        {d: {"spearman": sp, "kendall": kd, "pearson": pe}
                               for d, (sp, kd, pe) in base_metrics.items()},
            "mean_spearman":  base_mean_sp,
            "mean_pearson":   base_mean_pe,
        },
        "test_best_sigma": {
            "sigma":          float(FIXED_SIGMA),
            "per_dim":        {d: {"spearman": sp, "kendall": kd, "pearson": pe}
                               for d, (sp, kd, pe) in test_metrics.items()},
            "mean_spearman":  test_mean_sp,
            "mean_pearson":   test_mean_pe,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
