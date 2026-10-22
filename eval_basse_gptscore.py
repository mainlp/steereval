"""
GPTScore-based INCLINE evaluation on HiTZ/BASSE.

Computes the log-likelihood of each candidate summary conditioned on a
dimension-specific prompt (GPTScore), with optional cross-lingual MLP steering
(vector or map) learned from parallel EN↔XX text.

Sigma is tuned on a held-out validation split; the best sigma is then reported
on the test split.  Pairwise ranking accuracy across system outputs for the same
document is also available via --pairwise.

Usage
-----
# Spanish, vector steering, target-language prompts:
python eval_basse_gptscore.py \
    --model CohereForAI/aya-expanse-8b \
    --lang es \
    --parallel-data ./data/ncwm_old/en-es \
    --method vector \
    --prompting-lang tgt \
    --gpu 0 \
    --out results/basse_gptscore/aya_es_vector.json

# Basque, no steering (baseline):
python eval_basse_gptscore.py \
    --model CohereForAI/aya-expanse-8b \
    --lang eu \
    --parallel-data ./data/ncwm_old/en-eu \
    --method baseline \
    --gpu 0 \
    --out results/basse_gptscore/aya_eu_baseline.json
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from constants import (
    BASSE_DIMS,
    BASSE_LANG2NAME,
    LANG2PROMPT_5w1h,
    LANG2PROMPT_coh,
    LANG2PROMPT_con,
    LANG2PROMPT_flu,
    LANG2PROMPT_rel,
    MAX_LENGTH,
    SIGMAS_BY_MODE,
)
from load_basse import load_basse
from map_utils import load_layer_map, save_layer_map
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
from vector_utils import load_vectors, save_vectors

# ── prompt templates ───────────────────────────────────────────────────────

_LANG_PROMPTS: Dict[str, Dict[str, str]] = {
    "coherence": LANG2PROMPT_coh,
    "consistency": LANG2PROMPT_con,
    "fluency": LANG2PROMPT_flu,
    "relevance": LANG2PROMPT_rel,
    "5W1H": LANG2PROMPT_5w1h,
}

_EN_PROMPTS: Dict[str, str] = {
    "coherence": "Generate a coherent summary for the following text in {lang}: {article}\nSummary:",
    "consistency": "Generate a factually consistent summary for the following text in {lang}: {article}\nSummary:",
    "fluency": "Generate a fluent summary for the following text in {lang}: {article}\nSummary:",
    "relevance": "Generate a relevant summary for the following text in {lang}: {article}\nSummary:",
    "5W1H": "Generate a who/what/when/where/why/how summary for the following text in {lang}: {article}\nSummary:",
}


def build_prompt(article: str, dimension: str, lang_name: str, prompting_lang: str) -> str:
    if prompting_lang == "en":
        return _EN_PROMPTS[dimension].format(lang=lang_name, article=article)
    return _LANG_PROMPTS[dimension][lang_name].format(article=article)


# ── steering parameter computation ────────────────────────────────────────


def _cache_path(parallel_dir: str, lang_key: str, method: str, max_pairs: int,
                model_name: str) -> str:
    safe_model = model_name.replace("/", "_")
    return os.path.join(
        "cache", "basse", f"{lang_key}_{method}_{max_pairs}_{safe_model}_residual.npy"
    )


def compute_steer_params(
    mt: ModelAndTokenizer,
    parallel_dir: str,
    lang_key: str,
    method: str,
    max_pairs: int,
) -> List:
    """Return cached or freshly computed steering vectors / maps."""
    path = _cache_path(parallel_dir, lang_key, method, max_pairs, mt.model_name)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        print(f"Loaded steer params from {path}", flush=True)
        return list(data)

    en_path = os.path.join(parallel_dir, "train.en")
    xx_path = os.path.join(parallel_dir, f"train.{lang_key}")
    if not os.path.exists(en_path) or not os.path.exists(xx_path):
        raise FileNotFoundError(
            f"Parallel data not found. Expected:\n  {en_path}\n  {xx_path}\n"
            "Provide EN↔XX parallel sentences (one per line) in the directory."
        )

    with open(en_path, encoding="utf-8") as f:
        en_lines = [l.strip() for l in f]
    with open(xx_path, encoding="utf-8") as f:
        xx_lines = [l.strip() for l in f]

    EN_reps: List[np.ndarray] = []
    XX_reps: List[np.ndarray] = []
    i, cap = 0, max_pairs
    while len(EN_reps) < cap and i < min(len(en_lines), len(xx_lines)):
        if not en_lines[i] or not xx_lines[i]:
            cap += 1
            i += 1
            continue
        for text, store in ((en_lines[i], EN_reps), (xx_lines[i], XX_reps)):
            enc = mt.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
            )
            rep = get_residual_mean_over_tokens(
                mt.model,
                enc["input_ids"].to(mt.model.device),
                mt.num_layers,
                mt.model_name,
                enc.get("attention_mask", None).to(mt.model.device),
            )
            store.append(rep)
        i += 1
        if i % 100 == 0:
            print(f"  encoded {len(EN_reps)} pairs…", flush=True)

    EN = np.stack(EN_reps)   # (N, L, H)
    XX = np.stack(XX_reps)

    if method == "vector":
        params = [EN[:, l, :].mean(0) - XX[:, l, :].mean(0) for l in range(mt.num_layers)]
    else:  # map
        params = []
        for l in range(mt.num_layers):
            W, *_ = np.linalg.lstsq(XX[:, l, :], EN[:, l, :], rcond=None)
            params.append(W.astype(np.float32))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.array(params, dtype=object), allow_pickle=True)
    print(f"Saved steer params to {path}", flush=True)
    return params


# ── prompt caching ─────────────────────────────────────────────────────────


def build_cache(rows, lang_name, prompting_lang, mt, intervention_layers):
    """Pre-compute per-(article, dimension) MLP state and patch spec."""
    cache = {}
    for row in rows:
        key = (row["article"], row["dimension"])
        if key in cache:
            continue
        prompt = build_prompt(row["article"], row["dimension"], lang_name, prompting_lang)
        input_ids, attention_mask = tokenize_input(prompt, mt)
        residual_last = get_residual_last_token(
            mt.model, input_ids, mt.num_layers, mt.model_name, attention_mask
        )
        ntoks = input_ids.shape[1]
        states = [
            (ntoks - 1, get_layer_name(l, mt.model_name))
            for l in intervention_layers
        ]
        cache[key] = (input_ids, residual_last, states)
    return cache


# ── evaluation ─────────────────────────────────────────────────────────────


def eval_correlation(
    rows: List[dict],
    steer_params: Optional[List],
    sigma: float,
    intervention_layers: List[int],
    mt: ModelAndTokenizer,
    lang_name: str,
    prompting_lang: str,
    cache: Optional[dict] = None,
) -> Dict[str, Tuple[float, float, float]]:
    if cache is None:
        cache = build_cache(rows, lang_name, prompting_lang, mt, intervention_layers)

    by_dim: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))
    for row in rows:
        key = (row["article"], row["dimension"])
        prompt_ids, residual_last, states = cache[key]
        steered = apply_steering(residual_last, steer_params, sigma, intervention_layers)
        score = conditional_avg_logprob_with_patch(
            mt, prompt_ids, row["candidate"], steered, states
        )
        by_dim[row["dimension"]][0].append(score)
        by_dim[row["dimension"]][1].append(row["human_score"])

    return {
        dim: safe_corr(scores, humans)
        for dim, (scores, humans) in by_dim.items()
    }


def eval_pairwise_accuracy(
    rows: List[dict],
    steer_params: Optional[List],
    sigma: float,
    intervention_layers: List[int],
    mt: ModelAndTokenizer,
    lang_name: str,
    prompting_lang: str,
) -> Dict[str, Tuple[int, int]]:
    """Pairwise accuracy: check if GPTScore agrees with human ranking across
    different model outputs for the same (article, dimension) pair."""
    cache = build_cache(rows, lang_name, prompting_lang, mt, intervention_layers)

    groups: Dict[Tuple, List] = defaultdict(list)
    for row in rows:
        groups[(row["article"], row["dimension"])].append(row)

    by_dim: Dict[str, List[bool]] = defaultdict(list)
    for (article, dim), items in groups.items():
        if len(items) < 2:
            continue
        key = (article, dim)
        prompt_ids, residual_last, states = cache[key]
        steered = apply_steering(residual_last, steer_params, sigma, intervention_layers)
        scored = []
        for r in items:
            s = conditional_avg_logprob_with_patch(
                mt, prompt_ids, r["candidate"], steered, states
            )
            scored.append((s, r["human_score"]))
        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                mi, hi = scored[i]
                mj, hj = scored[j]
                if hi == hj:
                    continue
                by_dim[dim].append((mi > mj) == (hi > hj))

    return {dim: (sum(v), len(v)) for dim, v in by_dim.items()}


# ── per-layer sigma tuning ─────────────────────────────────────────────────


def select_best_layer(
    val_rows: List[dict],
    steer_params: Optional[List],
    candidate_layers: List[int],
    sigma_list: List[float],
    mt: "ModelAndTokenizer",
    lang_name: str,
    prompting_lang: str,
    val_cache: dict,
) -> Tuple[int, float, dict]:
    """For each candidate layer, sweep sigma with only that layer active.
    Returns (best_layer, best_sigma, per_layer_results) where best is the
    (layer, sigma) pair with the highest mean val Spearman."""
    per_layer_results: Dict[int, Tuple[float, float]] = {}  # layer -> (best_sigma, best_sp)
    for l in candidate_layers:
        layer_best_sigma, layer_best_sp = 0.0, -float("inf")
        for sigma in sigma_list:
            dim_corrs = eval_correlation(
                val_rows, steer_params, sigma, [l],
                mt, lang_name, prompting_lang, val_cache,
            )
            mean_sp = float(np.mean([sp for sp, _, _ in dim_corrs.values()]))
            print(f"    sigma={sigma:+.2f}  mean_Spearman={mean_sp:.4f}", flush=True)
            if mean_sp > layer_best_sp:
                layer_best_sp = mean_sp
                layer_best_sigma = sigma
        per_layer_results[l] = (layer_best_sigma, layer_best_sp)
        print(f"  layer {l}: best_sigma={layer_best_sigma:+.2f}  val_Spearman={layer_best_sp:.4f}", flush=True)
    best_layer = max(per_layer_results, key=lambda l: per_layer_results[l][1])
    best_sigma, _ = per_layer_results[best_layer]
    return best_layer, best_sigma, per_layer_results


# ── main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="GPTScore INCLINE evaluation on HiTZ/BASSE"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--lang", choices=["es", "eu"], default="es",
        help="BASSE language split (es=Spanish, eu=Basque)",
    )
    parser.add_argument(
        "--parallel-data", required=True,
        help="Directory with train.en and train.<lang> for computing steering params",
    )
    parser.add_argument(
        "--method", choices=["baseline", "vector", "map"], default="vector",
    )
    parser.add_argument(
        "--prompting-lang", choices=["en", "tgt"], default="tgt",
        help="en: English prompts; tgt: target-language prompts",
    )
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--val-frac", type=float, default=0.5)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--pairwise", action="store_true",
        help="Also compute pairwise ranking accuracy on the test set",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="Subset of layers to intervene on; default: all layers",
    )
    parser.add_argument("--out", default="results/basse_gptscore/results.json")
    parser.add_argument(
        "--select-layer", action="store_true",
        help="Sweep each candidate layer individually on val set, pick the single best (layer, sigma) pair",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_grad_enabled(False)

    lang_name = BASSE_LANG2NAME[args.lang]
    print(f"Language: {lang_name} ({args.lang})", flush=True)
    print(f"Method:   {args.method}", flush=True)

    mt = ModelAndTokenizer(args.model)
    intervention_layers = (
        list(range(mt.num_layers)) if args.layers is None else sorted(args.layers)
    )

    # Steering parameters
    if args.method == "baseline":
        steer_params = None
    else:
        print(f"\nComputing {args.method} steer params from {args.parallel_data}…", flush=True)
        steer_params = compute_steer_params(
            mt, args.parallel_data, args.lang, args.method, args.max_pairs
        )

    # Load BASSE
    val_rows, test_rows = load_basse(args.lang, args.val_frac)
    sigma_list = SIGMAS_BY_MODE[args.method]
    print(f"Sigma sweep: {sigma_list}", flush=True)

    # ── Val sweep ──────────────────────────────────────────────────────────
    print("  Building val prompt cache…", flush=True)
    val_cache = build_cache(val_rows, lang_name, args.prompting_lang, mt, intervention_layers)

    if args.select_layer:
        print("\n=== Layer selection sweep (layers tested individually) ===", flush=True)
        best_layer, best_sigma, layer_sweep_results = select_best_layer(
            val_rows, steer_params, intervention_layers, sigma_list,
            mt, lang_name, args.prompting_lang, val_cache,
        )
        print(f"\nSelected layer {best_layer} with sigma={best_sigma:+.2f}", flush=True)
        # Test uses only the single selected layer
        intervention_layers = [best_layer]
        val_results = None
        sigma_by_dim = None
    else:
        layer_sweep_results = None
        print("\n=== Val sigma sweep ===", flush=True)
        val_results: Dict[float, dict] = {}
        for sigma in sigma_list:
            dim_corrs = eval_correlation(
                val_rows, steer_params, sigma, intervention_layers,
                mt, lang_name, args.prompting_lang, val_cache,
            )
            mean_sp = float(np.mean([sp for sp, _, _ in dim_corrs.values()]))
            print(f"  sigma={sigma:+.2f}  mean_Spearman={mean_sp:.4f}")
            for dim, (sp, kd, pe) in dim_corrs.items():
                print(f"    [{dim}] Sp={sp:.4f} Kd={kd:.4f} Pe={pe:.4f}")
            val_results[sigma] = {
                "mean_spearman": mean_sp,
                "per_dim": {
                    d: {"spearman": sp, "kendall": kd, "pearson": pe}
                    for d, (sp, kd, pe) in dim_corrs.items()
                },
            }
        best_sigma = max(val_results, key=lambda s: val_results[s]["mean_spearman"])
        print(f"\nBest sigma (val): {best_sigma:+.2f}  "
              f"mean_Spearman={val_results[best_sigma]['mean_spearman']:.4f}", flush=True)
        sigma_by_dim: Dict[str, float] = {}
        for sigma, entry in val_results.items():
            for dim, corrs in entry["per_dim"].items():
                if dim not in sigma_by_dim or corrs["spearman"] > val_results[sigma_by_dim[dim]]["per_dim"][dim]["spearman"]:
                    sigma_by_dim[dim] = sigma
        print("\nPer-dimension best sigma (val):", flush=True)
        for dim, sigma in sigma_by_dim.items():
            sp = val_results[sigma]["per_dim"][dim]["spearman"]
            print(f"  [{dim}] sigma={sigma:+.2f}  val_Spearman={sp:.4f}")

    # ── Test evaluation ────────────────────────────────────────────────────
    print("\nBuilding test prompt cache…", flush=True)
    test_cache = build_cache(test_rows, lang_name, args.prompting_lang, mt, intervention_layers)

    print("\n=== Test baseline (sigma=0.0) ===", flush=True)
    base_corrs = eval_correlation(
        test_rows, None, 0.0, intervention_layers, mt, lang_name, args.prompting_lang,
        test_cache,
    )
    base_mean = float(np.mean([sp for sp, _, _ in base_corrs.values()]))
    for dim, (sp, kd, pe) in base_corrs.items():
        print(f"  [{dim}] Sp={sp:.4f} Kd={kd:.4f} Pe={pe:.4f}")
    print(f"  mean_Spearman={base_mean:.4f}")

    # ── Test: best sigma ───────────────────────────────────────────────────
    sigma_label = (
        "per-layer" if isinstance(best_sigma, dict)
        else f"{best_sigma:+.2f}"
    )
    print(f"\n=== Test best sigma ({sigma_label}) ===", flush=True)
    test_corrs = eval_correlation(
        test_rows, steer_params, best_sigma, intervention_layers,
        mt, lang_name, args.prompting_lang, test_cache,
    )
    test_mean = float(np.mean([sp for sp, _, _ in test_corrs.values()]))
    for dim, (sp, kd, pe) in test_corrs.items():
        print(f"  [{dim}] Sp={sp:.4f} Kd={kd:.4f} Pe={pe:.4f}")
    print(f"  mean_Spearman={test_mean:.4f}")

    # ── Optional pairwise accuracy ─────────────────────────────────────────
    pairwise_results = None
    if args.pairwise:
        print(f"\n=== Pairwise accuracy on test ({sigma_label}) ===", flush=True)
        pairwise_results = eval_pairwise_accuracy(
            test_rows, steer_params, best_sigma, intervention_layers,
            mt, lang_name, args.prompting_lang,
        )
        for dim, (correct, total) in pairwise_results.items():
            acc = correct / total if total > 0 else 0.0
            print(f"  [{dim}] Accuracy={acc:.4f} ({correct}/{total})")

    # ── Save ───────────────────────────────────────────────────────────────
    output = {
        "model": args.model,
        "lang": args.lang,
        "method": args.method,
        "prompting_lang": args.prompting_lang,
        "select_layer_mode": args.select_layer,
        "intervention_layers": intervention_layers,
        "best_sigma": float(best_sigma),
        "test_baseline": {
            "sigma": 0.0,
            "per_dim": {
                d: {"spearman": sp, "kendall": kd, "pearson": pe}
                for d, (sp, kd, pe) in base_corrs.items()
            },
            "mean_spearman": base_mean,
        },
        "test_best_sigma": {
            "sigma": float(best_sigma),
            "per_dim": {
                d: {"spearman": sp, "kendall": kd, "pearson": pe}
                for d, (sp, kd, pe) in test_corrs.items()
            },
            "mean_spearman": test_mean,
        },
    }
    if layer_sweep_results is not None:
        output["layer_sweep"] = {
            str(l): {"best_sigma": float(s), "val_spearman": float(sp)}
            for l, (s, sp) in layer_sweep_results.items()
        }
    if val_results is not None:
        output["val_results"] = {str(s): v for s, v in val_results.items()}
    if sigma_by_dim is not None:
        output["sigma_by_dim"] = {dim: float(s) for dim, s in sigma_by_dim.items()}
    if pairwise_results is not None:
        output["test_pairwise_accuracy"] = {
            dim: {
                "correct": c,
                "total": t,
                "accuracy": c / t if t > 0 else 0.0,
            }
            for dim, (c, t) in pairwise_results.items()
        }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
