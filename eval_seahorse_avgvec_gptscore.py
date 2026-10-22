"""GPTScore SEAHORSE evaluation with a language-averaged steering vector.

Loads cached XX->EN steering vectors for all five source languages, averages
them at a fixed layer, and evaluates with a fixed sigma.  This tests whether
hyperparameters found on individual languages generalise to a universal vector.

Fixed hyperparameters per model
--------------------------------
  Llama-3.2-3B      layer=27  sigma_en=-2.5  sigma_tgt=+2.5
  aya-expanse-8b    layer=31  sigma_en=-2.5  sigma_tgt=-2.5

Usage
-----
python eval_seahorse_avgvec_gptscore.py \\
    --model          meta-llama/Llama-3.2-3B \\
    --lang           de \\
    --prompting-lang en \\
    --data-dir       ./data/seahorse \\
    --data-root      ./data/ncwm_flores \\
    --gpu            0 \\
    --out            results/seahorse_avgvec_gptscore_llama3b/llama3b_de_avgvec_en.json
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from constants import MAX_LENGTH
from load_seahorse import SEAHORSE_DIMS, SEAHORSE_LANG2NAME, load_seahorse
from model_and_tok import ModelAndTokenizer
from utils import (
    apply_steering,
    conditional_avg_logprob_with_patch,
    get_residual_last_token,
    get_layer_name,
    get_residual_mean_over_tokens,
    tokenize_input,
)

# ── fixed hyperparameters ─────────────────────────────────────────────────────

SOURCE_LANGS = ["de", "es", "ru", "tr", "vi"]

FIXED_LAYER: Dict[str, int] = {
    "meta-llama/Llama-3.2-3B":   27,
    "CohereForAI/aya-expanse-8b": 31,
}

# sigma indexed by [model_name][prompting_lang]
FIXED_SIGMA: Dict[str, Dict[str, float]] = {
    "meta-llama/Llama-3.2-3B": {
        "en":  -2.5,
        "tgt": +2.5,
    },
    "CohereForAI/aya-expanse-8b": {
        "en":  -2.5,
        "tgt": -2.5,
    },
}

# ── prompt templates ──────────────────────────────────────────────────────────

_EN_PROMPTS: Dict[str, str] = {
    "question1": "Generate a comprehensible summary for the following article: {article}\nSummary:",
    "question2": "Generate a summary without unnecessary repetition for the following article: {article}\nSummary:",
    "question3": "Generate a grammatically correct summary for the following article: {article}\nSummary:",
    "question4": "Generate a faithful summary for the following article: {article}\nSummary:",
    "question5": "Generate a summary capturing the main ideas of the following article: {article}\nSummary:",
    "question6": "Generate a concise summary for the following article: {article}\nSummary:",
}

_TGT_PROMPTS: Dict[str, Dict[str, str]] = {
    "German": {
        "question1": "Erstelle eine verständliche Zusammenfassung des folgenden Artikels: {article}\nZusammenfassung:",
        "question2": "Erstelle eine Zusammenfassung ohne unnötige Wiederholungen des folgenden Artikels: {article}\nZusammenfassung:",
        "question3": "Erstelle eine grammatisch korrekte Zusammenfassung des folgenden Artikels: {article}\nZusammenfassung:",
        "question4": "Erstelle eine quelltreue Zusammenfassung des folgenden Artikels: {article}\nZusammenfassung:",
        "question5": "Erstelle eine Zusammenfassung mit den Hauptideen des folgenden Artikels: {article}\nZusammenfassung:",
        "question6": "Erstelle eine prägnante Zusammenfassung des folgenden Artikels: {article}\nZusammenfassung:",
    },
    "Spanish": {
        "question1": "Genera un resumen comprensible del siguiente artículo: {article}\nResumen:",
        "question2": "Genera un resumen sin repeticiones innecesarias del siguiente artículo: {article}\nResumen:",
        "question3": "Genera un resumen gramaticalmente correcto del siguiente artículo: {article}\nResumen:",
        "question4": "Genera un resumen fiel al contenido del siguiente artículo: {article}\nResumen:",
        "question5": "Genera un resumen con las ideas principales del siguiente artículo: {article}\nResumen:",
        "question6": "Genera un resumen conciso del siguiente artículo: {article}\nResumen:",
    },
    "Russian": {
        "question1": "Создайте понятное резюме следующей статьи: {article}\nРезюме:",
        "question2": "Создайте резюме без лишних повторений следующей статьи: {article}\nРезюме:",
        "question3": "Создайте грамматически правильное резюме следующей статьи: {article}\nРезюме:",
        "question4": "Создайте достоверное резюме следующей статьи: {article}\nРезюме:",
        "question5": "Создайте резюме с основными идеями следующей статьи: {article}\nРезюме:",
        "question6": "Создайте краткое резюме следующей статьи: {article}\nРезюме:",
    },
    "Turkish": {
        "question1": "Aşağıdaki makale için anlaşılır bir özet oluşturun: {article}\nÖzet:",
        "question2": "Aşağıdaki makale için gereksiz tekrar içermeyen bir özet oluşturun: {article}\nÖzet:",
        "question3": "Aşağıdaki makale için dilbilgisel açıdan doğru bir özet oluşturun: {article}\nÖzet:",
        "question4": "Aşağıdaki makale için kaynağa sadık bir özet oluşturun: {article}\nÖzet:",
        "question5": "Aşağıdaki makalenin ana fikirlerini içeren bir özet oluşturun: {article}\nÖzet:",
        "question6": "Aşağıdaki makale için kısa ve öz bir özet oluşturun: {article}\nÖzet:",
    },
    "Vietnamese": {
        "question1": "Tạo bản tóm tắt dễ hiểu cho bài viết sau: {article}\nTóm tắt:",
        "question2": "Tạo bản tóm tắt không có thông tin lặp lại không cần thiết cho bài viết sau: {article}\nTóm tắt:",
        "question3": "Tạo bản tóm tắt đúng ngữ pháp cho bài viết sau: {article}\nTóm tắt:",
        "question4": "Tạo bản tóm tắt trung thực với bài viết sau: {article}\nTóm tắt:",
        "question5": "Tạo bản tóm tắt nắm bắt ý chính của bài viết sau: {article}\nTóm tắt:",
        "question6": "Tạo bản tóm tắt súc tích cho bài viết sau: {article}\nTóm tắt:",
    },
}


def build_prompt(article: str, dimension: str, lang_name: str, prompting_lang: str) -> str:
    if prompting_lang == "en":
        return _EN_PROMPTS[dimension].format(article=article)
    return _TGT_PROMPTS[lang_name][dimension].format(article=article)


# ── vector loading / computation ──────────────────────────────────────────────

def _cache_path(lang: str, model_name: str, max_pairs: int) -> str:
    safe = model_name.replace("/", "_")
    return os.path.join("cache", "seahorse", f"{lang}_vector_{max_pairs}_{safe}_residual.npy")


def _load_or_compute(lang: str, model_name: str, data_root: str,
                     max_pairs: int, mt=None) -> Optional[List[np.ndarray]]:
    path = _cache_path(lang, model_name, max_pairs)
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
) -> np.ndarray:
    """Return the averaged steering vector at fixed_layer across all source languages.

    Also returns a full steer_params list (indexed by layer) compatible with
    apply_steering, where only index fixed_layer is populated.
    """
    # First pass: try cache only
    per_lang = {l: _load_or_compute(l, model_name, data_root, max_pairs) for l in SOURCE_LANGS}
    missing = [l for l in SOURCE_LANGS if per_lang[l] is None]

    if missing:
        import torch as _torch
        from model_and_tok import ModelAndTokenizer as _MT
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"  Loading model to compute vectors for: {missing}...", flush=True)
        mt = _MT(model_name)
        _torch.set_grad_enabled(False)
        for lang in missing:
            per_lang[lang] = _load_or_compute(lang, model_name, data_root, max_pairs, mt)
        del mt
        _torch.cuda.empty_cache()

    available = [v for v in per_lang.values() if v is not None]
    if not available:
        raise RuntimeError("No steering vectors could be loaded or computed.")

    # Average the steering vector at fixed_layer across source languages
    layer_vecs = [np.asarray(v[fixed_layer], dtype=np.float32) for v in available]
    avg_vec = np.mean(layer_vecs, axis=0)
    print(f"  Averaged {len(layer_vecs)} language vectors at layer {fixed_layer}.", flush=True)

    # Build a full-length steer_params list; only index fixed_layer is used
    n_layers = len(available[0])
    dummy = np.zeros_like(avg_vec)
    steer_params = [dummy] * n_layers
    steer_params[fixed_layer] = avg_vec
    return steer_params


# ── metrics ───────────────────────────────────────────────────────────────────

def safe_seahorse_metrics(scores, labels) -> Tuple[float, float]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(scores) < 2:
        return float("nan"), float("nan")
    pe = float(pearsonr(scores, labels)[0])
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) >= 2 else float("nan")
    return pe, auc


def _mean_auc(dim_metrics: Dict[str, Tuple[float, float]]) -> float:
    vals = [auc for _, auc in dim_metrics.values() if not np.isnan(auc)]
    return float(np.mean(vals)) if vals else float("nan")


def _mean_pearson(dim_metrics: Dict[str, Tuple[float, float]]) -> float:
    vals = [pe for pe, _ in dim_metrics.values() if not np.isnan(pe)]
    return float(np.mean(vals)) if vals else float("nan")


# ── prompt cache and evaluation ───────────────────────────────────────────────

def build_cache(rows, lang_name, prompting_lang, mt, intervention_layers):
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
        states = [(ntoks - 1, get_layer_name(l, mt.model_name)) for l in intervention_layers]
        cache[key] = (input_ids, residual_last, states)
    return cache


def eval_metrics(rows, steer_params, sigma, intervention_layers,
                 mt, cache) -> Dict[str, Tuple[float, float]]:
    by_dim = defaultdict(lambda: ([], []))
    for row in rows:
        key = (row["article"], row["dimension"])
        prompt_ids, residual_last, states = cache[key]
        steered = apply_steering(residual_last, steer_params, sigma, intervention_layers)
        score = conditional_avg_logprob_with_patch(mt, prompt_ids, row["candidate"], steered, states)
        by_dim[row["dimension"]][0].append(score)
        by_dim[row["dimension"]][1].append(row["human_score"])
    metrics = {dim: safe_seahorse_metrics(scores, labels)
               for dim, (scores, labels) in by_dim.items()}
    samples = {dim: {"scores": [float(s) for s in scores], "labels": [int(l) for l in labels]}
               for dim, (scores, labels) in by_dim.items()}
    return metrics, samples


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPTScore SEAHORSE evaluation with language-averaged steering vector"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang", choices=["de", "es", "ru", "tr", "vi"], required=True,
                        help="SEAHORSE evaluation language")
    parser.add_argument("--prompting-lang", choices=["en", "tgt"], default="en")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with SEAHORSE validation.tsv / test.tsv")
    parser.add_argument("--data-root", default="./data/ncwm_flores",
                        help="Root for parallel data (en-{lang}/ subdirs)")
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--layer", type=int, default=None,
                        help="Override fixed steering layer (required for models not in FIXED_LAYER)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Override fixed sigma (required for models not in FIXED_SIGMA)")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Resolve layer and sigma: CLI flags take precedence over hardcoded defaults.
    if args.layer is not None:
        fixed_layer = args.layer
    elif args.model in FIXED_LAYER:
        fixed_layer = FIXED_LAYER[args.model]
    else:
        raise ValueError(
            f"--layer required for model '{args.model}' (not in FIXED_LAYER). "
            f"Known models: {list(FIXED_LAYER)}"
        )

    if args.sigma is not None:
        fixed_sigma = args.sigma
    elif args.model in FIXED_SIGMA:
        fixed_sigma = FIXED_SIGMA[args.model][args.prompting_lang]
    else:
        raise ValueError(
            f"--sigma required for model '{args.model}' (not in FIXED_SIGMA). "
            f"Known models: {list(FIXED_SIGMA)}"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_grad_enabled(False)
    lang_name      = SEAHORSE_LANG2NAME[args.lang]
    prompting_lang = args.prompting_lang
    print(f"Language:      {lang_name} ({args.lang})", flush=True)
    print(f"Prompting:     {prompting_lang}", flush=True)
    print(f"Fixed layer:   {fixed_layer}", flush=True)
    print(f"Fixed sigma:   {fixed_sigma:+.2f}", flush=True)

    # ── Build averaged vector ────────────────────────────────────────────────
    print("\nLoading / computing source language vectors...", flush=True)
    steer_params = build_averaged_steer_params(
        args.model, args.data_root, args.max_pairs, fixed_layer, args.gpu
    )

    # ── Load model (may already be in memory from vector computation) ────────
    print("\nLoading model for evaluation...", flush=True)
    mt = ModelAndTokenizer(args.model)
    intervention_layers = [fixed_layer]

    # ── SEAHORSE data ────────────────────────────────────────────────────────
    _, test_rows = load_seahorse(args.data_dir, lang=args.lang)
    if not test_rows:
        raise RuntimeError(f"No usable test rows for lang='{args.lang}'.")

    print("\nBuilding test prompt cache...", flush=True)
    test_cache = build_cache(test_rows, lang_name, prompting_lang, mt, intervention_layers)

    # ── Baseline (sigma=0) ───────────────────────────────────────────────────
    print("\n=== Test baseline (sigma=0.0) ===", flush=True)
    base_metrics, base_samples = eval_metrics(test_rows, None, 0.0, intervention_layers, mt, test_cache)
    base_mean_auc = _mean_auc(base_metrics)
    base_mean_pe  = _mean_pearson(base_metrics)
    for dim, (pe, auc) in base_metrics.items():
        print(f"  [{dim}] Pearson={pe:.4f}  AUC={auc:.4f}")
    print(f"  mean_AUC={base_mean_auc:.4f}  mean_Pearson={base_mean_pe:.4f}")

    # ── Steered (fixed sigma) ────────────────────────────────────────────────
    print(f"\n=== Test averaged vector (sigma={fixed_sigma:+.2f}) ===", flush=True)
    test_metrics, test_samples = eval_metrics(
        test_rows, steer_params, fixed_sigma, intervention_layers, mt, test_cache
    )
    test_mean_auc = _mean_auc(test_metrics)
    test_mean_pe  = _mean_pearson(test_metrics)
    for dim, (pe, auc) in test_metrics.items():
        print(f"  [{dim}] Pearson={pe:.4f}  AUC={auc:.4f}")
    print(f"  mean_AUC={test_mean_auc:.4f}  mean_Pearson={test_mean_pe:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "model":               args.model,
        "lang":                args.lang,
        "method":              "avgvec",
        "prompting_lang":      prompting_lang,
        "source_langs":        SOURCE_LANGS,
        "intervention_layers": intervention_layers,
        "best_sigma":          float(fixed_sigma),
        "test_baseline": {
            "sigma":        0.0,
            "per_dim":      {d: {"pearson": pe, "auc": auc}
                             for d, (pe, auc) in base_metrics.items()},
            "mean_auc":     base_mean_auc,
            "mean_pearson": base_mean_pe,
        },
        "test_best_sigma": {
            "sigma":        float(fixed_sigma),
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
