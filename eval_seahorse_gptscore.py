"""GPTScore-based INCLINE evaluation on SEAHORSE.

Evaluates cross-lingual MLP steering (vector or map) on the SEAHORSE
multilingual summarization quality dataset.  Sigma and injection layer are
selected jointly on SEAHORSE's own validation split via layer-by-layer sweep
(matching Lu et al., EMNLP 2025), then reported on the test split.

Following the SEAHORSE paper (Clark et al., EMNLP 2023), evaluation uses
Pearson r and AUC-ROC, both appropriate for binary (Yes/No) human labels.
Layer and sigma are selected by maximising mean AUC-ROC across questions on
the validation split.

Usage
-----
# Spanish, vector steering, target-language prompts:
python eval_seahorse_gptscore.py \\
    --model      meta-llama/Llama-3.2-3B \\
    --lang       es \\
    --data-dir   ./data/seahorse \\
    --parallel-data ./data/ncwm_old/en-es \\
    --method     vector \\
    --prompting-lang tgt \\
    --layers     21 22 23 24 25 26 27 \\
    --select-layer \\
    --gpu        0 \\
    --out        results/seahorse_gptscore/llama3b_es_vector_tgt.json
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

from constants import MAX_LENGTH, SIGMAS_BY_MODE
from load_seahorse import (
    SEAHORSE_DIMS,
    SEAHORSE_LANG2NAME,
    load_seahorse,
)
from model_and_tok import ModelAndTokenizer
from utils import (
    apply_steering,
    conditional_avg_logprob_with_patch,
    get_residual_last_token,
    get_layer_name,
    get_residual_mean_over_tokens,
    tokenize_input,
)

# ── prompt templates ────────────────────────────────────────────────────────

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
    "English": {
        "question1": "Generate a comprehensible summary for the following article: {article}\nSummary:",
        "question2": "Generate a summary without unnecessary repetition for the following article: {article}\nSummary:",
        "question3": "Generate a grammatically correct summary for the following article: {article}\nSummary:",
        "question4": "Generate a faithful summary for the following article: {article}\nSummary:",
        "question5": "Generate a summary capturing the main ideas of the following article: {article}\nSummary:",
        "question6": "Generate a concise summary for the following article: {article}\nSummary:",
    },
}


def build_prompt(article: str, dimension: str, lang_name: str, prompting_lang: str) -> str:
    if prompting_lang == "en":
        return _EN_PROMPTS[dimension].format(article=article)
    return _TGT_PROMPTS[lang_name][dimension].format(article=article)


# ── metrics ─────────────────────────────────────────────────────────────────


def safe_seahorse_metrics(scores, labels) -> Tuple[float, float]:
    """Pearson r and AUC-ROC for binary (0/1) human labels.

    Both metrics are appropriate for SEAHORSE binary annotations (Clark et al.,
    EMNLP 2023).  Returns (pearson, auc).  AUC is NaN when only one class is
    present in labels.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(scores) < 2:
        return float("nan"), float("nan")
    pe = float(pearsonr(scores, labels)[0])
    if len(np.unique(labels)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels, scores))
    return pe, auc


# ── steering parameter computation ─────────────────────────────────────────


def _cache_path(parallel_dir: str, lang_key: str, method: str, max_pairs: int,
                model_name: str) -> str:
    safe_model = model_name.replace("/", "_")
    return os.path.join(
        "cache", "seahorse", f"{lang_key}_{method}_{max_pairs}_{safe_model}_residual.npy"
    )


def compute_steer_params(mt, parallel_dir: str, lang_key: str,
                         method: str, max_pairs: int) -> List:
    path = _cache_path(parallel_dir, lang_key, method, max_pairs, mt.model_name)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        print(f"Loaded steer params from {path}", flush=True)
        return list(data)

    en_path = os.path.join(parallel_dir, "train.en")
    xx_path = os.path.join(parallel_dir, f"train.{lang_key}")
    if not os.path.exists(en_path) or not os.path.exists(xx_path):
        raise FileNotFoundError(
            f"Parallel data not found. Expected:\n  {en_path}\n  {xx_path}"
        )

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
            print(f"  encoded {len(EN_reps)} pairs…", flush=True)

    EN = np.stack(EN_reps)
    XX = np.stack(XX_reps)

    if method == "vector":
        params = [EN[:, l, :].mean(0) - XX[:, l, :].mean(0) for l in range(mt.num_layers)]
    else:
        params = []
        for l in range(mt.num_layers):
            W, *_ = np.linalg.lstsq(XX[:, l, :], EN[:, l, :], rcond=None)
            params.append(W.astype(np.float32))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.array(params, dtype=object), allow_pickle=True)
    print(f"Saved steer params to {path}", flush=True)
    return params


# ── prompt caching ──────────────────────────────────────────────────────────


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


# ── evaluation ──────────────────────────────────────────────────────────────


def eval_metrics(rows, steer_params, sigma, intervention_layers, mt,
                 lang_name, prompting_lang, cache=None) -> Dict[str, Tuple[float, float]]:
    """Returns {dim: (pearson, auc_roc)} for each question dimension."""
    if cache is None:
        cache = build_cache(rows, lang_name, prompting_lang, mt, intervention_layers)
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


def _mean_auc(dim_metrics: Dict[str, Tuple[float, float]]) -> float:
    """Mean AUC-ROC across dimensions, ignoring NaN values."""
    vals = [auc for _, auc in dim_metrics.values() if not np.isnan(auc)]
    return float(np.mean(vals)) if vals else float("nan")


def _mean_pearson(dim_metrics: Dict[str, Tuple[float, float]]) -> float:
    vals = [pe for pe, _ in dim_metrics.values() if not np.isnan(pe)]
    return float(np.mean(vals)) if vals else float("nan")


# ── layer + sigma selection ─────────────────────────────────────────────────


def select_best_layer(val_rows, steer_params, candidate_layers, sigma_list,
                      mt, lang_name, prompting_lang, val_cache):
    """Sweep each layer individually on val; pick the (layer, sigma) with best mean Pearson r."""
    per_layer_results = {}
    for l in candidate_layers:
        layer_best_sigma, layer_best_pe = 0.0, -float("inf")
        for sigma in sigma_list:
            dim_metrics, _ = eval_metrics(
                val_rows, steer_params, sigma, [l],
                mt, lang_name, prompting_lang, val_cache,
            )
            mean_auc = _mean_auc(dim_metrics)
            mean_pe = _mean_pearson(dim_metrics)
            print(f"    sigma={sigma:+.2f}  mean_AUC={mean_auc:.4f}  mean_Pearson={mean_pe:.4f}", flush=True)
            if mean_pe > layer_best_pe:
                layer_best_pe = mean_pe
                layer_best_sigma = sigma
        per_layer_results[l] = (layer_best_sigma, layer_best_pe)
        print(f"  layer {l}: best_sigma={layer_best_sigma:+.2f}  val_Pearson={layer_best_pe:.4f}", flush=True)
    best_layer = max(per_layer_results, key=lambda l: per_layer_results[l][1])
    best_sigma, _ = per_layer_results[best_layer]
    return best_layer, best_sigma, per_layer_results


# ── main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="GPTScore INCLINE evaluation on SEAHORSE"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang", choices=["de", "en", "es", "ru", "tr", "vi"], required=True)
    parser.add_argument("--data-dir", required=True, help="Directory with SEAHORSE validation.tsv / test.tsv")
    parser.add_argument("--parallel-data", default=None,
                        help="Directory with train.en + train.<lang> (not needed for baseline)")
    parser.add_argument("--method", choices=["baseline", "vector", "map"], default="vector")
    parser.add_argument("--prompting-lang", choices=["en", "tgt"], default="tgt")
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--select-layer", action="store_true",
                        help="Sweep each candidate layer independently, pick the best one")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--out", default="results/seahorse_gptscore/results.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_grad_enabled(False)

    lang_name = SEAHORSE_LANG2NAME[args.lang]
    print(f"Language: {lang_name} ({args.lang})", flush=True)
    print(f"Method:   {args.method}", flush=True)

    mt = ModelAndTokenizer(args.model)
    intervention_layers = list(range(mt.num_layers)) if args.layers is None else sorted(args.layers)

    if args.method == "baseline":
        steer_params = None
    else:
        if args.parallel_data is None:
            raise ValueError("--parallel-data is required for non-baseline methods")
        print(f"\nComputing {args.method} steer params…", flush=True)
        steer_params = compute_steer_params(
            mt, args.parallel_data, args.lang, args.method, args.max_pairs
        )

    sigma_list = SIGMAS_BY_MODE[args.method]
    print(f"Sigma candidates: {sigma_list}", flush=True)

    val_rows, test_rows = load_seahorse(args.data_dir, lang=args.lang)
    if not val_rows:
        raise RuntimeError(
            f"No usable val rows for lang='{args.lang}'. "
            "All rows may be wikilingua (skipped by default). "
            "Check the gem_id sources printed above."
        )
    if not test_rows:
        raise RuntimeError(f"No usable test rows for lang='{args.lang}'.")

    # ── Val: layer + sigma selection ────────────────────────────────────────
    print("\n  Building val prompt cache…", flush=True)
    val_cache = build_cache(val_rows, lang_name, args.prompting_lang, mt, intervention_layers)

    if args.select_layer:
        print("\n=== Layer selection sweep ===", flush=True)
        best_layer, best_sigma, layer_sweep_results = select_best_layer(
            val_rows, steer_params, intervention_layers, sigma_list,
            mt, lang_name, args.prompting_lang, val_cache,
        )
        print(f"\nSelected layer {best_layer} with sigma={best_sigma:+.2f}", flush=True)
        intervention_layers = [best_layer]
        layer_sweep_out = {
            str(l): {"best_sigma": float(s), "val_auc": float(auc)}
            for l, (s, auc) in layer_sweep_results.items()
        }
        val_results = None
    else:
        print("\n=== Val sigma sweep ===", flush=True)
        val_results = {}
        for sigma in sigma_list:
            dim_metrics, _ = eval_metrics(
                val_rows, steer_params, sigma, intervention_layers,
                mt, lang_name, args.prompting_lang, val_cache,
            )
            mean_auc = _mean_auc(dim_metrics)
            mean_pe = _mean_pearson(dim_metrics)
            print(f"  sigma={sigma:+.2f}  mean_AUC={mean_auc:.4f}  mean_Pearson={mean_pe:.4f}")
            val_results[sigma] = {
                "mean_auc": mean_auc,
                "mean_pearson": mean_pe,
                "per_dim": {
                    d: {"pearson": pe, "auc": auc}
                    for d, (pe, auc) in dim_metrics.items()
                },
            }
        best_sigma = max(val_results, key=lambda s: val_results[s]["mean_pearson"])
        print(f"\nBest sigma (val): {best_sigma:+.2f}  mean_Pearson={val_results[best_sigma]['mean_pearson']:.4f}", flush=True)
        layer_sweep_out = None

    # ── Test evaluation ─────────────────────────────────────────────────────
    print("\nBuilding test prompt cache…", flush=True)
    test_cache = build_cache(test_rows, lang_name, args.prompting_lang, mt, intervention_layers)

    print("\n=== Test baseline (sigma=0.0) ===", flush=True)
    base_metrics, base_samples = eval_metrics(
        test_rows, None, 0.0, intervention_layers,
        mt, lang_name, args.prompting_lang, test_cache,
    )
    base_mean_auc = _mean_auc(base_metrics)
    base_mean_pe = _mean_pearson(base_metrics)
    for dim, (pe, auc) in base_metrics.items():
        print(f"  [{dim}] Pearson={pe:.4f}  AUC={auc:.4f}")
    print(f"  mean_AUC={base_mean_auc:.4f}  mean_Pearson={base_mean_pe:.4f}")

    sigma_label = f"{best_sigma:+.2f}"
    print(f"\n=== Test best sigma ({sigma_label}) ===", flush=True)
    test_metrics, test_samples = eval_metrics(
        test_rows, steer_params, best_sigma, intervention_layers,
        mt, lang_name, args.prompting_lang, test_cache,
    )
    test_mean_auc = _mean_auc(test_metrics)
    test_mean_pe = _mean_pearson(test_metrics)
    for dim, (pe, auc) in test_metrics.items():
        print(f"  [{dim}] Pearson={pe:.4f}  AUC={auc:.4f}")
    print(f"  mean_AUC={test_mean_auc:.4f}  mean_Pearson={test_mean_pe:.4f}")

    # ── Save ────────────────────────────────────────────────────────────────
    output = {
        "model":              args.model,
        "lang":               args.lang,
        "method":             args.method,
        "prompting_lang":     args.prompting_lang,
        "select_layer_mode":  args.select_layer,
        "intervention_layers": intervention_layers,
        "best_sigma":         float(best_sigma),
        "test_baseline": {
            "sigma": 0.0,
            "per_dim": {d: {"pearson": pe, "auc": auc}
                        for d, (pe, auc) in base_metrics.items()},
            "mean_auc": base_mean_auc,
            "mean_pearson": base_mean_pe,
        },
        "test_best_sigma": {
            "sigma": float(best_sigma),
            "per_dim": {d: {"pearson": pe, "auc": auc}
                        for d, (pe, auc) in test_metrics.items()},
            "mean_auc": test_mean_auc,
            "mean_pearson": test_mean_pe,
        },
        "test_samples": {
            "baseline":   base_samples,
            "best_sigma": test_samples,
        },
    }
    if layer_sweep_out is not None:
        output["layer_sweep"] = layer_sweep_out
    if val_results is not None:
        output["val_results"] = {str(s): v for s, v in val_results.items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
