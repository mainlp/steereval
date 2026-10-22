"""G-eval-based INCLINE evaluation on SEAHORSE.

Evaluates cross-lingual MLP steering (vector or map) on the SEAHORSE
multilingual summarization quality dataset using G-eval scoring (binary 0/1):
the model is prompted with the SEAHORSE annotation question and the score is
P("1") normalised over {P("0"), P("1")} after MLP patching.

Following the SEAHORSE paper (Clark et al., EMNLP 2023), evaluation uses
Pearson r and AUC-ROC, both appropriate for binary (Yes/No) human labels.
Layer and sigma are selected by maximising mean AUC-ROC across questions on
the validation split.

Usage
-----
# Spanish, vector steering, target-language prompts:
python eval_seahorse_geval.py \\
    --model      meta-llama/Llama-3.2-3B \\
    --lang       es \\
    --data-dir   ./data/seahorse \\
    --parallel-data ./data/ncwm_flores/en-es \\
    --method     vector \\
    --prompting-lang tgt \\
    --layers     21 22 23 24 25 26 27 \\
    --select-layer \\
    --gpu        0 \\
    --out        results/seahorse_geval/llama3b_es_vector_tgt.json
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from baukit.nethook import TraceDict
from constants import MAX_LENGTH, SIGMAS_BY_MODE
from load_seahorse import (
    SEAHORSE_DIMS,
    SEAHORSE_LANG2NAME,
    load_seahorse,
)
from model_and_tok import ModelAndTokenizer
from utils import (
    apply_steering,
    get_residual_last_token,
    get_layer_name,
    get_residual_mean_over_tokens,
    tokenize_input,
)

MAX_LENGTH_GEVAL = 1024 * 3

# ── prompt templates ─────────────────────────────────────────────────────────
# Each prompt is structured as:
#   cache prefix:  {task_intro}\n{question}\n\n{article_label}: {article}\n
#   full prompt :  {cache_prefix}{summary_label}: {summary}\n{answer_cue}
#
# The binary questions mirror the SEAHORSE paper's annotation questions
# (Clark et al., EMNLP 2023).

_EN_TASK_INTRO = "Answer with 1 (yes) or 0 (no).\n"

_EN_QUESTION: Dict[str, str] = {
    "question1": "Is the summary comprehensible?",
    "question2": "Does the summary contain no unnecessary repetition?",
    "question3": "Is the summary grammatically correct?",
    "question4": "Is the summary factually consistent with the article?",
    "question5": "Does the summary capture the main ideas of the article?",
    "question6": "Is the summary concise?",
}

_EN_ANSWER_CUE = "Answer (1=yes, 0=no):"

_TGT_TASK_INTRO: Dict[str, str] = {
    "German":     "Antworten Sie mit 1 (ja) oder 0 (nein).\n",
    "Spanish":    "Responda con 1 (sí) o 0 (no).\n",
    "Russian":    "Ответьте 1 (да) или 0 (нет).\n",
    "Turkish":    "1 (evet) veya 0 (hayır) ile yanıtlayın.\n",
    "Vietnamese": "Trả lời bằng 1 (có) hoặc 0 (không).\n",
    "English":    _EN_TASK_INTRO,
}

_TGT_QUESTION: Dict[str, Dict[str, str]] = {
    "German": {
        "question1": "Ist die Zusammenfassung verständlich?",
        "question2": "Enthält die Zusammenfassung keine unnötigen Wiederholungen?",
        "question3": "Ist die Zusammenfassung grammatikalisch korrekt?",
        "question4": "Ist die Zusammenfassung faktisch konsistent mit dem Artikel?",
        "question5": "Erfasst die Zusammenfassung die Hauptideen des Artikels?",
        "question6": "Ist die Zusammenfassung prägnant?",
    },
    "Spanish": {
        "question1": "¿Es el resumen comprensible?",
        "question2": "¿El resumen no contiene repeticiones innecesarias?",
        "question3": "¿Es el resumen gramaticalmente correcto?",
        "question4": "¿Es el resumen factualmente consistente con el artículo?",
        "question5": "¿El resumen captura las ideas principales del artículo?",
        "question6": "¿Es el resumen conciso?",
    },
    "Russian": {
        "question1": "Является ли резюме понятным?",
        "question2": "Не содержит ли резюме ненужных повторений?",
        "question3": "Является ли резюме грамматически правильным?",
        "question4": "Является ли резюме фактически согласованным со статьёй?",
        "question5": "Отражает ли резюме основные идеи статьи?",
        "question6": "Является ли резюме кратким?",
    },
    "Turkish": {
        "question1": "Özet anlaşılır mı?",
        "question2": "Özet gereksiz tekrar içermiyor mu?",
        "question3": "Özet dilbilgisi açısından doğru mu?",
        "question4": "Özet, makaleyle olgusal olarak tutarlı mı?",
        "question5": "Özet, makalenin ana fikirlerini yansıtıyor mu?",
        "question6": "Özet kısa ve öz mu?",
    },
    "Vietnamese": {
        "question1": "Bản tóm tắt có dễ hiểu không?",
        "question2": "Bản tóm tắt có không chứa sự lặp lại không cần thiết không?",
        "question3": "Bản tóm tắt có đúng ngữ pháp không?",
        "question4": "Bản tóm tắt có nhất quán về mặt thực tế với bài viết không?",
        "question5": "Bản tóm tắt có nắm bắt được các ý chính của bài viết không?",
        "question6": "Bản tóm tắt có súc tích không?",
    },
    "English": _EN_QUESTION,
}

_TGT_ARTICLE_LABEL: Dict[str, str] = {
    "German":     "Artikel",
    "Spanish":    "Artículo",
    "Russian":    "Статья",
    "Turkish":    "Makale",
    "Vietnamese": "Bài viết",
    "English":    "Article",
}

_TGT_SUMMARY_LABEL: Dict[str, str] = {
    "German":     "Zusammenfassung",
    "Spanish":    "Resumen",
    "Russian":    "Резюме",
    "Turkish":    "Özet",
    "Vietnamese": "Bản tóm tắt",
    "English":    "Summary",
}

_TGT_ANSWER_CUE: Dict[str, str] = {
    "German":     "Antwort (1=ja, 0=nein):",
    "Spanish":    "Respuesta (1=sí, 0=no):",
    "Russian":    "Ответ (1=да, 0=нет):",
    "Turkish":    "Yanıt (1=evet, 0=hayır):",
    "Vietnamese": "Câu trả lời (1=có, 0=không):",
    "English":    _EN_ANSWER_CUE,
}


def build_prompt(
    article: str,
    summary: Optional[str],
    dimension: str,
    lang_name: str,
    prompting_lang: str,
) -> str:
    """Build a binary G-eval prompt for SEAHORSE.

    If summary is None, returns the cache prefix (up to and including the article).
    If summary is provided, returns the full prompt ending with the answer cue.
    """
    if prompting_lang == "en":
        task_intro    = _EN_TASK_INTRO
        question      = _EN_QUESTION[dimension]
        article_label = "Article"
        summary_label = "Summary"
        answer_cue    = _EN_ANSWER_CUE
    else:
        task_intro    = _TGT_TASK_INTRO[lang_name]
        question      = _TGT_QUESTION[lang_name][dimension]
        article_label = _TGT_ARTICLE_LABEL[lang_name]
        summary_label = _TGT_SUMMARY_LABEL[lang_name]
        answer_cue    = _TGT_ANSWER_CUE[lang_name]

    prefix = task_intro + question + f"\n\n{article_label}: " + article + "\n"
    if summary is None:
        return prefix
    return prefix + f"{summary_label}: " + summary + f"\n{answer_cue}"


# ── G-eval scoring ───────────────────────────────────────────────────────────


def geval_with_patch(mt, full_prompt: str, steered_mlp, states_to_patch) -> float:
    """Run a forward pass with MLP patches and return P("1") / (P("0") + P("1"))."""
    input_ids, attention_mask = tokenize_input(
        full_prompt, mt, max_length=MAX_LENGTH_GEVAL
    )

    patch_spec = defaultdict(list)
    for t, layer in states_to_patch:
        patch_spec[layer].append(t)
    new_residual_t = torch.tensor(steered_mlp, device=mt.model.device, dtype=mt.model.dtype)

    def patch_rep(x, layer):
        if layer not in patch_spec:
            return x
        h = x[0] if isinstance(x, tuple) else x
        for t in patch_spec[layer]:
            ly = int(re.search(r"\.(\d+)$", layer).group(1))
            h[0, t] = new_residual_t[ly]
        if isinstance(x, tuple):
            return (h,) + x[1:]
        return h

    with torch.no_grad(), TraceDict(
        mt.model, list(patch_spec.keys()), edit_output=patch_rep
    ):
        logits = mt.model(input_ids=input_ids, attention_mask=attention_mask).logits

    next_token_logits = logits[0, -1]
    log_probs = F.log_softmax(next_token_logits, dim=-1)
    id_0 = mt.tokenizer("0", add_special_tokens=False).input_ids[0]
    id_1 = mt.tokenizer("1", add_special_tokens=False).input_ids[0]
    binary_logprobs = torch.tensor([log_probs[id_0], log_probs[id_1]]).cpu()
    probs = torch.softmax(binary_logprobs, dim=0)
    return probs[1].item()  # P("1" | prompt), normalised over {0, 1}


# ── metrics ──────────────────────────────────────────────────────────────────


def safe_seahorse_metrics(scores, labels) -> Tuple[float, float]:
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


# ── steering parameter computation ──────────────────────────────────────────


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


# ── prompt caching ────────────────────────────────────────────────────────────


def build_cache(rows, lang_name, prompting_lang, mt, intervention_layers):
    """Cache (mlp_last, states_to_patch) for the article-only prefix of each (article, dimension) pair."""
    cache = {}
    for row in rows:
        key = (row["article"], row["dimension"])
        if key in cache:
            continue
        prefix = build_prompt(row["article"], None, row["dimension"], lang_name, prompting_lang)
        input_ids, attention_mask = tokenize_input(prefix, mt, max_length=MAX_LENGTH_GEVAL)
        residual_last = get_residual_last_token(
            mt.model, input_ids, mt.num_layers, mt.model_name, attention_mask
        )
        ntoks = input_ids.shape[1]
        states = [(ntoks - 1, get_layer_name(l, mt.model_name)) for l in intervention_layers]
        cache[key] = (residual_last, states)
    return cache


# ── evaluation ───────────────────────────────────────────────────────────────


def eval_metrics(rows, steer_params, sigma, intervention_layers, mt,
                 lang_name, prompting_lang, cache=None) -> Dict[str, Tuple[float, float]]:
    """Returns {dim: (pearson, auc_roc)} for each question dimension."""
    if cache is None:
        cache = build_cache(rows, lang_name, prompting_lang, mt, intervention_layers)
    by_dim = defaultdict(lambda: ([], []))
    for row in rows:
        key = (row["article"], row["dimension"])
        residual_last, states = cache[key]
        steered = apply_steering(residual_last, steer_params, sigma, intervention_layers)
        full_prompt = build_prompt(
            row["article"], row["candidate"], row["dimension"], lang_name, prompting_lang
        )
        score = geval_with_patch(mt, full_prompt, steered, states)
        by_dim[row["dimension"]][0].append(score)
        by_dim[row["dimension"]][1].append(row["human_score"])
    metrics = {dim: safe_seahorse_metrics(scores, labels)
               for dim, (scores, labels) in by_dim.items()}
    samples = {dim: {"scores": [float(s) for s in scores], "labels": [int(l) for l in labels]}
               for dim, (scores, labels) in by_dim.items()}
    return metrics, samples


def _mean_auc(dim_metrics: Dict[str, Tuple[float, float]]) -> float:
    vals = [auc for _, auc in dim_metrics.values() if not np.isnan(auc)]
    return float(np.mean(vals)) if vals else float("nan")


def _mean_pearson(dim_metrics: Dict[str, Tuple[float, float]]) -> float:
    vals = [pe for pe, _ in dim_metrics.values() if not np.isnan(pe)]
    return float(np.mean(vals)) if vals else float("nan")


# ── layer + sigma selection ──────────────────────────────────────────────────


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


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="G-eval INCLINE evaluation on SEAHORSE"
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
    parser.add_argument("--out", default="results/seahorse_geval/results.json")
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

    # ── Val: layer + sigma selection ─────────────────────────────────────────
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

    # ── Test evaluation ──────────────────────────────────────────────────────
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

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "model":               args.model,
        "lang":                args.lang,
        "method":              args.method,
        "prompting_lang":      args.prompting_lang,
        "select_layer_mode":   args.select_layer,
        "intervention_layers": intervention_layers,
        "best_sigma":          float(best_sigma),
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
