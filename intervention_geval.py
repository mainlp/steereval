import os

import re
import json
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from tqdm import tqdm

from model_and_tok import ModelAndTokenizer

from constants import (
    LANGS,
    SIGMAS_BY_MODE,
    LANG2PROMPT_article_cache_geval,
    LANG2PROMPT_article_summary_geval,
    LANG2PROMPT_coh_criteria,
    LANG2PROMPT_com_criteria,
    LANG2PROMPT_task_intro,
    MAX_PAIRS,
)
from utils import (
    get_residual_last_token,
    get_layer_name,
    get_steer_params,
    load_annotations_csv,
    prep_parser,
    safe_corr,
    tokenize_input,
    apply_steering,
)
from baukit.nethook import TraceDict

MAX_LENGTH_GEVAL = 1024*3


def geval_with_patch(mt, full_prompt, steered_mlp, states_to_patch):
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

    # G-eval score calculation:
    next_token_logits = logits[0, -1]
    log_probs = F.log_softmax(next_token_logits, dim=-1)
    candidates = ["1", "2", "3", "4", "5"]
    candidate_token_ids = [
        mt.tokenizer(c, add_special_tokens=False).input_ids[0] for c in candidates
    ]

    candidate_logprobs = torch.tensor(
        [log_probs[token_id] for token_id in candidate_token_ids]
    ).cpu()

    normalized_probs = torch.softmax(candidate_logprobs, dim=0)

    scores = torch.tensor(
        [float(c) for c in candidates], device=normalized_probs.device
    )
    weighted_score = torch.sum(normalized_probs * scores).item()

    return weighted_score


def build_prompt(article, summary, dimension, lang, prompting_lang, cache=False):
    if prompting_lang == "en":
        task_intro = (
            f"You will be given an article and a summary in {lang}.\n"
            "Your task is to rate the summary on one metric.\n"
            "Please read the instructions carefully.\n"
        )

        if dimension == "coherence":
            criteria = (
                "Evaluation Criteria:\n"
                "Coherence (1–5) – the collective quality of all sentences. "
                "The summary should be well-structured and well-organized. "
                "The summary should not be a heap of related information, "
                "but should build from sentence to sentence into a coherent body of information about a topic.\n"
            )
        elif dimension == "completeness":
            criteria = (
                "Evaluation Criteria:\n"
                "Completeness (1–5) – the degree to which the summary covers all important information "
                "from the article without omitting key details or adding irrelevant content.\n"
            )

        if cache:
            # same prefix for caching: task intro + criteria + Article line
            prompt = task_intro + criteria + "\nArticle: " + article + "\n"
        else:
            prompt = (
                task_intro
                + criteria
                + "\nArticle: "
                + article
                + "\nSummary: "
                + summary
                + "\nYour Evaluation (1-5): "
            )

    elif prompting_lang == "tgt":
        # target-language prompts: use your LANG2PROMPT_* dictionaries
        if dimension == "coherence":
            criteria = LANG2PROMPT_coh_criteria[lang]
        elif dimension == "completeness":
            criteria = LANG2PROMPT_com_criteria[lang]

        if cache:
            # For caching, keep exactly the same prefix text as when evaluating in tgt language:
            prompt = (
                LANG2PROMPT_task_intro[lang]
                + "\n"
                + criteria
                + "\n"
                + LANG2PROMPT_article_cache_geval[lang].format(article=article)
                + "\n"
            )
        else:
            prompt = (
                LANG2PROMPT_task_intro[lang]
                + "\n"
                + criteria
                + "\n"
                + LANG2PROMPT_article_summary_geval[lang].format(article=article, summary=summary)
            )

    return prompt


def build_prompt_cached(
    article, lang, prompting_lang, mt, dimension, intervention_layers
):
    prompt = build_prompt(article, None, dimension, lang, prompting_lang, cache=True)
    input_ids, attention_mask = tokenize_input(
        prompt,
        mt,
        max_length=MAX_LENGTH_GEVAL,
    )
    residual_last = get_residual_last_token(
        mt.model, input_ids, mt.num_layers, mt.model_name, attention_mask
    )
    ntoks = input_ids.shape[1]
    states_to_patch = [
        (ntoks - 1, get_layer_name(l, mt.model_name)) for l in intervention_layers
    ]
    return prompt, input_ids, residual_last, states_to_patch


def prep_cache(
    lang, prompting_lang, mt, annotations_path, metric, dimension, intervention_layers
):
    ann = load_annotations_csv(annotations_path, metric, dimension)
    cache = {}
    for row in ann:
        art = row["article"]
        if art not in cache:
            _, prompt_ids, residual_last, states_to_patch = build_prompt_cached(
                art, lang, prompting_lang, mt, dimension, intervention_layers
            )
            cache[art] = (prompt_ids, residual_last, states_to_patch)
    return ann, cache


def run_eval_correlation(
    lang,
    prompting_lang,
    steer_params,
    mt,
    annotations_path,
    sigma_list,
    dimension,
    intervention_layers,
):
    metric = "correlation"
    ann, cache = prep_cache(
        lang,
        prompting_lang,
        mt,
        annotations_path,
        metric,
        dimension,
        intervention_layers,
    )

    results = []
    for sigma in sigma_list:
        model_scores, human_scores = [], []
        for row in ann:
            _, residual_last, states_to_patch = cache[row["article"]]
            steered_mlp = apply_steering(
                residual_last, steer_params, sigma, intervention_layers
            )
            full_prompt = build_prompt(
                row["article"],
                row["candidate"],
                dimension,
                lang,
                prompting_lang,
                cache=False,
            )
            s = geval_with_patch(mt, full_prompt, steered_mlp, states_to_patch)
            model_scores.append(s)
            human_scores.append(row["human_score"])
        sp, kd, pe = safe_corr(model_scores, human_scores)
        print(f"[{lang}] sigma={sigma:+.2f} Spearman={sp:.4f}")
        results.append((sigma, sp, kd, pe))
    return results


def run_eval_accuracy(
    lang: str,
    prompting_lang: str,
    steer_params: List[np.ndarray],
    mt,
    annotations_path: str,
    sigma_list: List[float],
    dimension: str,
    intervention_layers: List[int],
):
    metric = "accuracy"
    ann, cache = prep_cache(
        lang,
        prompting_lang,
        mt,
        annotations_path,
        metric,
        dimension,
        intervention_layers,
    )
    results = []
    for sigma in sigma_list:
        correct, total = 0, 0
        skipped_same = 0

        for row in ann:
            _, residual_last, states_to_patch = cache[row["article"]]
            steered_mlp = apply_steering(
                residual_last, steer_params, sigma, intervention_layers
            )

            # Skip pairs where the "original" and "corrupted" summaries are identical
            if (
                row["gpt_orig"].strip() == row["gpt_corr"].strip()
                and row["gem_orig"].strip() == row["gem_corr"].strip()
            ):
                skipped_same += 1
                continue

            try:
                gpt_o_prompt = build_prompt(
                    row["article"],
                    row["gpt_orig"],
                    dimension,
                    lang,
                    prompting_lang,
                    cache=False,
                )
                gpt_c_prompt = build_prompt(
                    row["article"],
                    row["gpt_corr"],
                    dimension,
                    lang,
                    prompting_lang,
                    cache=False,
                )
                gem_o_prompt = build_prompt(
                    row["article"],
                    row["gem_orig"],
                    dimension,
                    lang,
                    prompting_lang,
                    cache=False,
                )
                gem_c_prompt = build_prompt(
                    row["article"],
                    row["gem_corr"],
                    dimension,
                    lang,
                    prompting_lang,
                    cache=False,
                )

                gpt_o = geval_with_patch(
                    mt,
                    gpt_o_prompt,
                    steered_mlp,
                    states_to_patch,
                )
                gpt_c = geval_with_patch(
                    mt,
                    gpt_c_prompt,
                    steered_mlp,
                    states_to_patch,
                )
                gem_o = geval_with_patch(
                    mt,
                    gem_o_prompt,
                    steered_mlp,
                    states_to_patch,
                )
                gem_c = geval_with_patch(
                    mt,
                    gem_c_prompt,
                    steered_mlp,
                    states_to_patch,
                )
            except Exception as e:
                print(f"Skipping row due to error: {e}")
                continue

            # Count correct predictions
            if row["gpt_orig"].strip() != row["gpt_corr"].strip():
                correct += int(gpt_o > gpt_c)
                total += 1
            if row["gem_orig"].strip() != row["gem_corr"].strip():
                correct += int(gem_o > gem_c)
                total += 1

        acc = correct / total if total > 0 else 0
        print(
            f"[{lang}] sigma={sigma:+.2f} → Accuracy={acc:.4f} ({correct}/{total})  | Skipped identical pairs: {skipped_same}"
        )
        results.append((sigma, acc))

    return results


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = prep_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    model_name = args.model
    mt = ModelAndTokenizer(model_name)

    if args.layers is None:
        intervention_layers = list(range(mt.num_layers))
    else:
        intervention_layers = sorted(args.layers)
        print("Intervening only on layers:", intervention_layers)

    sigma_list = SIGMAS_BY_MODE[args.mode]
    out_dir = f"results/geval/{args.mode}/{args.data}/{args.prompting_lang}"
    os.makedirs(out_dir, exist_ok=True)
    for lang in tqdm(LANGS):
        save_suffix = f"{model_name.split('/')[1]}_{args.dimension}_{lang}{f'_{args.mode}' if args.mode == 'vector' else ''}.json"
        annotation_path = f"./data/ngrams/{args.dimension}/{lang.lower()}.csv"
        steer_params = get_steer_params(
            mt, lang, args.data, args.mode, max_pairs=MAX_PAIRS
        )

        if args.metric == "correlation" or args.metric == "all":
            print(f"\n=== Likelihood–human correlation: {lang} ===")
            results = run_eval_correlation(
                lang,
                args.prompting_lang,
                steer_params,
                mt,
                annotation_path,
                sigma_list=sigma_list,
                dimension=args.dimension,
                intervention_layers=intervention_layers,
            )
            out_path = f"{out_dir}/corr_results_{save_suffix}"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

        if args.metric == "accuracy" or args.metric == "all":
            print(f"\n=== Likelihood–human accuracy: {lang} ===")
            results = run_eval_accuracy(
                lang,
                args.prompting_lang,
                steer_params,
                mt,
                annotation_path,
                sigma_list=sigma_list,
                dimension=args.dimension,
                intervention_layers=intervention_layers,
            )
            out_path = f"{out_dir}/acc_results_{save_suffix}"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
