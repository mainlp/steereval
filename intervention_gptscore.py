import os

import json
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from model_and_tok import ModelAndTokenizer
from constants import LANGS, LANG2PROMPT_coh, LANG2PROMPT_com, SIGMAS_BY_MODE, MAX_PAIRS
from utils import (
    apply_steering,
    conditional_avg_logprob_with_patch,
    get_residual_last_token,
    get_layer_name,
    get_steer_params,
    load_annotations_csv,
    prep_parser,
    safe_corr,
    tokenize_input,
)


def build_prompt(article, dimension, lang, prompting_lang):
    if prompting_lang == "en":
        if dimension == "coherence":
            return f"Generate a coherent summary for the following text in {lang}: {article}\nTLDR: "
        elif dimension == "completeness":
            return f"Generate a complete summary for the following text in {lang}: {article}\nTLDR: "
    elif prompting_lang == "tgt":
        if dimension == "coherence":
            return LANG2PROMPT_coh[lang].format(article=article)
        elif dimension == "completeness":
            return LANG2PROMPT_com[lang].format(article=article)


def build_prompt_cached(
    article,
    lang,
    prompting_lang,
    mt,
    dimension,
    intervention_layers,
):
    prompt = build_prompt(article, dimension, lang, prompting_lang)
    input_ids, attention_mask = tokenize_input(prompt, mt)
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
            prompt_ids, residual_last, states_to_patch = cache[row["article"]]
            steered_mlp = apply_steering(
                residual_last, steer_params, sigma, intervention_layers
            )
            s = conditional_avg_logprob_with_patch(
                mt, prompt_ids, row["candidate"], steered_mlp, states_to_patch
            )
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
            prompt_ids, residual_last, states_to_patch = cache[row["article"]]
            steered_mlp = apply_steering(
                residual_last, steer_params, sigma, intervention_layers
            )

            if (
                row["gpt_orig"].strip() == row["gpt_corr"].strip()
                and row["gem_orig"].strip() == row["gem_corr"].strip()
            ):
                skipped_same += 1
                continue

            try:
                gpt_o = conditional_avg_logprob_with_patch(
                    mt, prompt_ids, row["gpt_orig"], steered_mlp, states_to_patch
                )
                gpt_c = conditional_avg_logprob_with_patch(
                    mt, prompt_ids, row["gpt_corr"], steered_mlp, states_to_patch
                )
                gem_o = conditional_avg_logprob_with_patch(
                    mt, prompt_ids, row["gem_orig"], steered_mlp, states_to_patch
                )
                gem_c = conditional_avg_logprob_with_patch(
                    mt, prompt_ids, row["gem_corr"], steered_mlp, states_to_patch
                )
            except Exception as e:
                print(f"Skipping row due to error: {e}")
                continue

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
    out_dir = f"results/gptscore/{args.mode}/{args.data}/{args.prompting_lang}"
    os.makedirs(out_dir, exist_ok=True)
    for lang in tqdm(LANGS):
        save_suffix = f"{model_name.split('/')[1]}_{args.dimension}_{lang}.json"
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
                annotation_path,
                sigma_list=sigma_list,
                dimension=args.dimension,
                intervention_layers=intervention_layers,
            )
            out_path = f"{out_dir}/corr_results_{save_suffix}"
            print("saving on", out_path)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

        if args.metric == "accuracy" or args.metric == "all":
            print(f"\n=== Likelihood–human accuracy: {lang} ===")
            results = run_eval_accuracy(
                lang,
                args.prompting_lang,
                steer_params,
                annotation_path,
                sigma_list=sigma_list,
                dimension=args.dimension,
                intervention_layers=intervention_layers,
            )
            out_path = f"{out_dir}/acc_results_{save_suffix}"
            print("saving on", out_path)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
