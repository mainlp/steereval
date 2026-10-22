import argparse
import ast
from collections import defaultdict
import re
from scipy.stats import spearmanr, kendalltau, pearsonr
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from statistics import mean

from map_utils import load_layer_map, save_layer_map
from vector_utils import load_vectors, save_vectors
from constants import MAX_LENGTH, LANG2LANGKEY, MAX_PAIRS
from baukit import TraceDict
from model_and_tok import ModelAndTokenizer


def tokenize_input(prompt: str, mt: ModelAndTokenizer, max_length=MAX_LENGTH):
    if "aya" in mt.model_name:
        messages = [{"role": "user", "content": prompt}]
        result = mt.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        # Newer transformers versions return BatchEncoding; older return a plain tensor.
        input_ids = result["input_ids"] if hasattr(result, "__getitem__") and not isinstance(result, torch.Tensor) else result
    else:
        input_ids = mt.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        )["input_ids"]

    attention_mask = torch.ones_like(input_ids)
    if mt.model.device:
        input_ids = input_ids.to(mt.model.device)
        attention_mask = attention_mask.to(mt.model.device)
    return input_ids, attention_mask


def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return (
        float(spearmanr(x, y).correlation),
        float(kendalltau(x, y).correlation),
        float(pearsonr(x, y)[0]),
    )


def conditional_avg_logprob_with_patch(
    mt,
    prompt_ids,
    candidate_text,
    steered_mlp,
    states_to_patch,
):
    cand_ids, cand_mask = tokenize_input(candidate_text, mt)
    cand_ids, cand_mask, prompt_ids = (
        cand_ids.to(mt.model.device),
        cand_mask.to(mt.model.device),
        prompt_ids.to(mt.model.device),
    )
    input_ids = torch.cat([prompt_ids, cand_ids[:, 1:]], dim=1)
    attention_mask = torch.ones_like(input_ids, device=mt.model.device)
    patch_spec = defaultdict(list)
    for t, layer in states_to_patch:
        patch_spec[layer].append(t)

    new_residual_t = torch.tensor(steered_mlp, device=mt.model.device, dtype=mt.model.dtype)

    def patch_rep(x, layer):
        if layer not in patch_spec:
            return x
        h = _untuple(x)
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

    log_probs = F.log_softmax(logits, dim=-1)
    prompt_len = prompt_ids.shape[1]
    candidate_token_ids = input_ids[0, prompt_len:]
    start = prompt_len - 1
    end = start + candidate_token_ids.shape[0]
    candidate_log_probs = log_probs[0, start:end, :]
    positions = torch.arange(candidate_token_ids.shape[0], device=mt.model.device)
    token_log_probs = candidate_log_probs[positions, candidate_token_ids]
    return token_log_probs.mean().item()


def _untuple(x):
    return x[0] if isinstance(x, tuple) else x


def get_layer_name(layer_idx, model_name):
    if "bloom" in model_name:
        return f"transformer.h.{layer_idx}"
    else:
        return f"model.layers.{layer_idx}"


def get_residual_last_token(
    model, input_ids, n_layers, model_name, attention_mask=None, index=-1
):
    model.eval()
    input_ids = input_ids.to(model.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    layers = [get_layer_name(i, model_name) for i in range(n_layers)]
    with torch.no_grad():
        with TraceDict(model, layers) as ret:
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            residual = [
                _untuple(ret[name].output)[0, index, :].detach().cpu().float().numpy()
                for name in layers
            ]
    return np.array(residual)


def get_residual_mean_over_tokens(
    model, input_ids, n_layers, model_name, attention_mask=None
):
    model.eval()
    input_ids = input_ids.to(model.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    layers = [get_layer_name(i, model_name) for i in range(n_layers)]
    with torch.no_grad():
        with TraceDict(model, layers) as ret:
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            residual = [
                _untuple(ret[name].output)[0].detach().cpu().float().numpy().mean(axis=0)
                for name in layers
            ]
    return np.array(residual)


def load_annotations_csv(path, metric, dimension):
    df = pd.read_csv(path)

    # correlation mode
    if metric == "correlation":
        if dimension == "coherence":
            score_cols = ["coherence_gpt", "coherence_gemini"]
            candidate_cols = ["gpt_corrupted_summary", "gemini_corrupted_summary"]
        elif dimension == "completeness":
            score_cols = ["gpt_grade", "gemini_grade"]
            candidate_cols = ["gpt_corrupted_summary", "gemini_corrupted_summary"]

        results = []
        for score_col, candidate_col in zip(score_cols, candidate_cols):
            if score_col in df.columns and candidate_col in df.columns:
                subset = df[[score_col, candidate_col, "text"]].dropna()
                for _, row in subset.iterrows():
                    try:
                        score = row[score_col]
                        if isinstance(score, str):
                            score = ast.literal_eval(score)
                        if isinstance(score, list):
                            score = mean([float(s) for s in score])
                        else:
                            score = float(score)
                        results.append(
                            {
                                "article": row["text"],
                                "candidate": row[candidate_col],
                                "human_score": score,
                            }
                        )
                    except Exception:
                        continue
        return results

    # accuracy mode
    elif metric in {"accuracy", "acc"}:
        required = [
            "text",
            "orig_gpt_prediction",
            "gpt_corrupted_summary",
            "orig_gemini_prediction",
            "gemini_corrupted_summary",
        ]
        results = []
        for _, row in df[required].dropna().iterrows():
            article = str(row["text"]).strip()
            gpt_orig = str(row["orig_gpt_prediction"]).strip()
            gpt_corr = str(row["gpt_corrupted_summary"]).strip()
            gem_orig = str(row["orig_gemini_prediction"]).strip()
            gem_corr = str(row["gemini_corrupted_summary"]).strip()
            if not all([article, gpt_orig, gpt_corr, gem_orig, gem_corr]):
                continue
            results.append(
                {
                    "article": article,
                    "gpt_orig": gpt_orig,
                    "gpt_corr": gpt_corr,
                    "gem_orig": gem_orig,
                    "gem_corr": gem_corr,
                }
            )
        return results


def get_en_xx(mt, lang, data, max_pairs=MAX_PAIRS):
    en_path = f"./data/{data}/en-{LANG2LANGKEY[lang]}/train.en"
    xx_path = f"./data/{data}/en-{LANG2LANGKEY[lang]}/train.{LANG2LANGKEY[lang]}"
    with open(en_path, encoding="utf-8") as f:
        en_lines = f.readlines()
    with open(xx_path, encoding="utf-8") as g:
        xx_lines = g.readlines()

    train_en, train_xx = [], []
    ind = 0
    # so that we can modify cap inside loop but save with actual max pairs
    cap = max_pairs
    while ind < cap and ind < min(len(en_lines), len(xx_lines)):
        if xx_lines[ind].strip() == "" or en_lines[ind].strip() == "":
            cap += 1
            ind += 1
            continue
        enc_xx = mt.tokenizer(
            xx_lines[ind].strip(),
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        residual_xx = get_residual_mean_over_tokens(
            mt.model,
            enc_xx["input_ids"].to(mt.model.device),
            mt.num_layers,
            mt.model_name,
            enc_xx.get("attention_mask", None).to(mt.model.device),
        )
        train_xx.append(residual_xx)
        enc_en = mt.tokenizer(
            en_lines[ind].strip(),
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        residual_en = get_residual_mean_over_tokens(
            mt.model,
            enc_en["input_ids"].to(mt.model.device),
            mt.num_layers,
            mt.model_name,
            enc_en.get("attention_mask", None).to(mt.model.device),
        )
        train_en.append(residual_en)
        ind += 1
    return np.stack(train_en), np.stack(train_xx)


def apply_steering(residual, steer_params, sigma, intervention_layers):
    if steer_params is None:
        return residual
    sample = steer_params[0]
    if not isinstance(sample, torch.Tensor):
        sample = torch.tensor(np.array(sample, dtype=np.float32))
    if sample.dim() == 2:
        return steer_from_residual_map(
            residual, steer_params, sigma, intervention_layers
        )
    elif sample.dim() == 1:
        return steer_from_residual_vector(
            residual, steer_params, sigma, intervention_layers
        )
    else:
        raise ValueError(f"Unrecognized steering parameter shape: {sample.shape}")


# here to avoid circular import in map_utils.py
def steer_from_residual_map(residual, Ws, sigma, intervention_layers):
    steered = residual.copy()
    for l in intervention_layers:
        sig = sigma[l] if isinstance(sigma, dict) else sigma
        direction = residual[l].dot(Ws[l])
        steered[l] = (1 - sig) * residual[l] + sig * direction
    return steered


# here to avoid circular import in vector_utils.py
def steer_from_residual_vector(residual, mean_diffs, sigma, intervention_layers):
    steered = residual.copy()
    for l in intervention_layers:
        sig = sigma[l] if isinstance(sigma, dict) else sigma
        direction = mean_diffs[l]
        steered[l] = residual[l] + sig * direction
    return steered


def get_steer_params(mt, lang, data, mode, max_pairs=MAX_PAIRS):
    if mode == "vector":
        return compute_crosslingual_vector(mt, lang, data, max_pairs=max_pairs)
    elif mode == "map":
        return learn_crosslingual_layer_map(mt, lang, data, max_pairs=max_pairs)
    elif mode == "baseline":
        return None
    else:
        raise ValueError(f"Unknown mode: {mode}")


def learn_crosslingual_layer_map(mt, lang, data, max_pairs=MAX_PAIRS):
    try:
        Ws = load_layer_map(lang, max_pairs, data, mt.model_name)
        if Ws is not None:
            return Ws
    except:
        pass

    print("Learning maps for", lang)
    train_en, train_xx = get_en_xx(mt, lang, data, max_pairs=max_pairs)
    Ws = []
    for l in range(mt.num_layers):
        W, *_ = np.linalg.lstsq(train_xx[:, l, :], train_en[:, l, :], rcond=None)
        Ws.append(W)
    save_layer_map(Ws, lang, max_pairs, data, mt.model_name)
    return Ws


def compute_crosslingual_vector(mt, lang, data, max_pairs=MAX_PAIRS):
    try:
        Vs = load_vectors(lang, max_pairs, data, mt.model_name)
        if Vs is not None:
            return Vs
    except:
        pass
    print("Computing intervention vectors for", lang)
    en, xx = get_en_xx(mt, lang, data, max_pairs=max_pairs)
    mean_diffs = []
    for l in range(mt.num_layers):
        diff = en[:, l, :].mean(axis=0) - xx[:, l, :].mean(axis=0)
        mean_diffs.append(diff)
        magnitude = np.linalg.norm(diff)
        print(f"{lang}: layer {l} mean diff magnitude: {magnitude:.4f}")
    save_vectors(mean_diffs, lang, max_pairs, data, mt.model_name)
    return mean_diffs


def prep_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument(
        "dimension",
        choices=["completeness", "coherence"],
        help="Which dimension to evaluate",
    )
    parser.add_argument("data")
    parser.add_argument(
        "prompting_lang",
        choices=["en", "tgt"],
        help="Language to use for prompting (en: English; tgt: target language)",
    )
    parser.add_argument(
        "metric",
        choices=["correlation", "accuracy", "all"],
        help="Which metric to evaluate",
    )
    parser.add_argument("gpu_id")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="List of layer indices to intervene on; default would steer all layers if unset",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "vector", "map"],
        help="Which intervention to run (baseline: no intervention; vector: mean difference vector; map: learned linear map)",
    )
    return parser
