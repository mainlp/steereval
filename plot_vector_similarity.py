#!/usr/bin/env python3
"""Plot pairwise cosine similarity heatmaps of XX→EN steering vectors.

Usage
-----
python plot_vector_similarity.py
python plot_vector_similarity.py --gpu 1 --out figures/vector_similarity.pdf
python plot_vector_similarity.py --data-root ./data/ncwm_flores --max-pairs 500
"""

import argparse
import itertools
import os
from typing import Dict, List, Optional

import numpy as np

# ── config ────────────────────────────────────────────────────────────────────

LANGS = ["de", "es", "ru", "tr", "vi"]

MODELS = {
    #"Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    "aya-expanse-8b": "CohereForAI/aya-expanse-8b",
}

CACHE_DIR = "cache/seahorse"
METHOD = "vector"

# Layer ranges used in the respective seahorse sigma search scripts
SEARCH_LAYERS = {
    #"meta-llama/Llama-3.2-3B": list(range(21, 28)),      # 21–27
    "CohereForAI/aya-expanse-8b": list(range(24, 32)),   # 24–31
}


# ── cache paths ───────────────────────────────────────────────────────────────

def _cache_path(lang: str, max_pairs: int, model_name: str) -> str:
    safe = model_name.replace("/", "_")
    return os.path.join(
        CACHE_DIR,
        f"{lang}_{METHOD}_{max_pairs}_{safe}_residual.npy",
    )


def _en_mean_cache_path(lang: str, max_pairs: int, model_name: str) -> str:
    safe = model_name.replace("/", "_")
    return os.path.join(
        CACHE_DIR,
        f"{lang}_{METHOD}_{max_pairs}_{safe}_en_mean.npy",
    )


def _load_cached(
    lang: str,
    max_pairs: int,
    model_name: str,
) -> Optional[List[np.ndarray]]:
    path = _cache_path(lang, max_pairs, model_name)

    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    return list(data)


# ── vector computation ────────────────────────────────────────────────────────

def _compute_and_cache(
    mt,
    parallel_dir: str,
    lang_key: str,
    max_pairs: int,
) -> List[np.ndarray]:

    from constants import MAX_LENGTH
    from utils import get_residual_mean_over_tokens

    en_path = os.path.join(parallel_dir, "train.en")
    xx_path = os.path.join(parallel_dir, f"train.{lang_key}")

    if not os.path.exists(en_path) or not os.path.exists(xx_path):
        raise FileNotFoundError(
            f"Parallel data not found at {parallel_dir} for lang={lang_key}"
        )

    with open(en_path, encoding="utf-8") as f:
        en_lines = [l.strip() for l in f]

    with open(xx_path, encoding="utf-8") as f:
        xx_lines = [l.strip() for l in f]

    EN_reps = []
    XX_reps = []

    i = 0
    cap = max_pairs

    while len(EN_reps) < cap and i < min(len(en_lines), len(xx_lines)):

        if not en_lines[i] or not xx_lines[i]:
            cap += 1
            i += 1
            continue

        for text, store in (
            (en_lines[i], EN_reps),
            (xx_lines[i], XX_reps),
        ):
            enc = mt.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
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

        if i % 50 == 0:
            print(f"    encoded {len(EN_reps)} pairs…", flush=True)

    EN = np.stack(EN_reps)
    XX = np.stack(XX_reps)

    en_means = [EN[:, l, :].mean(0) for l in range(mt.num_layers)]

    params = [
        en_means[l] - XX[:, l, :].mean(0)
        for l in range(mt.num_layers)
    ]

    path = _cache_path(lang_key, max_pairs, mt.model_name)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    np.save(path, np.array(params, dtype=object), allow_pickle=True)

    en_mean_path = _en_mean_cache_path(lang_key, max_pairs, mt.model_name)
    np.save(en_mean_path, np.array(en_means, dtype=object), allow_pickle=True)

    print(f"    Saved → {path}", flush=True)

    return params


def ensure_vectors(
    model_name: str,
    data_root: str,
    max_pairs: int,
    gpu: str,
) -> Dict[str, Optional[List[np.ndarray]]]:
    """Return {lang: vecs_per_layer}."""

    missing = [
        lang
        for lang in LANGS
        if _load_cached(lang, max_pairs, model_name) is None
    ]

    mt = None

    if missing:
        import torch
        from model_and_tok import ModelAndTokenizer

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        print(f"  Loading model {model_name}…", flush=True)

        mt = ModelAndTokenizer(model_name)

        torch.set_grad_enabled(False)

    result: Dict[str, Optional[List[np.ndarray]]] = {}

    for lang in LANGS:

        cached = _load_cached(lang, max_pairs, model_name)

        if cached is not None:
            result[lang] = cached
            continue

        parallel_dir = os.path.join(data_root, f"en-{lang}")

        if not os.path.isdir(parallel_dir):
            print(
                f"  [{lang}] parallel data not found at {parallel_dir}, skipping."
            )
            result[lang] = None
        else:
            print(f"  [{lang}] computing vectors…", flush=True)

            result[lang] = _compute_and_cache(
                mt,
                parallel_dir,
                lang,
                max_pairs,
            )

    if mt is not None:
        import torch
        del mt
        torch.cuda.empty_cache()

    return result


def ensure_en_means(
    model_name: str,
    data_root: str,
    max_pairs: int,
    gpu: str,
) -> Optional[List[np.ndarray]]:
    """Return per-layer EN mean representations, computing and caching if needed."""

    for lang in LANGS:
        en_mean_path = _en_mean_cache_path(lang, max_pairs, model_name)
        if os.path.exists(en_mean_path):
            return list(np.load(en_mean_path, allow_pickle=True))

    # Find the first language with parallel data to compute EN means from
    compute_lang = None
    for lang in LANGS:
        parallel_dir = os.path.join(data_root, f"en-{lang}")
        if (
            os.path.exists(os.path.join(parallel_dir, "train.en"))
            and os.path.exists(os.path.join(parallel_dir, f"train.{lang}"))
        ):
            compute_lang = lang
            break

    if compute_lang is None:
        return None

    import torch
    from model_and_tok import ModelAndTokenizer
    from constants import MAX_LENGTH
    from utils import get_residual_mean_over_tokens

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    print(f"  Loading model {model_name} to compute EN means…", flush=True)

    mt = ModelAndTokenizer(model_name)
    torch.set_grad_enabled(False)

    parallel_dir = os.path.join(data_root, f"en-{compute_lang}")

    with open(os.path.join(parallel_dir, "train.en"), encoding="utf-8") as f:
        en_lines = [l.strip() for l in f]

    with open(os.path.join(parallel_dir, f"train.{compute_lang}"), encoding="utf-8") as f:
        xx_lines = [l.strip() for l in f]

    EN_reps = []
    i = 0
    cap = max_pairs

    while len(EN_reps) < cap and i < min(len(en_lines), len(xx_lines)):

        if not en_lines[i] or not xx_lines[i]:
            cap += 1
            i += 1
            continue

        enc = mt.tokenizer(
            en_lines[i],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        rep = get_residual_mean_over_tokens(
            mt.model,
            enc["input_ids"].to(mt.model.device),
            mt.num_layers,
            mt.model_name,
            enc.get("attention_mask", None).to(mt.model.device),
        )

        EN_reps.append(rep)
        i += 1

        if i % 50 == 0:
            print(f"    encoded {len(EN_reps)} EN reps…", flush=True)

    EN = np.stack(EN_reps)
    en_means = [EN[:, l, :].mean(0) for l in range(mt.num_layers)]

    en_mean_path = _en_mean_cache_path(compute_lang, max_pairs, mt.model_name)
    os.makedirs(os.path.dirname(en_mean_path), exist_ok=True)
    np.save(en_mean_path, np.array(en_means, dtype=object), allow_pickle=True)

    print(f"    Saved EN means → {en_mean_path}", flush=True)

    del mt
    torch.cuda.empty_cache()

    return en_means


# ── similarity ────────────────────────────────────────────────────────────────

def cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    u = u.astype(float)
    v = v.astype(float)

    denom = np.linalg.norm(u) * np.linalg.norm(v)

    return float(np.dot(u, v) / denom) if denom > 0 else 0.0


def avg_pairwise_cosine_per_layer(
    lang_vectors: Dict[str, Optional[List[np.ndarray]]],
) -> List[float]:

    available = [v for v in lang_vectors.values() if v is not None]

    if not available:
        return []

    n_layers = len(available[0])

    result = []

    for l in range(n_layers):

        vecs = [
            np.asarray(v[l], dtype=float)
            for v in available
            if l < len(v)
        ]

        if len(vecs) < 2:
            result.append(float("nan"))
        else:
            pairs = [
                cosine_sim(u, v)
                for u, v in itertools.combinations(vecs, 2)
            ]
            result.append(float(np.mean(pairs)))

    return result


def best_layer_in_range(
    sims: List[float],
    search_layers: List[int],
) -> int:
    """Return layer with highest avg cosine sim in search range."""

    valid = [
        (l, sims[l])
        for l in search_layers
        if l < len(sims) and not np.isnan(sims[l])
    ]

    if not valid:
        return search_layers[0]

    return max(valid, key=lambda x: x[1])[0]


def pairwise_cosine_matrix(
    lang_vectors: Dict[str, Optional[List[np.ndarray]]],
    layer: int,
    langs: List[str] = None,
) -> np.ndarray:
    """Return pairwise cosine similarity matrix."""

    if langs is None:
        langs = LANGS

    n = len(langs)

    mat = np.full((n, n), float("nan"))

    for i, li in enumerate(langs):
        for j, lj in enumerate(langs):

            vi = lang_vectors.get(li)
            vj = lang_vectors.get(lj)

            if (
                vi is not None
                and vj is not None
                and layer < len(vi)
                and layer < len(vj)
            ):
                mat[i, j] = cosine_sim(
                    np.asarray(vi[layer], dtype=float),
                    np.asarray(vj[layer], dtype=float),
                )

    return mat


# ── main ──────────────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser(
        description="Plot pairwise cosine similarity heatmaps"
    )

    parser.add_argument(
        "--data-root",
        default="./data/ncwm_flores",
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--gpu",
        default="0",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: show interactively)",
    )

    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_models = len(MODELS)

    fig = plt.figure(figsize=(6 * n_models, 4))

    gs = gridspec.GridSpec(
        1,
        n_models,
        wspace=0.35,
    )

    heatmap_axes = [
        fig.add_subplot(gs[0, i])
        for i in range(n_models)
    ]

    for (label, model_name), ax_hm in zip(
        MODELS.items(),
        heatmap_axes,
    ):

        print(f"\n=== {label} ===", flush=True)

        lang_data = ensure_vectors(
            model_name,
            args.data_root,
            args.max_pairs,
            args.gpu,
        )

        available = next(
            (v for v in lang_data.values() if v is not None),
            None,
        )

        if available is None:
            print(f"  No vectors available for {label}, skipping.")
            continue

        # Compute average pairwise similarity across layers
        sims = avg_pairwise_cosine_per_layer(lang_data)

        # Choose best layer within search range
        search = SEARCH_LAYERS[model_name]
        best = best_layer_in_range(sims, search)

        # Add random baseline vector
        available = [
            v for v in lang_data.values()
            if v is not None
        ]

        if available:
            hidden_dim = np.asarray(
                available[0][best],
                dtype=float,
            ).shape[0]

            n_layers = len(available[0])

            en_means = ensure_en_means(
                model_name,
                args.data_root,
                args.max_pairs,
                args.gpu,
            )

            rng = np.random.default_rng(0)
            rand_vecs = []
            for l in range(n_layers):
                en_mean_l = np.asarray(en_means[l], dtype=np.float32)
                rand_xx = rng.standard_normal((args.max_pairs, hidden_dim)).astype(np.float32)
                rand_vecs.append(en_mean_l - rand_xx.mean(0))
            lang_data["rand"] = rand_vecs

        heatmap_langs = LANGS + (["rand"] if "rand" in lang_data else [])

        mat = pairwise_cosine_matrix(
            lang_data,
            best,
            langs=heatmap_langs,
        )

        im = ax_hm.imshow(
            mat,
            vmin=-1,
            vmax=1,
            cmap="RdYlGn",
        )

        ax_hm.set_xticks(range(len(heatmap_langs)))
        ax_hm.set_yticks(range(len(heatmap_langs)))

        ax_hm.set_xticklabels(heatmap_langs)
        ax_hm.set_yticklabels(heatmap_langs)

        ax_hm.set_title(
            f"{label}\nlayer {best} "
            f"(best in {search[0]}–{search[-1]})",
            fontsize=9,
        )

        for i in range(len(heatmap_langs)):
            for j in range(len(heatmap_langs)):

                if not np.isnan(mat[i, j]):

                    ax_hm.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

        plt.colorbar(
            im,
            ax=ax_hm,
            fraction=0.046,
            pad=0.04,
        )

    if args.out:

        os.makedirs(
            os.path.dirname(os.path.abspath(args.out)),
            exist_ok=True,
        )

        fig.savefig(
            args.out,
            dpi=150,
            bbox_inches="tight",
        )

        print(f"\nSaved → {args.out}")

    else:
        plt.show()


if __name__ == "__main__":
    main()