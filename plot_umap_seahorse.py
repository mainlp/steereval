#!/usr/bin/env python3
"""PCA of final-layer representations with steering direction arrows.

For each source language (de, es, ru, tr, vi), loads or computes per-sentence
hidden states at the fixed layer (mean-pooled over tokens) for both English
(EN) and target-language (XX) parallel sentences, then projects to 2-D with PCA.

Because PCA is linear, the arrow from the XX centroid to the EN centroid in
PCA space is exactly the 2-D projection of the steering vector (EN_mean − XX_mean).
The direction and relative length are geometrically faithful.

Usage
-----
python plot_umap_seahorse.py
python plot_umap_seahorse.py --gpu 1 --out figures/pca_seahorse.pdf
python plot_umap_seahorse.py --max-pairs 200
"""

import argparse
import os

import numpy as np

# ── config ────────────────────────────────────────────────────────────────────

LANGS = ["de", "es", "ru", "tr", "vi"]
LANG_NAMES = {
    "de": "German", "es": "Spanish", "ru": "Russian",
    "tr": "Turkish", "vi": "Vietnamese",
}

MODELS = {
    "Llama-3.2-3B":   "meta-llama/Llama-3.2-3B",
    #"aya-expanse-8b": "CohereForAI/aya-expanse-8b",
}

FIXED_LAYER = {
    "meta-llama/Llama-3.2-3B":   27,
    #"CohereForAI/aya-expanse-8b": 31,
}

CACHE_DIR = "cache/seahorse"

LANG_COLORS = {
    "de": "#e41a1c",
    "es": "#377eb8",
    "ru": "#4daf4a",
    "tr": "#984ea3",
    "vi": "#ff7f00",
}


# ── cache paths ───────────────────────────────────────────────────────────────

def _rep_cache_path(kind: str, lang: str, max_pairs: int, model_name: str, layer: int) -> str:
    safe = model_name.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{kind}_reps_{lang}_{max_pairs}_{safe}_l{layer}.npy")


# ── loading ───────────────────────────────────────────────────────────────────

def _load_rep(kind, lang, max_pairs, model_name, layer):
    path = _rep_cache_path(kind, lang, max_pairs, model_name, layer)
    if os.path.exists(path):
        return np.load(path)
    return None


# ── computation ───────────────────────────────────────────────────────────────

def _compute_reps(mt, data_root, lang, max_pairs, layer):
    """Extract EN and XX representations at `layer` and cache them."""
    from constants import MAX_LENGTH
    from utils import get_residual_mean_over_tokens

    parallel_dir = os.path.join(data_root, f"en-{lang}")
    if not os.path.isdir(parallel_dir):
        print(f"  [{lang}] parallel data not found at {parallel_dir}, skipping.", flush=True)
        return None, None

    with open(os.path.join(parallel_dir, "train.en"), encoding="utf-8") as f:
        en_lines = [line.strip() for line in f]
    with open(os.path.join(parallel_dir, f"train.{lang}"), encoding="utf-8") as f:
        xx_lines = [line.strip() for line in f]

    EN_reps, XX_reps = [], []
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
            all_layers = get_residual_mean_over_tokens(
                mt.model,
                enc["input_ids"].to(mt.model.device),
                mt.num_layers,
                mt.model_name,
                enc.get("attention_mask", None).to(mt.model.device),
            )
            store.append(np.asarray(all_layers[layer], dtype=np.float32))
        i += 1
        if i % 50 == 0:
            print(f"    encoded {len(EN_reps)} pairs…", flush=True)

    EN = np.stack(EN_reps)
    XX = np.stack(XX_reps)

    for kind, arr in (("en", EN), ("xx", XX)):
        path = _rep_cache_path(kind, lang, max_pairs, mt.model_name, layer)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, arr)
        print(f"    Saved {kind} reps [{arr.shape}] → {path}", flush=True)

    return EN, XX


def ensure_reps(model_name, data_root, max_pairs, layer, gpu):
    """Return (en_reps, xx_reps) dicts keyed by lang."""
    missing = [
        lang for lang in LANGS
        if (_load_rep("en", lang, max_pairs, model_name, layer) is None
            or _load_rep("xx", lang, max_pairs, model_name, layer) is None)
    ]

    mt = None
    if missing:
        import torch
        from model_and_tok import ModelAndTokenizer
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"  Loading model {model_name} to compute reps for: {missing}…", flush=True)
        mt = ModelAndTokenizer(model_name)
        torch.set_grad_enabled(False)

    en_reps, xx_reps = {}, {}
    for lang in LANGS:
        en = _load_rep("en", lang, max_pairs, model_name, layer)
        xx = _load_rep("xx", lang, max_pairs, model_name, layer)
        if en is None or xx is None:
            en, xx = _compute_reps(mt, data_root, lang, max_pairs, layer)
        en_reps[lang] = en
        xx_reps[lang] = xx

    if mt is not None:
        import torch
        del mt
        torch.cuda.empty_cache()

    return en_reps, xx_reps


# ── PCA ───────────────────────────────────────────────────────────────────────

def build_matrix(en_reps, xx_reps):
    """Stack EN and XX sentence reps; return (X, langs, kinds)."""
    points, langs, kinds = [], [], []

    for lang in LANGS:
        if en_reps.get(lang) is not None:
            for rep in en_reps[lang]:
                points.append(rep)
                langs.append(lang)
                kinds.append("en")

        if xx_reps.get(lang) is not None:
            for rep in xx_reps[lang]:
                points.append(rep)
                langs.append(lang)
                kinds.append("xx")

    if not points:
        return None, None, None

    return np.stack(points).astype(np.float32), langs, kinds

    # Random baseline: n_pairs independent Gaussian vectors (seed=0, reproducible).
    # These stand in for a "language" whose representations are pure noise.
    # The arrow from their centroid to the EN centroid is the random steering direction.
    rng = np.random.default_rng(0)
    random_reps = rng.standard_normal((n_pairs, hidden_dim)).astype(np.float32)
    for rep in random_reps:
        points.append(rep)
        langs.append("random")
        kinds.append("random")

    return np.stack(points).astype(np.float32), langs, kinds


def fit_pca(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(X)
    return emb, pca.explained_variance_ratio_


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_model(ax, emb, langs, kinds, model_label, layer, var_ratio):
    """Scatter EN/XX points and draw XX→EN centroid arrows + random baseline arrow."""
    kind_cfg = {
        "en": dict(marker="o", s=10, alpha=0.35, zorder=1, edgecolors="none", lw=0),
        "xx": dict(marker="^", s=10, alpha=0.35, zorder=1, edgecolors="none", lw=0),
    }

    for kind in ("en", "xx"):
        for lang in LANGS:
            idx = [i for i, (k, l) in enumerate(zip(kinds, langs))
                   if k == kind and l == lang]
            if not idx:
                continue
            color = "grey" if kind == "en" else LANG_COLORS[lang]
            ax.scatter(emb[idx, 0], emb[idx, 1], c=color, **kind_cfg[kind])

    # Arrow from XX centroid → EN centroid.
    # Because PCA is linear: P@EN_mean − P@XX_mean = P@(EN_mean − XX_mean),
    # so this arrow is the exact 2-D projection of the steering vector.
    for lang in LANGS:
        en_idx = [i for i, (k, l) in enumerate(zip(kinds, langs))
                  if k == "en" and l == lang]
        xx_idx = [i for i, (k, l) in enumerate(zip(kinds, langs))
                  if k == "xx" and l == lang]
        if not en_idx or not xx_idx:
            continue

        en_c = emb[en_idx].mean(axis=0)
        xx_c = emb[xx_idx].mean(axis=0)

        ax.annotate(
            "",
            xy=en_c, xytext=xx_c,
            arrowprops=dict(
                arrowstyle="-|>",
                color=LANG_COLORS[lang],
                lw=2.0,
                mutation_scale=14,
            ),
            zorder=4,
        )

        mid = (xx_c + en_c) / 2
        ax.text(
            mid[0], mid[1], lang,
            fontsize=7, fontweight="bold", color=LANG_COLORS[lang],
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0),
            zorder=5,
        )

    ax.set_title(f"{model_label}  (layer {layer})", fontsize=10)
    ax.set_xlabel(f"PC 1  ({var_ratio[0]:.1%} var)", fontsize=8)
    ax.set_ylabel(f"PC 2  ({var_ratio[1]:.1%} var)", fontsize=8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def build_legend_handles():
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    lang_patches = [
        mpatches.Patch(color=LANG_COLORS[l], label=LANG_NAMES[l]) for l in LANGS
    ]

    kind_handles = [
        mlines.Line2D([], [], color="grey", marker="o", linestyle="none",
                      markersize=6, label="EN sentences"),
        mlines.Line2D([], [], color="grey", marker="^", linestyle="none",
                      markersize=6, label="XX sentences"),
        mlines.Line2D([], [], color="grey", marker=">", linestyle="-",
                      markersize=6, label="XX→EN steering direction"),
    ]

    return lang_patches + kind_handles


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PCA of final-layer representations with steering direction arrows"
    )
    parser.add_argument("--data-root", default="./data/ncwm_flores")
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--gpu",       default="0")
    parser.add_argument("--out",       default=None,
                        help="Output path (default: show interactively)")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for ax, (label, model_name) in zip(axes, MODELS.items()):
        layer = FIXED_LAYER[model_name]
        print(f"\n=== {label}  layer={layer} ===", flush=True)

        en_reps, xx_reps = ensure_reps(
            model_name, args.data_root, args.max_pairs, layer, args.gpu
        )

        X, langs, kinds = build_matrix(en_reps, xx_reps)
        if X is None:
            ax.set_title(f"{label}\n(no data)")
            continue

        print(f"  Fitting PCA on {X.shape[0]} × {X.shape[1]} matrix…", flush=True)
        emb, var_ratio = fit_pca(X)
        plot_model(ax, emb, langs, kinds, label, layer, var_ratio)

    fig.legend(
        handles=build_legend_handles(),
        loc="lower center",
        ncol=len(LANGS) + 3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    plt.suptitle(
        "PCA of final-layer representations — arrows show XX→EN steering direction",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
