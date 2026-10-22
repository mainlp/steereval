"""
Evaluate COMET-based INCLINE intervention on HiTZ/BASSE summarization data.

Supports both the Spanish ('es') and Basque ('eu') language splits.
Sigma is tuned on a validation split (first half of documents by URL),
then the best sigma is applied to the held-out test split.

Usage:
    python eval_hitz_summarization_es.py \
        --lang es \
        --parallel-data ./data/ncwm_old/en-es \
        --method vector \
        --max-pairs 500 \
        --out results_hitz_es_vector.json
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from types import MethodType

from load_basse import load_basse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance", "5W1H"]

SIGMAS_BY_METHOD: Dict[str, List[float]] = {
    "vector": [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "map": [0.0, 0.25, 0.5, 0.75, 1.0],
}


# ─────────────────────────── COMET setup ────────────────────────────

def _load_comet():
    from comet import download_model, load_from_checkpoint
    from comet.models.pooling_utils import average_pooling, max_pooling

    print("Loading COMET model...", flush=True)
    ckpt = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(ckpt).to(DEVICE).eval()
    tok = model.encoder.tokenizer
    torch.set_grad_enabled(False)

    def _compute_sentence_embedding(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        *,
        add_delta=None,
        method=None,
    ):
        # Call the underlying HuggingFace model directly with return_dict=True to
        # avoid the 3-tuple unpacking bug in COMET's XLMREncoder.forward, which
        # assumes a pooler_output middle element that XLM-R may not return.
        raw = self.encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        enc = {
            "sentemb": raw.last_hidden_state[:, 0, :],
            "all_layers": raw.hidden_states,
        }

        if self.layerwise_attention:
            embeddings = self.layerwise_attention(enc["all_layers"], attention_mask)
        elif 0 <= self.hparams.layer < self.encoder.num_layers:
            embeddings = enc["all_layers"][self.hparams.layer]
        else:
            raise Exception(f"Invalid model layer {self.hparams.layer}.")

        pool = self.hparams.pool
        if pool == "default":
            sentemb = enc["sentemb"]
        elif pool == "max":
            sentemb = max_pooling(input_ids, embeddings, self.encoder.tokenizer.pad_token_id)
        elif pool == "avg":
            sentemb = average_pooling(
                input_ids, embeddings, attention_mask,
                self.encoder.tokenizer.pad_token_id,
                getattr(self.encoder.tokenizer, "sep_token_id", None),
                self.use_context,
            )
        elif pool == "cls":
            sentemb = embeddings[:, 0, :]
        else:
            raise Exception("Invalid pooling technique.")

        if add_delta is not None:
            add_delta = add_delta.to(sentemb.dtype).to(sentemb.device)
            if method == "map":
                sentemb = add_delta.unsqueeze(0)
            else:
                sentemb = sentemb + add_delta.unsqueeze(0)
        return sentemb

    def _get_sentence_embedding(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        *,
        add_delta=None,
        method=None,
    ):
        if self.caching and add_delta is None:
            return self.retrieve_sentence_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        return self.compute_sentence_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            add_delta=add_delta,
            method=method,
        )

    model.compute_sentence_embedding = MethodType(_compute_sentence_embedding, model)
    model.get_sentence_embedding = MethodType(_get_sentence_embedding, model)
    return model, tok


# ─────────────────────────── embedding helpers ──────────────────────

@torch.no_grad()
def _embed(text: str, comet, tok) -> torch.Tensor:
    batch = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    return comet.get_sentence_embedding(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        add_delta=None,
    ).squeeze(0)


def _build_delta(text: str, W: np.ndarray, sigma: float, method: str,
                 comet, tok, cache: dict) -> torch.Tensor:
    if method == "map":
        if text not in cache:
            p = _embed(text, comet, tok).detach().cpu().numpy()
            cache[text] = (p, p @ W)
        p_np, mapped_np = cache[text]
        p_t = torch.from_numpy(p_np).to(DEVICE).float()
        m_t = torch.from_numpy(mapped_np).to(DEVICE).float()
        return (1 - sigma) * p_t + sigma * m_t
    elif method == "vector":
        return sigma * torch.from_numpy(W).to(DEVICE).float()
    else:
        raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def _comet_score(
    hyp: str, ref: str, W: np.ndarray, sigma: float, method: str,
    comet, tok, cache: dict,
) -> float:
    src_batch = tok("", return_tensors="pt").to(DEVICE)
    hyp_batch = tok(hyp, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    ref_batch = tok(ref, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)

    d_hyp = _build_delta(hyp, W, sigma, method, comet, tok, cache)
    d_ref = _build_delta(ref, W, sigma, method, comet, tok, cache)

    src_emb = comet.get_sentence_embedding(
        input_ids=src_batch["input_ids"],
        attention_mask=src_batch["attention_mask"],
        add_delta=None,
    )
    hyp_emb = comet.get_sentence_embedding(
        input_ids=hyp_batch["input_ids"],
        attention_mask=hyp_batch["attention_mask"],
        add_delta=d_hyp, method=method,
    )
    ref_emb = comet.get_sentence_embedding(
        input_ids=ref_batch["input_ids"],
        attention_mask=ref_batch["attention_mask"],
        add_delta=d_ref, method=method,
    )
    return comet.estimate(src_emb, hyp_emb, ref_emb).score.detach().mean().item()


# ─────────────────────────── mapping learning ───────────────────────

def learn_parallel_mapping(parallel_dir: str, method: str, max_pairs: int,
                           comet, tok, lang_key: str = "es") -> np.ndarray:
    en_path = f"{parallel_dir}/train.en"
    xx_path = f"{parallel_dir}/train.{lang_key}"

    with open(en_path, encoding="utf-8") as f:
        en_lines = [l.strip() for l in f]
    with open(xx_path, encoding="utf-8") as f:
        xx_lines = [l.strip() for l in f]

    EN, XX = [], []
    i, cap = 0, max_pairs
    while len(EN) < cap and i < min(len(en_lines), len(xx_lines)):
        if not en_lines[i] or not xx_lines[i]:
            cap += 1
            i += 1
            continue
        EN.append(_embed(en_lines[i], comet, tok).detach().cpu().numpy())
        XX.append(_embed(xx_lines[i], comet, tok).detach().cpu().numpy())
        i += 1
        if i % 100 == 0:
            print(f"  encoded {len(EN)} pairs...", flush=True)

    EN = np.stack(EN)
    XX = np.stack(XX)

    if method == "map":
        W, *_ = np.linalg.lstsq(XX, EN, rcond=None)
        return W.astype(np.float32)
    elif method == "vector":
        return (EN.mean(0) - XX.mean(0)).astype(np.float32)
    else:
        raise ValueError(f"Unknown method: {method}")


# ─────────────────────────── data loading ───────────────────────────
# Delegated to load_basse.load_basse(); see that module for details.


# ─────────────────────────── evaluation ─────────────────────────────

def safe_corr(x, y) -> Tuple[float, float, float]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    sp = float(spearmanr(x, y).correlation)
    kd = float(kendalltau(x, y).correlation)
    pe = float(pearsonr(x, y)[0])
    return sp, kd, pe


def eval_correlation(
    rows: List[dict],
    W: np.ndarray,
    sigma: float,
    method: str,
    comet,
    tok,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute per-dimension Spearman/Kendall/Pearson for a given sigma.

    Each unique (candidate, article) pair is scored only once even though the
    same pair appears once per quality dimension in `rows`.
    """
    emb_cache: dict = {}   # text -> (p_np, mapped_np) for map; unused for vector
    score_cache: dict = {} # (candidate, article) -> comet score

    by_dim: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))

    for r in rows:
        key = (r["candidate"], r["article"])
        if key not in score_cache:
            score_cache[key] = _comet_score(
                r["candidate"], r["article"], W, sigma, method, comet, tok, emb_cache
            )
        by_dim[r["dimension"]][0].append(score_cache[key])
        by_dim[r["dimension"]][1].append(r["human_score"])

    return {dim: safe_corr(scores, humans) for dim, (scores, humans) in by_dim.items()}


def eval_pairwise_accuracy(
    rows: List[dict],
    W: np.ndarray,
    sigma: float,
    method: str,
    comet,
    tok,
) -> Dict[str, Tuple[int, int]]:
    """
    Pairwise accuracy: for each document×dimension, create (better, worse)
    pairs from different summaries and check if COMET agrees with human ranking.
    """
    cache: dict = {}

    # Group by (article, dimension)
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["article"], r["dimension"])].append(r)

    by_dim: Dict[str, List[bool]] = defaultdict(list)
    for (article, dim), items in groups.items():
        if len(items) < 2:
            continue
        scored = [
            (_comet_score(r["candidate"], article, W, sigma, method, comet, tok, cache),
             r["human_score"])
            for r in items
        ]
        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                m_i, h_i = scored[i]
                m_j, h_j = scored[j]
                if h_i == h_j:
                    continue
                correct = (m_i > m_j) == (h_i > h_j)
                by_dim[dim].append(correct)

    return {
        dim: (sum(v), len(v))
        for dim, v in by_dim.items()
    }


# ─────────────────────────── sigma sweep ────────────────────────────

def sweep_sigma(
    rows: List[dict],
    W: np.ndarray,
    sigma_list: List[float],
    method: str,
    comet,
    tok,
    split_label: str = "val",
) -> Dict[float, dict]:
    results = {}
    for sigma in sigma_list:
        dim_corrs = eval_correlation(rows, W, sigma, method, comet, tok)
        mean_sp = float(np.mean([sp for sp, _, _ in dim_corrs.values()]))
        entry = {
            "mean_spearman": mean_sp,
            "per_dim": {
                dim: {"spearman": sp, "kendall": kd, "pearson": pe}
                for dim, (sp, kd, pe) in dim_corrs.items()
            },
        }
        print(
            f"[{split_label}][{method}] sigma={sigma:+.2f}  mean_Spearman={mean_sp:.4f}",
            flush=True,
        )
        for dim, (sp, kd, pe) in dim_corrs.items():
            print(f"    [{dim}] Sp={sp:.4f} Kd={kd:.4f} Pe={pe:.4f}", flush=True)
        results[sigma] = entry
    return results


def best_sigma_per_dim(val_results: Dict[float, dict]) -> Dict[str, float]:
    """For each dimension, pick the sigma with the highest val Spearman."""
    best: Dict[str, Tuple[float, float]] = {}  # dim -> (sigma, spearman)
    for sigma, entry in val_results.items():
        for dim, corrs in entry["per_dim"].items():
            sp = corrs["spearman"]
            if dim not in best or sp > best[dim][1]:
                best[dim] = (sigma, sp)
    return {dim: sigma for dim, (sigma, _) in best.items()}


def eval_correlation_per_dim_sigma(
    rows: List[dict],
    W: np.ndarray,
    sigma_by_dim: Dict[str, float],
    method: str,
    comet,
    tok,
) -> Dict[str, Tuple[float, float, float]]:
    """Score each dimension's rows with that dimension's own sigma.

    Groups scoring by unique sigma value so each (candidate, article) pair is
    only run through the COMET forward once per unique sigma.
    """
    unique_sigmas = set(sigma_by_dim.values())

    emb_cache: dict = {}
    # sigma -> {(candidate, article) -> score}
    scores_by_sigma: Dict[float, Dict[Tuple[str, str], float]] = {}
    for sigma in unique_sigmas:
        score_cache: Dict[Tuple[str, str], float] = {}
        for r in rows:
            key = (r["candidate"], r["article"])
            if key not in score_cache:
                score_cache[key] = _comet_score(
                    r["candidate"], r["article"], W, sigma, method, comet, tok, emb_cache
                )
        scores_by_sigma[sigma] = score_cache

    by_dim: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    for r in rows:
        dim = r["dimension"]
        sigma = sigma_by_dim.get(dim, 0.0)
        key = (r["candidate"], r["article"])
        by_dim[dim][0].append(scores_by_sigma[sigma][key])
        by_dim[dim][1].append(r["human_score"])

    return {dim: safe_corr(scores, humans) for dim, (scores, humans) in by_dim.items()}


# ─────────────────────────── main ───────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Eval INCLINE COMET on HiTZ/BASSE with val sigma tuning"
    )
    parser.add_argument(
        "--lang", choices=["es", "eu"], default="es",
        help="BASSE language split to evaluate (es=Spanish, eu=Basque)",
    )
    parser.add_argument(
        "--parallel-data", required=True,
        help="Dir containing train.en and train.<lang> parallel files for XX→EN mapping"
    )
    parser.add_argument("--method", choices=["vector", "map"], default="vector")
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--val-frac", type=float, default=0.5,
                        help="Fraction of unique documents to use for validation")
    parser.add_argument("--out", default="results_hitz_basse.json")
    parser.add_argument("--pairwise", action="store_true",
                        help="Also compute pairwise accuracy on the test set")
    args = parser.parse_args()

    comet, tok = _load_comet()

    # Learn XX→EN mapping from parallel text
    print(f"\nLearning {args.method} mapping from {args.parallel_data} ...", flush=True)
    W = learn_parallel_mapping(args.parallel_data, args.method, args.max_pairs, comet, tok, lang_key=args.lang)

    # Load and split the BASSE dataset for the selected language
    val_rows, test_rows = load_basse(args.lang, args.val_frac)

    sigma_list = SIGMAS_BY_METHOD[args.method]

    # ── Val sweep: tune sigma ──────────────────────────────────────
    print("\n=== Val sigma sweep ===", flush=True)
    val_results = sweep_sigma(val_rows, W, sigma_list, args.method, comet, tok, "val")

    # Shared best sigma (maximises mean Spearman across all dimensions)
    shared_best_sigma = max(val_results, key=lambda s: val_results[s]["mean_spearman"])
    print(
        f"\nShared best sigma (val): {shared_best_sigma:+.2f}  "
        f"mean_Spearman={val_results[shared_best_sigma]['mean_spearman']:.4f}",
        flush=True,
    )

    # Per-dimension best sigma
    sigma_by_dim = best_sigma_per_dim(val_results)
    print("\nPer-dimension best sigma (val):", flush=True)
    for dim, sigma in sigma_by_dim.items():
        val_sp = val_results[sigma]["per_dim"][dim]["spearman"]
        print(f"  [{dim}] sigma={sigma:+.2f}  val_Spearman={val_sp:.4f}", flush=True)

    # ── Baseline (sigma=0) on test ─────────────────────────────────
    baseline_sigma = 0.0
    print("\n=== Test baseline (sigma=0.0) ===", flush=True)
    baseline_corrs = eval_correlation(test_rows, W, baseline_sigma, args.method, comet, tok)
    baseline_mean = float(np.mean([sp for sp, _, _ in baseline_corrs.values()]))
    for dim, (sp, kd, pe) in baseline_corrs.items():
        print(f"  [{dim}] Sp={sp:.4f} Kd={kd:.4f} Pe={pe:.4f}", flush=True)
    print(f"  mean_Spearman={baseline_mean:.4f}", flush=True)

    # ── Test with shared best sigma ────────────────────────────────
    print(f"\n=== Test with shared best sigma ({shared_best_sigma:+.2f}) ===", flush=True)
    test_corrs_shared = eval_correlation(test_rows, W, shared_best_sigma, args.method, comet, tok)
    test_mean_shared = float(np.mean([sp for sp, _, _ in test_corrs_shared.values()]))
    for dim, (sp, kd, pe) in test_corrs_shared.items():
        print(f"  [{dim}] Sp={sp:.4f} Kd={kd:.4f} Pe={pe:.4f}", flush=True)
    print(f"  mean_Spearman={test_mean_shared:.4f}", flush=True)

    # ── Test with per-dimension best sigma ────────────────────────
    print("\n=== Test with per-dimension best sigma ===", flush=True)
    test_corrs_per_dim = eval_correlation_per_dim_sigma(
        test_rows, W, sigma_by_dim, args.method, comet, tok
    )
    test_mean_per_dim = float(np.mean([sp for sp, _, _ in test_corrs_per_dim.values()]))
    for dim, (sp, kd, pe) in test_corrs_per_dim.items():
        sigma = sigma_by_dim.get(dim, 0.0)
        print(f"  [{dim}] sigma={sigma:+.2f}  Sp={sp:.4f} Kd={kd:.4f} Pe={pe:.4f}", flush=True)
    print(f"  mean_Spearman={test_mean_per_dim:.4f}", flush=True)

    # ── Optional: pairwise accuracy on test ───────────────────────
    pairwise_test = None
    if args.pairwise:
        print("\n=== Pairwise accuracy on test (per-dimension sigma) ===", flush=True)
        # Run once per unique sigma to avoid redundant scoring
        unique_sigmas = set(sigma_by_dim.values())
        pairwise_by_sigma = {
            sigma: eval_pairwise_accuracy(test_rows, W, sigma, args.method, comet, tok)
            for sigma in unique_sigmas
        }
        pairwise_test = {}
        for dim, sigma in sigma_by_dim.items():
            if dim in pairwise_by_sigma[sigma]:
                pairwise_test[dim] = pairwise_by_sigma[sigma][dim]
        for dim, (correct, total) in pairwise_test.items():
            acc = correct / total if total > 0 else 0.0
            sigma = sigma_by_dim.get(dim, 0.0)
            print(f"  [{dim}] sigma={sigma:+.2f}  Accuracy={acc:.4f} ({correct}/{total})", flush=True)

    # ── Save results ───────────────────────────────────────────────
    output = {
        "method": args.method,
        "val_results": {str(s): v for s, v in val_results.items()},
        "sigma_by_dim": {dim: float(s) for dim, s in sigma_by_dim.items()},
        "shared_best_sigma": shared_best_sigma,
        "test_baseline": {
            "sigma": baseline_sigma,
            "per_dim": {
                dim: {"spearman": sp, "kendall": kd, "pearson": pe}
                for dim, (sp, kd, pe) in baseline_corrs.items()
            },
            "mean_spearman": baseline_mean,
        },
        "test_shared_sigma": {
            "sigma": shared_best_sigma,
            "per_dim": {
                dim: {"spearman": sp, "kendall": kd, "pearson": pe}
                for dim, (sp, kd, pe) in test_corrs_shared.items()
            },
            "mean_spearman": test_mean_shared,
        },
        "test_per_dim_sigma": {
            "sigma_by_dim": {dim: float(s) for dim, s in sigma_by_dim.items()},
            "per_dim": {
                dim: {"spearman": sp, "kendall": kd, "pearson": pe}
                for dim, (sp, kd, pe) in test_corrs_per_dim.items()
            },
            "mean_spearman": test_mean_per_dim,
        },
    }
    if pairwise_test is not None:
        output["test_pairwise_accuracy"] = {
            dim: {"correct": c, "total": t, "accuracy": c / t if t > 0 else 0.0}
            for dim, (c, t) in pairwise_test.items()
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
