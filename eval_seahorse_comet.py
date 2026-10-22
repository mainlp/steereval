"""COMET-based INCLINE evaluation on SEAHORSE.

Uses Unbabel/wmt22-comet-da with cross-lingual pooling-space intervention
(vector or map) applied to both candidate and reference embeddings.
References are loaded from the SEAHORSE TSV (rows with model=reference).

Sigma is selected on SEAHORSE's own validation split by maximising mean
AUC-ROC across the six quality dimensions.  Final results are reported on
the test split.

Usage
-----
# Spanish, vector steering:
python eval_seahorse_comet.py \\
    --lang           es \\
    --data-dir       ./data/seahorse \\
    --parallel-data  ./data/ncwm_flores/en-es \\
    --method         vector \\
    --gpu            0 \\
    --out            results/seahorse_comet/es_vector.json
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from intervention_comet import (
    comet,
    tok,
    DEVICE,
    MAX_LEN,
    pooled_delta,
    pooled_delta_vector,
)
from load_seahorse import SEAHORSE_LANG2NAME, load_seahorse
from constants import MAX_PAIRS, SIGMAS_BY_MODE as SIGMAS
from comet.models.pooling_utils import average_pooling, max_pooling
from types import MethodType

# XLMREncoder.forward unpacks self.model() as (last_hidden, _, all_layers), but
# newer transformers skips None fields in ModelOutput iteration, so the unpack
# fails. Fix: replace compute_sentence_embedding to call self.encoder.model
# directly with return_dict=True, bypassing the encoder's own forward entirely.
def _fixed_compute_sentence_embedding(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None,
    *,
    add_delta: Optional[torch.Tensor] = None,
    method: Optional[str] = None,
) -> torch.Tensor:
    out = self.encoder.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    all_layers = list(out.hidden_states)
    if self.layerwise_attention:
        embeddings = self.layerwise_attention(all_layers, attention_mask)
    elif 0 <= self.hparams.layer < self.encoder.num_layers:
        embeddings = all_layers[self.hparams.layer]
    else:
        raise Exception(f"Invalid model layer {self.hparams.layer}.")

    if self.hparams.pool == "default":
        sentemb = out.last_hidden_state[:, 0, :]
    elif self.hparams.pool == "max":
        sentemb = max_pooling(input_ids, embeddings, self.encoder.tokenizer.pad_token_id)
    elif self.hparams.pool == "avg":
        sentemb = average_pooling(
            input_ids, embeddings, attention_mask,
            self.encoder.tokenizer.pad_token_id,
            getattr(self.encoder.tokenizer, "sep_token_id", None),
            self.use_context,
        )
    elif self.hparams.pool == "cls":
        sentemb = embeddings[:, 0, :]
    else:
        raise Exception(f"Invalid pooling technique: {self.hparams.pool}")

    if add_delta is not None:
        add_delta = add_delta.to(sentemb.dtype).to(sentemb.device)
        sentemb = add_delta.unsqueeze(0) if method == "map" else sentemb + add_delta.unsqueeze(0)

    return sentemb

comet.compute_sentence_embedding = MethodType(_fixed_compute_sentence_embedding, comet)


def pooled_embedding(text: str) -> torch.Tensor:
    batch = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        return comet.get_sentence_embedding(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            add_delta=None,
        ).squeeze(0)

_COMET_CACHE_TAG = "comet_wmt22da"


# ── reference loading ─────────────────────────────────────────────────────────


def load_references(data_dir: str) -> Dict[str, str]:
    """Load reference summaries from SEAHORSE TSVs, keyed by gem_id."""
    refs: Dict[str, str] = {}
    for filename in ("validation.tsv", "test.tsv"):
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("model") == "reference":
                    refs[row["gem_id"]] = row["summary"]
    return refs


# ── steering parameter computation ───────────────────────────────────────────


def _cache_path(lang_key: str, method: str, max_pairs: int) -> str:
    return os.path.join(
        "cache", "seahorse_comet",
        f"{lang_key}_{method}_{max_pairs}_{_COMET_CACHE_TAG}.npy",
    )


def compute_comet_params(parallel_dir: str, lang_key: str,
                         method: str, max_pairs: int) -> np.ndarray:
    """Learn cross-lingual COMET pooling-space params from parallel text.

    Returns a 1-D float32 array (mean-diff vector) for method='vector',
    or a 2-D float32 array (linear map W) for method='map'.
    """
    path = _cache_path(lang_key, method, max_pairs)
    if os.path.exists(path):
        params = np.load(path, allow_pickle=True)
        print(f"Loaded COMET params from {path}", flush=True)
        return params

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

    EN, XX = [], []
    i, cap = 0, max_pairs
    while len(EN) < cap and i < min(len(en_lines), len(xx_lines)):
        en, xx = en_lines[i], xx_lines[i]
        if not en or not xx:
            cap += 1
            i += 1
            continue
        EN.append(pooled_embedding(en).detach().cpu().numpy())
        XX.append(pooled_embedding(xx).detach().cpu().numpy())
        i += 1
        if len(EN) % 100 == 0:
            print(f"  encoded {len(EN)} pairs…", flush=True)

    EN_arr = np.stack(EN)  # [N, H]
    XX_arr = np.stack(XX)  # [N, H]

    if method == "vector":
        params = (EN_arr.mean(0) - XX_arr.mean(0)).astype(np.float32)
    else:  # map
        W, *_ = np.linalg.lstsq(XX_arr, EN_arr, rcond=None)
        params = W.astype(np.float32)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, params)
    print(f"Saved COMET params to {path}", flush=True)
    return params


# ── COMET scoring ─────────────────────────────────────────────────────────────


def _comet_score(hyp: str, ref: str,
                 steer_params: Optional[np.ndarray],
                 sigma: float, method: str) -> float:
    """Score a (hypothesis, reference) pair with COMET + optional steering.

    Cross-lingual steering is applied identically to both the candidate and
    reference embeddings.  Source is left empty (reference-based, source-free).
    """
    src_batch = tok("", return_tensors="pt").to(DEVICE)
    hyp_batch = tok(hyp, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    ref_batch = tok(ref, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)

    if steer_params is None or sigma == 0.0:
        d_hyp = d_ref = None
    elif method == "vector":
        d_hyp = pooled_delta_vector(sigma, steer_params)
        d_ref = pooled_delta_vector(sigma, steer_params)
    else:  # map
        d_hyp = pooled_delta(hyp, steer_params, sigma)
        d_ref = pooled_delta(ref, steer_params, sigma)

    effective_method = method if (d_hyp is not None) else None

    with torch.no_grad():
        src_emb = comet.get_sentence_embedding(
            input_ids=src_batch["input_ids"],
            attention_mask=src_batch["attention_mask"],
            add_delta=None,
        )
        hyp_emb = comet.get_sentence_embedding(
            input_ids=hyp_batch["input_ids"],
            attention_mask=hyp_batch["attention_mask"],
            add_delta=d_hyp,
            method=effective_method,
        )
        ref_emb = comet.get_sentence_embedding(
            input_ids=ref_batch["input_ids"],
            attention_mask=ref_batch["attention_mask"],
            add_delta=d_ref,
            method=effective_method,
        )
        score = comet.estimate(src_emb, hyp_emb, ref_emb)
    return score.score.detach().mean().item()


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


# ── evaluation ────────────────────────────────────────────────────────────────


def eval_metrics(rows: List[dict], references: Dict[str, str],
                 steer_params: Optional[np.ndarray],
                 sigma: float, method: str) -> Dict[str, Tuple[float, float]]:
    """Score all non-reference candidate rows and return {dim: (pearson, auc_roc)}.

    Reference rows (model=reference) are excluded: scoring a summary against
    itself would give a trivially high COMET score and distort correlation.
    """
    skipped_self = 0
    skipped_no_ref = 0
    by_dim: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))
    for row in rows:
        if row.get("model") == "reference":
            skipped_self += 1
            continue
        ref = references.get(row["gem_id"])
        if ref is None:
            skipped_no_ref += 1
            continue
        score = _comet_score(row["candidate"], ref, steer_params, sigma, method)
        by_dim[row["dimension"]][0].append(score)
        by_dim[row["dimension"]][1].append(row["human_score"])
    if skipped_self:
        print(f"  (skipped {skipped_self} reference-as-candidate rows)", flush=True)
    if skipped_no_ref:
        print(f"  (skipped {skipped_no_ref} rows with no matching reference)", flush=True)
    metrics = {dim: safe_seahorse_metrics(scores, labels)
               for dim, (scores, labels) in by_dim.items()}
    samples = {dim: {"scores": [float(s) for s in scores], "labels": [int(l) for l in labels]}
               for dim, (scores, labels) in by_dim.items()}
    return metrics, samples


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="COMET INCLINE evaluation on SEAHORSE")
    parser.add_argument("--lang", choices=["de", "en", "es", "ru", "tr", "vi"], required=True)
    parser.add_argument("--data-dir", required=True,
                        help="Directory with SEAHORSE validation.tsv / test.tsv")
    parser.add_argument("--parallel-data", default=None,
                        help="Directory with train.en + train.<lang>; not needed for baseline")
    parser.add_argument("--method", choices=["baseline", "vector", "map"], default="vector")
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--out", default="results/seahorse_comet/results.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    lang_name = SEAHORSE_LANG2NAME[args.lang]
    print(f"Language: {lang_name} ({args.lang})  method: {args.method}", flush=True)

    val_rows, test_rows = load_seahorse(args.data_dir, lang=args.lang)
    if not val_rows:
        raise RuntimeError(f"No usable val rows for lang='{args.lang}'.")
    if not test_rows:
        raise RuntimeError(f"No usable test rows for lang='{args.lang}'.")

    print("Loading SEAHORSE references…", flush=True)
    references = load_references(args.data_dir)
    print(f"  Loaded {len(references)} references", flush=True)

    if args.method == "baseline":
        steer_params = None
    else:
        if args.parallel_data is None:
            raise ValueError("--parallel-data is required for non-baseline methods")
        print(f"\nComputing COMET {args.method} params for {args.lang}…", flush=True)
        steer_params = compute_comet_params(
            args.parallel_data, args.lang, args.method, args.max_pairs
        )

    sigma_list = SIGMAS[args.method]

    # ── Val: sigma selection ────────────────────────────────────────────────
    print("\n=== Val sigma sweep ===", flush=True)
    val_results: Dict = {}
    for sigma in sigma_list:
        dim_metrics, _ = eval_metrics(val_rows, references, steer_params, sigma, args.method)
        mean_auc = _mean_auc(dim_metrics)
        mean_pe = _mean_pearson(dim_metrics)
        print(f"  sigma={sigma:+.2f}  mean_AUC={mean_auc:.4f}  mean_Pearson={mean_pe:.4f}", flush=True)
        val_results[sigma] = {
            "mean_auc": mean_auc,
            "mean_pearson": mean_pe,
            "per_dim": {d: {"pearson": pe, "auc": auc} for d, (pe, auc) in dim_metrics.items()},
        }
    best_sigma = max(val_results, key=lambda s: val_results[s]["mean_pearson"])
    print(
        f"\nBest sigma (val): {best_sigma:+.2f}  "
        f"mean_Pearson={val_results[best_sigma]['mean_pearson']:.4f}",
        flush=True,
    )

    # ── Test: unsteered baseline + best sigma ───────────────────────────────
    print("\n=== Test baseline (sigma=0.0) ===", flush=True)
    base_metrics, base_samples = eval_metrics(test_rows, references, None, 0.0, args.method)
    base_mean_auc = _mean_auc(base_metrics)
    base_mean_pe = _mean_pearson(base_metrics)
    for dim, (pe, auc) in sorted(base_metrics.items()):
        print(f"  [{dim}] Pearson={pe:.4f}  AUC={auc:.4f}")
    print(f"  mean_AUC={base_mean_auc:.4f}  mean_Pearson={base_mean_pe:.4f}")

    print(f"\n=== Test best sigma ({best_sigma:+.2f}) ===", flush=True)
    test_metrics, test_samples = eval_metrics(test_rows, references, steer_params, best_sigma, args.method)
    test_mean_auc = _mean_auc(test_metrics)
    test_mean_pe = _mean_pearson(test_metrics)
    for dim, (pe, auc) in sorted(test_metrics.items()):
        print(f"  [{dim}] Pearson={pe:.4f}  AUC={auc:.4f}")
    print(f"  mean_AUC={test_mean_auc:.4f}  mean_Pearson={test_mean_pe:.4f}")

    # ── Save ────────────────────────────────────────────────────────────────
    output = {
        "lang":       args.lang,
        "method":     args.method,
        "max_pairs":  args.max_pairs,
        "best_sigma": float(best_sigma),
        "val_results": {str(s): v for s, v in val_results.items()},
        "test_baseline": {
            "sigma":      0.0,
            "per_dim":    {d: {"pearson": pe, "auc": auc} for d, (pe, auc) in base_metrics.items()},
            "mean_auc":   base_mean_auc,
            "mean_pearson": base_mean_pe,
        },
        "test_best_sigma": {
            "sigma":      float(best_sigma),
            "per_dim":    {d: {"pearson": pe, "auc": auc} for d, (pe, auc) in test_metrics.items()},
            "mean_auc":   test_mean_auc,
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
