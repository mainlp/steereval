import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

from constants import LANG2LANGKEY, LANGS, MAX_PAIRS

from utils import safe_corr, load_annotations_csv

from comet import download_model, load_from_checkpoint
from types import MethodType
from comet.models.pooling_utils import average_pooling, max_pooling

MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

comet_ckpt = download_model("Unbabel/wmt22-comet-da")
comet = load_from_checkpoint(comet_ckpt).to(DEVICE).eval()

tok = comet.encoder.tokenizer
HIDDEN = comet.encoder.model.config.hidden_size


@torch.no_grad()
def pooled_embedding(text: str) -> torch.Tensor:
    batch = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(
        DEVICE
    )

    emb = comet.get_sentence_embedding(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        add_delta=None,
    )
    return emb.squeeze(0)  # [H]


def learn_crosslingual_pooled_map(
    lang: str,
    method: str,
    max_pairs: int = MAX_PAIRS,
    base_dir: str = "./data/ncwm_old",
) -> np.ndarray:

    xx = LANG2LANGKEY[lang]
    en_path = os.path.join(base_dir, f"en-{xx}", "train.en")
    xx_path = os.path.join(base_dir, f"en-{xx}", f"train.{xx}")

    with open(en_path, encoding="utf-8") as f:
        en_lines = [ln.strip() for ln in f]
    with open(xx_path, encoding="utf-8") as g:
        xx_lines = [ln.strip() for ln in g]

    EN, XX = [], []
    i = 0
    while len(EN) < max_pairs and i < min(len(en_lines), len(xx_lines)):
        en = en_lines[i]
        xx_line = xx_lines[i]
        if not en or not xx_line:
            i += 1
            max_pairs += 1
            continue

        EN.append(pooled_embedding(en).detach().cpu().numpy())
        XX.append(pooled_embedding(xx_line).detach().cpu().numpy())
        i += 1

    if method == "map":
        EN = np.stack(EN)  # [N, H]
        XX = np.stack(XX)  # [N, H]

        W, *_ = np.linalg.lstsq(XX, EN, rcond=None)
        return W.astype(np.float32)
    elif method == "vector":
        EN = np.mean(EN, axis=0)
        XX = np.mean(XX, axis=0)
        V = EN - XX
        return V.astype(np.float32)
    else:
        raise ValueError("method must be 'map' or 'vector'")


@torch.no_grad()
def pooled_delta(text: str, W: np.ndarray, sigma: float) -> torch.Tensor:
    p = pooled_embedding(text)  # [H]
    d = torch.from_numpy(p.detach().cpu().numpy() @ W).to(DEVICE)  # [H]
    return ((1 - sigma) * p) + (sigma * d)


@torch.no_grad()
def pooled_delta_vector(sigma: float, V: np.ndarray) -> torch.Tensor:
    d = torch.from_numpy(V).to(DEVICE).float()
    return sigma * d


def _compute_sentence_embedding_pooled(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None,
    *,
    add_delta: Optional[torch.Tensor] = None,
    method: Optional[str] = None,
) -> torch.Tensor:

    enc = self.encoder(input_ids, attention_mask, token_type_ids=token_type_ids)

    if self.layerwise_attention:
        embeddings = self.layerwise_attention(enc["all_layers"], attention_mask)
    elif 0 <= self.hparams.layer < self.encoder.num_layers:
        embeddings = enc["all_layers"][self.hparams.layer]
    else:
        raise Exception(f"Invalid model layer {self.hparams.layer}.")

    if self.hparams.pool == "default":
        sentemb = enc["sentemb"]
    elif self.hparams.pool == "max":
        sentemb = max_pooling(
            input_ids, embeddings, self.encoder.tokenizer.pad_token_id
        )
    elif self.hparams.pool == "avg":
        sentemb = average_pooling(
            input_ids,
            embeddings,
            attention_mask,
            self.encoder.tokenizer.pad_token_id,
            getattr(self.encoder.tokenizer, "sep_token_id", None),
            self.use_context,
        )
    elif self.hparams.pool == "cls":
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


def _get_sentence_embedding_pooled(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None,
    *,
    add_delta: Optional[torch.Tensor] = None,
    method: Optional[str] = None,
) -> torch.Tensor:

    if self.caching and add_delta is None:
        return self.retrieve_sentence_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    else:
        return self.compute_sentence_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            add_delta=add_delta,
            method=method,
        )


comet.compute_sentence_embedding = MethodType(_compute_sentence_embedding_pooled, comet)
comet.get_sentence_embedding = MethodType(_get_sentence_embedding_pooled, comet)


def forward_summarization_crosslingual_pooled_intervention(
    self,
    *,
    hypothesis_text: str,
    reference_text: str,
    params_by_lang: Dict[str, np.ndarray],
    hyp_lang: str,
    ref_lang: str,
    method: str,
    sigma_hyp: float = 0.5,
    sigma_ref: float = 0.5,
):

    if method == "map":
        d_hyp = pooled_delta(hypothesis_text, params_by_lang[hyp_lang], sigma_hyp)
        d_ref = pooled_delta(reference_text, params_by_lang[ref_lang], sigma_ref)

    elif method == "vector":
        d_hyp = pooled_delta_vector(sigma_hyp, params_by_lang[hyp_lang])
        d_ref = pooled_delta_vector(sigma_ref, params_by_lang[ref_lang])

    else:
        raise ValueError("method must be 'map' or 'vector'")

    src_batch = tok("", return_tensors="pt").to(DEVICE)
    hyp_batch = tok(
        hypothesis_text, return_tensors="pt", truncation=True, max_length=MAX_LEN
    ).to(DEVICE)
    ref_batch = tok(
        reference_text, return_tensors="pt", truncation=True, max_length=MAX_LEN
    ).to(DEVICE)

    with torch.no_grad():
        src_emb = self.get_sentence_embedding(
            input_ids=src_batch["input_ids"],
            attention_mask=src_batch["attention_mask"],
            add_delta=None,
        )
        hyp_emb = self.get_sentence_embedding(
            input_ids=hyp_batch["input_ids"],
            attention_mask=hyp_batch["attention_mask"],
            add_delta=d_hyp,
            method=method,
        )
        ref_emb = self.get_sentence_embedding(
            input_ids=ref_batch["input_ids"],
            attention_mask=ref_batch["attention_mask"],
            add_delta=d_ref,
            method=method,
        )

    return self.estimate(src_emb, hyp_emb, ref_emb)


comet.forward_summarization_crosslingual_pooled_intervention = MethodType(
    forward_summarization_crosslingual_pooled_intervention, comet
)


def run_eval_correlation_comet_pooled(
    lang: str,
    W: np.ndarray,
    coherence_annotations_path: str,
    consistency_annotations_path: str,
    sigma_list: List[float],
    method: str,
):

    coh_rows = load_annotations_csv(
        coherence_annotations_path, metric="correlation", dimension="coherence"
    )
    cons_rows = load_annotations_csv(
        consistency_annotations_path, metric="correlation", dimension="completeness"
    )

    emb_cache: Dict[str, torch.Tensor] = {}

    def delta(text: str, sigma: float, method: str):
        if method == "map":
            if text not in emb_cache:
                p = pooled_embedding(text).detach().cpu().numpy()
                mapped = p @ W
                emb_cache[text] = (p, mapped)
            p_np, mapped_np = emb_cache[text]
            p_t = torch.from_numpy(p_np).to(DEVICE).float()
            mapped_t = torch.from_numpy(mapped_np).to(DEVICE).float()
            return (1 - sigma) * p_t + sigma * mapped_t

        elif method == "vector":
            return sigma * torch.from_numpy(W).to(DEVICE).float()

        else:
            raise ValueError("method must be 'map' or 'vector'")

    def eval_rows(rows: list, label: str):
        results = []

        for sigma in sigma_list:
            model_scores, human_scores = [], []

            for r in rows:
                hyp = r["candidate"]
                ref = r["article"]

                src_batch = tok("", return_tensors="pt").to(DEVICE)
                hyp_batch = tok(
                    hyp, return_tensors="pt", truncation=True, max_length=MAX_LEN
                ).to(DEVICE)
                ref_batch = tok(
                    ref, return_tensors="pt", truncation=True, max_length=MAX_LEN
                ).to(DEVICE)

                with torch.no_grad():
                    src_emb = comet.get_sentence_embedding(
                        input_ids=src_batch["input_ids"],
                        attention_mask=src_batch["attention_mask"],
                        add_delta=None,
                    )
                    hyp_emb = comet.get_sentence_embedding(
                        input_ids=hyp_batch["input_ids"],
                        attention_mask=hyp_batch["attention_mask"],
                        add_delta=delta(hyp, sigma, method),
                        method=method,
                    )
                    ref_emb = comet.get_sentence_embedding(
                        input_ids=ref_batch["input_ids"],
                        attention_mask=ref_batch["attention_mask"],
                        add_delta=delta(ref, sigma, method),
                        method=method,
                    )
                    score = comet.estimate(src_emb, hyp_emb, ref_emb)

                model_scores.append(score.score.detach().mean().item())
                human_scores.append(float(r["human_score"]))

            sp, kd, pe = safe_corr(model_scores, human_scores)
            print(
                f"[{lang}][{label}][{method}] sigma={sigma:.2f}  "
                f"Spearman={sp:.4f}  Kendall={kd:.4f}  Pearson={pe:.4f}"
            )
            results.append((sigma, sp, kd, pe))

        return results

    return {
        "coherence": eval_rows(coh_rows, "coherence"),
        "consistency": eval_rows(cons_rows, "consistency"),
    }


def run_eval_accuracy_comet_pooled(
    lang: str,
    W: np.ndarray,
    coherence_annotations_path: str,
    consistency_annotations_path: str,
    sigma_list: List[float],
    method: str,
):

    coh_rows = load_annotations_csv(
        coherence_annotations_path,
        metric="accuracy",
        dimension="coherence",
    )
    cons_rows = load_annotations_csv(
        consistency_annotations_path,
        metric="accuracy",
        dimension="completeness",
    )

    emb_cache: Dict[str, torch.Tensor] = {}

    def delta(text: str, sigma: float, method: str):
        if method == "map":
            if text not in emb_cache:
                p = pooled_embedding(text).detach().cpu().numpy()
                mapped = p @ W
                emb_cache[text] = (p, mapped)
            p_np, mapped_np = emb_cache[text]
            p_t = torch.from_numpy(p_np).to(DEVICE).float()
            mapped_t = torch.from_numpy(mapped_np).to(DEVICE).float()
            return (1 - sigma) * p_t + sigma * mapped_t

        elif method == "vector":
            return sigma * torch.from_numpy(W).to(DEVICE).float()

        else:
            raise ValueError("method must be 'map' or 'vector'")

    def comet_score(hyp: str, ref: str, sigma: float) -> float:
        src_batch = tok("", return_tensors="pt").to(DEVICE)
        hyp_batch = tok(
            hyp, return_tensors="pt", truncation=True, max_length=MAX_LEN
        ).to(DEVICE)
        ref_batch = tok(
            ref, return_tensors="pt", truncation=True, max_length=MAX_LEN
        ).to(DEVICE)

        with torch.no_grad():
            src_emb = comet.get_sentence_embedding(
                input_ids=src_batch["input_ids"],
                attention_mask=src_batch["attention_mask"],
                add_delta=None,
            )
            hyp_emb = comet.get_sentence_embedding(
                input_ids=hyp_batch["input_ids"],
                attention_mask=hyp_batch["attention_mask"],
                add_delta=delta(hyp, sigma, method),
                method=method,
            )
            ref_emb = comet.get_sentence_embedding(
                input_ids=ref_batch["input_ids"],
                attention_mask=ref_batch["attention_mask"],
                add_delta=delta(ref, sigma, method),
                method=method,
            )
            score = comet.estimate(src_emb, hyp_emb, ref_emb)

        return score.score.detach().mean().item()

    def eval_rows(rows: list, label: str):
        results = []
        for sigma in sigma_list:
            correct, total = 0, 0
            skipped_same = 0

            for r in rows:
                article = r["article"]
                gpt_o = r["gpt_orig"].strip()
                gpt_c = r["gpt_corr"].strip()
                gem_o = r["gem_orig"].strip()
                gem_c = r["gem_corr"].strip()

                if gpt_o == gpt_c and gem_o == gem_c:
                    skipped_same += 1
                    continue

                try:
                    if gpt_o != gpt_c:
                        s_o = comet_score(gpt_o, article, sigma)
                        s_c = comet_score(gpt_c, article, sigma)
                        correct += int(s_o > s_c)
                        total += 1

                    if gem_o != gem_c:
                        s_o = comet_score(gem_o, article, sigma)
                        s_c = comet_score(gem_c, article, sigma)
                        correct += int(s_o > s_c)
                        total += 1
                except Exception as e:
                    print(f"Skipping row due to error: {e}")
                    continue

            acc = correct / total if total > 0 else 0.0
            print(
                f"[{lang}][{label}][{method}] sigma={sigma:.2f} → "
                f"Accuracy={acc:.4f} ({correct}/{total})  | Skipped identical pairs: {skipped_same}"
            )
            results.append((sigma, acc))

        return results

    return {
        "coherence": eval_rows(coh_rows, "coherence"),
        "consistency": eval_rows(cons_rows, "consistency"),
    }


if __name__ == "__main__":
    log_path = "comet_output.log"
    with open(log_path, "w", encoding="utf-8") as f:
        stdout_original = sys.stdout
        sys.stdout = f

        try:
            METHODS = ["vector", "map"]

            SIGMAS_BY_METHOD = {
                "vector": [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "map": [0.0, 0.25, 0.5, 0.75, 1.0],
            }

            for method in METHODS:
                print(f"\n=== Running method: {method} ===")
                sigma_list = SIGMAS_BY_METHOD[method]

                W_by_lang: Dict[str, np.ndarray] = {}

                for lang in LANGS:
                    print(f"Learning pooled {method} for {lang} ...")
                    W_by_lang[lang] = learn_crosslingual_pooled_map(
                        lang=lang,
                        method=method,
                        max_pairs=MAX_PAIRS,
                    )

                for lang in LANGS:
                    coh_path = f"./data/ngrams/coherence/{lang.lower()}.csv"
                    cons_path = f"./data/ngrams/completeness/{lang.lower()}.csv"

                    print(
                        f"\n=== COMET pooled-intervention {method} correlation: {lang} ==="
                    )
                    _corr_res = run_eval_correlation_comet_pooled(
                        lang=lang,
                        W=W_by_lang[lang],
                        coherence_annotations_path=coh_path,
                        consistency_annotations_path=cons_path,
                        sigma_list=sigma_list,
                        method=method,
                    )

                    print(
                        f"\n=== COMET pooled-intervention {method} accuracy: {lang} ==="
                    )
                    _acc_res = run_eval_accuracy_comet_pooled(
                        lang=lang,
                        W=W_by_lang[lang],
                        coherence_annotations_path=coh_path,
                        consistency_annotations_path=cons_path,
                        sigma_list=sigma_list,
                        method=method,
                    )
        finally:
            sys.stdout = stdout_original

    print(f"Logs saved to: {log_path}")
