"""Loader for the Beyond-Ngrams summarization quality dataset.

Data is in data/ngrams/{coherence,completeness}/{lang}.csv.
Each CSV row yields two candidate summaries (one from Gemini, one from GPT)
with corresponding human quality grades.

Returns a flat list of dicts with keys:
    article      – source document text
    candidate    – candidate summary text
    dimension    – "coherence" or "completeness"
    human_score  – mean of annotator grades (float)
    annotator    – "gemini" or "gpt" (which LLM produced the summary)
"""

import ast
import csv
import os
from typing import List, Optional

DIMS = ["coherence", "completeness"]

LANG2NAME = {
    "arabic":    "Arabic",
    "chinese":   "Chinese",
    "hebrew":    "Hebrew",
    "japanese":  "Japanese",
    "spanish":   "Spanish",
    "turkish":   "Turkish",
    "ukrainian": "Ukrainian",
    "yoruba":    "Yoruba",
}

_SCORE_COL = {
    # (dim, annotator) -> column name in CSV
    ("coherence",    "gemini"): "coherence_gemini",
    ("coherence",    "gpt"):    "coherence_gpt",
    ("completeness", "gemini"): "gemini_grade",
    ("completeness", "gpt"):    "gpt_grade",
}


def _mean_grade(raw) -> Optional[float]:
    if not raw:
        return None
    try:
        vals = ast.literal_eval(raw)
        nums = [float(v) for v in vals if v is not None]
        return float(sum(nums) / len(nums)) if nums else None
    except Exception:
        try:
            return float(raw)
        except Exception:
            return None


def load_ngrams(
    lang: str,
    data_dir: str = "data/ngrams",
    dims: Optional[List[str]] = None,
) -> List[dict]:
    """Load all rows for *lang* across all requested dimensions.

    Parameters
    ----------
    lang:     Language key, e.g. "arabic", "spanish".
    data_dir: Root directory containing {coherence,completeness}/ subdirs.
    dims:     Dimensions to load; defaults to all (coherence + completeness).

    Returns
    -------
    List of row dicts.  Rows with missing or non-parseable scores are dropped.
    """
    if dims is None:
        dims = DIMS

    rows: List[dict] = []
    for dim in dims:
        path = os.path.join(data_dir, dim, f"{lang}.csv")
        if not os.path.exists(path):
            print(f"  [load_ngrams] file not found: {path}", flush=True)
            continue

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                article = row.get("text", "").strip()
                if not article:
                    continue

                for annotator in ("gemini", "gpt"):
                    summary_key = f"{annotator}_corrupted_summary"
                    score_key   = _SCORE_COL[(dim, annotator)]

                    candidate = row.get(summary_key, "").strip()
                    score     = _mean_grade(row.get(score_key, ""))

                    if not candidate or score is None:
                        continue

                    rows.append({
                        "article":     article,
                        "candidate":   candidate,
                        "dimension":   dim,
                        "human_score": score,
                        "annotator":   annotator,
                    })

    print(f"  [load_ngrams] {lang}: {len(rows)} rows loaded", flush=True)
    return rows
