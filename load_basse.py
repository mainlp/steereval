"""Shared loader for the HiTZ/BASSE summarization evaluation dataset.

Supports both the Spanish ('es') and Basque ('eu') language splits.
Returns rows suitable for correlation and pairwise-accuracy evaluation.
"""

from typing import List, Optional, Tuple

import numpy as np

BASSE_DIMENSIONS = ["coherence", "consistency", "fluency", "relevance", "5W1H"]


def _avg_score(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return None if np.isnan(f) else f
    if isinstance(v, (list, tuple)):
        vals = [float(x) for x in v if x is not None]
        return float(np.mean(vals)) if vals else None
    return None


def load_basse(
    lang_key: str,
    val_frac: float = 0.5,
    dimensions: Optional[List[str]] = None,
) -> Tuple[List[dict], List[dict]]:
    """Load HiTZ/BASSE for *lang_key* ('es' or 'eu') and split into val / test.

    The split is deterministic: documents are sorted by their URL (or idx when
    URL is absent) and the first *val_frac* fraction forms the validation set.

    Returns
    -------
    val_rows, test_rows
        Each row is a dict with keys:
            article     – source document text
            candidate   – candidate summary text
            dimension   – quality dimension name (one of BASSE_DIMENSIONS)
            human_score – mean of annotator scores (float, 1–5)
            model       – model that produced the summary
    """
    from datasets import load_dataset

    if dimensions is None:
        dimensions = BASSE_DIMENSIONS

    print(f"Loading HiTZ/BASSE ({lang_key})…", flush=True)
    ds = load_dataset("HiTZ/BASSE", lang_key, split="test")
    print(f"  Columns: {ds.column_names}", flush=True)

    doc_col = "document" if "document" in ds.column_names else "text"
    sum_col = "summary" if "summary" in ds.column_names else "generated_summary"
    id_col = "url" if "url" in ds.column_names else "idx"

    all_ids = sorted(set(str(item[id_col]) for item in ds))
    n_val = max(1, int(len(all_ids) * val_frac))
    val_ids = set(all_ids[:n_val])
    test_ids = set(all_ids[n_val:])
    print(
        f"  {len(all_ids)} unique documents → {len(val_ids)} val, {len(test_ids)} test",
        flush=True,
    )

    val_rows: List[dict] = []
    test_rows: List[dict] = []

    for item in ds:
        doc_id = str(item[id_col])
        article = item[doc_col]
        candidate = item[sum_col]
        bucket = val_rows if doc_id in val_ids else test_rows

        for dim in dimensions:
            score = _avg_score(item.get(dim))
            if score is None:
                continue
            bucket.append(
                {
                    "article": article,
                    "candidate": candidate,
                    "dimension": dim,
                    "human_score": score,
                    "model": item.get("model", ""),
                }
            )

    print(f"  Val rows: {len(val_rows)}, Test rows: {len(test_rows)}", flush=True)
    return val_rows, test_rows
