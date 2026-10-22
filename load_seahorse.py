"""Loader for the SEAHORSE multilingual summarization evaluation dataset.

Loads SEAHORSE TSV annotation files and joins article text from GEM via
HuggingFace (xsum, mlsum, xlsum, wiki_lingua).

Download the SEAHORSE annotations from:
  https://storage.googleapis.com/seahorse-public/seahorse_data.zip
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

SEAHORSE_DIMS = [
    "question1", "question2", "question3", "question4", "question5", "question6"
]
SEAHORSE_LANGS = ["de", "en", "es", "ru", "tr", "vi"]
SEAHORSE_LANG2NAME = {
    "de": "German", "en": "English", "es": "Spanish",
    "ru": "Russian", "tr": "Turkish", "vi": "Vietnamese",
}

_SCORE_MAP = {"Yes": 1.0, "No": 0.0}  # "Unsure" → None (row skipped)

_SPLIT_MAP = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


def _parse_gem_id(gem_id: str):
    """Parse a GEM id into (dataset_key, hf_split, index).

    Handles both formats seen in practice:
      wiki_lingua_english_en-val-11175   (no gem- prefix, short split name)
      gem-mlsum_de-validation-10         (gem- prefix, full split name)
    """
    s = gem_id[len("gem-"):] if gem_id.startswith("gem-") else gem_id
    parts = s.split("-")
    # Scan from right: find split token, then integer index after it
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] in _SPLIT_MAP:
            try:
                idx = int(parts[i + 1])
            except (IndexError, ValueError):
                continue
            return "-".join(parts[:i]), _SPLIT_MAP[parts[i]], idx
    return None, None, None


def _hf_dataset_args(dataset_key: str):
    """Return (hf_dataset_name, config) for a GEM dataset key."""
    if dataset_key == "xsum":
        return "GEM/xsum", None
    if dataset_key.startswith("mlsum_"):
        return "GEM/mlsum", dataset_key[len("mlsum_"):]
    if dataset_key.startswith("xlsum_"):
        return "GEM/xlsum", dataset_key[len("xlsum_"):]
    if dataset_key.startswith("wiki_lingua_"):
        # e.g. "wiki_lingua_english_en" → config "en"
        lang_code = dataset_key.split("_")[-1]
        return "GEM/wiki_lingua", lang_code
    return None, None  # unknown


def _article_field(column_names):
    for f in ("document", "text", "article", "source"):
        if f in column_names:
            return f
    return None


def load_gem_articles(gem_ids) -> Dict[str, str]:
    """Fetch article text from HuggingFace GEM datasets for the given gem_ids."""
    from datasets import load_dataset

    groups: Dict[tuple, List[tuple]] = defaultdict(list)
    for gem_id in set(gem_ids):
        dataset_key, split, idx = _parse_gem_id(gem_id)
        if dataset_key is None:
            continue
        hf_name, config = _hf_dataset_args(dataset_key)
        if hf_name is None:
            print(f"  Warning: unknown dataset key '{dataset_key}' for gem_id {gem_id}", flush=True)
            continue
        groups[(hf_name, config, split)].append((gem_id, idx))

    articles: Dict[str, str] = {}
    for (hf_name, config, split), items in groups.items():
        try:
            ds = load_dataset(hf_name, config, split=split) if config else \
                 load_dataset(hf_name, split=split)
        except Exception as e:
            print(f"  Warning: could not load {hf_name} ({config}, {split}): {e}", flush=True)
            continue

        field = _article_field(ds.column_names)
        if field is None:
            print(f"  Warning: no article field in {hf_name} columns {ds.column_names}")
            continue

        if "gem_id" in ds.column_names:
            target_ids = {gid for gid, _ in items}
            for row in ds:
                if row["gem_id"] in target_ids:
                    articles[row["gem_id"]] = row[field]
        else:
            ds_list = list(ds)
            for gem_id, idx in items:
                if 0 <= idx < len(ds_list):
                    articles[gem_id] = ds_list[idx][field]

    return articles


def _load_tsv(path: str) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_seahorse(
    data_dir: str,
    lang: Optional[str] = None,
    dimensions: Optional[List[str]] = None,
) -> Tuple[List[dict], List[dict]]:
    """Load SEAHORSE val/test rows with GEM articles joined in.

    Parameters
    ----------
    data_dir : str
        Directory containing validation.tsv and test.tsv.
    lang : str, optional
        Filter to one language code (de, en, es, ru, tr, vi).
    dimensions : list, optional
        Subset of question1..question6; defaults to all six.

    Returns
    -------
    val_rows, test_rows
        Each row: article, candidate, dimension, human_score (0.0/1.0),
        model, lang, gem_id.
    """
    if dimensions is None:
        dimensions = SEAHORSE_DIMS

    print(f"Loading SEAHORSE from {data_dir}…", flush=True)
    splits_raw = {}
    for split_name, filename in [("val", "validation.tsv"), ("test", "test.tsv")]:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SEAHORSE file not found: {path}")
        splits_raw[split_name] = _load_tsv(path)
        print(f"  {split_name}: {len(splits_raw[split_name])} raw rows", flush=True)

    # Show what language values are present so mismatches are obvious
    seen_langs = sorted({row["worker_lang"] for rows in splits_raw.values() for row in rows})
    seen_prefixes = sorted({v.split("-")[0] for v in seen_langs})
    print(f"  worker_lang values in data: {seen_langs}", flush=True)
    if lang is not None and lang not in seen_prefixes:
        raise ValueError(
            f"Language '{lang}' not found in dataset. Available prefixes: {seen_prefixes}"
        )

    # Gather only the gem_ids we actually need; show source breakdown
    needed_gem_ids = {
        row["gem_id"]
        for rows in splits_raw.values()
        for row in rows
        if lang is None or row["worker_lang"].split("-")[0] == lang
    }
    sources = defaultdict(int)
    for gid in needed_gem_ids:
        dk, _, _ = _parse_gem_id(gid)
        sources[dk or "unparseable"] += 1
    print(f"  gem_id sources: { {k: v for k, v in sorted(sources.items())} }", flush=True)
    print(f"  Fetching GEM articles for {len(needed_gem_ids)} gem_ids…", flush=True)
    articles = load_gem_articles(needed_gem_ids)
    print(f"  Retrieved {len(articles)} articles", flush=True)

    result = {}
    for split_name, raw_rows in splits_raw.items():
        out = []
        for row in raw_rows:
            if lang is not None and row["worker_lang"].split("-")[0] != lang:
                continue
            gem_id = row["gem_id"]
            if gem_id not in articles:
                continue
            for dim in dimensions:
                score = _SCORE_MAP.get((row.get(dim) or "").strip())
                if score is None:
                    continue
                out.append({
                    "article":      articles[gem_id],
                    "candidate":    row["summary"],
                    "dimension":    dim,
                    "human_score":  score,
                    "model":        row.get("model", ""),
                    "lang":         row["worker_lang"],
                    "gem_id":       gem_id,
                })
        result[split_name] = out
        print(f"  {split_name}: {len(out)} usable rows", flush=True)

    return result["val"], result["test"]
