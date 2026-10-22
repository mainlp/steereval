#!/usr/bin/env python3
"""Paired t-test (per SEAHORSE dimension) for all main-text experiments.

For each (setup, language, dimension), runs a paired t-test over the test
samples in that dimension, comparing model scores at the optimal steering
sigma against scores at sigma=0.  Requires the 'test_samples' field in
each result JSON — produced by the updated eval_seahorse_*.py scripts.

When test_samples is missing from a JSON, the script automatically calls
rescore_seahorse.py to patch that JSON before running the t-tests.

Covers:
  - Small models, vector steer, EN-prompt
  - Small + large models, avgvec, EN-prompt
  - COMET, vector steer + avgvec

Usage
-----
python ttest_seahorse.py                 # plain-text output
python ttest_seahorse.py --latex         # LaTeX tabular output
python ttest_seahorse.py --alpha 0.05    # significance threshold (default 0.05)
python ttest_seahorse.py --summary       # fraction of sig dims per (setup, lang)
python ttest_seahorse.py --no-rerun      # skip reruns; print missing JSONs and exit

# Pass data paths when reruns are needed (defaults match the project layout):
python ttest_seahorse.py --data-dir ./data/seahorse --data-root ./data/ncwm_flores
"""

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from scipy import stats

# ── constants ─────────────────────────────────────────────────────────────────

DIMS = ["question1", "question2", "question3", "question4", "question5", "question6"]
DIM_LABELS = {
    "question1": "Comprehensible",
    "question2": "Non-repetitive",
    "question3": "Grammatical",
    "question4": "Attributable",
    "question5": "Main ideas",
    "question6": "Concise",
}
LANGS = ["de", "es", "ru", "tr", "vi"]

MODEL_LABEL = {
    "llama3b": "Llama-3.2-3B",
    "aya":     "aya-expanse-8b",
    "llama8b": "Llama-3.1-8B",
    "aya32b":  "aya-expanse-32b",
}
SCORING_LABEL = {"gptscore": "GPTScore", "geval": "G-eval"}

# ── experiment registry ───────────────────────────────────────────────────────

def _vec_fname(model, lang):
    return f"{model}_{lang}_vector_en.json"

def _avg_fname(model, lang):
    return f"{model}_{lang}_avgvec_en.json"

def _comet_vec_fname(lang):
    return f"{lang}_vector.json"

def _comet_avg_fname(lang):
    return f"{lang}_avgvec.json"


SMALL_MODELS = [
    ("gptscore", "llama3b"),
    ("geval",    "llama3b"),
    ("gptscore", "aya"),
    ("geval",    "aya"),
]

VECTOR_DIRS = {
    ("gptscore", "llama3b"): "results/seahorse_gptscore_llama3b",
    ("geval",    "llama3b"): "results/seahorse_geval_llama3b",
    ("gptscore", "aya"):     "results/seahorse_gptscore_aya",
    ("geval",    "aya"):     "results/seahorse_geval_aya",
}

AVGVEC_DIRS_SMALL = {
    ("gptscore", "llama3b"): "results/seahorse_avgvec_gptscore_llama3b",
    ("geval",    "llama3b"): "results/seahorse_avgvec_geval_llama3b",
    ("gptscore", "aya"):     "results/seahorse_avgvec_gptscore_aya",
    ("geval",    "aya"):     "results/seahorse_avgvec_geval_aya",
}

LARGE_MODELS = [
    ("gptscore", "llama8b"),
    ("gptscore", "aya32b"),
]

AVGVEC_DIRS_LARGE = {
    ("gptscore", "llama8b"): "results/seahorse_avgvec_gptscore_llama8b",
    ("gptscore", "aya32b"):  "results/seahorse_avgvec_gptscore_aya32b",
}

COMET_DIR        = "results/seahorse_comet"
COMET_AVGVEC_DIR = "results/seahorse_avgvec_comet"

# ── loading ───────────────────────────────────────────────────────────────────

def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _needs_patch(result: Optional[dict]) -> bool:
    return result is not None and "test_samples" not in result


def _patch_missing(paths: List[str], data_dir: str, data_root: str,
                   gpu: str, max_pairs: int):
    """Call rescore_seahorse.py for each path that is missing test_samples."""
    rescore = os.path.join(os.path.dirname(__file__), "rescore_seahorse.py")
    cmd = [
        sys.executable, rescore,
        "--json-paths", *paths,
        "--data-dir",  data_dir,
        "--data-root", data_root,
        "--gpu",       gpu,
        "--max-pairs", str(max_pairs),
    ]
    print(f"Running rescore for {len(paths)} JSON(s)…", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"rescore_seahorse.py failed (exit {result.returncode})")


def _dim_samples(result: Optional[dict], section_key: str, dim: str
                 ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Return (baseline_scores, steered_scores) for one dimension."""
    if result is None:
        return None, None
    ts = result.get("test_samples")
    if ts is None:
        return None, None
    base = ts.get("baseline", {}).get(dim)
    steer = ts.get(section_key, {}).get(dim)
    if base is None or steer is None:
        return None, None
    return base["scores"], steer["scores"]

# ── t-test ────────────────────────────────────────────────────────────────────

def run_ttest(base_scores: List[float], steer_scores: List[float]
              ) -> Tuple[float, float, float, int]:
    """Paired t-test; returns (mean_delta, t_stat, p_value, n)."""
    n = len(base_scores)
    assert n == len(steer_scores), "score lists must be same length"
    mean_delta = sum(s - b for s, b in zip(steer_scores, base_scores)) / n
    t_stat, p_value = stats.ttest_rel(steer_scores, base_scores)
    return mean_delta, float(t_stat), float(p_value), n


def sig_stars(p: float, alpha: float) -> str:
    if p < alpha / 50:
        return "***"
    if p < alpha / 5:
        return "**"
    if p < alpha:
        return "*"
    return ""

# ── result dataclass ──────────────────────────────────────────────────────────

class DimResult:
    def __init__(self, group, lang, dim, delta, t, p, n):
        self.group = group
        self.lang  = lang
        self.dim   = dim
        self.delta = delta
        self.t     = t
        self.p     = p
        self.n     = n

# ── collect all results ───────────────────────────────────────────────────────

def _collect_group(group: str, result: dict, lang: str,
                   steer_key: str = "best_sigma") -> List[DimResult]:
    rows = []
    for dim in DIMS:
        base_sc, steer_sc = _dim_samples(result, steer_key, dim)
        if base_sc is None:
            continue
        delta, t, p, n = run_ttest(base_sc, steer_sc)
        rows.append(DimResult(group, lang, dim, delta, t, p, n))
    return rows


def _all_json_paths() -> Dict[str, str]:
    """Return {path: group_label} for every JSON the t-test needs."""
    paths = {}
    for scoring, model in SMALL_MODELS:
        group = f"{SCORING_LABEL[scoring]} / {MODEL_LABEL[model]} / vector steer"
        rdir  = VECTOR_DIRS[(scoring, model)]
        for lang in LANGS:
            paths[os.path.join(rdir, _vec_fname(model, lang))] = group
    for scoring, model in SMALL_MODELS:
        group = f"{SCORING_LABEL[scoring]} / {MODEL_LABEL[model]} / avgvec"
        avd   = AVGVEC_DIRS_SMALL[(scoring, model)]
        for lang in LANGS:
            paths[os.path.join(avd, _avg_fname(model, lang))] = group
    for scoring, model in LARGE_MODELS:
        group = f"{SCORING_LABEL[scoring]} / {MODEL_LABEL[model]} / avgvec"
        avd   = AVGVEC_DIRS_LARGE[(scoring, model)]
        for lang in LANGS:
            paths[os.path.join(avd, _avg_fname(model, lang))] = group
    for lang in LANGS:
        paths[os.path.join(COMET_DIR, _comet_vec_fname(lang))] = \
            "COMET / wmt22-comet-da / vector steer"
        paths[os.path.join(COMET_AVGVEC_DIR, _comet_avg_fname(lang))] = \
            "COMET / wmt22-comet-da / avgvec"
    return paths


def collect(alpha: float, no_rerun: bool,
            data_dir: str, data_root: str,
            gpu: str, max_pairs: int) -> List[DimResult]:

    all_paths = _all_json_paths()

    # ── detect missing test_samples ───────────────────────────────────────────
    need_patch = [
        p for p in all_paths
        if _needs_patch(_load(p))
    ]
    missing_entirely = [
        p for p in all_paths
        if not os.path.exists(p)
    ]

    if missing_entirely:
        print(f"WARNING: {len(missing_entirely)} JSON(s) not found (experiment not run yet):",
              flush=True)
        for p in missing_entirely:
            print(f"  {p}", flush=True)

    if need_patch:
        if no_rerun:
            print(f"\n{len(need_patch)} JSON(s) missing test_samples (use without --no-rerun "
                  "to patch automatically):", flush=True)
            for p in need_patch:
                print(f"  {p}", flush=True)
            sys.exit(1)
        _patch_missing(need_patch, data_dir, data_root, gpu, max_pairs)

    # ── collect t-test results ─────────────────────────────────────────────────
    rows: List[DimResult] = []

    for scoring, model in SMALL_MODELS:
        group = f"{SCORING_LABEL[scoring]} / {MODEL_LABEL[model]} / vector steer"
        rdir  = VECTOR_DIRS[(scoring, model)]
        for lang in LANGS:
            r = _load(os.path.join(rdir, _vec_fname(model, lang)))
            rows.extend(_collect_group(group, r, lang))

    for scoring, model in SMALL_MODELS:
        group = f"{SCORING_LABEL[scoring]} / {MODEL_LABEL[model]} / avgvec"
        avd   = AVGVEC_DIRS_SMALL[(scoring, model)]
        for lang in LANGS:
            r = _load(os.path.join(avd, _avg_fname(model, lang)))
            rows.extend(_collect_group(group, r, lang))

    for scoring, model in LARGE_MODELS:
        group = f"{SCORING_LABEL[scoring]} / {MODEL_LABEL[model]} / avgvec"
        avd   = AVGVEC_DIRS_LARGE[(scoring, model)]
        for lang in LANGS:
            r = _load(os.path.join(avd, _avg_fname(model, lang)))
            rows.extend(_collect_group(group, r, lang))

    group = "COMET / wmt22-comet-da / vector steer"
    for lang in LANGS:
        r = _load(os.path.join(COMET_DIR, _comet_vec_fname(lang)))
        rows.extend(_collect_group(group, r, lang))

    group = "COMET / wmt22-comet-da / avgvec"
    for lang in LANGS:
        r = _load(os.path.join(COMET_AVGVEC_DIR, _comet_avg_fname(lang)))
        rows.extend(_collect_group(group, r, lang))

    return rows

# ── plain-text output ─────────────────────────────────────────────────────────

def _fmt_delta(v: float) -> str:
    s = f"{v:+.3f}"
    if s.startswith("+0."):
        return "+" + s[2:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def print_plain(rows: List[DimResult], alpha: float):
    print(f"Per-dimension paired t-test  (α={alpha})")
    print(f"Significance: * p<{alpha}  ** p<{alpha/5:.3f}  *** p<{alpha/50:.4f}\n")

    # Organise: group -> lang -> dim -> DimResult
    data: Dict[str, Dict[str, Dict[str, DimResult]]] = defaultdict(
        lambda: defaultdict(dict))
    for r in rows:
        data[r.group][r.lang][r.dim] = r

    for group, lang_data in data.items():
        print(f"── {group}")
        # header
        lang_w = 14
        header = f"  {'Dimension':<18}" + "".join(f"  {l:>{lang_w}}" for l in LANGS)
        print(header)
        print("  " + "-" * (18 + len(LANGS) * (lang_w + 2)))
        for dim in DIMS:
            dlabel = DIM_LABELS[dim]
            cells = []
            for lang in LANGS:
                r = lang_data.get(lang, {}).get(dim)
                if r is None:
                    cells.append("---")
                else:
                    stars = sig_stars(r.p, alpha)
                    cells.append(f"{_fmt_delta(r.delta)}(p={r.p:.3f}{stars})")
            print(f"  {dlabel:<18}" + "".join(f"  {c:>{lang_w}}" for c in cells))
        print()


# ── summary output ────────────────────────────────────────────────────────────

def print_summary(rows: List[DimResult], alpha: float):
    """One row per (group, lang): n_sig / 6 dimensions significant."""
    print(f"Summary: significant dimensions per (setup, language)  (α={alpha})\n")
    data: Dict[str, Dict[str, List[DimResult]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r.group][r.lang].append(r)

    col = 55
    hdr = f"{'Setup':<{col}}  {'lang':4}  {'sig/6':6}  {'pos/neg':9}  {'min_p':7}"
    print(hdr)
    print("-" * len(hdr))
    for group, lang_data in data.items():
        for lang in LANGS:
            dim_rows = lang_data.get(lang, [])
            if not dim_rows:
                continue
            n_sig    = sum(1 for r in dim_rows if r.p < alpha)
            n_pos    = sum(1 for r in dim_rows if r.delta > 0)
            n_neg    = len(dim_rows) - n_pos
            min_p    = min(r.p for r in dim_rows)
            print(f"{group:<{col}}  {lang:4}  {n_sig}/{len(dim_rows):1}      "
                  f"{n_pos}+/{n_neg}-     {min_p:.4f}")
        print()


# ── LaTeX output ──────────────────────────────────────────────────────────────

def print_latex(rows: List[DimResult], alpha: float):
    """One table per setup: rows=dims, cols=langs, cell=Δ(p*)."""
    data: Dict[str, Dict[str, Dict[str, DimResult]]] = defaultdict(
        lambda: defaultdict(dict))
    for r in rows:
        data[r.group][r.lang][r.dim] = r

    col_spec = "l|" + "|".join("c" for _ in LANGS)
    lang_header = " & ".join(f"\\textbf{{{l}}}" for l in LANGS)

    for group, lang_data in data.items():
        safe_label = group.lower().replace(" ", "-").replace("/", "").replace("--", "-")
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\resizebox{\\columnwidth}{!}{%",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            f"\\textbf{{Dimension}} & {lang_header} \\\\",
            "\\midrule",
        ]
        for dim in DIMS:
            dlabel = DIM_LABELS[dim]
            cells = []
            for lang in LANGS:
                r = lang_data.get(lang, {}).get(dim)
                if r is None:
                    cells.append("---")
                else:
                    stars = sig_stars(r.p, alpha)
                    delta_s = _fmt_delta(r.delta)
                    p_s = f"{r.p:.3f}".lstrip("0") or ".000"
                    cells.append(f"${delta_s}$ (${p_s}${stars})")
            lines.append(f"{dlabel} & " + " & ".join(cells) + " \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            f"\\caption{{Per-dimension paired t-test ($n\\approx{rows[0].n if rows else '?'}$ test "
            f"samples) for \\textit{{{group}}}. Each cell: mean $\\Delta$score and "
            f"two-tailed $p$-value. "
            f"Significance: {sig_stars(alpha*0.999,alpha)}~$p<{alpha}$, "
            f"{sig_stars(alpha/5*0.999,alpha)}~$p<{alpha/5:.2f}$, "
            f"{sig_stars(alpha/50*0.999,alpha)}~$p<{alpha/50:.3f}$.}}",
            f"\\label{{tab:ttest_{safe_label}}}",
            "\\end{table}",
            "",
        ]
        print("\n".join(lines))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-dimension paired t-test for main-text SEAHORSE experiments"
    )
    parser.add_argument("--latex",     action="store_true", help="LaTeX table output")
    parser.add_argument("--summary",   action="store_true",
                        help="Compact summary per (setup, lang)")
    parser.add_argument("--alpha",     type=float, default=0.05,
                        help="Significance level (default 0.05)")
    parser.add_argument("--no-rerun",  action="store_true",
                        help="Do not trigger reruns; print missing JSONs and exit")
    parser.add_argument("--data-dir",  default="./data/seahorse",
                        help="SEAHORSE data dir (default: ./data/seahorse)")
    parser.add_argument("--data-root", default="./data/ncwm_flores",
                        help="Parallel data root (default: ./data/ncwm_flores)")
    parser.add_argument("--gpu",       default="0",
                        help="GPU id passed to rescore_seahorse.py (default: 0)")
    parser.add_argument("--max-pairs", type=int, default=500,
                        help="Max parallel pairs for steer param computation (default: 500)")
    args = parser.parse_args()

    rows = collect(args.alpha, args.no_rerun,
                   args.data_dir, args.data_root,
                   args.gpu, args.max_pairs)

    if not rows:
        print("No per-sample data found.")
        return

    if args.latex:
        print_latex(rows, args.alpha)
    elif args.summary:
        print_summary(rows, args.alpha)
    else:
        print_plain(rows, args.alpha)


if __name__ == "__main__":
    main()
