#!/usr/bin/env python3
"""Summarize SEAHORSE COMET results into a LaTeX table.

Reads JSON files produced by eval_seahorse_comet.py and
eval_seahorse_avgvec_comet.py, and formats them into tables mirroring the
paper's style.

Usage
-----
# Per-dimension table (vector only):
python summarize_seahorse_comet.py --results-dir results/seahorse_comet

# Per-dimension table (vector + avgvec):
python summarize_seahorse_comet.py --results-dir results/seahorse_comet \\
    --avgvec-dir results/seahorse_avgvec_comet

# Mean over all dimensions:
python summarize_seahorse_comet.py --results-dir results/seahorse_comet --mean-only

# Condensed mean±std table with color-coded steering:
python summarize_seahorse_comet.py --condensed

# AUC-ROC instead of Pearson r:
python summarize_seahorse_comet.py --results-dir results/seahorse_comet --metric auc
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

# ── dataset constants ─────────────────────────────────────────────────────────

DIMS = ["question1", "question2", "question3", "question4", "question5", "question6"]
DIM_LABEL = {
    "question1": "Comprehensible",
    "question2": "Non-repetitive",
    "question3": "Grammatical",
    "question4": "Attributable",
    "question5": "Main ideas",
    "question6": "Concise",
}
LANGS = ["de", "es", "ru", "tr", "vi"]

# Each entry: (display label, JSON section key, source)
# source="vector" -> per-language steering, files from --results-dir
# source="avgvec" -> cross-lingual zero-shot, files from --avgvec-dir
ROWS_ALL = [
    ("Baseline",           "test_baseline",   "vector"),
    ("+ vector steer",     "test_best_sigma", "vector"),
    ("+ avgvec zero-shot", "test_best_sigma", "avgvec"),
]

ROWS_VECTOR = [
    ("Baseline",       "test_baseline",   "vector"),
    ("+ vector steer", "test_best_sigma", "vector"),
]

DEFAULT_RESULTS_DIR = "results/seahorse_comet"
DEFAULT_AVGVEC_DIR  = "results/seahorse_avgvec_comet"


# ── file loading ──────────────────────────────────────────────────────────────

def load_result(results_dir: str, lang: str) -> Optional[dict]:
    fname = f"{lang}_baseline.json" if lang == "en" else f"{lang}_vector.json"
    path = os.path.join(results_dir, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_avgvec_result(avgvec_dir: str, lang: str) -> Optional[dict]:
    path = os.path.join(avgvec_dir, f"{lang}_avgvec.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── value extraction ──────────────────────────────────────────────────────────

def get_val(result: Optional[dict], section: str, dim: str, metric: str) -> Optional[float]:
    if result is None:
        return None
    v = result.get(section, {}).get("per_dim", {}).get(dim, {}).get(metric)
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


def mean_over_dims(result: Optional[dict], section: str, metric: str) -> Optional[float]:
    if result is None:
        return None
    vals = [get_val(result, section, d, metric) for d in DIMS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def std_over_dims(result: Optional[dict], section: str, metric: str) -> Optional[float]:
    if result is None:
        return None
    vals = [get_val(result, section, d, metric) for d in DIMS]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


# ── formatting ────────────────────────────────────────────────────────────────

def fmt(v: Optional[float]) -> str:
    if v is None:
        return "---"
    s = f"{v:.3f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def bold(s: str) -> str:
    return f"\\textbf{{{s}}}"


def maybe_bold(v: Optional[float], best: Optional[float], s: str) -> str:
    if v is not None and best is not None and abs(v - best) < 5e-4:
        return bold(s)
    return s


def fmt_cell(mean: Optional[float], std: Optional[float], is_bold: bool = False) -> str:
    if mean is None:
        return "---"
    m = fmt(mean)
    if is_bold:
        m = f"\\mathbf{{{m}}}"
    s = fmt(std) if std is not None else "?"
    return f"${m}_{{\\pm{s}}}$"


# ── helpers ───────────────────────────────────────────────────────────────────

def _get(results: Dict[str, Dict[str, Optional[dict]]], lang: str, source: str) -> Optional[dict]:
    return results.get(lang, {}).get(source)


# ── table builders ────────────────────────────────────────────────────────────

def per_dim_table(
    results: Dict[str, Dict[str, Optional[dict]]],
    metric: str,
    rows: list,
) -> str:
    """Full table: one column group per SEAHORSE dimension, sub-columns = languages."""
    n_lang = len(LANGS)
    n_dim  = len(DIMS)

    data: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for label, section, source in rows:
        data[label] = {
            d: {l: get_val(_get(results, l, source), section, d, metric) for l in LANGS}
            for d in DIMS
        }

    best: Dict[Tuple[str, str], Optional[float]] = {}
    for d in DIMS:
        for l in LANGS:
            vals = [data[lbl][d][l] for lbl, *_ in rows if data[lbl][d][l] is not None]
            best[(d, l)] = max(vals) if vals else None

    lang_header = " & ".join(f"\\textbf{{{l}}}" for l in LANGS)
    col_spec = "l" + ("|" + "c" * n_lang) * n_dim
    metric_name = "Pearson $r$" if metric == "pearson" else "AUC-ROC"

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "& " + " & ".join(
            f"\\multicolumn{{{n_lang}}}{{c}}{{\\textbf{{{DIM_LABEL[d]}}}}}"
            for d in DIMS
        ) + " \\\\",
        "& " + " & ".join(lang_header for _ in DIMS) + " \\\\",
        "\\toprule",
        "\\multicolumn{" + str(1 + n_lang * n_dim) + "}{l}{\\textbf{COMET}} \\\\",
        "\\midrule",
    ]

    for label, section, source in rows:
        cells = []
        for d in DIMS:
            for l in LANGS:
                v = data[label][d][l]
                cells.append(maybe_bold(v, best[(d, l)], fmt(v)))
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        f"\\caption{{SEAHORSE {metric_name} for COMET with cross-lingual vector "
        "steering (\\texttt{wmt22-comet-da}). "
        "Baseline uses $\\sigma{=}0$; steered uses the best $\\sigma$ selected "
        "on the validation split by Pearson~$r$; "
        "avgvec zero-shot uses a fixed vector averaged across all source languages. "
        "Bold indicates the highest value per language and dimension.}}",
        "\\label{tab:seahorse_comet}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def mean_only_table(
    results: Dict[str, Dict[str, Optional[dict]]],
    metric: str,
    rows: list,
) -> str:
    """Compact table: one column per language, values = mean over all 6 dimensions."""
    data: Dict[str, Dict[str, Optional[float]]] = {}
    for label, section, source in rows:
        data[label] = {l: mean_over_dims(_get(results, l, source), section, metric) for l in LANGS}

    best: Dict[str, Optional[float]] = {}
    for l in LANGS:
        vals = [data[lbl][l] for lbl, *_ in rows if data[lbl][l] is not None]
        best[l] = max(vals) if vals else None

    lang_header = " & ".join(f"\\textbf{{{l}}}" for l in LANGS)
    col_spec = "l" + "c" * len(LANGS)
    metric_name = "Pearson $r$" if metric == "pearson" else "AUC-ROC"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"& {lang_header} \\\\",
        "\\midrule",
        "\\multicolumn{" + str(1 + len(LANGS)) + "}{l}{\\textbf{COMET}} \\\\",
        "\\midrule",
    ]

    for label, section, source in rows:
        cells = [maybe_bold(data[label][l], best[l], fmt(data[label][l])) for l in LANGS]
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{SEAHORSE {metric_name} (mean over 6 dimensions) for COMET "
        "with cross-lingual vector steering.}}",
        "\\label{tab:seahorse_comet_mean}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def condensed_table(
    results: Dict[str, Dict[str, Optional[dict]]],
    metric: str,
    rows: list,
) -> str:
    """Condensed table: mean±std per language, color-coded steering rows.

    Requires \\usepackage{xcolor} in the LaTeX preamble.
    Bold = highest value per language across all rows.
    Green/↑ (red/↓) on steered rows = improvement (degradation) vs baseline.
    """
    metric_name = "Pearson $r$" if metric == "pearson" else "AUC-ROC"
    lang_header = " & ".join(f"\\textbf{{{l}}}" for l in LANGS)
    col_spec = "l|" + "|".join("c" for _ in LANGS)

    means: Dict[str, Dict[str, Optional[float]]] = {}
    stds:  Dict[str, Dict[str, Optional[float]]] = {}
    for label, section, source in rows:
        means[label] = {l: mean_over_dims(_get(results, l, source), section, metric) for l in LANGS}
        stds[label]  = {l: std_over_dims(_get(results, l, source),  section, metric) for l in LANGS}

    global_best: Dict[str, Optional[float]] = {}
    for lang in LANGS:
        vals = [means[lbl][lang] for lbl, *_ in rows if means[lbl][lang] is not None]
        global_best[lang] = max(vals) if vals else None

    base_label = rows[0][0]
    steer_indices = {i for i, (_, sec, _) in enumerate(rows) if sec == "test_best_sigma"}

    delta: Dict[Tuple[int, str], Optional[float]] = {}
    for i, (label, section, source) in enumerate(rows):
        if i in steer_indices:
            for lang in LANGS:
                m_s = means[label][lang]
                m_b = means[base_label][lang]
                delta[(i, lang)] = (m_s - m_b) if (m_s is not None and m_b is not None) else None

    lines = [
        "% Requires \\usepackage{xcolor} in your LaTeX preamble.",
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"& {lang_header} \\\\",
        "\\toprule",
        "\\multicolumn{" + str(1 + len(LANGS)) + "}{l}{\\textbf{COMET}} \\\\",
        "\\midrule",
    ]

    for i, (label, section, source) in enumerate(rows):
        cells = []
        for lang in LANGS:
            mean = means[label][lang]
            std  = stds[label][lang]
            is_bold = (mean is not None and global_best[lang] is not None
                       and abs(mean - global_best[lang]) < 5e-4)
            cell = fmt_cell(mean, std, is_bold=is_bold)
            if i in steer_indices:
                d = delta.get((i, lang))
                if d is not None and d > 0:
                    cell = f"\\textcolor{{green}}{{{cell}$\\uparrow$}}"
                elif d is not None and d < 0:
                    cell = f"\\textcolor{{red}}{{{cell}$\\downarrow$}}"
            cells.append(cell)
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{SEAHORSE {metric_name} (mean$_{{\\pm\\text{{std}}}}$ over 6 dimensions) "
        "for COMET (\\texttt{wmt22-comet-da}) with cross-lingual vector steering. "
        "Baseline uses $\\sigma{=}0$; steered uses the best $\\sigma$ selected on the "
        "validation split; avgvec zero-shot uses a fixed vector averaged across all source languages. "
        "\\textbf{Bold} indicates the highest value per language. "
        "\\textcolor{green}{Green}/$\\uparrow$ (\\textcolor{red}{red}/$\\downarrow$) "
        "indicates improvement (degradation) relative to the baseline.}}",
        "\\label{tab:seahorse_comet_condensed}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Summarize SEAHORSE COMET results as LaTeX")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Directory with eval_seahorse_comet.py JSON outputs")
    parser.add_argument("--avgvec-dir", default=None,
                        help="Directory with eval_seahorse_avgvec_comet.py JSON outputs "
                             f"(default: {DEFAULT_AVGVEC_DIR})")
    parser.add_argument("--metric", choices=["pearson", "auc"], default="pearson")
    parser.add_argument("--mean-only", action="store_true",
                        help="Show mean over all 6 dimensions instead of per-dimension columns")
    parser.add_argument("--condensed", action="store_true",
                        help="Condensed mean±std table with color-coded steering rows")
    parser.add_argument("--avgvec-only", action="store_true",
                        help="Omit per-language vector rows; show baseline + avgvec only")
    args = parser.parse_args()

    avgvec_dir = args.avgvec_dir or DEFAULT_AVGVEC_DIR

    results: Dict[str, Dict[str, Optional[dict]]] = {}
    missing: List[str] = []

    for lang in LANGS:
        results[lang] = {
            "vector": load_result(args.results_dir, lang),
            "avgvec": load_avgvec_result(avgvec_dir, lang),
        }
        if results[lang]["vector"] is None:
            missing.append(f"{lang}/vector")
        if results[lang]["avgvec"] is None:
            missing.append(f"{lang}/avgvec")

    if missing:
        print(f"% Warning: no result files found for: {missing}")

    if args.avgvec_only:
        rows = [
            ("Baseline",           "test_baseline",   "vector"),
            ("+ avgvec zero-shot", "test_best_sigma", "avgvec"),
        ]
    else:
        rows = ROWS_ALL

    if args.condensed:
        print(condensed_table(results, args.metric, rows))
    elif args.mean_only:
        print(mean_only_table(results, args.metric, rows))
    else:
        print(per_dim_table(results, args.metric, rows))


if __name__ == "__main__":
    main()
