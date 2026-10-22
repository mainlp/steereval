#!/usr/bin/env python3
"""Summarize SEAHORSE GPTScore or G-eval results into a LaTeX table.

Reads JSON files produced by eval_seahorse_gptscore.py / eval_seahorse_geval.py
and eval_seahorse_avgvec_gptscore.py, and formats them into a table with SEAHORSE
quality dimensions as column groups and languages as sub-columns.

Files produced with an old sigma range (identifiable by the "val_spearman" key
in layer_sweep entries) are silently omitted.

Usage
-----
# GPTScore per-dimension table (vector + avgvec rows):
python summarize_seahorse_llm.py --scoring gptscore

# Mean-only table:
python summarize_seahorse_llm.py --scoring geval --mean-only

# AUC-ROC instead of Pearson r:
python summarize_seahorse_llm.py --scoring gptscore --metric auc

# Custom directories:
python summarize_seahorse_llm.py --scoring gptscore \\
    --results-dir results/seahorse_gptscore_llama3b \\
    --avgvec-dir  results/seahorse_avgvec_gptscore_llama3b
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

# Each entry: (display label, JSON section key, prompting_lang, source)
# source="vector"  -> per-language steering, files from --results-dir
# source="avgvec"  -> cross-lingual zero-shot, files from --avgvec-dir
# Baseline values come from the vector-en/tgt files' test_baseline section.
ROWS_ALL = [
    ("Baseline (EN-prompt)",             "test_baseline",   "en",  "vector"),
    ("Baseline (TGT-prompt)",            "test_baseline",   "tgt", "vector"),
    ("+ vector steer (EN-prompt)",       "test_best_sigma", "en",  "vector"),
    ("+ vector steer (TGT-prompt)",      "test_best_sigma", "tgt", "vector"),
    ("+ avgvec zero-shot (EN-prompt)",   "test_best_sigma", "en",  "avgvec"),
    ("+ avgvec zero-shot (TGT-prompt)",  "test_best_sigma", "tgt", "avgvec"),
]

ROWS_AVGVEC = [
    ("Baseline (EN-prompt)",             "test_baseline",   "en",  "vector"),
    ("Baseline (TGT-prompt)",            "test_baseline",   "tgt", "vector"),
    ("+ avgvec zero-shot (EN-prompt)",   "test_best_sigma", "en",  "avgvec"),
    ("+ avgvec zero-shot (TGT-prompt)",  "test_best_sigma", "tgt", "avgvec"),
]

DEFAULT_DIRS = {
    ("gptscore", "llama3b"): "results/seahorse_gptscore_llama3b",
    ("geval",    "llama3b"): "results/seahorse_geval_llama3b",
    ("gptscore", "aya"):     "results/seahorse_gptscore_aya",
    ("geval",    "aya"):     "results/seahorse_geval_aya",
}
DEFAULT_AVGVEC_DIRS = {
    ("gptscore", "llama3b"): "results/seahorse_avgvec_gptscore_llama3b",
    ("gptscore", "aya"):     "results/seahorse_avgvec_gptscore_aya",
    # geval avgvec not yet run; files will be absent and treated as missing
    ("geval",    "llama3b"): "results/seahorse_avgvec_geval_llama3b",
    ("geval",    "aya"):     "results/seahorse_avgvec_geval_aya",
}
SCORING_LABEL = {
    "gptscore": "GPTScore",
    "geval":    "G-eval",
}
MODEL_TEXTTT = {
    "llama3b": "Llama-3.2-3B",
    "aya":     "aya-expanse-8b",
}

# ── condensed (combined) table constants ──────────────────────────────────────

CONDENSED_ROWS = [
    ("Baseline (EN-prompt)",        "test_baseline",   "en",  "vector"),
    ("+ vector steer (EN-prompt)",  "test_best_sigma", "en",  "vector"),
    ("Baseline (src-prompt)",       "test_baseline",   "tgt", "vector"),
    ("+ vector steer (src-prompt)", "test_best_sigma", "tgt", "vector"),
]

# Pairs of (steer_row_index, baseline_row_index) in CONDENSED_ROWS for delta coloring
_STEER_PAIRS = [(1, 0), (3, 2)]

# Condensed table variant: baseline + avgvec zero-shot (EN-prompt only)
CONDENSED_AVGVEC_ROWS = [
    ("Baseline (EN-prompt)",           "test_baseline",   "en", "vector"),
    ("+ avgvec zero-shot (EN-prompt)", "test_best_sigma", "en", "avgvec"),
]

SCORING_SECTIONS_ORDER = [
    ("GPTScore",         "gptscore"),
    ("Direct Prompting", "geval"),
]

MODEL_ORDER = [
    ("llama3b", "Llama-3.2-3B"),
    ("aya",     "aya-expanse-8b"),
]

# Large models — avgvec EN-prompt only, GPTScore only
MODEL_ORDER_LARGE = [
    ("llama8b", "Llama-3.1-8B-Instruct"),
    ("aya32b",  "aya-expanse-32b"),
]

DEFAULT_AVGVEC_DIRS_LARGE = {
    ("gptscore", "llama8b"): "results/seahorse_avgvec_gptscore_llama8b",
    ("gptscore", "aya32b"):  "results/seahorse_avgvec_gptscore_aya32b",
}

# Both rows from avgvec files — large models have no separate vector/baseline files
CONDENSED_LARGE_ROWS = [
    ("Baseline (EN-prompt)",           "test_baseline",   "en", "avgvec"),
    ("+ avgvec zero-shot (EN-prompt)", "test_best_sigma", "en", "avgvec"),
]


# ── file loading ──────────────────────────────────────────────────────────────

def _fname_vector(prefix: str, lang: str, plang: str) -> str:
    return f"{prefix}_{lang}_vector_{plang}.json"


def _fname_avgvec(prefix: str, lang: str, plang: str) -> str:
    return f"{prefix}_{lang}_avgvec_{plang}.json"


def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def has_full_sigma_range(result: Optional[dict]) -> bool:
    """Return False if the file pre-dates the negative-sigma range.

    Old files store "val_spearman" in layer_sweep entries; new files store
    "val_auc".  avgvec files have neither, so they pass through as True.
    """
    if result is None:
        return False
    method = result.get("method", "")
    if method in ("baseline", "avgvec"):
        return True

    layer_sweep = result.get("layer_sweep")
    if layer_sweep:
        for layer_data in layer_sweep.values():
            if "val_spearman" in layer_data:
                return False
        return True

    val_results = result.get("val_results")
    if val_results:
        sigmas = [float(k) for k in val_results.keys()]
        return any(s < 0 for s in sigmas)

    return True


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


def fmt_cell(mean: Optional[float], std: Optional[float],
             is_bold: bool = False, is_underline: bool = False) -> str:
    """Format a mean±std cell with optional bold and/or underline on the mean."""
    if mean is None:
        return "---"
    m = fmt(mean)
    if is_underline:
        m = f"\\underline{{{m}}}"
    if is_bold:
        m = f"\\mathbf{{{m}}}"
    s = fmt(std) if std is not None else "?"
    return f"${m}_{{\\pm{s}}}$"


# ── table builders ────────────────────────────────────────────────────────────

def _get(results, lang, plang, source):
    return results[lang][plang][source]


def per_dim_table(
    results: Dict[str, Dict[str, Dict[str, Optional[dict]]]],
    metric: str,
    scoring: str,
    model: str,
    rows: list,
    avgvec_only: bool = False,
) -> str:
    """Full table: one column group per dimension, sub-columns = languages."""
    n_lang = len(LANGS)
    n_dim  = len(DIMS)

    data: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for label, section, plang, source in rows:
        data[label] = {
            d: {l: get_val(_get(results, l, plang, source), section, d, metric)
                for l in LANGS}
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
    scoring_label = SCORING_LABEL[scoring]
    model_tt = MODEL_TEXTTT[model]

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
        "\\multicolumn{" + str(1 + n_lang * n_dim) + "}{l}"
        f"{{\\textbf{{{scoring_label}}}}} \\\\",
        "\\midrule",
    ]

    for label, section, plang, source in rows:
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
        (
            f"\\caption{{SEAHORSE {metric_name} for {scoring_label} with cross-lingual "
            f"averaged-vector steering (\\texttt{{{model_tt}}}). "
            "Baseline uses $\\sigma{=}0$; avgvec uses a fixed vector averaged across "
            "all source languages with pre-set $\\sigma$. "
            "Bold indicates the highest value per language and dimension.}}"
            if avgvec_only else
            f"\\caption{{SEAHORSE {metric_name} for {scoring_label} with vector steering "
            f"(\\texttt{{{model_tt}}}). "
            "Baseline uses $\\sigma{=}0$; vector steer uses the best $\\sigma$ selected "
            "on the validation split by Pearson~$r$ with a language-specific X$\\to$EN vector; "
            "avgvec zero-shot uses a fixed vector averaged across all source languages. "
            "Bold indicates the highest value per language and dimension.}}"
        ),
        f"\\label{{tab:seahorse_{scoring}_{model}}}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def mean_only_table(
    results: Dict[str, Dict[str, Dict[str, Optional[dict]]]],
    metric: str,
    scoring: str,
    model: str,
    rows: list,
    avgvec_only: bool = False,
) -> str:
    """Compact table: one column per language, values = mean over all 6 dimensions."""
    data: Dict[str, Dict[str, Optional[float]]] = {}
    for label, section, plang, source in rows:
        data[label] = {
            l: mean_over_dims(_get(results, l, plang, source), section, metric)
            for l in LANGS
        }

    best: Dict[str, Optional[float]] = {}
    for l in LANGS:
        vals = [data[lbl][l] for lbl, *_ in rows if data[lbl][l] is not None]
        best[l] = max(vals) if vals else None

    lang_header = " & ".join(f"\\textbf{{{l}}}" for l in LANGS)
    col_spec = "l" + "c" * len(LANGS)
    metric_name = "Pearson $r$" if metric == "pearson" else "AUC-ROC"
    scoring_label = SCORING_LABEL[scoring]
    model_tt = MODEL_TEXTTT[model]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"& {lang_header} \\\\",
        "\\midrule",
        "\\multicolumn{" + str(1 + len(LANGS)) + "}{l}"
        f"{{\\textbf{{{scoring_label}}}}} \\\\",
        "\\midrule",
    ]

    for label, section, plang, source in rows:
        cells = [maybe_bold(data[label][l], best[l], fmt(data[label][l])) for l in LANGS]
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        (
            f"\\caption{{SEAHORSE {metric_name} (mean over 6 dimensions) for {scoring_label} "
            f"with cross-lingual averaged-vector steering (\\texttt{{{model_tt}}}). "
            "Avgvec uses a fixed vector averaged across all source languages.}}"
            if avgvec_only else
            f"\\caption{{SEAHORSE {metric_name} (mean over 6 dimensions) for {scoring_label} "
            f"with vector steering (\\texttt{{{model_tt}}}). "
            "Vector steer uses a language-specific X$\\to$EN vector; "
            "avgvec zero-shot uses a fixed averaged vector.}}"
        ),
        f"\\label{{tab:seahorse_{scoring}_{model}_mean}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── hyperparameter summary table ──────────────────────────────────────────────

def params_table(
    results: Dict[str, Dict[str, Dict[str, Optional[dict]]]],
    scoring: str,
    model: str,
    rows: list,
) -> str:
    """Layer and sigma for each steered row × language."""
    steer_rows = [(label, plang, source) for label, section, plang, source in rows
                  if section == "test_best_sigma"]

    n_lang = len(LANGS)
    col_spec = "l" + ("|" + "cc") * n_lang
    scoring_label = SCORING_LABEL[scoring]

    lang_groups = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{{l}}}}}" for l in LANGS
    )
    sub_header = " & ".join("layer & $\\sigma$" for _ in LANGS)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"& {lang_groups} \\\\",
        f"& {sub_header} \\\\",
        "\\toprule",
        f"\\multicolumn{{{1 + 2 * n_lang}}}{{l}}"
        f"{{\\textbf{{{scoring_label}}}}} \\\\",
        "\\midrule",
    ]

    for label, plang, source in steer_rows:
        cells = []
        for l in LANGS:
            r = _get(results, l, plang, source)
            if r is None:
                cells += ["---", "---"]
            else:
                layer = r.get("intervention_layers", [None])[0]
                sigma = r.get("best_sigma")
                layer_str = str(layer) if layer is not None else "---"
                sigma_str = f"${sigma:+.1f}$" if sigma is not None else "---"
                cells += [layer_str, sigma_str]
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    model_tt = MODEL_TEXTTT[model]
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Selected layer and $\\sigma$ for {scoring_label} vector steering "
        f"(\\texttt{{{model_tt}}}). Vector rows: chosen by val Pearson~$r$. "
        "Avgvec rows: fixed hyperparameters.}",
        f"\\label{{tab:seahorse_{scoring}_{model}_params}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── condensed combined table ──────────────────────────────────────────────────

def condensed_table(
    results_all: Dict,
    metric: str,
    prompt_lang: str = "en",
    rows: list = None,
    is_avgvec: bool = False,
    model_order: list = None,
    scoring_sections: list = None,
    label_tag: str = None,
) -> str:
    """One half of the condensed table, filtered to a single prompting language.

    prompt_lang:      "en" -> EN-prompt rows; "tgt" -> source-language-prompt rows.
    rows:             row definitions (defaults to CONDENSED_ROWS).
    is_avgvec:        if True, use avgvec caption text.
    model_order:      list of (model_key, display_name); defaults to MODEL_ORDER.
    scoring_sections: list of (label, scoring_key); defaults to SCORING_SECTIONS_ORDER.
    label_tag:        override the \\label suffix (auto-derived if None).
    results_all[(scoring, model_key)] -> {lang: {plang: {source: result}}}
    Bold      = highest value per language across ALL rows in the table.
    Underline = highest value per language within each model's rows.
    Steering rows: green text + up-arrow for improvement, red + down-arrow for harm.
    Requires \\usepackage{xcolor} in the LaTeX preamble.
    """
    if rows is None:
        rows = CONDENSED_ROWS
    if model_order is None:
        model_order = MODEL_ORDER
    if scoring_sections is None:
        scoring_sections = SCORING_SECTIONS_ORDER

    # Filter to the requested prompting language; order is [baseline, steer]
    active_rows = [(lbl, sec, pl, src)
                   for lbl, sec, pl, src in rows if pl == prompt_lang]
    # With filtered rows: index 0 = baseline, index 1 = steer
    steer_indices = {1}

    n_rows = len(active_rows)          # 2
    n_total_cols = 2 + len(LANGS)
    col_spec = "ll|" + "|".join("c" for _ in LANGS)
    lang_header = " & ".join(f"\\textbf{{{l}}}" for l in LANGS)
    metric_name  = "Pearson $r$" if metric == "pearson" else "AUC-ROC"
    _pl          = "en" if prompt_lang == "en" else "src"
    if label_tag is not None:
        label_suffix = label_tag
    else:
        label_suffix = f"avgvec_{_pl}" if is_avgvec else _pl
    prompt_desc  = "English" if prompt_lang == "en" else "source-language"

    # ── delta lookup: (scoring, model_key, steer_idx=1, lang) -> delta ───────
    delta_map: Dict = {}
    for _, scoring in scoring_sections:
        for model_key, _ in model_order:
            r_dict = results_all.get((scoring, model_key), {})
            _, steer_sec, steer_pl, steer_src = active_rows[1]
            _, base_sec,  base_pl,  base_src  = active_rows[0]
            for lang in LANGS:
                r_s = r_dict.get(lang, {}).get(steer_pl, {}).get(steer_src)
                r_b = r_dict.get(lang, {}).get(base_pl,  {}).get(base_src)
                m_s = mean_over_dims(r_s, steer_sec, metric)
                m_b = mean_over_dims(r_b, base_sec,  metric)
                d = (m_s - m_b) if (m_s is not None and m_b is not None) else None
                delta_map[(scoring, model_key, 1, lang)] = d

    # ── global best per language across ALL sections and models (for bold) ────
    global_best: Dict[str, Optional[float]] = {}
    for lang in LANGS:
        vals = []
        for _, scoring in scoring_sections:
            for model_key, _ in model_order:
                r_dict = results_all.get((scoring, model_key), {})
                for _, section_key, plang, source in active_rows:
                    r = r_dict.get(lang, {}).get(plang, {}).get(source)
                    v = mean_over_dims(r, section_key, metric)
                    if v is not None:
                        vals.append(v)
        global_best[lang] = max(vals) if vals else None

    # ── table ─────────────────────────────────────────────────────────────────
    lines = [
        "% Requires \\usepackage{xcolor} in your LaTeX preamble.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"\\textbf{{}} & \\textbf{{}} & {lang_header} \\\\",
        "\\toprule",
    ]

    first_section = True
    for section_label, scoring in scoring_sections:
        if not first_section:
            lines.append("\\midrule")
        first_section = False

        lines.append(
            f"\\multicolumn{{{n_total_cols}}}{{l}}{{\\textbf{{{section_label}}}}} \\\\"
        )
        lines.append("\\midrule")

        first_model = True
        for model_key, model_name in model_order:
            if not first_model:
                lines.append("\\midrule")
            first_model = False

            r_dict = results_all.get((scoring, model_key), {})

            # Best within this model (for underline)
            model_best: Dict[str, Optional[float]] = {}
            for lang in LANGS:
                vals = []
                for _, section_key, plang, source in active_rows:
                    r = r_dict.get(lang, {}).get(plang, {}).get(source)
                    v = mean_over_dims(r, section_key, metric)
                    if v is not None:
                        vals.append(v)
                model_best[lang] = max(vals) if vals else None

            for i, (label, section_key, plang, source) in enumerate(active_rows):
                cells = []
                for lang in LANGS:
                    r = r_dict.get(lang, {}).get(plang, {}).get(source)
                    mean = mean_over_dims(r, section_key, metric)
                    std  = std_over_dims(r, section_key, metric)
                    is_bold = (mean is not None and global_best[lang] is not None
                               and abs(mean - global_best[lang]) < 5e-4)
                    is_under = (mean is not None and model_best[lang] is not None
                                and abs(mean - model_best[lang]) < 5e-4)
                    cell = fmt_cell(mean, std, is_bold, is_under)
                    if i in steer_indices:
                        d = delta_map.get((scoring, model_key, 1, lang))
                        if d is not None and d > 0:
                            cell = f"\\textcolor{{green}}{{{cell}$\\uparrow$}}"
                        elif d is not None and d < 0:
                            cell = f"\\textcolor{{red}}{{{cell}$\\downarrow$}}"
                    cells.append(cell)

                model_cell = (
                    f"\\multirow{{{n_rows}}}{{*}}{{\\texttt{{{model_name}}}}}"
                    if i == 0 else ""
                )
                lines.append(f"{model_cell} & {label} & {' & '.join(cells)} \\\\")

    _steer_desc = (
        "avgvec zero-shot uses a fixed steering vector averaged across all source "
        "languages with a pre-set $\\sigma$."
        if is_avgvec else
        "steered uses the best $\\sigma$ selected on the validation split."
    )
    caption = (
        f"SEAHORSE {metric_name} averaged over 6 quality dimensions "
        "(Comprehensible, Non-repetitive, Grammatical, Attributable, Main ideas, Concise) "
        f"with standard deviation as subscripts ($\\text{{mean}}_{{\\pm\\text{{std}}}}$), "
        f"per language, using {prompt_desc} prompts. "
        f"Baseline uses $\\sigma{{=}}0$ (no steering); {_steer_desc} "
        "\\textbf{Bold} indicates the single highest value per language across the "
        "entire table; \\underline{underline} indicates the highest value per language "
        "within each model. "
        "\\textcolor{green}{Green}/$\\uparrow$ (\\textcolor{red}{red}/$\\downarrow$) "
        "on steered rows indicates improvement (degradation) relative to the baseline."
    )
    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:seahorse_condensed_{label_suffix}}}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


# ── large-model helpers ───────────────────────────────────────────────────────

def _load_large_results() -> Dict:
    """Load avgvec GPTScore results for large models into results_all format."""
    results_all: Dict = {}
    for model_key, _ in MODEL_ORDER_LARGE:
        avd = DEFAULT_AVGVEC_DIRS_LARGE[("gptscore", model_key)]
        res: Dict[str, Dict[str, Dict[str, Optional[dict]]]] = {}
        for lang in LANGS:
            res[lang] = {"en": {}}
            avg_path = os.path.join(avd, _fname_avgvec(model_key, lang, "en"))
            res[lang]["en"]["avgvec"] = _load(avg_path)
        results_all[("gptscore", model_key)] = res
    return results_all


def large_avgvec_table(metric: str) -> str:
    return condensed_table(
        _load_large_results(), metric, prompt_lang="en",
        rows=CONDENSED_LARGE_ROWS, is_avgvec=True,
        model_order=MODEL_ORDER_LARGE,
        scoring_sections=[("GPTScore", "gptscore")],
        label_tag="avgvec_large_en",
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Summarize SEAHORSE GPTScore/G-eval results as LaTeX"
    )
    parser.add_argument("--scoring", choices=["gptscore", "geval"], default=None,
                        help="Scoring method (required unless --condensed)")
    parser.add_argument("--model", choices=["llama3b", "aya"], default="llama3b")
    parser.add_argument("--results-dir", default=None,
                        help="Directory with per-language vector JSON outputs")
    parser.add_argument("--avgvec-dir", default=None,
                        help="Directory with avgvec zero-shot JSON outputs")
    parser.add_argument("--metric", choices=["pearson", "auc"], default="pearson")
    parser.add_argument("--mean-only", action="store_true",
                        help="Show mean over all 6 dimensions instead of per-dimension columns")
    parser.add_argument("--avgvec-only", action="store_true",
                        help="Only show cross-lingual zero-shot rows (baseline + avgvec)")
    parser.add_argument("--condensed", action="store_true",
                        help="Combined table: both scoring methods x both models, mean+/-std")
    parser.add_argument("--condensed-large", action="store_true",
                        help="Avgvec condensed table for large models (llama8b, aya32b), "
                             "GPTScore only, EN-prompt only")
    args = parser.parse_args()

    # ── condensed large-model avgvec mode ─────────────────────────────────────
    if args.condensed_large:
        print(large_avgvec_table(args.metric))
        return

    # ── condensed mode ────────────────────────────────────────────────────────
    if args.condensed:
        if args.avgvec_only:
            # Load vector files (baseline) and avgvec files (steer)
            results_all: Dict = {}
            for scoring in ("gptscore", "geval"):
                for model_key in ("llama3b", "aya"):
                    rd  = DEFAULT_DIRS[(scoring, model_key)]
                    avd = DEFAULT_AVGVEC_DIRS[(scoring, model_key)]
                    res: Dict[str, Dict[str, Dict[str, Optional[dict]]]] = {}
                    for lang in LANGS:
                        res[lang] = {"en": {}}
                        vec_path = os.path.join(rd, _fname_vector(model_key, lang, "en"))
                        r = _load(vec_path)
                        res[lang]["en"]["vector"] = r if (r is not None and has_full_sigma_range(r)) else None
                        avg_path = os.path.join(avd, _fname_avgvec(model_key, lang, "en"))
                        res[lang]["en"]["avgvec"] = _load(avg_path)
                    results_all[(scoring, model_key)] = res
            print(condensed_table(results_all, args.metric, prompt_lang="en",
                                  rows=CONDENSED_AVGVEC_ROWS, is_avgvec=True))
            print()
            print(large_avgvec_table(args.metric))
            return

        results_all = {}
        for scoring in ("gptscore", "geval"):
            for model_key in ("llama3b", "aya"):
                rd = DEFAULT_DIRS[(scoring, model_key)]
                res = {}
                for lang in LANGS:
                    res[lang] = {"en": {}, "tgt": {}}
                    for plang in ("en", "tgt"):
                        vec_path = os.path.join(rd, _fname_vector(model_key, lang, plang))
                        r = _load(vec_path)
                        if r is None or not has_full_sigma_range(r):
                            res[lang][plang]["vector"] = None
                        else:
                            res[lang][plang]["vector"] = r
                results_all[(scoring, model_key)] = res
        print(condensed_table(results_all, args.metric, prompt_lang="en"))
        print()
        print(condensed_table(results_all, args.metric, prompt_lang="tgt"))
        print()
        print(large_avgvec_table(args.metric))
        return

    if args.scoring is None:
        parser.error("--scoring is required unless --condensed is used")

    prefix      = args.model
    results_dir = args.results_dir or DEFAULT_DIRS[(args.scoring, args.model)]
    avgvec_dir  = args.avgvec_dir  or DEFAULT_AVGVEC_DIRS[(args.scoring, args.model)]

    # Load: results[lang][plang][source]
    results: Dict[str, Dict[str, Dict[str, Optional[dict]]]] = {}
    missing: List[str] = []
    skipped: List[str] = []

    for lang in LANGS:
        results[lang] = {"en": {}, "tgt": {}}
        for plang in ("en", "tgt"):
            # vector files
            vec_path = os.path.join(results_dir, _fname_vector(prefix, lang, plang))
            r = _load(vec_path)
            if r is None:
                missing.append(f"{lang}/{plang}/vector")
                results[lang][plang]["vector"] = None
            elif not has_full_sigma_range(r):
                skipped.append(f"{lang}/{plang} ({os.path.basename(vec_path)})")
                results[lang][plang]["vector"] = None
            else:
                results[lang][plang]["vector"] = r

            # avgvec files
            avg_path = os.path.join(avgvec_dir, _fname_avgvec(prefix, lang, plang))
            r = _load(avg_path)
            if r is None:
                missing.append(f"{lang}/{plang}/avgvec")
                results[lang][plang]["avgvec"] = None
            else:
                results[lang][plang]["avgvec"] = r

    if missing:
        print(f"% Warning: no file found for: {missing}")
    if skipped:
        print(f"% Warning: skipped (old sigma range, pre-negative): {skipped}")

    rows = ROWS_AVGVEC if args.avgvec_only else ROWS_ALL

    if args.mean_only:
        print(mean_only_table(results, args.metric, args.scoring, args.model, rows,
                              avgvec_only=args.avgvec_only))
    else:
        print(per_dim_table(results, args.metric, args.scoring, args.model, rows,
                            avgvec_only=args.avgvec_only))

    print()
    print(params_table(results, args.scoring, args.model, rows))


if __name__ == "__main__":
    main()
