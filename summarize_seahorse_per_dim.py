#!/usr/bin/env python3
"""Per-dimension SEAHORSE LaTeX tables combining GPTScore, G-eval, and COMET.

Generates one table per SEAHORSE quality dimension in the condensed
multi-metric format: GPTScore / G-eval / COMET × models × (baseline, steer).
Columns = languages (de, es, ru, tr, vi). Values are single per-dim scores
(no std, since each dimension contributes one number).

Requires \\usepackage{xcolor,multirow,booktabs} in LaTeX preamble.

Usage
-----
python summarize_seahorse_per_dim.py                  # all 6 dims to stdout
python summarize_seahorse_per_dim.py --dim question3  # single dim
python summarize_seahorse_per_dim.py --metric auc
python summarize_seahorse_per_dim.py --prompt-lang tgt
python summarize_seahorse_per_dim.py --avgvec         # include avgvec rows
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

# ── constants ─────────────────────────────────────────────────────────────────

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

# ── file paths ────────────────────────────────────────────────────────────────

LLM_DIRS = {
    ("gptscore", "llama3b"): "results/seahorse_gptscore_llama3b",
    ("geval",    "llama3b"): "results/seahorse_geval_llama3b",
    ("gptscore", "aya"):     "results/seahorse_gptscore_aya",
    ("geval",    "aya"):     "results/seahorse_geval_aya",
}
LLM_AVGVEC_DIRS = {
    ("gptscore", "llama3b"): "results/seahorse_avgvec_gptscore_llama3b",
    ("geval",    "llama3b"): "results/seahorse_avgvec_geval_llama3b",
    ("gptscore", "aya"):     "results/seahorse_avgvec_gptscore_aya",
    ("geval",    "aya"):     "results/seahorse_avgvec_geval_aya",
}
COMET_DIR         = "results/seahorse_comet"
COMET_AVGVEC_DIR  = "results/seahorse_avgvec_comet"

MODEL_TEXTTT = {"llama3b": "Llama-3.2-3B", "aya": "aya-expanse-8b"}

# Sentinel prompting-language key for COMET (no prompting in COMET scoring)
_COMET_PLANG = "_"
COMET_MODEL_KEY  = "comet_da"
COMET_MODEL_NAME = "wmt22-comet-da"

# Sections: (scoring_key, display_label, [(model_key, display_name)])
SCORING_SECTIONS = [
    ("gptscore", "GPTScore",
     [("llama3b", MODEL_TEXTTT["llama3b"]), ("aya", MODEL_TEXTTT["aya"])]),
    ("geval", "Direct Prompting",
     [("llama3b", MODEL_TEXTTT["llama3b"]), ("aya", MODEL_TEXTTT["aya"])]),
    ("comet", "COMET",
     [(COMET_MODEL_KEY, COMET_MODEL_NAME)]),
]

# ── row templates: (label, section_key, plang, source) ───────────────────────

_LLM_ROWS = {
    "en": [
        ("Baseline (EN-prompt)",       "test_baseline",   "en", "vector"),
        ("+ vector steer (EN-prompt)", "test_best_sigma", "en", "vector"),
    ],
    "tgt": [
        ("Baseline (src-prompt)",       "test_baseline",   "tgt", "vector"),
        ("+ vector steer (src-prompt)", "test_best_sigma", "tgt", "vector"),
    ],
}
_LLM_ROWS_AVGVEC = {
    "en": [
        ("Baseline (EN-prompt)",            "test_baseline",   "en", "vector"),
        ("+ vector steer (EN-prompt)",       "test_best_sigma", "en", "vector"),
        ("+ avgvec zero-shot (EN-prompt)",   "test_best_sigma", "en", "avgvec"),
    ],
    "tgt": [
        ("Baseline (src-prompt)",            "test_baseline",   "tgt", "vector"),
        ("+ vector steer (src-prompt)",       "test_best_sigma", "tgt", "vector"),
        ("+ avgvec zero-shot (src-prompt)",   "test_best_sigma", "tgt", "avgvec"),
    ],
}
_COMET_ROWS = [
    ("Baseline",           "test_baseline",   _COMET_PLANG, "vector"),
    ("+ vector steer",     "test_best_sigma", _COMET_PLANG, "vector"),
]
_COMET_ROWS_AVGVEC = [
    ("Baseline",           "test_baseline",   _COMET_PLANG, "vector"),
    ("+ vector steer",     "test_best_sigma", _COMET_PLANG, "vector"),
    ("+ avgvec zero-shot", "test_best_sigma", _COMET_PLANG, "avgvec"),
]


# ── file loading ──────────────────────────────────────────────────────────────

def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _has_full_sigma_range(result: Optional[dict]) -> bool:
    """Return False for old result files that predate the negative-sigma range."""
    if result is None:
        return False
    method = result.get("method", "")
    if method in ("baseline", "avgvec"):
        return True
    layer_sweep = result.get("layer_sweep")
    if layer_sweep:
        for entry in layer_sweep.values():
            if "val_spearman" in entry:
                return False
        return True
    val_results = result.get("val_results")
    if val_results:
        return any(float(k) < 0 for k in val_results)
    return True


def load_all_results(include_avgvec: bool = False) -> dict:
    """Load all result JSONs into results_all[(scoring, model)][lang][plang][source]."""
    results_all: Dict[Tuple[str, str], Dict] = {}
    missing: List[str] = []
    skipped: List[str] = []

    for scoring in ("gptscore", "geval"):
        for model_key in ("llama3b", "aya"):
            prefix = model_key
            rd  = LLM_DIRS[(scoring, model_key)]
            avd = LLM_AVGVEC_DIRS[(scoring, model_key)]
            res: Dict = {}
            for lang in LANGS:
                res[lang] = {"en": {}, "tgt": {}}
                for plang in ("en", "tgt"):
                    vpath = os.path.join(rd, f"{prefix}_{lang}_vector_{plang}.json")
                    r = _load(vpath)
                    if r is None:
                        missing.append(f"{scoring}/{model_key}/{lang}/{plang}/vector")
                        res[lang][plang]["vector"] = None
                    elif not _has_full_sigma_range(r):
                        skipped.append(vpath)
                        res[lang][plang]["vector"] = None
                    else:
                        res[lang][plang]["vector"] = r
                    if include_avgvec:
                        apath = os.path.join(avd, f"{prefix}_{lang}_avgvec_{plang}.json")
                        res[lang][plang]["avgvec"] = _load(apath)
            results_all[(scoring, model_key)] = res

    comet_res: Dict = {}
    for lang in LANGS:
        comet_res[lang] = {_COMET_PLANG: {}}
        vpath = os.path.join(COMET_DIR, f"{lang}_vector.json")
        comet_res[lang][_COMET_PLANG]["vector"] = _load(vpath)
        if include_avgvec:
            apath = os.path.join(COMET_AVGVEC_DIR, f"{lang}_avgvec.json")
            comet_res[lang][_COMET_PLANG]["avgvec"] = _load(apath)
    results_all[("comet", COMET_MODEL_KEY)] = comet_res

    if missing:
        print(f"% Warning: missing files for: {missing[:6]}{'...' if len(missing) > 6 else ''}")
    if skipped:
        print(f"% Warning: skipped (old sigma range): {[os.path.basename(p) for p in skipped]}")

    return results_all


# ── value extraction ──────────────────────────────────────────────────────────

def get_val(result: Optional[dict], section: str, dim: str, metric: str) -> Optional[float]:
    if result is None:
        return None
    v = result.get(section, {}).get("per_dim", {}).get(dim, {}).get(metric)
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


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


def fmt_cell(v: Optional[float], is_bold: bool = False,
             is_under: bool = False) -> str:
    if v is None:
        return "---"
    s = fmt(v)
    if is_under:
        s = f"\\underline{{{s}}}"
    if is_bold:
        s = f"\\mathbf{{{s}}}"
    return f"${s}$"


# ── table builder ─────────────────────────────────────────────────────────────

def per_dim_table(
    results_all: dict,
    dim: str,
    metric: str,
    prompt_lang: str,
    include_avgvec: bool,
) -> str:
    col_spec    = "ll|" + "|".join("c" for _ in LANGS)
    lang_header = " & ".join(f"\\textbf{{{l}}}" for l in LANGS)
    metric_name = "Pearson $r$" if metric == "pearson" else "AUC-ROC"
    n_total_cols = 2 + len(LANGS)

    llm_rows   = (_LLM_ROWS_AVGVEC   if include_avgvec else _LLM_ROWS)[prompt_lang]
    comet_rows = (_COMET_ROWS_AVGVEC if include_avgvec else _COMET_ROWS)

    steer_idx_llm   = {i for i, (_, sec, _, _) in enumerate(llm_rows)   if sec == "test_best_sigma"}
    steer_idx_comet = {i for i, (_, sec, _, _) in enumerate(comet_rows) if sec == "test_best_sigma"}

    def _val(r, sec):
        return get_val(r, sec, dim, metric)

    def _rows_and_steer(scoring):
        if scoring == "comet":
            return comet_rows, steer_idx_comet
        return llm_rows, steer_idx_llm

    # Global best per language across all rows in the whole table
    global_best: Dict[str, Optional[float]] = {l: None for l in LANGS}
    for scoring, _, model_order in SCORING_SECTIONS:
        active_rows, _ = _rows_and_steer(scoring)
        for model_key, _ in model_order:
            r_dict = results_all.get((scoring, model_key), {})
            for lang in LANGS:
                for _, sec, pl, src in active_rows:
                    r = r_dict.get(lang, {}).get(pl, {}).get(src)
                    v = _val(r, sec)
                    if v is not None and (global_best[lang] is None or v > global_best[lang]):
                        global_best[lang] = v

    # Table header
    lines = [
        "% Requires \\usepackage{xcolor,multirow,booktabs} in LaTeX preamble.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"\\textbf{{}} & \\textbf{{}} & {lang_header} \\\\",
        "\\toprule",
    ]

    first_section = True
    for scoring, section_label, model_order in SCORING_SECTIONS:
        active_rows, steer_idx = _rows_and_steer(scoring)
        n_rows = len(active_rows)

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

            # Per-model best per language (for underline)
            model_best: Dict[str, Optional[float]] = {l: None for l in LANGS}
            for lang in LANGS:
                for _, sec, pl, src in active_rows:
                    r = r_dict.get(lang, {}).get(pl, {}).get(src)
                    v = _val(r, sec)
                    if v is not None and (model_best[lang] is None or v > model_best[lang]):
                        model_best[lang] = v

            base_label, base_sec, base_pl, base_src = active_rows[0]

            for i, (label, sec, pl, src) in enumerate(active_rows):
                cells = []
                for lang in LANGS:
                    r   = r_dict.get(lang, {}).get(pl,      {}).get(src)
                    r_b = r_dict.get(lang, {}).get(base_pl, {}).get(base_src)
                    v   = _val(r,   sec)
                    v_b = _val(r_b, base_sec)

                    is_bold  = (v is not None and global_best[lang] is not None
                                and abs(v - global_best[lang]) < 5e-4)
                    is_under = (v is not None and model_best[lang] is not None
                                and abs(v - model_best[lang]) < 5e-4)
                    cell = fmt_cell(v, is_bold, is_under)

                    if i in steer_idx and v is not None and v_b is not None:
                        delta = v - v_b
                        if delta > 0:
                            cell = f"\\textcolor{{blue!70!black}}{{{cell}$\\uparrow$}}"
                        elif delta < 0:
                            cell = f"\\textcolor{{red!70!black}}{{{cell}$\\downarrow$}}"
                    cells.append(cell)

                model_col = (
                    f"\\multirow{{{n_rows}}}{{*}}{{\\texttt{{{model_name}}}}}"
                    if i == 0 else ""
                )
                lines.append(f"{model_col} & {label} & {' & '.join(cells)} \\\\")

    prompt_desc = "English" if prompt_lang == "en" else "source-language"
    steer_desc = (
        "steered uses the best $\\sigma$ selected on the validation split"
        + ("; avgvec zero-shot uses a fixed vector averaged across all source languages" if include_avgvec else "")
        + "."
    )
    caption = (
        f"SEAHORSE {metric_name} for dimension \\textbf{{{DIM_LABEL[dim]}}} per language, "
        f"using {prompt_desc} prompts. "
        f"Baseline uses $\\sigma{{=}}0$ (no steering); {steer_desc} "
        "\\textbf{Bold} indicates the highest value per language across the entire table; "
        "\\underline{underline} indicates the highest value per language within each model. "
        "\\textcolor{blue!70!black}{Blue}/$\\uparrow$ "
        "(\\textcolor{red!70!black}{red}/$\\downarrow$) "
        "on steered rows indicates improvement (degradation) relative to the baseline."
    )

    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:seahorse_perdim_{dim}_{prompt_lang}}}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-dimension SEAHORSE LaTeX tables (GPTScore + G-eval + COMET)"
    )
    parser.add_argument("--dim", choices=DIMS, default=None,
                        help="Single dimension (default: all 6, separated by blank lines)")
    parser.add_argument("--metric", choices=["pearson", "auc"], default="pearson")
    parser.add_argument("--prompt-lang", choices=["en", "tgt"], default="en",
                        help="Prompting language for LLM rows (COMET rows are unaffected)")
    parser.add_argument("--avgvec", action="store_true",
                        help="Include avgvec zero-shot rows (3 rows per model instead of 2)")
    args = parser.parse_args()

    results_all = load_all_results(include_avgvec=args.avgvec)

    dims = [args.dim] if args.dim else DIMS
    for i, dim in enumerate(dims):
        if i > 0:
            print()
            print()
        print(per_dim_table(results_all, dim, args.metric, args.prompt_lang, args.avgvec))


if __name__ == "__main__":
    main()
