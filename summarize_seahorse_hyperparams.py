#!/usr/bin/env python3
"""Report best hyperparameters (layer and sigma) for all SEAHORSE setups.

Reads result JSONs for GPTScore, G-eval, and COMET and emits two LaTeX tables:

  1. LLM table — best intervention layer + best sigma per (scoring, model,
     prompt_lang, source_language).
  2. COMET table — best sigma per source language (no per-layer selection in
     COMET's pooling-space steering).

Usage
-----
python summarize_seahorse_hyperparams.py          # both tables
python summarize_seahorse_hyperparams.py --llm-only
python summarize_seahorse_hyperparams.py --comet-only
python summarize_seahorse_hyperparams.py --avgvec   # include avgvec rows
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

# ── constants ─────────────────────────────────────────────────────────────────

LANGS = ["de", "es", "ru", "tr", "vi"]

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
COMET_DIR        = "results/seahorse_comet"
COMET_AVGVEC_DIR = "results/seahorse_avgvec_comet"

MODEL_TEXTTT = {"llama3b": "Llama-3.2-3B", "aya": "aya-expanse-8b"}

SCORING_LABEL = {"gptscore": "GPTScore", "geval": "Direct Prompting"}
SCORING_SECTIONS_ORDER = [("gptscore", "GPTScore"), ("geval", "Direct Prompting")]
MODEL_ORDER = [("llama3b", MODEL_TEXTTT["llama3b"]), ("aya", MODEL_TEXTTT["aya"])]


# ── file loading ──────────────────────────────────────────────────────────────

def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _has_full_sigma_range(result: Optional[dict]) -> bool:
    if result is None:
        return False
    method = result.get("method", "")
    if method in ("baseline", "avgvec"):
        return True
    layer_sweep = result.get("layer_sweep")
    if layer_sweep:
        return all("val_spearman" not in e for e in layer_sweep.values())
    val_results = result.get("val_results")
    if val_results:
        return any(float(k) < 0 for k in val_results)
    return True


# ── hyperparameter extraction ─────────────────────────────────────────────────

def _layer(result: Optional[dict]) -> str:
    if result is None:
        return "---"
    layers = result.get("intervention_layers")
    if layers:
        return str(layers[0])
    return "---"


def _sigma(result: Optional[dict]) -> str:
    if result is None:
        return "---"
    s = result.get("best_sigma")
    if s is None:
        return "---"
    return f"${float(s):+.1f}$"


# ── LLM hyperparameter table ──────────────────────────────────────────────────

def llm_hyperparams_table(include_avgvec: bool = False) -> str:
    n_lang = len(LANGS)
    col_spec = "ll|" + "|".join("cc" for _ in LANGS)
    lang_groups = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{{l}}}}}" for l in LANGS
    )
    sub_header = " & ".join("layer & $\\sigma$" for _ in LANGS)
    n_total_cols = 2 + 2 * n_lang

    # Row definitions: (label, plang, source)
    vector_rows = [
        ("EN-prompt",  "en",  "vector"),
        ("src-prompt", "tgt", "vector"),
    ]
    avgvec_rows = [
        ("avgvec (EN)",  "en",  "avgvec"),
        ("avgvec (src)", "tgt", "avgvec"),
    ]
    all_row_defs = vector_rows + (avgvec_rows if include_avgvec else [])
    n_rows_per_model = len(all_row_defs)

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f" & & {lang_groups} \\\\",
        f" & & {sub_header} \\\\",
        "\\toprule",
    ]

    first_section = True
    for scoring, section_label in SCORING_SECTIONS_ORDER:
        if not first_section:
            lines.append("\\midrule")
        first_section = False

        lines.append(
            f"\\multicolumn{{{n_total_cols}}}{{l}}{{\\textbf{{{section_label}}}}} \\\\"
        )
        lines.append("\\midrule")

        first_model = True
        for model_key, model_name in MODEL_ORDER:
            if not first_model:
                lines.append("\\midrule")
            first_model = False

            rd  = LLM_DIRS[(scoring, model_key)]
            avd = LLM_AVGVEC_DIRS[(scoring, model_key)]
            prefix = model_key

            for i, (label, plang, source) in enumerate(all_row_defs):
                cells = []
                for lang in LANGS:
                    if source == "vector":
                        path = os.path.join(rd, f"{prefix}_{lang}_vector_{plang}.json")
                        r = _load(path)
                        if r is not None and not _has_full_sigma_range(r):
                            r = None
                    else:
                        path = os.path.join(avd, f"{prefix}_{lang}_avgvec_{plang}.json")
                        r = _load(path)
                    cells += [_layer(r), _sigma(r)]

                model_col = (
                    f"\\multirow{{{n_rows_per_model}}}{{*}}{{\\texttt{{{model_name}}}}}"
                    if i == 0 else ""
                )
                lines.append(f"{model_col} & {label} & {' & '.join(cells)} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\caption{Best intervention layer and steering $\\sigma$ for LLM-based "
        "SEAHORSE evaluation. Layer and $\\sigma$ are selected jointly on the "
        "validation split by mean Pearson~$r$ across six quality dimensions. "
        "Avgvec rows use a fixed $\\sigma$ and the layer that maximised "
        "cross-lingual vector similarity.}",
        "\\label{tab:seahorse_hyperparams_llm}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


# ── COMET hyperparameter table ────────────────────────────────────────────────

def comet_hyperparams_table(include_avgvec: bool = False) -> str:
    n_lang = len(LANGS)
    col_spec = "l|" + "|".join("c" for _ in LANGS)
    lang_header = " & ".join(f"\\textbf{{{l}}}" for l in LANGS)
    n_total_cols = 1 + n_lang

    row_defs: List[Tuple[str, str, str]] = [("vector steer", COMET_DIR, "{lang}_vector.json")]
    if include_avgvec:
        row_defs.append(("avgvec zero-shot", COMET_AVGVEC_DIR, "{lang}_avgvec.json"))

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"& {lang_header} \\\\",
        "\\toprule",
        f"\\multicolumn{{{n_total_cols}}}{{l}}{{\\textbf{{COMET (wmt22-comet-da)}}}} \\\\",
        "\\midrule",
    ]

    for label, result_dir, fname_tpl in row_defs:
        cells = []
        for lang in LANGS:
            fname = fname_tpl.format(lang=lang)
            r = _load(os.path.join(result_dir, fname))
            cells.append(_sigma(r))
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Best steering $\\sigma$ for COMET (\\texttt{wmt22-comet-da}) "
        "SEAHORSE evaluation. COMET steers the pooling-space embedding directly "
        "(no per-layer selection); $\\sigma$ is chosen on the validation split "
        "by mean Pearson~$r$ across six quality dimensions.}",
        "\\label{tab:seahorse_hyperparams_comet}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── plain-text summary ────────────────────────────────────────────────────────

def plain_summary(include_avgvec: bool = False) -> str:
    rows: List[str] = []

    header = f"{'Setup':<50}  {'Lang':<4}  {'Layer':>5}  {'Sigma':>6}"
    rows.append(header)
    rows.append("-" * len(header))

    for scoring, section_label in SCORING_SECTIONS_ORDER:
        for model_key, model_name in MODEL_ORDER:
            rd  = LLM_DIRS[(scoring, model_key)]
            avd = LLM_AVGVEC_DIRS[(scoring, model_key)]
            prefix = model_key

            for plang in ("en", "tgt"):
                setup_label = f"{section_label} / {model_name} / {plang}-prompt / vector"
                for lang in LANGS:
                    path = os.path.join(rd, f"{prefix}_{lang}_vector_{plang}.json")
                    r = _load(path)
                    if r is not None and not _has_full_sigma_range(r):
                        r = None
                    layer = _layer(r).replace("$", "")
                    sigma = _sigma(r).replace("$", "")
                    rows.append(f"{setup_label:<50}  {lang:<4}  {layer:>5}  {sigma:>6}")

            if include_avgvec:
                for plang in ("en", "tgt"):
                    setup_label = f"{section_label} / {model_name} / {plang}-prompt / avgvec"
                    for lang in LANGS:
                        path = os.path.join(avd, f"{prefix}_{lang}_avgvec_{plang}.json")
                        r = _load(path)
                        layer = _layer(r).replace("$", "")
                        sigma = _sigma(r).replace("$", "")
                        rows.append(f"{setup_label:<50}  {lang:<4}  {layer:>5}  {sigma:>6}")

        rows.append("")

    rows.append("COMET / wmt22-comet-da / vector")
    for lang in LANGS:
        r = _load(os.path.join(COMET_DIR, f"{lang}_vector.json"))
        sigma = _sigma(r).replace("$", "")
        rows.append(f"  {lang}: sigma={sigma}")

    if include_avgvec:
        rows.append("COMET / wmt22-comet-da / avgvec")
        for lang in LANGS:
            r = _load(os.path.join(COMET_AVGVEC_DIR, f"{lang}_avgvec.json"))
            sigma = _sigma(r).replace("$", "")
            rows.append(f"  {lang}: sigma={sigma}")

    return "\n".join(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Report best layer + sigma for all SEAHORSE setups"
    )
    parser.add_argument("--llm-only",   action="store_true")
    parser.add_argument("--comet-only", action="store_true")
    parser.add_argument("--avgvec",     action="store_true",
                        help="Include avgvec rows")
    parser.add_argument("--plain", action="store_true",
                        help="Print plain-text summary instead of LaTeX")
    args = parser.parse_args()

    if args.plain:
        print(plain_summary(args.avgvec))
        return

    if not args.comet_only:
        print(llm_hyperparams_table(args.avgvec))

    if not args.llm_only:
        if not args.comet_only:
            print()
        print(comet_hyperparams_table(args.avgvec))


if __name__ == "__main__":
    main()
