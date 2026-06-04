"""
tables.py — Export DataFrames (or summary CSVs) to LaTeX tables.

Does NOT produce images. Requires \\usepackage{booktabs} (when booktabs=True)
and \\usepackage{longtable} (when longtable=True) in the LaTeX preamble.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd


def save_table_latex(
    source: str | Path | pd.DataFrame,
    out_path: str | Path,
    columns: list[str] | None = None,
    max_rows: int | None = None,
    sort_by: str | list[str] | None = None,
    ascending: bool = True,
    rename: dict[str, str] | None = None,
    float_format: str = "%.3g",
    caption: str | None = None,
    label: str | None = None,
    index: bool = False,
    booktabs: bool = True,
    longtable: bool = False,
    escape: bool = True,
    na_rep: str = "--",
) -> Path:
    """
    Write a LaTeX table from a DataFrame or summary CSV path.

    Pipeline: load → select columns → sort → truncate → rename → to_latex.

    Parameters
    ----------
    source      : CSV path or DataFrame.
    out_path    : destination .tex file (parent dirs created automatically).
    columns     : subset/order of columns; None = all.
    max_rows    : cap rows after sorting (ignored when longtable=True).
    sort_by     : column(s) to sort by before applying max_rows.
    ascending   : sort direction.
    rename      : {original_col: display_name} mapping.
    float_format: printf-style format for float columns.
    caption     : LaTeX table caption (optional).
    label       : LaTeX \\label{} (optional).
    index       : include DataFrame index.
    booktabs    : replace \\hline with \\toprule/\\midrule/\\bottomrule.
    longtable   : use longtable environment (paginated; ignores max_rows).
    escape      : escape special LaTeX characters in cell values.
    na_rep      : replacement string for NaN values.

    Returns
    -------
    Path of the written .tex file.
    """
    # 1. Load
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        from rc_lab.viz.io import load_summary
        df = load_summary(Path(source))

    # 2. Select columns
    if columns is not None:
        df = df[list(columns)]

    # 3. Sort
    if sort_by is not None:
        df = df.sort_values(sort_by, ascending=ascending)

    # 4. Truncate (longtable paginates itself)
    if max_rows is not None and not longtable:
        df = df.head(max_rows)

    # 5. Rename
    if rename is not None:
        df = df.rename(columns=rename)

    # 6. Generate LaTeX via DataFrame.to_latex
    float_fn = lambda x: float_format % x  # noqa: E731

    buf = io.StringIO()
    df.to_latex(
        buf=buf,
        index=index,
        na_rep=na_rep,
        float_format=float_fn,
        longtable=longtable,
        escape=escape,
        caption=caption,
        label=label,
    )
    latex_str = buf.getvalue()

    # 7. Post-process booktabs: replace \hline sequence with booktabs rules
    if booktabs:
        latex_str = _apply_booktabs(latex_str)

    # 8. Write
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex_str, encoding="utf-8")
    return out_path


def _apply_booktabs(latex: str) -> str:
    """
    Replace the first \\hline with \\toprule, the second with \\midrule,
    and the last with \bottomrule inside tabular/longtable environments.
    """
    lines = latex.split("\n")
    in_env = False
    hline_count = 0
    last_hline_idx = -1

    # Find all \hline positions inside the table environment
    hline_indices: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if r"\begin{tabular}" in stripped or r"\begin{longtable}" in stripped:
            in_env = True
        if in_env and stripped == r"\hline":
            hline_indices.append(i)
        if r"\end{tabular}" in stripped or r"\end{longtable}" in stripped:
            in_env = False

    if not hline_indices:
        return latex

    replacements: dict[int, str] = {}
    if len(hline_indices) == 1:
        replacements[hline_indices[0]] = r"\toprule"
    elif len(hline_indices) == 2:
        replacements[hline_indices[0]] = r"\toprule"
        replacements[hline_indices[1]] = r"\bottomrule"
    else:
        replacements[hline_indices[0]] = r"\toprule"
        for idx in hline_indices[1:-1]:
            replacements[idx] = r"\midrule"
        replacements[hline_indices[-1]] = r"\bottomrule"

    new_lines = [
        replacements[i] if i in replacements else line
        for i, line in enumerate(lines)
    ]
    return "\n".join(new_lines)
