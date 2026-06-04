"""
Tests for viz.tables.save_table_latex.

All tests use synthetic DataFrames/CSVs; no large runs required.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE = pd.DataFrame({
    "model": ["esn", "lstm", "rnn", "tdr"],
    "nmse": [0.123456, 0.0987654, 0.234567, 0.345678],
    "memory": [85.1, 72.3, 60.0, 91.2],
    "rank": [2, 1, 3, 4],
})


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    p = tmp_path / "sample.csv"
    _SAMPLE.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_contains_tabular(tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "out.tex"
    save_table_latex(_SAMPLE, out)
    content = _read(out)
    assert r"\begin{tabular}" in content or r"\begin{longtable}" in content


def test_longtable_env(tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "out.tex"
    save_table_latex(_SAMPLE, out, longtable=True)
    content = _read(out)
    assert r"\begin{longtable}" in content


def test_columns_subset(tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "out.tex"
    save_table_latex(_SAMPLE, out, columns=["model", "nmse"])
    content = _read(out)
    assert "nmse" in content
    assert "memory" not in content


def test_max_rows_truncates(tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "out.tex"
    save_table_latex(_SAMPLE, out, max_rows=2)
    content = _read(out)
    # Only 2 data rows → 'rnn' and 'tdr' should be absent
    assert "rnn" not in content
    assert "tdr" not in content


def test_max_rows_ignored_for_longtable(tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "out.tex"
    save_table_latex(_SAMPLE, out, max_rows=1, longtable=True)
    content = _read(out)
    # All 4 models should appear
    assert "esn" in content
    assert "tdr" in content


def test_sort_by_orders(tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "out.tex"
    save_table_latex(_SAMPLE, out, sort_by="nmse", ascending=True)
    content = _read(out)
    # lstm (0.098) appears before esn (0.123) when sorted ascending by nmse
    assert content.index("lstm") < content.index("esn")


def test_sort_by_max_rows(tmp_path: Path) -> None:
    """sort_by + max_rows → top-N rows after sorting."""
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "out.tex"
    # sort ascending by nmse, keep top-2: lstm (0.098) and esn (0.123)
    save_table_latex(_SAMPLE, out, sort_by="nmse", ascending=True, max_rows=2)
    content = _read(out)
    assert "lstm" in content
    assert "esn" in content
    assert "rnn" not in content
    assert "tdr" not in content


def test_rename_changes_headers(tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "out.tex"
    save_table_latex(_SAMPLE, out, rename={"nmse": "NMSE", "memory": "MC"})
    content = _read(out)
    assert "NMSE" in content
    assert "MC" in content
    assert "nmse" not in content.split(r"\begin{tabular}")[1]  # not in table body


def test_float_format_applied(tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "out.tex"
    save_table_latex(_SAMPLE, out, float_format="%.2f")
    content = _read(out)
    # nmse=0.123456 formatted as %.2f → "0.12"
    assert "0.12" in content


def test_na_rep_in_output(tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    df = _SAMPLE.copy()
    df.loc[0, "nmse"] = float("nan")
    out = tmp_path / "out.tex"
    save_table_latex(df, out, na_rep="N/A")
    content = _read(out)
    assert "N/A" in content


def test_from_csv_path(sample_csv: Path, tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "out.tex"
    save_table_latex(sample_csv, out)
    content = _read(out)
    assert r"\begin{tabular}" in content or r"\begin{longtable}" in content
    assert "esn" in content


def test_creates_parent_dirs(tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "sub" / "deep" / "out.tex"
    result = save_table_latex(_SAMPLE, out)
    assert result.exists()


def test_returns_path(tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "out.tex"
    result = save_table_latex(_SAMPLE, out)
    assert isinstance(result, Path)
    assert result == out


def test_booktabs_replaces_hlines(tmp_path: Path) -> None:
    from rc_lab.viz.tables import save_table_latex

    out = tmp_path / "out.tex"
    save_table_latex(_SAMPLE, out, booktabs=True)
    content = _read(out)
    assert r"\toprule" in content
    assert r"\midrule" in content or r"\bottomrule" in content
