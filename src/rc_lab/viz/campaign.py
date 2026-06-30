"""campaign.py — Reproducible figures and LaTeX tables for final campaign results.

This module is intentionally post-hoc: it consumes persisted campaign summaries
(`summary.csv` for the multitask sweep and `comparison_summary.csv` for design)
and produces manuscript-ready artifacts without depending on the runners.

Sweep batch (regenerated):
  - F_sweep_error_vs_rho      curvas de descenso de error vs rho por s_in (S1)
  - F_sweep_task_heatmaps     planos rho x s_in por tarea (S2)
  - F_sweep_aggregate_heatmaps  meseta de compromiso (rank agregado)
  - F_sweep_tradeoff_mc_mg / _mc_narma  frontera memoria <-> no-linealidad (S3)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from rc_lab.viz.io import pivot_plane
from rc_lab.viz.primitives import add_colorbar, categorical_vline, heatmap_display_grid, heatmap_plane
from rc_lab.viz.style import CMAPS, PALETTE, apply_style, save_figure
from rc_lab.viz.tables import save_table_latex


# ---------------------------------------------------------------------------
# Constants / labels
# ---------------------------------------------------------------------------

FAMILY_ORDER = [
    "random_sparse_baseline",
    "cycle_scr",
    "cycle_jump_j3",
    "cycle_jump_j7",
    "nonnormal_chain_g0_1",
    "nonnormal_chain_g0_3",
    "nonnormal_chain_g0_6",
    "multiscale_two_random",
    "multiscale_three_random",
]

FAMILY_LABELS = {
    "random_sparse_baseline": "random sparse",
    "cycle_scr": "SCR",
    "cycle_jump_j3": "CRJ j=3",
    "cycle_jump_j7": "CRJ j=7",
    "nonnormal_chain_g0_1": "cadena g=0.1",
    "nonnormal_chain_g0_3": "cadena g=0.3",
    "nonnormal_chain_g0_6": "cadena g=0.6",
    "multiscale_two_random": "multiescala 2",
    "multiscale_three_random": "multiescala 3",
}

TASK_LABELS = {
    "narma10": "NARMA-10",
    "mackey_glass": "Mackey–Glass",
    "memory_capacity": "MC total",
    "delay_recall": "Delay Recall",
    "aggregate": "agregado",
}

TASK_RANK_COLS = {
    "narma10": "global_rank_narma10",
    "mackey_glass": "global_rank_mg",
    "memory_capacity": "global_rank_mc",
}

TASK_METRIC_COLS = {
    "narma10": "narma10_val_nmse_mean",
    "mackey_glass": "mg_val_nmse_mean",
    "memory_capacity": "mc_total_mean",
}

# Clean, ordered palette for s_in lines (light, distinguishable; legend conveys order)
SIN_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

LAB_RHO = r"$\rho$"
LAB_SIN = r"$s_{\mathrm{in}}$"
LAB_ALPHA = r"$\alpha$"

# Design figure accent colours
ACCENT_RED = "#C44E52"
ACCENT_BLUE = "#6B8CBE"
CLOUD_GRAY = "#BBBBBB"

# ---------------------------------------------------------------------------
# External comparison — constants
# ---------------------------------------------------------------------------

MODEL_ORDER: list[str] = [
    "random_sparse_baseline",
    "cycle_scr",
    "cycle_jump_j7",
    "nonnormal_chain_g0_3",
    "multiscale_three_random",
    "simple_rnn",
    "lstm",
    "tapped_delay_ridge",
]

MODEL_LABELS: dict[str, str] = {
    **FAMILY_LABELS,
    "simple_rnn": "RNN simple",
    "lstm": "LSTM",
    "tapped_delay_ridge": "tapped-delay + ridge",
}

KIND_COLORS: dict[str, str] = {
    "esn": "#4C72B0",
    "torch_simple_rnn": "#C44E52",
    "torch_lstm": "#DD8452",
    "tapped_delay_ridge": "#7f7f7f",
}

# (task_key, display_label, val_col, test_col, maximize)
TASK_SPECS_EXTERNAL: list[tuple[str, str, str, str, bool]] = [
    (
        "delay_recall", "Delay Recall",
        "delay_recall_memory_corr_total_mean",
        "delay_recall_test_memory_corr_total_mean",
        True,
    ),
    (
        "narma10", "NARMA-10",
        "narma10_val_nmse_mean",
        "narma10_test_nmse_mean",
        False,
    ),
    (
        "mackey_glass", "Mackey–Glass",
        "mg_val_nmse_mean",
        "mg_test_nmse_mean",
        False,
    ),
]

# Metric key inside runs/<config>_seed*.json > test_metrics
_RUNS_METRIC_KEY: dict[str, str] = {
    "delay_recall": "memory_corr_total",
    "narma10": "nmse",
    "mackey_glass": "nmse",
}


# ---------------------------------------------------------------------------
# Loading / normalization
# ---------------------------------------------------------------------------

def load_sweep_summary(path_or_dir: str | Path) -> pd.DataFrame:
    """Load final multitask sweep summary.csv from a file or directory."""
    p = Path(path_or_dir)
    csv_path = p / "summary.csv" if p.is_dir() else p
    df = pd.read_csv(csv_path)
    _require_columns(
        df,
        [
            "config_id", "spectral_radius", "input_scaling", "leak_rate",
            "narma10_val_nmse_mean", "mg_val_nmse_mean", "mc_total_mean",
            "rank_narma10", "rank_mg", "rank_mc", "aggregate_rank",
        ],
        "sweep summary",
    )
    return df


def load_design_summary(path_or_dir: str | Path) -> pd.DataFrame:
    """Load design comparison_summary.csv from a file or directory."""
    p = Path(path_or_dir)
    csv_path = p / "comparison_summary.csv" if p.is_dir() else p
    df = pd.read_csv(csv_path)
    _require_columns(
        df,
        [
            "design_name", "reservoir_type", "config_id",
            "spectral_radius", "input_scaling", "leak_rate",
            "narma10_val_nmse_mean", "mg_val_nmse_mean", "mc_total_mean",
            "global_rank_narma10", "global_rank_mg", "global_rank_mc",
            "global_aggregate_rank",
        ],
        "design comparison summary",
    )
    return df


def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")


# ---------------------------------------------------------------------------
# External comparison — data model
# ---------------------------------------------------------------------------

@dataclass
class ExternalData:
    """Loaded external comparison results, ready for table/figure building."""
    candidates: pd.DataFrame          # Full comparison_summary.csv (all model×config rows)
    selected: dict[str, pd.DataFrame] # task_key -> per-model selected DataFrame
    has_std: bool                     # True if runs/ dirs were found and std was computed
    results_dir: Path


def load_external_summary(results_dir: str | Path) -> ExternalData:
    """Load external comparison results from comparison_summary.csv.

    results_dir: directory containing comparison_summary.csv and
    <model>/<task>/runs/*.json.  If runs/ is absent, std columns are NaN and
    a UserWarning is emitted.  Selection is always by validation metric.
    """
    import warnings

    d = Path(results_dir)
    csv_path = d / "comparison_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"comparison_summary.csv not found in {d}")

    candidates = pd.read_csv(csv_path)

    has_std = _detect_runs_dirs(d)
    if not has_std:
        warnings.warn(
            f"No runs/ subdirectories found under {d}; "
            "std deviation and error bars will be unavailable.",
            UserWarning,
            stacklevel=2,
        )

    selected: dict[str, pd.DataFrame] = {}
    for task, _label, val_col, test_col, maximize in TASK_SPECS_EXTERNAL:
        rows = _select_per_model(candidates, val_col, test_col, maximize, d, task, has_std)
        if rows:
            sel_df = pd.DataFrame(rows)
            test_vals = pd.to_numeric(sel_df["test"], errors="coerce")
            if maximize:
                sel_df["rank_1_8"] = test_vals.rank(
                    ascending=False, method="min", na_option="bottom"
                ).astype(int)
            else:
                sel_df["rank_1_8"] = test_vals.rank(
                    ascending=True, method="min", na_option="bottom"
                ).astype(int)
        else:
            sel_df = pd.DataFrame(
                columns=[
                    "model_name", "model_kind", "val", "test", "test_std",
                    "n_trainable_params", "tuning_s", "rank_1_8",
                ]
            )
        selected[task] = sel_df

    return ExternalData(
        candidates=candidates, selected=selected,
        has_std=has_std, results_dir=d,
    )


def _detect_runs_dirs(d: Path) -> bool:
    """Return True if any <model>/<task>/runs/ directory exists under d."""
    for child in d.iterdir():
        if not child.is_dir():
            continue
        for task in ("delay_recall", "narma10", "mackey_glass"):
            if (child / task / "runs").is_dir():
                return True
    return False


def _select_per_model(
    candidates: pd.DataFrame,
    val_col: str,
    test_col: str,
    maximize: bool,
    results_dir: Path,
    task: str,
    has_std: bool,
) -> list[dict[str, object]]:
    """For each model_name, pick best config by val_col; record test_col."""
    rows: list[dict[str, object]] = []
    for model_name, group in candidates.groupby("model_name"):
        vals = pd.to_numeric(group[val_col], errors="coerce")
        valid_idx = vals.dropna().index
        if valid_idx.empty:
            continue
        best_idx = vals.loc[valid_idx].idxmax() if maximize else vals.loc[valid_idx].idxmin()
        best = candidates.loc[best_idx]

        test_std = float("nan")
        if has_std:
            test_std = _load_test_std_from_runs(
                results_dir, task,
                str(best["model_name"]), str(best["config_id"]),
                _RUNS_METRIC_KEY[task],
            )

        rows.append({
            "model_name": str(model_name),
            "model_kind": str(best.get("model_kind", "")),
            "config_id": str(best.get("config_id", "")),
            "val": _to_float(best.get(val_col)),
            "test": _to_float(best.get(test_col)),
            "test_std": test_std,
            "n_trainable_params": _trainable_params_for_task(best, task),
            "tuning_s": _to_float(best.get("tuning_total_s_sum")),
        })
    return rows


def _load_test_std_from_runs(
    results_dir: Path,
    task: str,
    model_name: str,
    config_id: str,
    metric_key: str,
) -> float:
    """Read per-seed run JSONs and return std of the test metric (ddof=1)."""
    import json

    runs_dir = results_dir / model_name / task / "runs"
    if not runs_dir.exists():
        return float("nan")
    values: list[float] = []
    for path in runs_dir.glob(f"{config_id}_*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                run = json.load(f)
            val = run.get("test_metrics", {}).get(metric_key)
            if val is not None:
                fval = float(val)
                if np.isfinite(fval):
                    values.append(fval)
        except Exception:
            pass
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def _to_float(val: object) -> float:
    """Safe float conversion; returns NaN on failure."""
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Trainable-parameter helpers (full learned model, per task)
# ---------------------------------------------------------------------------

def _tapped_feature_count(n_lags: int, feature_mode: str) -> int:
    """Feature dimension for TappedDelayRidge with 1-dim input (lags 0..n_lags)."""
    base = n_lags + 1
    return base * 2 if feature_mode in ("quadratic", "linear_quadratic") else base


def _trainable_params_for_task(
    row: "pd.Series",
    task: str,
    *,
    kmax: int = 100,
    esn_N: int = 100,
) -> float:
    """Return the trainable parameter count of this candidate for one task.

    ESN: RidgeReadout uses fit_intercept=False; Phi shape is (T, N) for
    features='states', so n_trainable = N * n_outputs.  N must be provided
    explicitly (from the reservoir config) rather than inverted from the
    stored cross-task mean.

    Tapped-delay: computed from n_lags + feature_mode (from the CSV row).

    Torch: computed from the actual hidden size, layer count and task output
    size.  The stored n_trainable_params_mean is a cross-task average and is
    deliberately not used as the size of a concrete model.
    """
    kind = str(row.get("model_kind", ""))
    n_outputs = kmax if task == "delay_recall" else 1
    task_prefix = "mg" if task == "mackey_glass" else task
    persisted = _to_float(row.get(f"{task_prefix}_n_trainable_params"))
    if np.isfinite(persisted):
        return persisted

    if kind == "esn":
        return float(esn_N * n_outputs)

    if kind in {"torch_simple_rnn", "torch_lstm"}:
        hidden_raw = _to_float(row.get("hidden_size"))
        layers_raw = _to_float(row.get("num_layers"))
        if not np.isfinite(hidden_raw):
            return float("nan")
        hidden = int(hidden_raw)
        num_layers = int(layers_raw) if np.isfinite(layers_raw) else 1
        if num_layers < 1:
            return float("nan")

        if kind == "torch_simple_rnn":
            recurrent = hidden + hidden * hidden + 2 * hidden
            recurrent += (num_layers - 1) * (
                2 * hidden * hidden + 2 * hidden
            )
        else:
            recurrent = 4 * hidden + 4 * hidden * hidden + 8 * hidden
            recurrent += (num_layers - 1) * (
                8 * hidden * hidden + 8 * hidden
            )
        return float(recurrent + n_outputs * (hidden + 1))

    if kind in {"tapped_delay_ridge", "narx_ridge", "ng_rc"}:
        n_lags_raw = _to_float(row.get("n_lags"))
        if not np.isfinite(n_lags_raw):
            return float("nan")
        feature_mode = str(row.get("feature_mode", "raw"))
        return float(n_outputs * _tapped_feature_count(int(n_lags_raw), feature_mode))

    return _to_float(row.get("n_trainable_params_mean"))


# ---------------------------------------------------------------------------
# Tables as DataFrames
# ---------------------------------------------------------------------------

def sweep_representatives(df: pd.DataFrame) -> pd.DataFrame:
    """Representative rows mapping the two-arm narrative: best compromise plus the
    single-task specialists (memory / Mackey-Glass / NARMA)."""
    selectors = [
        ("Compromiso (mejor agregado)", "aggregate_rank", "min"),
        ("Especialista memoria (MC total)", "rank_mc", "min"),
        ("Especialista Mackey–Glass", "rank_mg", "min"),
        ("Especialista NARMA-10", "rank_narma10", "min"),
    ]
    rows: list[dict[str, object]] = []
    for label, col, direction in selectors:
        idx = df[col].idxmin() if direction == "min" else df[col].idxmax()
        rows.append(_sweep_display_row(df.loc[idx], label))
    return pd.DataFrame(rows)


def sweep_top_configs(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Top-n sweep configurations by aggregate rank."""
    out = df.sort_values("aggregate_rank", ascending=True).head(n).copy()
    rows = [_sweep_display_row(row, "") for _, row in out.iterrows()]
    result = pd.DataFrame(rows)
    if "rol" in result.columns:
        result = result.drop(columns=["rol"])
    return result


def design_best_by_task(df: pd.DataFrame) -> pd.DataFrame:
    """Best global design row overall and for each task (selection by test metric)."""
    narma_col = _best_col(df, "narma10_test_nmse_mean", "narma10_val_nmse_mean")
    mg_col    = _best_col(df, "mg_test_nmse_mean", "mg_val_nmse_mean")
    rows = [
        _design_display_row(df.loc[df["global_aggregate_rank"].idxmin()], "Mejor agregado"),
        _design_display_row(df.loc[df[narma_col].idxmin()],               "Mejor NARMA-10"),
        _design_display_row(df.loc[df[mg_col].idxmin()],                  "Mejor Mackey–Glass"),
        _design_display_row(df.loc[df["mc_total_mean"].idxmax()],          "Mejor MC total"),
    ]
    return pd.DataFrame(rows)


def design_best_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Best within-family configuration, ordered by global aggregate rank."""
    idxs = df.groupby("design_name")["aggregate_rank_within_design"].idxmin()
    out = df.loc[idxs].sort_values("global_aggregate_rank", ascending=True)
    return pd.DataFrame([_design_display_row(row, "") for _, row in out.iterrows()]).drop(columns=["rol"])


def design_family_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per family: best per task (by test metric) and best aggregate."""
    narma_col = _best_col(df, "narma10_test_nmse_mean", "narma10_val_nmse_mean")
    mg_col    = _best_col(df, "mg_test_nmse_mean", "mg_val_nmse_mean")
    task_specs = [
        ("memory_capacity", "mc_total_mean", True),
        ("narma10",         narma_col,       False),
        ("mackey_glass",    mg_col,          False),
    ]
    rows: list[dict[str, object]] = []
    for family in _ordered_families(df):
        sub = df[df["design_name"] == family]
        best_agg = sub.loc[sub["global_aggregate_rank"].idxmin()]
        row: dict[str, object] = {
            "familia": FAMILY_LABELS.get(family, family),
            "mejor agg.": int(best_agg["global_aggregate_rank"]),
        }
        for task, metric_col, maximize in task_specs:
            best = sub.loc[sub[metric_col].idxmax() if maximize else sub[metric_col].idxmin()]
            row[f"rank {TASK_LABELS[task]}"] = int(best[TASK_RANK_COLS[task]])
            row[f"métrica {TASK_LABELS[task]}"] = _round(best[metric_col], 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mejor agg.", ascending=True)


def save_campaign_tables(
    sweep_df: pd.DataFrame,
    design_df: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Save standard CSV + LaTeX tables for the final campaign."""
    out = Path(out_dir)
    csv_dir = out / "csv"
    tex_dir = out / "latex"
    csv_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "sweep_representatives": sweep_representatives(sweep_df),
        "sweep_top12": sweep_top_configs(sweep_df, n=12),
        "design_best_by_task": design_best_by_task(design_df),
        "design_best_by_family": design_best_by_family(design_df),
        "design_family_task_summary": design_family_task_summary(design_df),
        "design_mechanism": design_mechanism_table(design_df),
    }

    paths: dict[str, Path] = {}
    for name, table in tables.items():
        csv_path = csv_dir / f"{name}.csv"
        tex_path = tex_dir / f"{name}.tex"
        table.to_csv(csv_path, index=False)
        save_table_latex(table, tex_path, float_format="%.4g", index=False, escape=False, booktabs=True)
        paths[f"{name}_csv"] = csv_path
        paths[f"{name}_tex"] = tex_path
    return paths


def external_master_table(ext: ExternalData, *, kmax: int = 100, esn_N: int = 100) -> pd.DataFrame:
    """Master table: one row per model in MODEL_ORDER, test metrics + cost.

    Parameter columns use the candidate selected by validation for each task.
    They are separate because NARMA-10 and Mackey-Glass can select different
    architecture sizes even though both tasks have one output.

    Champion of each task is marked with a trailing ``*``.
    """
    # tuning_total_s_sum in comparison_summary is the 3-task total per model.
    tuning_by_model: dict[str, float] = {}
    for mn, grp in ext.candidates.groupby("model_name"):
        ts = _to_float(grp["tuning_total_s_sum"].dropna().iloc[0] if len(grp) > 0 else float("nan"))
        if np.isfinite(ts):
            tuning_by_model[str(mn)] = ts

    # Champion per task (best test metric from selected)
    champs: dict[str, str] = {}
    for task, _label, _vc, _tc, maximize in TASK_SPECS_EXTERNAL:
        sel = ext.selected.get(task, pd.DataFrame())
        if sel.empty:
            continue
        test_vals = pd.to_numeric(sel["test"], errors="coerce")
        idx = test_vals.idxmax() if maximize else test_vals.idxmin()
        if idx is not None and not pd.isna(idx):
            champs[task] = str(sel.loc[idx, "model_name"])

    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        row: dict[str, object] = {"modelo": MODEL_LABELS.get(model_name, model_name)}

        for task, _label, _vc, test_col, maximize in TASK_SPECS_EXTERNAL:
            sel = ext.selected.get(task, pd.DataFrame())
            mrow = sel[sel["model_name"] == model_name]
            col = _metric_label(test_col)
            if mrow.empty:
                row[col] = None
            else:
                r = mrow.iloc[0]
                ndigits = 3
                std_val = _to_float(r["test_std"]) if ext.has_std else float("nan")
                cell = _pm(_to_float(r["test"]), std_val, ndigits)
                if champs.get(task) == model_name and cell is not None:
                    cell = f"{cell} *"
                row[col] = cell

        for task, param_col in (
            ("narma10", "params (NARMA-10)"),
            ("mackey_glass", "params (Mackey–Glass)"),
            ("delay_recall", "params (delay recall)"),
        ):
            selected = ext.selected.get(task, pd.DataFrame())
            selected_row = selected[selected["model_name"] == model_name]
            if selected_row.empty:
                params = float("nan")
            else:
                config_id = str(selected_row.iloc[0].get("config_id", ""))
                candidate = ext.candidates[
                    (ext.candidates["model_name"] == model_name)
                    & (ext.candidates["config_id"].astype(str) == config_id)
                ]
                if candidate.empty:
                    params = _to_float(
                        selected_row.iloc[0].get("n_trainable_params")
                    )
                else:
                    params = _trainable_params_for_task(
                        candidate.iloc[0],
                        task,
                        kmax=kmax,
                        esn_N=esn_N,
                    )
            row[param_col] = int(params) if np.isfinite(params) else None

        total_s = tuning_by_model.get(model_name, float("nan"))
        row["tuning total (min)"] = (
            round(total_s / 60.0, 1) if np.isfinite(total_s) else None
        )
        rows.append(row)

    return pd.DataFrame(rows)


def save_external_tables(
    ext: ExternalData,
    out_dir: str | Path,
    *,
    kmax: int = 100,
    esn_N: int = 100,
) -> dict[str, Path]:
    """Save external master table (+ convergence table if runs/ available) as CSV + LaTeX."""
    out = Path(out_dir)
    csv_dir = out / "csv"
    tex_dir = out / "latex"
    csv_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    table = external_master_table(ext, kmax=kmax, esn_N=esn_N)
    csv_path = csv_dir / "external_master.csv"
    tex_path = tex_dir / "external_master.tex"
    table.to_csv(csv_path, index=False)
    save_table_latex(
        table, tex_path, float_format="%.4g", index=False, escape=False, booktabs=True
    )
    paths: dict[str, Path] = {
        "external_master_csv": csv_path,
        "external_master_tex": tex_path,
    }

    if ext.has_std:
        try:
            conv_table = external_convergence_table(ext.results_dir, ext=ext)
            if not conv_table.empty:
                conv_csv = csv_dir / "external_convergence.csv"
                conv_tex = tex_dir / "external_convergence.tex"
                conv_table.to_csv(conv_csv, index=False)
                save_table_latex(
                    conv_table, conv_tex,
                    float_format="%.4g", index=False, escape=False, booktabs=True,
                )
                paths["external_convergence_csv"] = conv_csv
                paths["external_convergence_tex"] = conv_tex
        except Exception as exc:
            import warnings
            warnings.warn(f"external_convergence_table skipped: {exc}", UserWarning, stacklevel=2)

    return paths


def external_convergence_table(
    results_dir: str | Path,
    ext: ExternalData | None = None,
) -> pd.DataFrame:
    """Post-hoc convergence table for torch RNN/LSTM (§4).

    One row per (architecture, task) for the candidate selected by validation,
    aggregating over seeds: best_epoch ± std, epochs_ran ± std, % early-stopped,
    best val-loss ± std, and a flag for whether any seed hit the epoch cap.
    """
    import json

    d = Path(results_dir)
    torch_kinds = {"torch_simple_rnn", "torch_lstm"}
    arch_labels = {"torch_simple_rnn": "RNN simple", "torch_lstm": "LSTM"}

    val_cols: dict[str, str] = {
        "delay_recall": "delay_recall_memory_corr_total_mean",
        "narma10": "narma10_val_nmse_mean",
        "mackey_glass": "mg_val_nmse_mean",
    }
    maximize_tasks = {"delay_recall"}

    # Identify torch models present in results_dir
    torch_models: dict[str, str] = {}  # model_name → model_kind
    for child in sorted(d.iterdir()):
        if not child.is_dir():
            continue
        if ext is not None:
            cand_row = ext.candidates[ext.candidates["model_name"] == child.name]
            if not cand_row.empty:
                mk = str(cand_row.iloc[0]["model_kind"])
                if mk in torch_kinds:
                    torch_models[child.name] = mk
        else:
            # Infer from metadata in first available run JSON
            for task_dir in child.iterdir():
                runs_dir = task_dir / "runs"
                if runs_dir.exists():
                    jfiles = sorted(runs_dir.glob("*.json"))
                    if jfiles:
                        with open(jfiles[0], encoding="utf-8") as f:
                            run = json.load(f)
                        mk = str(run.get("model_kind", ""))
                        if mk in torch_kinds:
                            torch_models[child.name] = mk
                        break

    if not torch_models:
        return pd.DataFrame()

    # For each (model, task), find selected config_id by val metric
    candidates_df = ext.candidates if ext is not None else _load_candidates_df(d)

    rows: list[dict[str, object]] = []
    for model_name, model_kind in sorted(torch_models.items()):
        for task, _label, _vc, _tc, maximize in TASK_SPECS_EXTERNAL:
            val_col = val_cols.get(task)
            if val_col is None or candidates_df is None:
                continue
            m_rows = candidates_df[candidates_df["model_name"] == model_name]
            if m_rows.empty or val_col not in m_rows.columns:
                continue
            vals = pd.to_numeric(m_rows[val_col], errors="coerce")
            valid = vals.dropna()
            if valid.empty:
                continue
            best_idx = valid.idxmax() if task in maximize_tasks else valid.idxmin()
            selected_config_id = str(candidates_df.loc[best_idx, "config_id"])
            config_point = {}
            if "config_id" in m_rows.columns:
                cp_row = m_rows[m_rows["config_id"] == selected_config_id]
                if not cp_row.empty:
                    r = cp_row.iloc[0]
                    config_point = {
                        "hidden_size": r.get("hidden_size"),
                        "learning_rate": r.get("learning_rate"),
                    }

            runs_dir = d / model_name / task / "runs"
            if not runs_dir.exists():
                continue
            seed_runs = [
                p for p in sorted(runs_dir.glob("*.json"))
                if p.stem.startswith(selected_config_id + "_")
            ]
            if not seed_runs:
                seed_runs = sorted(runs_dir.glob(f"{selected_config_id}_*.json"))
            if not seed_runs:
                continue

            epochs_ran_list: list[int] = []
            best_epoch_list: list[int] = []
            early_stopped_list: list[bool] = []
            best_val_list: list[float] = []
            max_epochs_val: int = 800
            tope_alcanzado = False

            for p in seed_runs:
                with open(p, encoding="utf-8") as f:
                    run = json.load(f)
                meta = run.get("metadata", {})
                cp = run.get("config_point", {})
                max_ep = int(cp.get("max_epochs", 800))
                max_epochs_val = max_ep
                max_t = cp.get("max_train_seconds_per_run")
                er = int(meta.get("epochs_ran", 0))
                be = int(meta.get("best_epoch", 0))
                es = bool(meta.get("early_stopped", False))
                bv = float(meta.get("best_val_loss", float("nan")))
                st = str(meta.get("status", "ok"))
                epochs_ran_list.append(er)
                best_epoch_list.append(be)
                early_stopped_list.append(es)
                if np.isfinite(bv):
                    best_val_list.append(bv)
                if er >= max_ep or not es or st == "timeout":
                    tope_alcanzado = True

            if not epochs_ran_list:
                continue

            er_arr = np.array(epochs_ran_list, dtype=float)
            be_arr = np.array(best_epoch_list, dtype=float)
            bv_arr = np.array(best_val_list, dtype=float)
            pct_early = 100.0 * sum(early_stopped_list) / len(early_stopped_list)

            hs = config_point.get("hidden_size", "?")
            lr = config_point.get("learning_rate", "?")

            row: dict[str, object] = {
                "arquitectura": arch_labels.get(model_kind, model_kind),
                "tarea": TASK_LABELS.get(task, task),
                "candidato (hidden, lr)": f"({hs}, {lr})",
                "tope épocas": max_epochs_val,
                "mejor época (media)": round(float(np.mean(be_arr)), 1),
                "mejor época (std)": round(float(np.std(be_arr, ddof=1)), 1) if len(be_arr) > 1 else 0.0,
                "épocas corridas (media)": round(float(np.mean(er_arr)), 1),
                "épocas corridas (std)": round(float(np.std(er_arr, ddof=1)), 1) if len(er_arr) > 1 else 0.0,
                "% early-stopped": round(pct_early, 0),
                "mejor val-loss (media)": round(float(np.mean(bv_arr)), 4) if len(bv_arr) > 0 else None,
                "mejor val-loss (std)": round(float(np.std(bv_arr, ddof=1)), 4) if len(bv_arr) > 1 else 0.0,
                "tope alcanzado": tope_alcanzado,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def _load_candidates_df(results_dir: Path) -> pd.DataFrame | None:
    """Fallback: load comparison_summary.csv from results_dir."""
    csv_path = results_dir / "comparison_summary.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def _pm(mean: object, std: object, ndigits: int) -> str | None:
    """Format 'mean ± std' compactly; None if mean is non-finite."""
    m = _round(mean, ndigits)
    if m is None:
        return None
    s = _round(std, ndigits)
    return f"{m:.{ndigits}f}" if s is None else f"{m:.{ndigits}f} ± {s:.{ndigits}f}"


def _sweep_display_row(r: pd.Series, role: str) -> dict[str, object]:
    return {
        "rol": role,
        "config": r["config_id"],
        LAB_RHO: _round(r["spectral_radius"], 3),
        LAB_SIN: _round(r["input_scaling"], 3),
        LAB_ALPHA: _round(r["leak_rate"], 3),
        "NARMA test": _pm(r.get("narma10_test_nmse_mean"), r.get("narma10_test_nmse_std"), 3),
        "MG test": _pm(r.get("mg_test_nmse_mean"), r.get("mg_test_nmse_std"), 3),
        "MC total": _pm(r.get("mc_total_mean"), r.get("mc_total_std"), 2),
        "rank NARMA": int(r["rank_narma10"]),
        "rank MG": int(r["rank_mg"]),
        "rank MC total": int(r["rank_mc"]),
        "rank agg.": _round(r["aggregate_rank"], 2),
    }


def _design_display_row(r: pd.Series, role: str) -> dict[str, object]:
    narma_val = r.get("narma10_test_nmse_mean", r.get("narma10_val_nmse_mean"))
    mg_val    = r.get("mg_test_nmse_mean",    r.get("mg_val_nmse_mean"))
    return {
        "rol": role,
        "familia": FAMILY_LABELS.get(str(r["design_name"]), str(r["design_name"])),
        "config": r["config_id"],
        LAB_RHO: _round(r["spectral_radius"], 3),
        LAB_SIN: _round(r["input_scaling"], 3),
        LAB_ALPHA: _round(r["leak_rate"], 3),
        "NARMA test": _round(narma_val, 4),
        "MG test": _round(mg_val, 4),
        "MC total": _round(r["mc_total_mean"], 3),
        "rank NARMA": int(r["global_rank_narma10"]),
        "rank MG": int(r["global_rank_mg"]),
        "rank MC total": int(r["global_rank_mc"]),
        "rank agg.": _round(r["global_aggregate_rank"], 2),
    }


def _round(value: object, ndigits: int) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return round(x, ndigits)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_sweep_figures(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """Build the regenerated sweep figure batch."""
    apply_style()
    out = Path(out_dir)
    paths: list[Path] = []
    paths += plot_sweep_error_vs_rho(df, out, leak=1.0, formats=formats)
    paths += plot_sweep_task_heatmaps(df, out, formats=formats)
    paths += plot_sweep_aggregate_heatmaps(df, out, formats=formats)
    paths += plot_sweep_tradeoff(df, out, y_metric="mg_test_nmse_mean", name="F_sweep_tradeoff_mc_mg", formats=formats)
    paths += plot_sweep_tradeoff(df, out, y_metric="narma10_test_nmse_mean", name="F_sweep_tradeoff_mc_narma", formats=formats)
    return paths


def build_design_figures(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """Build reservoir-design figures (D1-D5)."""
    apply_style()
    out = Path(out_dir)
    paths: list[Path] = []
    paths += plot_design_family_bars(df, out, formats=formats)       # D1
    paths += plot_design_niche_tradeoff(df, out, formats=formats)    # D2
    paths += plot_design_mechanism_2x2(df, out, formats=formats)     # D3
    paths += plot_design_region_by_task(df, out, formats=formats)    # D4
    paths += plot_design_gain_bars(df, out, formats=formats)         # D5
    return paths


# ---------------------------------------------------------------------------
# Sweep figures (regenerated)
# ---------------------------------------------------------------------------

def plot_sweep_error_vs_rho(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    leak: float = 1.0,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """S1 — error/metric vs rho with one line per s_in, at fixed leak.

    Shows the opposite-corner structure directly: NARMA/MC are best at rho~1 and
    low drive, Mackey-Glass at high drive and rho>1. Coverage is block-dependent
    (low s_in stops near rho=1.05; high s_in starts at rho=1.0) — that gap is the
    regime structure, not missing data.
    """
    import matplotlib.pyplot as plt

    d = df[np.isclose(df["leak_rate"].astype(float), leak)]
    sins = sorted(float(v) for v in d["input_scaling"].unique())
    panels = [
        ("narma10_test_nmse_mean", _metric_panel_title("narma10_test_nmse_mean"), True),
        ("mg_test_nmse_mean", _metric_panel_title("mg_test_nmse_mean"), True),
        ("mc_total_mean", _metric_panel_title("mc_total_mean"), False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (col, title, logy) in zip(axes, panels, strict=True):
        for color, s in zip(SIN_PALETTE, sins):
            sub = d[np.isclose(d["input_scaling"].astype(float), s)].sort_values("spectral_radius")
            if sub.empty:
                continue
            ax.plot(sub["spectral_radius"], sub[col], "-o", ms=4, lw=1.6, color=color, label=f"{s:g}")
        ax.axvline(1.0, ls=":", c="#999", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel(LAB_RHO)
        ax.set_ylabel(_metric_label(col))
        ax.grid(True, alpha=0.25)
        if logy:
            ax.set_yscale("log")
    axes[0].legend(title=LAB_SIN, fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    return save_figure(fig, out_dir, "F_sweep_error_vs_rho", formats=formats)


def plot_sweep_task_heatmaps(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """S2 — rho x s_in plane per task, each at a representative leak.

    Warm colormap for errors (light = low error = good); cool reversed for MC
    (light = high MC = good). Cell annotations use auto-contrast text; rho=1 marked.
    """
    import matplotlib.pyplot as plt

    # All panels on the SAME (rho, s_in) plane at alpha=1 for comparability.
    leak = 1.0
    specs = [
        ("narma10_test_nmse_mean", leak, _metric_panel_title("narma10_test_nmse_mean"), CMAPS["sequential_warm"]),
        ("mg_test_nmse_mean", leak, _metric_panel_title("mg_test_nmse_mean"), CMAPS["sequential_warm"]),
        ("mc_total_mean", leak, _metric_panel_title("mc_total_mean"), CMAPS["sequential_cool"] + "_r"),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(15.5, 4.1), squeeze=False)
    for ax, (col, leak, title, cmap) in zip(axes[0], specs, strict=True):
        grid, xs, ys, mask = pivot_plane(df, col, "spectral_radius", "input_scaling", {"leak_rate": leak})
        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))
        pcm = heatmap_plane(ax, grid, xs, ys, cmap, vmin=vmin, vmax=vmax, mask=mask)
        categorical_vline(ax, list(xs), 1.0, ls=":", color="#333", lw=1.0)
        ax.set_title(rf"{title}, $\alpha={leak:g}$")
        ax.set_xlabel(LAB_RHO)
        ax.set_ylabel(LAB_SIN)
        fmt = "{:.0f}" if col == "mc_total_mean" else "{:.2f}"
        _annotate_heatmap(ax, heatmap_display_grid(grid), vmin=vmin, vmax=vmax, cmap_name=cmap, fmt=fmt)
        add_colorbar(fig, ax, pcm, _metric_label(col))
    fig.subplots_adjust(wspace=0.55)
    return save_figure(fig, out_dir, "F_sweep_task_heatmaps", formats=formats)


def plot_sweep_aggregate_heatmaps(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """Meseta de compromiso — rank agregado en (rho, s_in) por leak.

    Light colormap (low rank = light = good); the contiguous light band is the
    robustness plateau. Auto-contrast annotations.
    """
    import matplotlib.pyplot as plt

    leaks = sorted(float(v) for v in df["leak_rate"].unique())
    cmap = CMAPS["sequential_cool"]  # low (good) -> light
    grids = []
    for leak in leaks:
        grid, xs, ys, mask = pivot_plane(df, "aggregate_rank", "spectral_radius", "input_scaling", {"leak_rate": leak})
        grids.append((grid, xs, ys, mask, leak))
    vmin = float(np.nanmin([np.nanmin(g[0]) for g in grids]))
    vmax = float(np.nanmax([np.nanmax(g[0]) for g in grids]))

    n = len(leaks)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.6), squeeze=False)
    last_pcm = None
    for ax, (grid, xs, ys, mask, leak) in zip(axes[0], grids, strict=True):
        last_pcm = heatmap_plane(ax, grid, xs, ys, cmap, vmin=vmin, vmax=vmax, mask=mask)
        categorical_vline(ax, list(xs), 1.0, ls=":", color="#333", lw=1.0)
        ax.set_title(rf"$\alpha={leak:g}$")
        ax.set_xlabel(LAB_RHO)
        ax.set_ylabel(LAB_SIN)
        _annotate_heatmap(ax, heatmap_display_grid(grid), vmin=vmin, vmax=vmax, cmap_name=cmap, fmt="{:.0f}")
    if last_pcm is not None:
        add_colorbar(fig, axes[0, -1], last_pcm, _metric_label("aggregate_rank"))
    return save_figure(fig, out_dir, "F_sweep_aggregate_heatmaps", formats=formats)


def plot_sweep_tradeoff(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    y_metric: str,
    name: str,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """S3 — memory <-> nonlinearity trade-off: MC (x) vs error (y).

    Single panel, one marker, single sequential gradient by s_in (no marker zoo),
    with the Pareto frontier (max MC for each error level) drawn and the three
    representative configs labelled. The empty bottom-right corner is the no-free-
    lunch result.
    """
    import matplotlib.pyplot as plt

    if y_metric.startswith("mg"):
        ylabel, title = _metric_label(y_metric), _metric_panel_title(y_metric)
    elif y_metric.startswith("narma"):
        ylabel, title = _metric_label(y_metric), _metric_panel_title(y_metric)
    else:
        ylabel, title = _metric_label(y_metric), _metric_panel_title(y_metric)

    sub = df.dropna(subset=["mc_total_mean", y_metric]).copy()
    x = sub["mc_total_mean"].to_numpy(float)
    y = sub[y_metric].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    sc = ax.scatter(x, y, c=sub["input_scaling"].astype(float), cmap="cividis",
                    s=34, alpha=0.85, edgecolors="white", linewidths=0.3)

    # Pareto frontier: for each point, non-dominated = no other has both higher MC and lower error.
    order = np.argsort(-x)  # high MC first
    best_err = np.inf
    front = []
    for i in order:
        if y[i] < best_err:
            front.append(i)
            best_err = y[i]
    front = sorted(front, key=lambda i: x[i])
    ax.plot(x[front], y[front], "-", color="#444", lw=1.2, alpha=0.7, zorder=1, label="frontera de Pareto")

    aggregate_idx = sub["aggregate_rank"].idxmin() if "aggregate_rank" in sub else None
    if aggregate_idx is not None:
        r = sub.loc[aggregate_idx]
        ax.annotate("mejor agregado", xy=(r["mc_total_mean"], r[y_metric]),
                    xytext=(8, -14), textcoords="offset points", fontsize=8, ha="left")

    ax.margins(x=0.12)

    ax.set_yscale("log")
    ax.set_xlabel(_metric_label("mc_total_mean"))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    leg = ax.legend(frameon=True, loc="upper left", fontsize=8)
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("black")
    frame.set_alpha(0.95)
    frame.set_linewidth(0.8)
    ax.grid(True, alpha=0.2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(LAB_SIN)
    return save_figure(fig, out_dir, name, formats=formats)


# ---------------------------------------------------------------------------
# Design figures
# ---------------------------------------------------------------------------

def plot_design_family_bars(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """D1 — Horizontal bars: best config per family, one panel per task."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullLocator

    narma_col = _best_col(df, "narma10_test_nmse_mean", "narma10_val_nmse_mean")
    mg_col = _best_col(df, "mg_test_nmse_mean", "mg_val_nmse_mean")
    families = _ordered_families(df)

    narma_best = {f: float(df[df["design_name"] == f][narma_col].min()) for f in families}
    mg_best    = {f: float(df[df["design_name"] == f][mg_col].min())    for f in families}
    mc_best    = {f: float(df[df["design_name"] == f]["mc_total_mean"].max()) for f in families}

    narma_champ = min((f for f in narma_best if np.isfinite(narma_best[f])), key=narma_best.__getitem__, default=None)
    mg_champ    = min((f for f in mg_best    if np.isfinite(mg_best[f])),    key=mg_best.__getitem__,    default=None)
    mc_champ    = max((f for f in mc_best    if np.isfinite(mc_best[f])),    key=mc_best.__getitem__,    default=None)

    panels = [
        (narma_best, narma_champ, _metric_panel_title(narma_col), _metric_label(narma_col), True),
        (mg_best,    mg_champ,    _metric_panel_title(mg_col),    _metric_label(mg_col),    True),
        (mc_best,    mc_champ,    _metric_panel_title("mc_total_mean"), _metric_label("mc_total_mean"), False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (data_dict, champ, title, xlabel, is_nmse) in zip(axes, panels):
        valid_fams = [f for f in families if np.isfinite(data_dict.get(f, np.nan))]
        if is_nmse:
            sorted_fams = sorted(valid_fams, key=lambda f: data_dict[f])
        else:
            sorted_fams = sorted(valid_fams, key=lambda f: data_dict[f], reverse=True)
        disp   = list(reversed(sorted_fams))
        values = [data_dict[f] for f in disp]
        colors = [ACCENT_RED if f == champ else ACCENT_BLUE for f in disp]
        labels = [FAMILY_LABELS.get(f, f) for f in disp]

        y_pos = list(range(len(disp)))
        ax.barh(y_pos, values, color=colors, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", alpha=0.2)
        if is_nmse:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_locator(NullLocator())

    fig.tight_layout()
    return save_figure(fig, out_dir, "F_design_family_bars", formats=formats)


def plot_design_niche_tradeoff(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """D2 — Per-family capability envelope: memory ceiling vs nonlinear ceiling.

    x = max(mc_total_mean) over all configs of that family.
    y = min(mg_test_nmse_mean) over stable configs (NMSE <= 1) of that family.

    The two coordinates come from *different* configs (best-MC vs best-MG), so
    families that perform well in both regimes appear high on both axes, softening
    the true trade-off. This shows achievable ceilings per dimension, not a single
    operating point.
    """
    import matplotlib.pyplot as plt

    mg_col     = _best_col(df, "mg_test_nmse_mean", "mg_val_nmse_mean")
    mg_stable  = df[df[mg_col].astype(float) <= 1.0]

    pts: list[tuple[float, float, str]] = []
    for family in _ordered_families(df):
        sub_mc = df[df["design_name"] == family].dropna(subset=["mc_total_mean"])
        sub_mg = mg_stable[mg_stable["design_name"] == family].dropna(subset=[mg_col])
        if sub_mc.empty or sub_mg.empty:
            continue
        pts.append((float(sub_mc["mc_total_mean"].max()),
                    float(sub_mg[mg_col].min()),
                    family))

    if not pts:
        return []

    accent_set = {max(pts, key=lambda p: p[0])[2], min(pts, key=lambda p: p[1])[2]}

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    label_pts: list[tuple[float, float, str]] = []
    for mc_val, mg_val, family in pts:
        color = ACCENT_RED if family in accent_set else ACCENT_BLUE
        ax.scatter(mc_val, mg_val, s=60, color=color, zorder=3,
                   edgecolors="white", linewidths=0.5)
        label_pts.append((mc_val, mg_val, FAMILY_LABELS.get(family, family)))

    _label_points(ax, label_pts)
    ax.set_yscale("log")
    ax.set_xlabel(f"{_metric_label('mc_total_mean')} (max. por familia)")
    ax.set_ylabel(f"{_metric_label(mg_col)} (min. por familia)")
    ax.set_title("Familias")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return save_figure(fig, out_dir, "F_design_niche_tradeoff", formats=formats)


def plot_design_mechanism_2x2(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """D3 - Mechanism map: W structure and drive against MC/MG metrics."""
    import matplotlib.pyplot as plt

    _require_columns(
        df,
        ["diag_transient_growth_max_mean", "input_scaling", "mc_total_mean"],
        "design diagnostics",
    )
    narma_col = _best_col(df, "narma10_test_nmse_mean", "narma10_val_nmse_mean")
    mg_col = _best_col(df, "mg_test_nmse_mean", "mg_val_nmse_mean")

    base = _stable_nmse_rows(df, [narma_col, mg_col]).copy()

    growth = pd.to_numeric(base["diag_transient_growth_max_mean"], errors="coerce")
    base["_log_growth"] = np.log10(np.clip(growth.to_numpy(float), 1e-12, None))
    base["_input_scaling_num"] = pd.to_numeric(base["input_scaling"], errors="coerce")

    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return float("nan")
        rho = pd.Series(x[mask]).corr(pd.Series(y[mask]), method="spearman")
        return float(rho) if rho is not None and np.isfinite(rho) else float("nan")

    def _draw_panel(
        ax,
        x_col: str,
        y_col: str,
        *,
        xlabel: str,
        ylabel: str,
        logy: bool = False,
    ) -> None:
        plot_df = base.dropna(subset=[x_col, y_col]).copy()
        x = pd.to_numeric(plot_df[x_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(plot_df[y_col], errors="coerce").to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if logy:
            mask &= y > 0
        x = x[mask]
        y = y[mask]
        rho_s = _spearman(x, y)
        strong = np.isfinite(rho_s) and abs(rho_s) >= 0.60
        accent = ACCENT_RED if strong else "#777777"

        ax.scatter(
            x, y,
            s=32, color=CLOUD_GRAY, alpha=0.78,
            edgecolors="white", linewidths=0.35, zorder=3,
        )
        if len(x) >= 2 and np.nanmax(x) > np.nanmin(x):
            fit_y = np.log10(y) if logy else y
            fit_mask = np.isfinite(fit_y)
            if fit_mask.sum() >= 2:
                coef = np.polyfit(x[fit_mask], fit_y[fit_mask], 1)
                xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
                yy = np.polyval(coef, xx)
                if logy:
                    yy = np.power(10.0, yy)
                ax.plot(xx, yy, color=accent, lw=1.4, alpha=0.9, zorder=4)

        rho_txt = "n/a" if not np.isfinite(rho_s) else f"{rho_s:.2f}"
        ax.text(
            0.04, 0.94, rf"$\rho_s={rho_txt}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=accent,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                  "edgecolor": accent, "linewidth": 0.8, "alpha": 0.92},
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(accent)
            spine.set_linewidth(1.2 if strong else 0.8)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.18)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), squeeze=False)
    x_growth_label = r"$\log_{10}\max_k\|W^k\|$"
    panels = [
        (axes[0, 0], "_log_growth", "mc_total_mean", x_growth_label, _metric_label("mc_total_mean"), False),
        (axes[0, 1], "_input_scaling_num", "mc_total_mean", LAB_SIN, _metric_label("mc_total_mean"), False),
        (axes[1, 0], "_log_growth", mg_col, x_growth_label, _metric_label(mg_col), True),
        (axes[1, 1], "_input_scaling_num", mg_col, LAB_SIN, _metric_label(mg_col), True),
    ]
    for ax, x_col, y_col, xlabel, ylabel, logy in panels:
        _draw_panel(ax, x_col, y_col, xlabel=xlabel, ylabel=ylabel, logy=logy)

    axes[0, 0].set_title("estructura de W")
    axes[0, 1].set_title("drive")
    fig.tight_layout()
    return save_figure(fig, out_dir, "F_design_mechanism_2x2", formats=formats)


def plot_design_property_curves(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """D3a — Mean metric per bin of log10(transient growth) with ±1 std band.

    Stable filter: rows with NMSE > 1 excluded for NARMA and Mackey-Glass before
    binning. Empty bins are skipped; the gap between cycles/random (x≈0) and
    nonnormal chains (x≈17–31) is real and informative — the axis is not broken.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    _require_columns(df, ["diag_transient_growth_max_mean"], "design diagnostics")
    narma_col = _best_col(df, "narma10_test_nmse_mean", "narma10_val_nmse_mean")
    mg_col    = _best_col(df, "mg_test_nmse_mean",    "mg_val_nmse_mean")
    x_col     = "diag_transient_growth_max_mean"

    panels = [
        ("mc_total_mean", _metric_panel_title("mc_total_mean"), False, None, _metric_label("mc_total_mean")),
        (narma_col,       _metric_panel_title(narma_col),       True,  1.0, _metric_label(narma_col)),
        (mg_col,          _metric_panel_title(mg_col),          True,  1.0, _metric_label(mg_col)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (y_col, title, logy, thresh, ylabel) in zip(axes, panels):
        base = df.dropna(subset=[x_col, y_col]).copy()
        if thresh is not None:
            base = base[base[y_col].astype(float) <= thresh]
        if len(base) < 5:
            ax.set_title(title)
            ax.set_xlabel(r"$\log_{10}\max_k\|W^k\|$")
            ax.set_ylabel(ylabel)
            continue

        x_v = np.log10(np.clip(base[x_col].astype(float).to_numpy(), 0.1, None))
        y_v = base[y_col].astype(float).to_numpy()

        rho_s, _ = spearmanr(x_v, y_v)

        n_bins = 8
        edges = np.linspace(x_v.min(), x_v.max(), n_bins + 1)
        xs_plot, means_plot, stds_plot = [], [], []
        for i in range(n_bins):
            mask = (x_v >= edges[i]) & (x_v <= edges[i + 1])
            y_bin = y_v[mask]
            if len(y_bin) >= 2:
                xs_plot.append(float((edges[i] + edges[i + 1]) / 2))
                means_plot.append(float(np.mean(y_bin)))
                stds_plot.append(float(np.std(y_bin, ddof=1)))

        if len(xs_plot) >= 2:
            xa = np.array(xs_plot)
            ma = np.array(means_plot)
            sa = np.array(stds_plot)
            lo_band = np.maximum(ma - sa, 1e-10) if logy else (ma - sa)
            ax.plot(xa, ma, "-o", color=ACCENT_BLUE, lw=1.8, ms=5, zorder=3)
            ax.fill_between(xa, lo_band, ma + sa, alpha=0.20, color=ACCENT_BLUE, zorder=2)

        ax.text(0.97, 0.95, rf"$\rho_s = {rho_s:.2f}$", transform=ax.transAxes,
                fontsize=7, ha="right", va="top")
        if logy:
            ax.set_yscale("log")
            y_fin = y_v[np.isfinite(y_v)]
            if len(y_fin) > 0:
                ax.set_ylim(float(y_fin.min()) * 0.5, 1.0)
        ax.set_title(title)
        ax.set_xlabel(r"$\log_{10}\max_k\|W^k\|$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.15)

    fig.tight_layout()
    return save_figure(fig, out_dir, "F_design_property_curves", formats=formats)


def plot_design_property_strip(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """D3b — Families ordered by median transient amplification, coloured by group.

    Three topological groups map to three specialist roles:
    cycles → memory (MC total), random/multiscale → balance (NARMA),
    nonnormal chains → nonlinear (MG).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _require_columns(df, ["diag_transient_growth_max_mean"], "design diagnostics")
    x_col = "diag_transient_growth_max_mean"

    _group_color = {
        "cycles":       "#4C72B0",
        "random_multi": "#DD8452",
        "nonnormal":    "#C44E52",
    }
    _family_group = {
        "cycle_scr":               "cycles",
        "cycle_jump_j3":           "cycles",
        "cycle_jump_j7":           "cycles",
        "random_sparse_baseline":  "random_multi",
        "multiscale_two_random":   "random_multi",
        "multiscale_three_random": "random_multi",
        "nonnormal_chain_g0_1":    "nonnormal",
        "nonnormal_chain_g0_3":    "nonnormal",
        "nonnormal_chain_g0_6":    "nonnormal",
    }
    _y_cycle = [-0.06, 0.03, 0.09, -0.03, 0.06, -0.09, 0.01, -0.05, 0.07]

    family_x: list[tuple[float, str]] = []
    for family in _ordered_families(df):
        sub = df[df["design_name"] == family].dropna(subset=[x_col])
        if sub.empty:
            continue
        med = float(sub[x_col].astype(float).median())
        family_x.append((float(np.log10(max(med, 0.1))), family))
    family_x.sort(key=lambda t: t[0])

    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    label_pts: list[tuple[float, float, str]] = []
    for i, (x_val, family) in enumerate(family_x):
        y_val = _y_cycle[i % len(_y_cycle)]
        group = _family_group.get(family, "random_multi")
        ax.scatter(x_val, y_val, s=80, color=_group_color[group], zorder=3,
                   edgecolors="white", linewidths=0.5)
        label_pts.append((x_val, y_val, FAMILY_LABELS.get(family, family)))

    _label_points(ax, label_pts, fontsize=8)

    # Region annotations at fixed x positions to avoid overlap between the
    # dense low-x cluster (cycles, random/multiscale near log T≈0) and chains.
    for group_key, region_label, x_pos, ha in [
        ("cycles",       "memoria (MC total)", 0.0,  "left"),
        ("random_multi", "balance (NARMA)", 3.0,  "left"),
        ("nonnormal",    "no lineal (MG)",  18.0, "center"),
    ]:
        ax.text(x_pos, -0.27, region_label,
                ha=ha, va="top", fontsize=8,
                color=_group_color[group_key], style="italic")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_group_color["cycles"],
               markersize=8, label="cycles (MC total)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_group_color["random_multi"],
               markersize=8, label="random / multiescala"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_group_color["nonnormal"],
               markersize=8, label="cadena nonnormal (MG)"),
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel(r"$\log_{10}\max_k\|W^k\|$ (mediana por familia)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.38, 0.30)
    ax.set_title("Topologías")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    return save_figure(fig, out_dir, "F_design_property_strip", formats=formats)


def plot_design_region_by_task(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """D4 — Heatmap of (rho, s_in) performance per task, averaged over topologies.

    Cell value = mean metric over all rows sharing that (spectral_radius, input_scaling),
    marginalising over families and leak rates. Stable filter (NMSE <= 1) applied
    before aggregation for NARMA and Mackey-Glass.

    In the design grid the compromise and nonlinear boxes are nearly disjoint in
    (rho, s_in); cells with no data are masked. Complements sweep heatmap S2 by
    showing the region holds across topologies.
    """
    import matplotlib.pyplot as plt

    narma_col   = _best_col(df, "narma10_test_nmse_mean", "narma10_val_nmse_mean")
    mg_col      = _best_col(df, "mg_test_nmse_mean",    "mg_val_nmse_mean")
    cmap_warm   = CMAPS["sequential_warm"]
    cmap_warm_r = CMAPS["sequential_warm"] + "_r"

    panels = [
        ("mc_total_mean", _metric_panel_title("mc_total_mean"), cmap_warm_r, None, "{:.0f}"),
        (narma_col,       _metric_panel_title(narma_col),       cmap_warm,   1.0, "{:.3f}"),
        (mg_col,          _metric_panel_title(mg_col),          cmap_warm,   1.0, "{:.3f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), squeeze=False)
    for ax, (col, title, cmap, thresh, fmt) in zip(axes[0], panels):
        sub = df.copy()
        if thresh is not None:
            sub = sub[sub[col].astype(float) <= thresh]
        agg = sub.groupby(["spectral_radius", "input_scaling"])[col].mean().reset_index()
        if agg.empty:
            ax.set_title(title)
            continue
        grid, xs, ys, mask = pivot_plane(agg, col, "spectral_radius", "input_scaling", {})
        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))
        pcm  = heatmap_plane(ax, grid, xs, ys, cmap, vmin=vmin, vmax=vmax, mask=mask)
        categorical_vline(ax, xs, 1.0, ls=":", color="#333", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel(LAB_RHO)
        ax.set_ylabel(LAB_SIN)
        _annotate_heatmap(ax, heatmap_display_grid(grid), vmin=vmin, vmax=vmax,
                          cmap_name=cmap, fmt=fmt)
        add_colorbar(fig, ax, pcm, _metric_label(col))

    fig.subplots_adjust(wspace=0.55)
    return save_figure(fig, out_dir, "F_design_region_by_task", formats=formats)


def plot_design_gain_bars(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """D5 — Relative improvement of the best specialist over random_sparse_baseline.

    Gain for MC = (specialist - baseline) / baseline.
    Gain for NMSE = (baseline - specialist) / baseline  (reduction).
    Stable configs only (NMSE <= 1) for NARMA and Mackey-Glass selection.
    """
    import matplotlib.pyplot as plt

    narma_col = _best_col(df, "narma10_test_nmse_mean", "narma10_val_nmse_mean")
    mg_col    = _best_col(df, "mg_test_nmse_mean",    "mg_val_nmse_mean")

    bl = df[df["design_name"] == "random_sparse_baseline"]
    if bl.empty:
        return []

    bl_mc    = float(bl["mc_total_mean"].max())
    bl_narma = float(bl[bl[narma_col].astype(float) <= 1.0][narma_col].min()
                     if not bl[bl[narma_col].astype(float) <= 1.0].empty
                     else bl[narma_col].min())
    bl_mg    = float(bl[bl[mg_col].astype(float) <= 1.0][mg_col].min()
                     if not bl[bl[mg_col].astype(float) <= 1.0].empty
                     else bl[mg_col].min())

    narma_s = df[df[narma_col].astype(float) <= 1.0]
    mg_s    = df[df[mg_col].astype(float) <= 1.0]

    mc_idx    = df["mc_total_mean"].idxmax()
    narma_idx = narma_s[narma_col].idxmin() if not narma_s.empty else df[narma_col].idxmin()
    mg_idx    = mg_s[mg_col].idxmin()       if not mg_s.empty    else df[mg_col].idxmin()

    global_mc    = float(df.loc[mc_idx,    "mc_total_mean"])
    global_narma = float(df.loc[narma_idx, narma_col])
    global_mg    = float(df.loc[mg_idx,    mg_col])

    mc_fam    = FAMILY_LABELS.get(str(df.loc[mc_idx,    "design_name"]), "?")
    narma_fam = FAMILY_LABELS.get(str(df.loc[narma_idx, "design_name"]), "?")
    mg_fam    = FAMILY_LABELS.get(str(df.loc[mg_idx,    "design_name"]), "?")

    gain_mc    = 100.0 * (global_mc - bl_mc)       / abs(bl_mc)    if bl_mc    != 0 else 0.0
    gain_narma = 100.0 * (bl_narma  - global_narma) / abs(bl_narma) if bl_narma != 0 else 0.0
    gain_mg    = 100.0 * (bl_mg     - global_mg)    / abs(bl_mg)    if bl_mg    != 0 else 0.0

    tasks   = [_metric_label("mc_total_mean"), "NARMA-10", "Mackey–Glass"]
    gains   = [gain_mc, gain_narma, gain_mg]
    fam_lbs = [mc_fam, narma_fam, mg_fam]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars  = ax.bar(range(3), gains, color=ACCENT_BLUE, width=0.5)
    y_top = max((g for g in gains if np.isfinite(g)), default=10.0)
    for bar, gain, lbl in zip(bars, gains, fam_lbs):
        if np.isfinite(gain):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_top * 0.02,
                    f"{lbl}\n+{gain:.0f}%",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(3))
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Mejora relativa (%)")
    ax.set_title("Ganancia relativa")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    return save_figure(fig, out_dir, "F_design_gain_bars", formats=formats)


# ---------------------------------------------------------------------------
# End-to-end builder
# ---------------------------------------------------------------------------

def build_campaign_artifacts(
    sweep_dir: str | Path,
    design_dir: str | Path,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    external_dir: str | Path | None = None,
    *,
    kmax: int = 100,
    esn_N: int = 100,
    run_diagnostic: bool = False,
    diagnostic_seeds: tuple[int, ...] = (42, 123, 456),
    diagnostic_device: str = "cpu",
) -> dict[str, list[Path] | dict[str, Path]]:
    """Build all standard tables and figures for sweep + design campaign.

    external_dir: if provided, load external comparison results, generate the
        master table (with corrected params, §2), E1 bars, E2a/E2b cost–tradeoff
        clouds, and the convergence table (§4) when runs/ are available.
    run_diagnostic: if True (requires external_dir), also re-runs the selected
        torch candidates with record_history=True (§5) to produce
        F_external_training_curves.  Seeds and device are configurable.
    """
    sweep_df = load_sweep_summary(sweep_dir)
    design_df = load_design_summary(design_dir)
    out = Path(out_dir)
    tables = save_campaign_tables(sweep_df, design_df, out / "tables")
    sweep_figs = build_sweep_figures(sweep_df, out / "figures" / "sweep", formats=formats)
    design_figs = build_design_figures(design_df, out / "figures" / "design", formats=formats)
    result: dict[str, list[Path] | dict[str, Path]] = {
        "tables": tables,
        "sweep_figures": sweep_figs,
        "design_figures": design_figs,
    }
    if external_dir is not None:
        ext = load_external_summary(external_dir)
        ext_tables = save_external_tables(ext, out / "tables", kmax=kmax, esn_N=esn_N)
        curves_dir: Path | None = None
        if run_diagnostic:
            curves_dir = run_torch_diagnostic(
                external_dir,
                seeds=diagnostic_seeds,
                device=diagnostic_device,
            )
        ext_figs = build_external_figures(
            ext, out / "figures" / "external",
            formats=formats, kmax=kmax, esn_N=esn_N, curves_dir=curves_dir,
        )
        result["external_tables"] = ext_tables
        result["external_figures"] = ext_figs
    return result


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ordered_families(df: pd.DataFrame) -> list[str]:
    present = [f for f in FAMILY_ORDER if f in set(df["design_name"].astype(str))]
    extras = sorted(set(df["design_name"].astype(str)) - set(present))
    return present + extras


def _best_col(df: pd.DataFrame, test_col: str, val_col: str) -> str:
    """Return test column if present in df, else val column."""
    return test_col if test_col in df.columns else val_col


def _metric_split_from_col(col: str) -> str | None:
    """Infer validation/test split from a persisted metric column name."""
    key = col.lower()
    if key in {"mc_total_mean", "aggregate_rank", "global_aggregate_rank"}:
        return None
    if "_test_" in key or key.endswith("_test"):
        return "test"
    if "_val_" in key or key.endswith("_val"):
        return "val"
    if key.startswith("delay_recall_"):
        return "val"
    return None


def _metric_label(col: str) -> str:
    """Readable metric label with split suffix where the metric has one."""
    key = col.lower()
    split = _metric_split_from_col(col)
    suffix = f" ({split})" if split is not None else ""

    if key == "mc_total_mean":
        return "MC total"
    if key in {"aggregate_rank", "global_aggregate_rank"}:
        return "Rank agregado"
    if key.startswith("narma10") and "nmse" in key:
        return f"NARMA-10 NMSE{suffix}"
    if key.startswith("mg") and "nmse" in key:
        return f"Mackey–Glass NMSE{suffix}"
    if key.startswith("delay_recall"):
        if "memory_corr_total" in key:
            base = "Delay Recall memory corr. total"
        elif "memory_eff_total" in key:
            base = "Delay Recall memory eff. total"
        else:
            base = "Delay Recall"
        return f"{base}{suffix}"
    return col


def _metric_panel_title(col: str) -> str:
    """Short task-oriented title for plot panels."""
    key = col.lower()
    if key == "mc_total_mean":
        return "MC total"
    if key.startswith("narma10"):
        return "NARMA-10"
    if key.startswith("mg"):
        return "Mackey–Glass"
    if key.startswith("delay_recall"):
        return "Delay Recall"
    if key in {"aggregate_rank", "global_aggregate_rank"}:
        return "Rank agregado"
    return col


def _stable_nmse_rows(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    """Keep rows whose available NMSE metrics are stable (<= 1)."""
    out = df.copy()
    for col in metric_cols:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        out = out[vals <= 1.0]
    return out


def _label_points(
    ax,
    points: list[tuple[float, float, str]],
    fontsize: int = 8,
) -> None:
    """Place text labels near scatter points without arrows.

    Sorts by y to reduce vertical overlap; alternates horizontal offset ±8 pt.
    """
    pts = sorted(points, key=lambda p: p[1])
    for idx, (x, y, label) in enumerate(pts):
        dx = 8 if idx % 2 == 0 else -8
        dy = 5 if idx % 2 == 0 else -7
        ha = "left" if dx > 0 else "right"
        ax.annotate(label, xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=fontsize,
                    ha=ha, va="bottom")


def design_mechanism_table(df: pd.DataFrame) -> pd.DataFrame:
    """Champion row per task (MC total, MG, NARMA-10, aggregate) with dynamic signature."""
    mc_col    = "mc_total_mean"
    narma_col = _best_col(df, "narma10_test_nmse_mean", "narma10_val_nmse_mean")
    mg_col    = _best_col(df, "mg_test_nmse_mean", "mg_val_nmse_mean")

    selectors = [
        ("MC total",     df[mc_col].idxmax(),                  mc_col,    3),
        ("Mackey–Glass", df[mg_col].idxmin(),                  mg_col,    4),
        ("NARMA-10",     df[narma_col].idxmin(),               narma_col, 4),
        ("Agregado",     df["global_aggregate_rank"].idxmin(), "global_aggregate_rank", 2),
    ]
    diag_map = {
        "diag_henrici_departure_mean":        "Henrici",
        "diag_singular_condition_number_mean": "κ",
        "diag_transient_growth_max_mean":      r"max$\|W^k\|$",
    }
    rows: list[dict[str, object]] = []
    for task_label, idx, metric_col, ndigits in selectors:
        r = df.loc[idx]
        row: dict[str, object] = {
            "tarea":   task_label,
            "familia": FAMILY_LABELS.get(str(r["design_name"]), str(r["design_name"])),
            LAB_RHO:   _round(r["spectral_radius"], 3),
            LAB_SIN:   _round(r["input_scaling"],   3),
            LAB_ALPHA: _round(r["leak_rate"],        3),
            "métrica": _round(r[metric_col],         ndigits),
        }
        for col, header in diag_map.items():
            if col in df.columns:
                val = float(r[col])
                if col == "diag_transient_growth_max_mean" and np.isfinite(val) and val > 1e4:
                    row[header] = f"{val:.1e}"
                else:
                    row[header] = _round(val, 4)
            else:
                row[header] = None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# External comparison — figures
# ---------------------------------------------------------------------------

def run_torch_diagnostic(
    results_dir: str | Path,
    seeds: tuple[int, ...] = (42, 123, 456),
    *,
    tasks_cfg: dict[str, dict] | None = None,
    device: str = "cpu",
) -> Path:
    """Re-run selected torch candidates with record_history=True (§5).

    For each (arch × task), identifies the candidate selected by validation
    (same criterion as the main campaign), retrains with the same loop
    (windowed BPTT, patience=50, max_epochs=800, Adam, grad_clip, normalisation)
    and record_history=True, then saves per-seed curves to:
        results_dir/diagnostics/curves/{arch}_{task}_seed{n}.json

    tasks_cfg: optional dict mapping task name → task config (n_train, n_val,
    n_test, washout; kmax for delay_recall).  If None, uses the standard
    external-campaign defaults (n_train=2400, n_val=600, n_test=1000,
    washout=1000, kmax=100).

    Returns the path to the curves directory.
    """
    import json

    from rc_lab.runners.runner import resolve_task
    from rc_lab.sequence_models.torch_models import fit_torch_sequence_model
    from rc_lab.utils.seeding import set_seed

    d = Path(results_dir)
    curves_dir = d / "diagnostics" / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    _default_task_cfg: dict[str, dict] = {
        "delay_recall": {
            "name": "delay_recall", "n_train": 2400, "n_val": 600,
            "n_test": 1000, "washout": 1000, "kmax": 100,
            "state_policy": "reset",
        },
        "narma10": {
            "name": "narma10", "n_train": 2400, "n_val": 600,
            "n_test": 1000, "washout": 1000, "state_policy": "reset",
        },
        "mackey_glass": {
            "name": "mackey_glass", "n_train": 2400, "n_val": 600,
            "n_test": 1000, "washout": 1000, "state_policy": "reset",
        },
    }
    if tasks_cfg is not None:
        _default_task_cfg.update(tasks_cfg)
    tasks_cfg = _default_task_cfg

    val_cols: dict[str, str] = {
        "delay_recall": "delay_recall_memory_corr_total_mean",
        "narma10": "narma10_val_nmse_mean",
        "mackey_glass": "mg_val_nmse_mean",
    }
    maximize_tasks = {"delay_recall"}
    torch_kinds = {"torch_simple_rnn", "torch_lstm"}

    candidates_df = _load_candidates_df(d)
    if candidates_df is None:
        raise FileNotFoundError(f"comparison_summary.csv not found in {d}")

    torch_mask = candidates_df["model_kind"].isin(torch_kinds)
    torch_models_df = candidates_df[torch_mask].drop_duplicates("model_name")

    for _, model_row in torch_models_df.iterrows():
        model_name = str(model_row["model_name"])
        model_kind = str(model_row["model_kind"])
        arch_key = "simple_rnn" if "rnn" in model_kind else "lstm"

        for task, _label, _vc, _tc, _ in TASK_SPECS_EXTERNAL:
            val_col = val_cols.get(task)
            if val_col is None:
                continue
            m_rows = candidates_df[candidates_df["model_name"] == model_name]
            vals = pd.to_numeric(m_rows[val_col], errors="coerce").dropna()
            if vals.empty:
                continue
            best_idx = (vals.idxmax() if task in maximize_tasks else vals.idxmin())
            best_row = candidates_df.loc[best_idx]
            selected_config_id = str(best_row["config_id"])

            # Load one run JSON to get the exact config_point used
            runs_dir = d / model_name / task / "runs"
            if not runs_dir.exists():
                continue
            ref_files = sorted(runs_dir.glob(f"{selected_config_id}_*.json"))
            if not ref_files:
                continue
            with open(ref_files[0], encoding="utf-8") as f:
                ref_run = json.load(f)
            config_point = ref_run["config_point"]
            metric_names = list(ref_run.get("val_metrics", {}).keys()) or ["nmse"]

            task_cfg = tasks_cfg[task]
            task_obj = resolve_task(
                task_cfg.get("name", task),
                state_policy=task_cfg.get("state_policy", "reset"),
                task_cfg=task_cfg,
            )

            for seed in seeds:
                out_path = curves_dir / f"{arch_key}_{task}_seed{seed}.json"
                if out_path.exists():
                    continue
                set_seed(seed)
                task_data = task_obj.generate(
                    n_train=task_cfg["n_train"],
                    n_val=task_cfg["n_val"],
                    n_test=task_cfg["n_test"],
                    washout=task_cfg["washout"],
                    seed=seed,
                )
                result = fit_torch_sequence_model(
                    kind=model_kind,
                    task_data=task_data,
                    config_point=config_point,
                    metrics=metric_names,
                    seed=seed,
                    device=device,
                    evaluate_test=False,
                    task_name=task,
                    task_cfg=task_cfg,
                    record_history=True,
                )
                curve_payload = {
                    "arch": arch_key,
                    "model_kind": model_kind,
                    "task": task,
                    "seed": seed,
                    "config_id": selected_config_id,
                    "config_point": config_point,
                    "best_epoch": result["metadata"].get("best_epoch"),
                    "epochs_ran": result["metadata"].get("epochs_ran"),
                    "early_stopped": result["metadata"].get("early_stopped"),
                    "history": result.get("history", []),
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(curve_payload, f, indent=2, ensure_ascii=False)

    return curves_dir


def plot_external_training_curves(
    curves_dir: str | Path,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """F_external_training_curves — 2×3 grid of val_loss vs epoch.

    Per panel: individual seed curves (thin, semi-transparent) + mean val_loss
    up to L_común (epochs where all seeds are present).  Best-epoch range shown
    as axvspan + mean line so noisy tasks reveal unstable stopping points.
    Train loss drawn in grey to show the train-val gap.
    """
    import json
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    d = Path(curves_dir)
    archs = ["simple_rnn", "lstm"]
    arch_labels = {"simple_rnn": "RNN simple", "lstm": "LSTM"}
    tasks_order = ["delay_recall", "narma10", "mackey_glass"]

    n_rows, n_cols = len(archs), len(tasks_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 6.5), squeeze=False)

    for ri, arch in enumerate(archs):
        for ci, task in enumerate(tasks_order):
            ax = axes[ri][ci]
            pattern = f"{arch}_{task}_seed*.json"
            seed_files = sorted(d.glob(pattern))
            if not seed_files:
                ax.set_title(f"{arch_labels.get(arch, arch)} / {TASK_LABELS.get(task, task)}")
                ax.set_visible(False)
                continue

            all_val_loss: list[np.ndarray] = []
            all_train_loss: list[np.ndarray] = []
            best_epochs: list[int] = []

            for p in seed_files:
                with open(p, encoding="utf-8") as f:
                    curve = json.load(f)
                hist = curve.get("history", [])
                if not hist:
                    continue
                vl = np.array([
                    h["val_loss"] if h.get("val_loss") is not None else float("nan")
                    for h in hist
                ], dtype=float)
                tl = np.array([
                    h["train_loss"] if h.get("train_loss") is not None else float("nan")
                    for h in hist
                ], dtype=float)
                all_val_loss.append(vl)
                all_train_loss.append(tl)
                be = curve.get("best_epoch")
                if be is not None:
                    best_epochs.append(int(be))

            if not all_val_loss:
                ax.set_title(f"{arch_labels.get(arch, arch)} / {TASK_LABELS.get(task, task)}")
                continue

            # L_común: epochs where ALL seeds contributed (no tail bias)
            L_comun = min(len(v) for v in all_val_loss)
            epoch_axis = np.arange(1, L_comun + 1)

            # Train loss tenue (individual seeds, grey)
            for tl in all_train_loss:
                ep_tl = np.arange(1, len(tl) + 1)
                ax.semilogy(ep_tl, tl, color="#bbbbbb", lw=0.6, alpha=0.25, zorder=1)

            # Individual val_loss curves (thin, semi-transparent)
            for vl in all_val_loss:
                ep_vl = np.arange(1, len(vl) + 1)
                ax.semilogy(ep_vl, vl, color=ACCENT_BLUE, lw=0.8, alpha=0.40, zorder=3)

            # Mean val_loss up to L_común only
            vl_common = np.vstack([v[:L_comun] for v in all_val_loss])
            mean_vl = np.nanmean(vl_common, axis=0)
            valid = np.isfinite(mean_vl)
            if valid.any():
                ax.semilogy(epoch_axis[valid], mean_vl[valid],
                            color=ACCENT_BLUE, lw=1.4, alpha=0.75, zorder=4)

            # Best-epoch: span over seed range + thin mean line (§2)
            if best_epochs:
                mean_be = float(np.mean(best_epochs))
                min_be, max_be = min(best_epochs), max(best_epochs)
                legend_label = f"restauración (rango/≈{mean_be:.0f})"
                if min_be < max_be:
                    ax.axvspan(min_be, max_be, alpha=0.12, color=ACCENT_RED,
                               zorder=2, label=legend_label)
                    ax.axvline(mean_be, ls="--", color=ACCENT_RED, lw=0.9,
                               alpha=0.75, zorder=5)
                else:
                    ax.axvline(mean_be, ls="--", color=ACCENT_RED, lw=1.0,
                               alpha=0.8, zorder=5, label=f"restauración ≈ {mean_be:.0f}")
                ax.legend(frameon=False, fontsize=7)

            # Budget cap marker
            ax.axvline(800, ls=":", color="#999", lw=0.8, alpha=0.6)
            ax.text(800, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 1.0,
                    " cap", fontsize=6, color="#777", va="top")

            ax.set_xlabel("época", fontsize=8)
            ax.set_ylabel("loss (val)", fontsize=8)
            ax.set_title(
                f"{arch_labels.get(arch, arch)} / {TASK_LABELS.get(task, task)}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.15)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

    fig.tight_layout()
    return save_figure(fig, out_dir, "F_external_training_curves", formats=formats)


def build_external_figures(
    ext: ExternalData,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    *,
    kmax: int = 100,
    esn_N: int = 100,
    curves_dir: str | Path | None = None,
) -> list[Path]:
    """Build external comparison figures: E1 (task bars), E2a (cost-time), E2b (cost-params).

    E3 (rank heatmap) was removed: superseded by E1.

    If curves_dir is provided and contains diagnostic curve JSONs (from
    run_torch_diagnostic), also builds F_external_training_curves.
    """
    apply_style()
    out = Path(out_dir)
    paths: list[Path] = []
    paths += plot_external_task_bars(ext, out, formats=formats)
    paths += plot_external_cost_tradeoff(ext, out, x_axis="time", kmax=kmax, esn_N=esn_N, formats=formats)
    paths += plot_external_cost_tradeoff(ext, out, x_axis="params", kmax=kmax, esn_N=esn_N, formats=formats)
    if curves_dir is not None and Path(curves_dir).exists():
        paths += plot_external_training_curves(curves_dir, out, formats=formats)
    return paths


def plot_external_task_bars(
    ext: ExternalData,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """E1 — Horizontal bars per model, one panel per task (F_external_task_bars).

    Models ordered by performance within each panel (best at top).
    NARMA/MG on log x-axis; Delay Recall on linear.  Champion gets a bold border.
    Error bars from per-seed std when available.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullLocator

    fig, axes = plt.subplots(1, len(TASK_SPECS_EXTERNAL), figsize=(13.5, 4.2))

    for ax, (task, label, _vc, test_col, maximize) in zip(axes, TASK_SPECS_EXTERNAL):
        sel = ext.selected.get(task, pd.DataFrame())
        if sel.empty:
            ax.set_title(label)
            continue

        sel_plot = sel.dropna(subset=["test"]).copy()
        sel_plot["test"] = pd.to_numeric(sel_plot["test"], errors="coerce")
        # Sort so best lands at the last (top) position in barh:
        #   NMSE (minimize): descending → smallest is last → top ✓
        #   Maximize tasks: ascending -> largest is last -> top.
        ascending = maximize
        sel_plot = sel_plot.sort_values("test", ascending=ascending)

        champ_name = (
            sel_plot.iloc[-1]["model_name"] if not sel_plot.empty else None
        )
        model_names = sel_plot["model_name"].tolist()
        values = sel_plot["test"].tolist()
        kinds = sel_plot["model_kind"].tolist()
        y_labels = [MODEL_LABELS.get(m, m) for m in model_names]
        colors = [KIND_COLORS.get(k, ACCENT_BLUE) for k in kinds]

        xerr: list[float] | None = None
        if ext.has_std:
            stds = [_to_float(v) for v in sel_plot["test_std"]]
            has_any = any(np.isfinite(s) and s > 0 for s in stds)
            if has_any:
                xerr = [s if (np.isfinite(s) and s > 0) else 0.0 for s in stds]

        y_pos = list(range(len(model_names)))
        bars = ax.barh(
            y_pos, values, color=colors, height=0.7,
            xerr=xerr,
            error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#555"},
        )
        for bar, mn in zip(bars, model_names):
            if mn == champ_name:
                bar.set_edgecolor("#222")
                bar.set_linewidth(1.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_title(label)
        ax.set_xlabel(_metric_label(test_col))
        ax.grid(True, axis="x", alpha=0.2)
        if not maximize:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_locator(NullLocator())

    fig.tight_layout()
    return save_figure(fig, out_dir, "F_external_task_bars", formats=formats)


def plot_external_cost_tradeoff(
    ext: ExternalData,
    out_dir: str | Path,
    *,
    x_axis: str = "time",
    kmax: int = 100,
    esn_N: int = 100,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """E2a/E2b — Cost vs performance clouds + Pareto frontier, one panel per task.

    Draws ALL candidates (6 per model) coloured by model kind.  Only the top-4
    ESN families per task are labelled (reducing clutter).  Both axes log-scale.

    x_axis="time"   → tuning_total_s_sum (minutes, model-level total)
                       → F_external_cost_time
    x_axis="params" → full n_trainable count for each task
                       → F_external_cost_params

    Pareto corner: low cost + best metric.
    Text annotation: aggregated cost ESN vs torch.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if x_axis not in {"time", "params"}:
        raise ValueError("x_axis must be 'time' or 'params'")

    use_time = x_axis == "time"
    name = "F_external_cost_time" if use_time else "F_external_cost_params"
    x_label = (
        "Tiempo total de ajuste (min)"
        if use_time
        else "Parámetros entrenables"
    )

    # val metric columns per task
    val_cols: dict[str, str] = {
        "delay_recall": "delay_recall_memory_corr_total_mean",
        "narma10": "narma10_val_nmse_mean",
        "mackey_glass": "mg_val_nmse_mean",
    }

    candidates = ext.candidates.copy()

    # Aggregated cost annotation (total tuning time ESN vs torch)
    esn_total_s = float("nan")
    torch_total_s = float("nan")
    esn_rows = candidates[candidates["model_kind"] == "esn"]
    if not esn_rows.empty:
        esn_models = esn_rows["model_name"].unique()
        esn_total_s = float(sum(
            candidates[candidates["model_name"] == mn]["tuning_total_s_sum"].dropna().iloc[0]
            for mn in esn_models
            if len(candidates[candidates["model_name"] == mn]["tuning_total_s_sum"].dropna()) > 0
        ))
    torch_rows = candidates[candidates["model_kind"].isin({"torch_simple_rnn", "torch_lstm"})]
    if not torch_rows.empty:
        torch_models = torch_rows["model_name"].unique()
        torch_total_s = float(sum(
            candidates[candidates["model_name"] == mn]["tuning_total_s_sum"].dropna().iloc[0]
            for mn in torch_models
            if len(candidates[candidates["model_name"] == mn]["tuning_total_s_sum"].dropna()) > 0
        ))

    fig, axes = plt.subplots(1, len(TASK_SPECS_EXTERNAL), figsize=(14.5, 4.4))

    for ax, (task, panel_label, _vc, _tc, maximize) in zip(axes, TASK_SPECS_EXTERNAL):
        val_col = val_cols.get(task)
        if val_col not in candidates.columns:
            ax.set_title(panel_label)
            continue

        plot_rows = candidates.dropna(subset=[val_col]).copy()
        if plot_rows.empty:
            ax.set_title(panel_label)
            continue

        # x values
        if use_time:
            plot_rows["_x"] = pd.to_numeric(
                plot_rows["tuning_total_s_sum"], errors="coerce"
            ) / 60.0
        else:
            plot_rows["_x"] = plot_rows.apply(
                lambda r: _trainable_params_for_task(r, task, kmax=kmax, esn_N=esn_N), axis=1
            )

        plot_rows["_y"] = pd.to_numeric(plot_rows[val_col], errors="coerce")
        plot_rows = plot_rows.dropna(subset=["_x", "_y"])
        plot_rows = plot_rows[plot_rows["_x"] > 0]
        if plot_rows.empty:
            ax.set_title(panel_label)
            continue

        x_arr = plot_rows["_x"].to_numpy(float)
        y_arr = plot_rows["_y"].to_numpy(float)
        kinds = plot_rows["model_kind"].tolist()
        mnames = plot_rows["model_name"].tolist()
        colors = [KIND_COLORS.get(k, ACCENT_BLUE) for k in kinds]

        ax.scatter(x_arr, y_arr, c=colors, s=30, zorder=3,
                   edgecolors="white", linewidths=0.3, alpha=0.85)

        # Top-4 ESN families by val metric (best candidate per family)
        esn_mask = plot_rows["model_kind"] == "esn"
        if esn_mask.any():
            esn_sub = plot_rows[esn_mask].copy()
            if maximize:
                best_per_family = esn_sub.loc[
                    esn_sub.groupby("model_name")["_y"].idxmax()
                ]
                top4 = best_per_family.nlargest(4, "_y")
            else:
                best_per_family = esn_sub.loc[
                    esn_sub.groupby("model_name")["_y"].idxmin()
                ]
                top4 = best_per_family.nsmallest(4, "_y")

            label_pts: list[tuple[float, float, str]] = []
            for _, lr in top4.iterrows():
                lx = float(lr["_x"])
                ly = float(lr["_y"])
                lname = MODEL_LABELS.get(str(lr["model_name"]), str(lr["model_name"]))
                label_pts.append((lx, ly, lname))
            _label_points(ax, label_pts, fontsize=6)

        # Pareto frontier (all points, task-appropriate corner)
        front = _pareto_cost_indices(x_arr, y_arr, maximize=maximize)
        if len(front) >= 2:
            fx, fy = x_arr[front], y_arr[front]
            order = np.argsort(fx)
            ax.plot(fx[order], fy[order], "-", color="#333", lw=1.1,
                    alpha=0.7, zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(_metric_label(val_col), fontsize=8)
        ax.set_title(panel_label)
        ax.grid(True, alpha=0.12)

    # Cost annotation on rightmost panel
    if use_time and np.isfinite(esn_total_s) and np.isfinite(torch_total_s):
        esn_min = esn_total_s / 60.0
        torch_min = torch_total_s / 60.0
        ann = (
            f"ESN total: {esn_min:.0f} min\n"
            f"RNN/LSTM total: {torch_min:.0f} min"
        )
        axes[-1].text(
            0.97, 0.05, ann,
            transform=axes[-1].transAxes,
            fontsize=7, ha="right", va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "#aaa", "alpha": 0.85},
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=KIND_COLORS["esn"], markersize=7, label="ESN"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=KIND_COLORS["torch_simple_rnn"], markersize=7,
               label="RNN simple"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=KIND_COLORS["torch_lstm"], markersize=7,
               label="LSTM"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=KIND_COLORS["tapped_delay_ridge"], markersize=7,
               label="tapped-delay"),
        Line2D([0], [0], color="#333", lw=1.1, label="Pareto"),
    ]
    axes[0].legend(handles=legend_elements, frameon=True, fontsize=6.5,
                   loc="best", framealpha=0.9)

    fig.tight_layout()
    return save_figure(fig, out_dir, name, formats=formats)


def _pareto_cost_indices(
    cost: np.ndarray,
    metric: np.ndarray,
    maximize: bool,
) -> np.ndarray:
    """Indices of Pareto-optimal (low-cost, best-metric) points.

    A point i is dominated if some j has cost[j]<=cost[i] AND metric[j] at
    least as good, with at least one strict inequality.
    """
    n = len(cost)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cost_leq = cost[j] <= cost[i]
            metric_geq = metric[j] >= metric[i] if maximize else metric[j] <= metric[i]
            cost_lt = cost[j] < cost[i]
            metric_better = metric[j] > metric[i] if maximize else metric[j] < metric[i]
            if cost_leq and metric_geq and (cost_lt or metric_better):
                dominated[i] = True
                break
    return np.where(~dominated)[0]


def plot_external_rank_heatmap(
    ext: ExternalData,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """E3 — Rank heatmap MODEL_ORDER × tasks, cell = rank_1_8 (F_external_rank_heatmap).

    Sequential-inverted colormap: rank 1 (best) is darkest.  Extra column
    shows mean rank across tasks.  Annotated with _annotate_heatmap style.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    task_keys = [t for t, *_ in TASK_SPECS_EXTERNAL]
    task_col_labels = [lbl for _, lbl, *_ in TASK_SPECS_EXTERNAL]
    col_labels = task_col_labels + ["media"]
    row_labels = [MODEL_LABELS.get(m, m) for m in MODEL_ORDER]
    n_models = len(MODEL_ORDER)
    n_tasks = len(task_keys)

    rank_matrix = np.full((n_models, n_tasks), np.nan)
    for j, task in enumerate(task_keys):
        sel = ext.selected.get(task, pd.DataFrame())
        for i, mn in enumerate(MODEL_ORDER):
            mrow = sel[sel["model_name"] == mn]
            if not mrow.empty:
                rank_matrix[i, j] = float(mrow.iloc[0]["rank_1_8"])

    mean_ranks = np.nanmean(rank_matrix, axis=1, keepdims=True)
    full_matrix = np.hstack([rank_matrix, mean_ranks])

    cmap_name = "YlGnBu_r"
    cmap = plt.get_cmap(cmap_name)
    vmin, vmax = 1.0, float(n_models)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    im = ax.imshow(full_matrix, cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Separator before "media" column
    ax.axvline(n_tasks - 0.5, color="white", lw=2.0, zorder=4)

    # Annotate cells
    for i in range(n_models):
        for j in range(len(col_labels)):
            val = full_matrix[i, j]
            if not np.isfinite(val):
                continue
            rgba = cmap(norm(val))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if brightness < 0.5 else "black"
            fmt = f"{val:.1f}" if j == n_tasks else f"{val:.0f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=8, color=text_color)

    add_colorbar(fig, ax, im, "Rank (1 = mejor)")
    ax.set_title("Rank")
    fig.tight_layout()
    return save_figure(fig, out_dir, "F_external_rank_heatmap", formats=formats)


def _annotate_heatmap(
    ax,
    grid: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap_name: str | None = None,
    fmt: str = "{:.0f}",
) -> None:
    """Annotate heatmap cells with auto-contrast text (white on dark, black on light)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return
    vmin = float(np.nanmin(grid)) if vmin is None else vmin
    vmax = float(np.nanmax(grid)) if vmax is None else vmax
    cmap = plt.get_cmap(cmap_name) if cmap_name else None
    norm = Normalize(vmin=vmin, vmax=vmax)
    for iy in range(grid.shape[0]):
        for ix in range(grid.shape[1]):
            val = grid[iy, ix]
            if not np.isfinite(val):
                continue
            color = "black"
            if cmap is not None:
                r, g, b, _ = cmap(norm(val))
                color = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"
            ax.text(ix, iy, fmt.format(val), ha="center", va="center", fontsize=6.5, color=color)
