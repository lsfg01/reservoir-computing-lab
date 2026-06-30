from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _minimal_design_df() -> pd.DataFrame:
    rows = [
        ("random_sparse_baseline", "random_sparse", "base", 0.9, 0.1, 1.0, 0.24, 0.42, 48.0, 1.0),
        ("cycle_scr", "cycle", "scr", 1.0, 0.1, 1.0, 0.20, 0.36, 82.0, 1.4),
        ("nonnormal_chain_g0_3", "nonnormal", "nn", 1.1, 1.0, 0.5, 0.18, 0.08, 58.0, 1.0e5),
        ("multiscale_three_random", "multiscale", "ms", 0.95, 0.4, 1.0, 0.16, 0.24, 64.0, 12.0),
    ]
    data = []
    for rank, (design, reservoir, config, rho, sin, leak, narma, mg, mc, growth) in enumerate(rows, start=1):
        data.append(
            {
                "design_name": design,
                "reservoir_type": reservoir,
                "config_id": config,
                "spectral_radius": rho,
                "input_scaling": sin,
                "leak_rate": leak,
                "narma10_val_nmse_mean": narma,
                "narma10_test_nmse_mean": narma * 1.05,
                "mg_val_nmse_mean": mg,
                "mg_test_nmse_mean": mg * 1.05,
                "mc_total_mean": mc,
                "global_rank_narma10": rank,
                "global_rank_mg": rank,
                "global_rank_mc": rank,
                "global_aggregate_rank": rank,
                "aggregate_rank_within_design": 1,
                "diag_transient_growth_max_mean": growth,
            }
        )
    return pd.DataFrame(data)


def test_metric_labels_distinguish_mc_and_delay_recall() -> None:
    from rc_lab.viz.campaign import _metric_label, _metric_split_from_col

    assert _metric_label("mc_total_mean") == "MC total"
    assert _metric_split_from_col("mc_total_mean") is None
    assert _metric_label("narma10_test_nmse_mean") == "NARMA-10 NMSE (test)"
    assert _metric_label("mg_val_nmse_mean") == "Mackey–Glass NMSE (val)"
    assert (
        _metric_label("delay_recall_memory_corr_total_mean")
        == "Delay Recall memory corr. total (val)"
    )
    assert (
        _metric_label("delay_recall_test_memory_corr_total_mean")
        == "Delay Recall memory corr. total (test)"
    )


@pytest.mark.parametrize(
    ("kind", "hidden_size", "task", "expected"),
    [
        ("torch_simple_rnn", 64, "narma10", 4353),
        ("torch_simple_rnn", 64, "delay_recall", 10788),
        ("torch_simple_rnn", 128, "narma10", 16897),
        ("torch_simple_rnn", 128, "delay_recall", 29668),
        ("torch_lstm", 64, "narma10", 17217),
        ("torch_lstm", 64, "delay_recall", 23652),
        ("torch_lstm", 128, "narma10", 67201),
        ("torch_lstm", 128, "delay_recall", 79972),
    ],
)
def test_torch_trainable_params_are_task_specific(
    kind: str,
    hidden_size: int,
    task: str,
    expected: int,
) -> None:
    from rc_lab.viz.campaign import _trainable_params_for_task

    row = pd.Series(
        {
            "model_kind": kind,
            "hidden_size": hidden_size,
            "num_layers": 1,
            "n_trainable_params_mean": -1,
        }
    )

    assert _trainable_params_for_task(row, task) == expected


def test_external_master_uses_each_tasks_selected_candidate(tmp_path: Path) -> None:
    from rc_lab.viz.campaign import (
        ExternalData,
        _select_per_model,
        external_master_table,
    )

    candidates = pd.DataFrame(
        [
            {
                "model_name": "simple_rnn",
                "model_kind": "torch_simple_rnn",
                "config_id": "h64",
                "hidden_size": 64,
                "num_layers": 1,
                "delay_recall_memory_corr_total_mean": 10.0,
                "delay_recall_test_memory_corr_total_mean": 9.0,
                "narma10_val_nmse_mean": 0.1,
                "narma10_test_nmse_mean": 0.11,
                "mg_val_nmse_mean": 0.1,
                "mg_test_nmse_mean": 0.11,
                "n_trainable_params_mean": 6498,
                "tuning_total_s_sum": 60.0,
            },
            {
                "model_name": "simple_rnn",
                "model_kind": "torch_simple_rnn",
                "config_id": "h128",
                "hidden_size": 128,
                "num_layers": 1,
                "delay_recall_memory_corr_total_mean": 20.0,
                "delay_recall_test_memory_corr_total_mean": 19.0,
                "narma10_val_nmse_mean": 0.2,
                "narma10_test_nmse_mean": 0.21,
                "mg_val_nmse_mean": 0.2,
                "mg_test_nmse_mean": 0.21,
                "n_trainable_params_mean": 21154,
                "tuning_total_s_sum": 60.0,
            },
        ]
    )
    selected = {
        "delay_recall": pd.DataFrame(
            _select_per_model(
                candidates,
                "delay_recall_memory_corr_total_mean",
                "delay_recall_test_memory_corr_total_mean",
                True,
                tmp_path,
                "delay_recall",
                False,
            )
        ),
        "narma10": pd.DataFrame(
            _select_per_model(
                candidates,
                "narma10_val_nmse_mean",
                "narma10_test_nmse_mean",
                False,
                tmp_path,
                "narma10",
                False,
            )
        ),
        "mackey_glass": pd.DataFrame(
            _select_per_model(
                candidates,
                "mg_val_nmse_mean",
                "mg_test_nmse_mean",
                False,
                tmp_path,
                "mackey_glass",
                False,
            )
        ),
    }
    ext = ExternalData(
        candidates=candidates,
        selected=selected,
        has_std=False,
        results_dir=tmp_path,
    )

    table = external_master_table(ext)
    row = table[table["modelo"] == "RNN simple"].iloc[0]

    assert row["params (delay recall)"] == 29668
    assert row["params (NARMA-10)"] == 4353
    assert row["params (Mackey–Glass)"] == 4353
    assert "params (NARMA/MG)" not in table.columns


def test_build_design_figures_replaces_property_figures(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    import rc_lab.viz.campaign as campaign

    saved: list[str] = []

    def _capture(fig, out_dir, name, formats=("png",)):
        saved.append(name)
        plt.close(fig)
        return [Path(out_dir) / f"{name}.{fmt}" for fmt in formats]

    monkeypatch.setattr(campaign, "save_figure", _capture)

    paths = campaign.build_design_figures(_minimal_design_df(), tmp_path, formats=("png",))

    assert "F_design_mechanism_2x2" in saved
    assert "F_design_property_curves" not in saved
    assert "F_design_property_strip" not in saved
    assert tmp_path / "F_design_mechanism_2x2.png" in paths


def test_campaign_figures_do_not_use_internal_suptitles() -> None:
    import rc_lab.viz.campaign as campaign

    source = Path(campaign.__file__).read_text(encoding="utf-8")
    assert "fig.suptitle" not in source
    assert "Parámetros entrenables (readout)" not in source
