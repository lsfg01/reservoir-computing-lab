"""
Tests unitarios para DesignComparisonRunner.aggregate_comparison.

Verifica la construcción de la tabla comparativa usando objetos
MultiTaskSweepSummary y MultiTaskConfigEntry construidos manualmente,
sin necesidad de ejecutar ningún sweep real.

Requisitos cubiertos: 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
"""

import pytest

from rc_lab.runners.design_comparison_runner import DesignComparisonRunner
from rc_lab.runners.multitask_sweep_runner import (
    MultiTaskConfigEntry,
    MultiTaskSweepSummary,
    RankingSpec,
)


# ---------------------------------------------------------------------------
# Helpers para construir objetos mock
# ---------------------------------------------------------------------------

def _make_entry(
    config_id: str,
    spectral_radius: float,
    input_scaling: float,
    leak_rate: float,
    narma10_val: float,
    narma10_test: float,
    mg_val: float,
    mg_test: float,
    mc_total: float,
    rank_narma10: int,
    rank_mg: int,
    rank_mc: int,
    aggregate_rank: float,
    n_seeds: int = 3,
) -> MultiTaskConfigEntry:
    return MultiTaskConfigEntry(
        config_id=config_id,
        config_point={
            "spectral_radius": spectral_radius,
            "input_scaling": input_scaling,
            "leak_rate": leak_rate,
        },
        n_seeds=n_seeds,
        narma10_primary_metric="nmse",
        narma10_val_mean=narma10_val,
        narma10_val_std=0.01,
        narma10_test_mean=narma10_test,
        narma10_test_std=0.01,
        mg_primary_metric="nmse",
        mg_val_mean=mg_val,
        mg_val_std=0.01,
        mg_test_mean=mg_test,
        mg_test_std=0.01,
        mc_total_mean=mc_total,
        mc_total_std=0.5,
        rank_narma10=rank_narma10,
        rank_mg=rank_mg,
        rank_mc=rank_mc,
        aggregate_rank=aggregate_rank,
    )


def _make_summary(sweep_name: str, entries: list[MultiTaskConfigEntry]) -> MultiTaskSweepSummary:
    return MultiTaskSweepSummary(
        sweep_name=sweep_name,
        n_configs=len(entries),
        n_seeds=entries[0].n_seeds if entries else 0,
        grid={},
        ranking_config={
            "narma10": RankingSpec(metric="nmse", direction="min"),
            "mackey_glass": RankingSpec(metric="nmse", direction="min"),
            "memory_capacity": RankingSpec(metric="mc_total", direction="max"),
        },
        configs=entries,
        shortlist=[e.config_id for e in entries[:3]],
        shortlist_top_n=3,
        timestamp="2024-01-01T00:00:00+00:00",
    )


def _make_runner(designs: list[dict], output_dir: str = "/tmp/test_output") -> DesignComparisonRunner:
    """Construye un DesignComparisonRunner mínimo sin ejecutar ningún sweep."""
    cfg = {
        "sweep": {
            "name": "test_comparison",
            "output_dir": output_dir,
            "seeds": [42, 123, 456],
        },
        "designs": designs,
        "grid": {
            "spectral_radius": [0.9],
            "input_scaling": [0.1, 0.2],
            "leak_rate": [1.0],
        },
        "tasks": {
            "narma10": {"n_train": 3000, "n_val": 1000, "n_test": 1500, "washout": 200, "state_policy": "reset"},
            "mackey_glass": {"n_train": 3000, "n_val": 1000, "n_test": 1500, "washout": 200, "state_policy": "reset", "tau": 17, "dt": 0.1},
            "memory_capacity": {"washout": 200, "input_length": 5000, "fit_fraction": 0.5, "kmax": 200, "ridge_param": 1e-6},
        },
        "readout": {"type": "ridge", "features": "states", "ridge_candidates": [1e-6, 1e-4, 1e-2]},
        "metrics": ["nmse", "rmse"],
        "ranking": {
            "shortlist_top_n": 3,
            "narma10": {"metric": "nmse", "direction": "min"},
            "mackey_glass": {"metric": "nmse", "direction": "min"},
            "memory_capacity": {"metric": "mc_total", "direction": "max"},
        },
    }
    return DesignComparisonRunner(cfg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_design_results():
    """
    Dos diseños (baseline + cycle) con dos config_ids cada uno.
    config_id "cfg_A": baseline tiene mejor NMSE, cycle tiene mejor MC.
    config_id "cfg_B": cycle tiene mejor NMSE, baseline tiene mejor MC.
    """
    designs = [
        {"name": "random_sparse_baseline", "reservoir": {"type": "random_sparse", "N": 100, "sparsity": 0.9, "bias_scaling": 0.0}},
        {"name": "cycle", "reservoir": {"type": "cycle", "N": 100, "cycle_weight": 1.0, "bias_scaling": 0.0}},
    ]

    # Baseline entries
    baseline_entries = [
        _make_entry("cfg_A", 0.9, 0.1, 1.0, narma10_val=0.10, narma10_test=0.11, mg_val=0.20, mg_test=0.21, mc_total=50.0, rank_narma10=1, rank_mg=1, rank_mc=2, aggregate_rank=4/3),
        _make_entry("cfg_B", 0.9, 0.2, 1.0, narma10_val=0.15, narma10_test=0.16, mg_val=0.25, mg_test=0.26, mc_total=60.0, rank_narma10=2, rank_mg=2, rank_mc=1, aggregate_rank=5/3),
    ]
    # Cycle entries
    cycle_entries = [
        _make_entry("cfg_A", 0.9, 0.1, 1.0, narma10_val=0.12, narma10_test=0.13, mg_val=0.22, mg_test=0.23, mc_total=55.0, rank_narma10=2, rank_mg=2, rank_mc=1, aggregate_rank=5/3),
        _make_entry("cfg_B", 0.9, 0.2, 1.0, narma10_val=0.08, narma10_test=0.09, mg_val=0.18, mg_test=0.19, mc_total=45.0, rank_narma10=1, rank_mg=1, rank_mc=2, aggregate_rank=4/3),
    ]

    design_results = [
        ("random_sparse_baseline", _make_summary("test_baseline", baseline_entries)),
        ("cycle", _make_summary("test_cycle", cycle_entries)),
    ]
    runner = _make_runner(designs)
    return runner, design_results


# ---------------------------------------------------------------------------
# Tests: columnas mínimas (Req 5.3, 5.4)
# ---------------------------------------------------------------------------

class TestMinimumColumns:
    """Verifica que la tabla contiene todas las columnas mínimas requeridas."""

    REQUIRED_COLUMNS = [
        "design_name", "reservoir_type", "config_id",
        "spectral_radius", "input_scaling", "leak_rate",
        "n_seeds",
        "narma10_val_nmse_mean", "narma10_test_nmse_mean",
        "mg_val_nmse_mean", "mg_test_nmse_mean",
        "mc_total_mean",
    ]

    def test_all_required_columns_present(self, two_design_results):
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        assert len(table) > 0
        for col in self.REQUIRED_COLUMNS:
            assert col in table[0], f"Columna requerida ausente: {col!r}"

    def test_row_count_equals_total_configs(self, two_design_results):
        """Debe haber una fila por (design_name, config_id)."""
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        # 2 diseños × 2 configs = 4 filas
        assert len(table) == 4

    def test_reservoir_type_extracted_correctly(self, two_design_results):
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        types_by_design = {r["design_name"]: r["reservoir_type"] for r in table}
        assert types_by_design["random_sparse_baseline"] == "random_sparse"
        assert types_by_design["cycle"] == "cycle"

    def test_metric_values_mapped_correctly(self, two_design_results):
        """narma10_val_nmse_mean debe coincidir con entry.narma10_val_mean."""
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        # Buscar fila baseline cfg_A
        row = next(r for r in table if r["design_name"] == "random_sparse_baseline" and r["config_id"] == "cfg_A")
        assert row["narma10_val_nmse_mean"] == pytest.approx(0.10)
        assert row["narma10_test_nmse_mean"] == pytest.approx(0.11)
        assert row["mg_val_nmse_mean"] == pytest.approx(0.20)
        assert row["mg_test_nmse_mean"] == pytest.approx(0.21)
        assert row["mc_total_mean"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Tests: ranks internos (Req 5.5)
# ---------------------------------------------------------------------------

class TestWithinDesignRanks:
    """Verifica que los ranks internos se heredan del MultiTaskSweepSummary."""

    WITHIN_RANK_COLUMNS = [
        "rank_narma10_within_design",
        "rank_mg_within_design",
        "rank_mc_within_design",
        "aggregate_rank_within_design",
    ]

    def test_within_rank_columns_present(self, two_design_results):
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        for col in self.WITHIN_RANK_COLUMNS:
            assert col in table[0], f"Columna de rank interno ausente: {col!r}"

    def test_within_ranks_match_entry_values(self, two_design_results):
        """Los ranks internos deben coincidir exactamente con los del entry."""
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        row = next(r for r in table if r["design_name"] == "random_sparse_baseline" and r["config_id"] == "cfg_A")
        assert row["rank_narma10_within_design"] == 1
        assert row["rank_mg_within_design"] == 1
        assert row["rank_mc_within_design"] == 2
        assert row["aggregate_rank_within_design"] == pytest.approx(4 / 3)


# ---------------------------------------------------------------------------
# Tests: ranks globales (Req 5.6)
# ---------------------------------------------------------------------------

class TestGlobalRanks:
    """Verifica el cálculo de ranks globales sobre el conjunto completo de filas."""

    GLOBAL_RANK_COLUMNS = [
        "global_rank_narma10",
        "global_rank_mg",
        "global_rank_mc",
        "global_aggregate_rank",
    ]

    def test_global_rank_columns_present(self, two_design_results):
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        for col in self.GLOBAL_RANK_COLUMNS:
            assert col in table[0], f"Columna de rank global ausente: {col!r}"

    def test_global_rank_narma10_ascending(self, two_design_results):
        """El menor narma10_val_nmse_mean debe tener global_rank_narma10 == 1."""
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        # cycle cfg_B tiene narma10_val=0.08, el menor de los 4
        row = next(r for r in table if r["design_name"] == "cycle" and r["config_id"] == "cfg_B")
        assert row["global_rank_narma10"] == 1

    def test_global_rank_mc_descending(self, two_design_results):
        """El mayor mc_total_mean debe tener global_rank_mc == 1."""
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        # baseline cfg_B tiene mc_total=60.0, el mayor de los 4
        row = next(r for r in table if r["design_name"] == "random_sparse_baseline" and r["config_id"] == "cfg_B")
        assert row["global_rank_mc"] == 1

    def test_global_aggregate_rank_is_mean_of_three(self, two_design_results):
        """global_aggregate_rank debe ser la media aritmética de los tres ranks globales."""
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        for row in table:
            expected = (row["global_rank_narma10"] + row["global_rank_mg"] + row["global_rank_mc"]) / 3.0
            assert row["global_aggregate_rank"] == pytest.approx(expected)

    def test_global_ranks_cover_all_positions(self, two_design_results):
        """Con 4 filas y valores distintos, los ranks deben cubrir 1..4."""
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        n10_ranks = sorted(r["global_rank_narma10"] for r in table)
        mg_ranks = sorted(r["global_rank_mg"] for r in table)
        mc_ranks = sorted(r["global_rank_mc"] for r in table)
        assert n10_ranks == [1, 2, 3, 4]
        assert mg_ranks == [1, 2, 3, 4]
        assert mc_ranks == [1, 2, 3, 4]

    def test_global_rank_uses_min_method_for_ties(self):
        """Con empates, method='min' asigna el menor rank a todos los empatados."""
        designs = [
            {"name": "random_sparse_baseline", "reservoir": {"type": "random_sparse", "N": 100, "sparsity": 0.9, "bias_scaling": 0.0}},
            {"name": "cycle", "reservoir": {"type": "cycle", "N": 100, "cycle_weight": 1.0, "bias_scaling": 0.0}},
        ]
        # Dos filas con el mismo narma10_val → empate
        baseline_entries = [
            _make_entry("cfg_A", 0.9, 0.1, 1.0, narma10_val=0.10, narma10_test=0.11, mg_val=0.20, mg_test=0.21, mc_total=50.0, rank_narma10=1, rank_mg=1, rank_mc=1, aggregate_rank=1.0),
        ]
        cycle_entries = [
            _make_entry("cfg_A", 0.9, 0.1, 1.0, narma10_val=0.10, narma10_test=0.11, mg_val=0.25, mg_test=0.26, mc_total=45.0, rank_narma10=1, rank_mg=1, rank_mc=1, aggregate_rank=1.0),
        ]
        design_results = [
            ("random_sparse_baseline", _make_summary("test_baseline", baseline_entries)),
            ("cycle", _make_summary("test_cycle", cycle_entries)),
        ]
        runner = _make_runner(designs)
        table = runner.aggregate_comparison(design_results)
        # Ambas filas tienen narma10_val=0.10 → ambas deben tener rank 1 (method="min")
        n10_ranks = [r["global_rank_narma10"] for r in table]
        assert all(rank == 1 for rank in n10_ranks), f"Se esperaba rank 1 para empates, obtenido: {n10_ranks}"


# ---------------------------------------------------------------------------
# Tests: ordenación final (Req 5.8)
# ---------------------------------------------------------------------------

class TestSorting:
    """Verifica que la tabla se ordena por global_aggregate_rank ascendente."""

    def test_table_sorted_by_global_aggregate_rank(self, two_design_results):
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        ranks = [r["global_aggregate_rank"] for r in table]
        assert ranks == sorted(ranks), f"Tabla no ordenada por global_aggregate_rank: {ranks}"


# ---------------------------------------------------------------------------
# Tests: deltas vs baseline (Req 5.7)
# ---------------------------------------------------------------------------

class TestDeltasVsBaseline:
    """Verifica el cálculo de deltas frente a random_sparse_baseline."""

    DELTA_COLUMNS = [
        "delta_vs_baseline_narma10_val_nmse",
        "delta_vs_baseline_mg_val_nmse",
        "delta_vs_baseline_mc_total",
    ]

    def test_delta_columns_present_for_non_baseline(self, two_design_results):
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        non_baseline = [r for r in table if r["design_name"] != "random_sparse_baseline"]
        assert len(non_baseline) > 0
        for row in non_baseline:
            for col in self.DELTA_COLUMNS:
                assert col in row, f"Columna delta ausente en fila no-baseline: {col!r}"

    def test_delta_narma10_sign_convention(self, two_design_results):
        """Negativo = mejora para NMSE (cycle cfg_B tiene menor NMSE que baseline cfg_B)."""
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        # cycle cfg_B: narma10_val=0.08, baseline cfg_B: narma10_val=0.15 → delta = -0.07
        row = next(r for r in table if r["design_name"] == "cycle" and r["config_id"] == "cfg_B")
        assert row["delta_vs_baseline_narma10_val_nmse"] == pytest.approx(-0.07)

    def test_delta_mc_sign_convention(self, two_design_results):
        """Positivo = mejora para MC (cycle cfg_A tiene mayor MC que baseline cfg_A)."""
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        # cycle cfg_A: mc_total=55.0, baseline cfg_A: mc_total=50.0 → delta = +5.0
        row = next(r for r in table if r["design_name"] == "cycle" and r["config_id"] == "cfg_A")
        assert row["delta_vs_baseline_mc_total"] == pytest.approx(5.0)

    def test_delta_values_correct(self, two_design_results):
        """Verifica los tres deltas para cycle cfg_A."""
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        row = next(r for r in table if r["design_name"] == "cycle" and r["config_id"] == "cfg_A")
        # cycle cfg_A: narma10_val=0.12, baseline cfg_A: narma10_val=0.10 → delta = +0.02
        assert row["delta_vs_baseline_narma10_val_nmse"] == pytest.approx(0.02)
        # cycle cfg_A: mg_val=0.22, baseline cfg_A: mg_val=0.20 → delta = +0.02
        assert row["delta_vs_baseline_mg_val_nmse"] == pytest.approx(0.02)
        # cycle cfg_A: mc_total=55.0, baseline cfg_A: mc_total=50.0 → delta = +5.0
        assert row["delta_vs_baseline_mc_total"] == pytest.approx(5.0)

    def test_baseline_rows_have_no_delta_columns(self, two_design_results):
        """Las filas del baseline no deben tener columnas delta."""
        runner, design_results = two_design_results
        table = runner.aggregate_comparison(design_results)
        baseline_rows = [r for r in table if r["design_name"] == "random_sparse_baseline"]
        for row in baseline_rows:
            for col in self.DELTA_COLUMNS:
                assert col not in row, f"Columna delta inesperada en fila baseline: {col!r}"

    def test_delta_none_when_no_matching_baseline(self):
        """Si no hay baseline con el mismo config_id, los deltas no se añaden."""
        designs = [
            {"name": "random_sparse_baseline", "reservoir": {"type": "random_sparse", "N": 100, "sparsity": 0.9, "bias_scaling": 0.0}},
            {"name": "cycle", "reservoir": {"type": "cycle", "N": 100, "cycle_weight": 1.0, "bias_scaling": 0.0}},
        ]
        # Baseline tiene cfg_A, cycle tiene cfg_B (config_ids distintos → sin match)
        baseline_entries = [
            _make_entry("cfg_A", 0.9, 0.1, 1.0, narma10_val=0.10, narma10_test=0.11, mg_val=0.20, mg_test=0.21, mc_total=50.0, rank_narma10=1, rank_mg=1, rank_mc=1, aggregate_rank=1.0),
        ]
        cycle_entries = [
            _make_entry("cfg_B", 0.9, 0.2, 1.0, narma10_val=0.12, narma10_test=0.13, mg_val=0.22, mg_test=0.23, mc_total=55.0, rank_narma10=1, rank_mg=1, rank_mc=1, aggregate_rank=1.0),
        ]
        design_results = [
            ("random_sparse_baseline", _make_summary("test_baseline", baseline_entries)),
            ("cycle", _make_summary("test_cycle", cycle_entries)),
        ]
        runner = _make_runner(designs)
        table = runner.aggregate_comparison(design_results)
        cycle_row = next(r for r in table if r["design_name"] == "cycle")
        for col in self.DELTA_COLUMNS:
            assert col not in cycle_row, f"Delta inesperado cuando no hay baseline con mismo config_id: {col!r}"


# ---------------------------------------------------------------------------
# Tests: tabla vacía (edge case)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Casos límite."""

    def test_empty_design_results_returns_empty_list(self):
        designs = [
            {"name": "random_sparse_baseline", "reservoir": {"type": "random_sparse", "N": 100, "sparsity": 0.9, "bias_scaling": 0.0}},
        ]
        runner = _make_runner(designs)
        result = runner.aggregate_comparison([])
        assert result == []

    def test_single_design_no_deltas(self):
        """Con un solo diseño (el baseline), no se calculan deltas."""
        designs = [
            {"name": "random_sparse_baseline", "reservoir": {"type": "random_sparse", "N": 100, "sparsity": 0.9, "bias_scaling": 0.0}},
        ]
        entries = [
            _make_entry("cfg_A", 0.9, 0.1, 1.0, narma10_val=0.10, narma10_test=0.11, mg_val=0.20, mg_test=0.21, mc_total=50.0, rank_narma10=1, rank_mg=1, rank_mc=1, aggregate_rank=1.0),
        ]
        design_results = [
            ("random_sparse_baseline", _make_summary("test_baseline", entries)),
        ]
        runner = _make_runner(designs)
        table = runner.aggregate_comparison(design_results)
        assert len(table) == 1
        row = table[0]
        assert row["global_rank_narma10"] == 1
        assert row["global_rank_mg"] == 1
        assert row["global_rank_mc"] == 1
        assert row["global_aggregate_rank"] == pytest.approx(1.0)
        # Sin deltas en fila baseline
        assert "delta_vs_baseline_narma10_val_nmse" not in row

    def test_diagnostics_skipped_gracefully_when_no_files(self, two_design_results, tmp_path):
        """Si no existen JSONs de diagnóstico, no se añaden columnas diag_*_mean."""
        runner, design_results = two_design_results
        # Usar un output_dir que no existe → no hay JSONs de diagnóstico
        runner._output_dir = tmp_path / "nonexistent"
        table = runner.aggregate_comparison(design_results)
        for row in table:
            for key in row:
                assert not key.startswith("diag_"), f"Columna diag_ inesperada cuando no hay archivos: {key!r}"
