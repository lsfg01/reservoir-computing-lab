"""
Tests unitarios para DesignComparisonRunner.save_comparison.

Verifica que los archivos comparison_summary.csv y comparison_summary.json
se crean correctamente con el contenido esperado.

Requisitos cubiertos: 5.9, 5.10, 5.11
"""

import csv
import json
from pathlib import Path

import pytest

from rc_lab.runners.design_comparison_runner import DesignComparisonRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(tmp_path: Path) -> DesignComparisonRunner:
    """Construye un DesignComparisonRunner mínimo apuntando a tmp_path."""
    cfg = {
        "sweep": {
            "name": "test_comparison",
            "output_dir": str(tmp_path),
            "seeds": [42, 123],
        },
        "designs": [
            {"name": "random_sparse_baseline", "reservoir": {"type": "random_sparse", "N": 100, "sparsity": 0.9, "bias_scaling": 0.0}},
            {"name": "cycle", "reservoir": {"type": "cycle", "N": 100, "cycle_weight": 1.0, "bias_scaling": 0.0}},
        ],
        "grid": {"spectral_radius": [0.9], "input_scaling": [0.1], "leak_rate": [1.0]},
        "tasks": {
            "narma10": {"n_train": 3000, "n_val": 1000, "n_test": 1500, "washout": 200, "state_policy": "reset"},
            "mackey_glass": {"n_train": 3000, "n_val": 1000, "n_test": 1500, "washout": 200, "state_policy": "reset", "tau": 17, "dt": 0.1},
            "memory_capacity": {"washout": 200, "input_length": 5000, "fit_fraction": 0.5, "kmax": 200, "ridge_param": 1e-6},
        },
        "readout": {"type": "ridge", "features": "states", "ridge_candidates": [1e-6, 1e-4]},
        "metrics": ["nmse"],
        "ranking": {
            "shortlist_top_n": 2,
            "narma10": {"metric": "nmse", "direction": "min"},
            "mackey_glass": {"metric": "nmse", "direction": "min"},
            "memory_capacity": {"metric": "mc_total", "direction": "max"},
        },
    }
    return DesignComparisonRunner(cfg)


def _make_table() -> list[dict]:
    """
    Tabla comparativa mínima con 4 filas (2 diseños × 2 configs).
    Ya ordenada por global_aggregate_rank ascendente.
    """
    return [
        {
            "design_name": "cycle",
            "reservoir_type": "cycle",
            "config_id": "cfg_B",
            "spectral_radius": 0.9,
            "input_scaling": 0.2,
            "leak_rate": 1.0,
            "n_seeds": 2,
            "narma10_val_nmse_mean": 0.08,
            "narma10_test_nmse_mean": 0.09,
            "mg_val_nmse_mean": 0.18,
            "mg_test_nmse_mean": 0.19,
            "mc_total_mean": 45.0,
            "rank_narma10_within_design": 1,
            "rank_mg_within_design": 1,
            "rank_mc_within_design": 2,
            "aggregate_rank_within_design": 4 / 3,
            "global_rank_narma10": 1,
            "global_rank_mg": 1,
            "global_rank_mc": 4,
            "global_aggregate_rank": 2.0,
            "delta_vs_baseline_narma10_val_nmse": -0.07,
            "delta_vs_baseline_mg_val_nmse": -0.07,
            "delta_vs_baseline_mc_total": -15.0,
        },
        {
            "design_name": "random_sparse_baseline",
            "reservoir_type": "random_sparse",
            "config_id": "cfg_A",
            "spectral_radius": 0.9,
            "input_scaling": 0.1,
            "leak_rate": 1.0,
            "n_seeds": 2,
            "narma10_val_nmse_mean": 0.10,
            "narma10_test_nmse_mean": 0.11,
            "mg_val_nmse_mean": 0.20,
            "mg_test_nmse_mean": 0.21,
            "mc_total_mean": 50.0,
            "rank_narma10_within_design": 1,
            "rank_mg_within_design": 1,
            "rank_mc_within_design": 2,
            "aggregate_rank_within_design": 4 / 3,
            "global_rank_narma10": 2,
            "global_rank_mg": 2,
            "global_rank_mc": 3,
            "global_aggregate_rank": 7 / 3,
        },
        {
            "design_name": "cycle",
            "reservoir_type": "cycle",
            "config_id": "cfg_A",
            "spectral_radius": 0.9,
            "input_scaling": 0.1,
            "leak_rate": 1.0,
            "n_seeds": 2,
            "narma10_val_nmse_mean": 0.12,
            "narma10_test_nmse_mean": 0.13,
            "mg_val_nmse_mean": 0.22,
            "mg_test_nmse_mean": 0.23,
            "mc_total_mean": 55.0,
            "rank_narma10_within_design": 2,
            "rank_mg_within_design": 2,
            "rank_mc_within_design": 1,
            "aggregate_rank_within_design": 5 / 3,
            "global_rank_narma10": 3,
            "global_rank_mg": 3,
            "global_rank_mc": 2,
            "global_aggregate_rank": 8 / 3,
            "delta_vs_baseline_narma10_val_nmse": 0.02,
            "delta_vs_baseline_mg_val_nmse": 0.02,
            "delta_vs_baseline_mc_total": 5.0,
        },
        {
            "design_name": "random_sparse_baseline",
            "reservoir_type": "random_sparse",
            "config_id": "cfg_B",
            "spectral_radius": 0.9,
            "input_scaling": 0.2,
            "leak_rate": 1.0,
            "n_seeds": 2,
            "narma10_val_nmse_mean": 0.15,
            "narma10_test_nmse_mean": 0.16,
            "mg_val_nmse_mean": 0.25,
            "mg_test_nmse_mean": 0.26,
            "mc_total_mean": 60.0,
            "rank_narma10_within_design": 2,
            "rank_mg_within_design": 2,
            "rank_mc_within_design": 1,
            "aggregate_rank_within_design": 5 / 3,
            "global_rank_narma10": 4,
            "global_rank_mg": 4,
            "global_rank_mc": 1,
            "global_aggregate_rank": 3.0,
        },
    ]


# ---------------------------------------------------------------------------
# Tests: archivos creados (Req 5.9)
# ---------------------------------------------------------------------------

class TestFilesCreated:
    """Verifica que los archivos se crean en la ubicación correcta."""

    def test_csv_file_created(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        csv_path, json_path = runner.save_comparison(table)
        assert csv_path.exists(), "comparison_summary.csv no fue creado"

    def test_json_file_created(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        csv_path, json_path = runner.save_comparison(table)
        assert json_path.exists(), "comparison_summary.json no fue creado"

    def test_csv_path_in_output_dir(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        csv_path, _ = runner.save_comparison(table)
        assert csv_path == tmp_path / "comparison_summary.csv"

    def test_json_path_in_output_dir(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        assert json_path == tmp_path / "comparison_summary.json"

    def test_output_dir_created_if_not_exists(self, tmp_path):
        """El directorio de salida debe crearse si no existe."""
        nested_dir = tmp_path / "nested" / "output"
        runner = _make_runner(nested_dir)
        table = _make_table()
        csv_path, json_path = runner.save_comparison(table)
        assert nested_dir.exists()
        assert csv_path.exists()
        assert json_path.exists()

    def test_returns_tuple_of_paths(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        result = runner.save_comparison(table)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Path)
        assert isinstance(result[1], Path)


# ---------------------------------------------------------------------------
# Tests: contenido CSV (Req 5.9)
# ---------------------------------------------------------------------------

class TestCSVContent:
    """Verifica el contenido del CSV generado."""

    def test_csv_has_correct_row_count(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        csv_path, _ = runner.save_comparison(table)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(table), f"Se esperaban {len(table)} filas, obtenidas {len(rows)}"

    def test_csv_has_all_columns_from_first_row(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        csv_path, _ = runner.save_comparison(table)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        expected_fields = list(table[0].keys())
        assert fieldnames == expected_fields

    def test_csv_values_match_table(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        csv_path, _ = runner.save_comparison(table)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Verificar primera fila: design_name y config_id
        assert rows[0]["design_name"] == table[0]["design_name"]
        assert rows[0]["config_id"] == table[0]["config_id"]
        # Verificar valor numérico (CSV lo guarda como string)
        assert float(rows[0]["narma10_val_nmse_mean"]) == pytest.approx(table[0]["narma10_val_nmse_mean"])

    def test_csv_preserves_row_order(self, tmp_path):
        """El CSV debe preservar el orden de la tabla (ordenada por global_aggregate_rank)."""
        runner = _make_runner(tmp_path)
        table = _make_table()
        csv_path, _ = runner.save_comparison(table)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        design_names = [r["design_name"] for r in rows]
        expected_names = [r["design_name"] for r in table]
        assert design_names == expected_names

    def test_csv_handles_optional_columns_gracefully(self, tmp_path):
        """
        Filas sin columnas delta (baseline) no deben causar error.
        extrasaction='ignore' garantiza que filas con menos claves no fallan.
        """
        runner = _make_runner(tmp_path)
        table = _make_table()
        # No debe lanzar excepción aunque algunas filas no tengan delta_*
        csv_path, _ = runner.save_comparison(table)
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# Tests: contenido JSON (Req 5.9, 5.10, 5.11)
# ---------------------------------------------------------------------------

class TestJSONContent:
    """Verifica el contenido del JSON generado."""

    def _load_json(self, json_path: Path) -> dict:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    def test_json_has_table_key(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        data = self._load_json(json_path)
        assert "table" in data

    def test_json_table_has_correct_row_count(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        data = self._load_json(json_path)
        assert len(data["table"]) == len(table)

    def test_json_has_best_overall_key(self, tmp_path):
        """Req 5.10: JSON debe incluir best_overall."""
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        data = self._load_json(json_path)
        assert "best_overall" in data

    def test_json_best_overall_is_lowest_global_aggregate_rank(self, tmp_path):
        """best_overall debe ser la fila con el menor global_aggregate_rank."""
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        data = self._load_json(json_path)
        best = data["best_overall"]
        min_rank = min(r["global_aggregate_rank"] for r in table)
        assert float(best["global_aggregate_rank"]) == pytest.approx(min_rank)

    def test_json_best_overall_matches_first_row(self, tmp_path):
        """Como la tabla ya está ordenada, best_overall debe coincidir con la primera fila."""
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        data = self._load_json(json_path)
        assert data["best_overall"]["design_name"] == table[0]["design_name"]
        assert data["best_overall"]["config_id"] == table[0]["config_id"]

    def test_json_has_best_by_design_key(self, tmp_path):
        """Req 5.11: JSON debe incluir best_by_design."""
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        data = self._load_json(json_path)
        assert "best_by_design" in data

    def test_json_best_by_design_has_entry_per_design(self, tmp_path):
        """best_by_design debe tener una entrada por cada design_name."""
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        data = self._load_json(json_path)
        design_names = {r["design_name"] for r in table}
        assert set(data["best_by_design"].keys()) == design_names

    def test_json_best_by_design_selects_lowest_aggregate_rank_within_design(self, tmp_path):
        """Para cada design_name, best_by_design debe ser la fila con menor aggregate_rank_within_design."""
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        data = self._load_json(json_path)
        for design_name, best_row in data["best_by_design"].items():
            rows_for_design = [r for r in table if r["design_name"] == design_name]
            min_rank = min(r["aggregate_rank_within_design"] for r in rows_for_design)
            assert float(best_row["aggregate_rank_within_design"]) == pytest.approx(min_rank), (
                f"best_by_design[{design_name!r}] no tiene el menor aggregate_rank_within_design"
            )

    def test_json_has_timestamp(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        data = self._load_json(json_path)
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)
        assert len(data["timestamp"]) > 0

    def test_json_has_n_designs(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        data = self._load_json(json_path)
        assert "n_designs" in data
        expected_n_designs = len({r["design_name"] for r in table})
        assert data["n_designs"] == expected_n_designs

    def test_json_has_n_configs_per_design(self, tmp_path):
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        data = self._load_json(json_path)
        assert "n_configs_per_design" in data
        n_designs = len({r["design_name"] for r in table})
        expected = len(table) / n_designs
        assert data["n_configs_per_design"] == pytest.approx(expected)

    def test_json_is_valid_json(self, tmp_path):
        """El archivo JSON debe ser parseable sin errores."""
        runner = _make_runner(tmp_path)
        table = _make_table()
        _, json_path = runner.save_comparison(table)
        # Si json.load no lanza excepción, el JSON es válido
        data = self._load_json(json_path)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Casos límite para save_comparison."""

    def test_raises_on_empty_table(self, tmp_path):
        runner = _make_runner(tmp_path)
        with pytest.raises(ValueError, match="comparison_table está vacía"):
            runner.save_comparison([])

    def test_single_row_table(self, tmp_path):
        """Una tabla con una sola fila debe funcionar correctamente."""
        runner = _make_runner(tmp_path)
        table = [_make_table()[0]]  # Solo la primera fila
        csv_path, json_path = runner.save_comparison(table)
        assert csv_path.exists()
        assert json_path.exists()
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(data["table"]) == 1
        assert data["n_designs"] == 1
