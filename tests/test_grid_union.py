"""
Tests del enhancement de unión de grids (grid: vs grids:) en el sweep.

Cubre:
1. Unitario — expand_grid y _resolve_grid_spec:
   - dict clásico → producto cartesiano idéntico al itertools.product original
   - Unión de 2 bloques sin solapamiento → conjunto esperado de puntos
   - Dedup en bloques solapados → primera aparición prevalece, sin duplicados
   - Bloques con ejes distintos → ValueError
   - Grid vacío (lista vacía) → ValueError
   - Eje vacío dentro de un bloque → ValueError
   - Ambos 'grid' y 'grids' → ValueError
   - Ninguno de 'grid' ni 'grids' → ValueError
2. Integración (invariante de alineación):
   - MultiTaskSweepRunner con grids: mínimo end-to-end →
     config_ids MC == NARMA/MG, agregador no lanza, n_configs == unión deduplicada
3. Regresión — pilot_multitask.yaml (grid:) expande igual que antes
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
import yaml

from rc_lab.runners.sweep_runner import (
    SweepRunner,
    _resolve_grid_spec,
    expand_grid,
    make_config_id,
)


# ---------------------------------------------------------------------------
# 1. Unitario — expand_grid y _resolve_grid_spec
# ---------------------------------------------------------------------------

class TestExpandGridDict:
    """expand_grid con dict → comportamiento clásico (retrocompatibilidad)."""

    def test_single_axis(self):
        grid = {"spectral_radius": [0.7, 0.9, 1.1]}
        points = expand_grid(grid)
        assert len(points) == 3
        assert [p["spectral_radius"] for p in points] == [0.7, 0.9, 1.1]

    def test_two_axes_cartesian(self):
        grid = {"spectral_radius": [0.7, 0.9], "input_scaling": [0.1, 0.2]}
        points = expand_grid(grid)
        assert len(points) == 4
        expected = [
            {"spectral_radius": 0.7, "input_scaling": 0.1},
            {"spectral_radius": 0.7, "input_scaling": 0.2},
            {"spectral_radius": 0.9, "input_scaling": 0.1},
            {"spectral_radius": 0.9, "input_scaling": 0.2},
        ]
        assert points == expected

    def test_identical_to_original_itertools_product(self):
        """expand_grid(dict) == resultado del itertools.product original."""
        grid = {
            "spectral_radius": [0.7, 0.9, 1.1],
            "input_scaling": [0.1, 0.2],
            "leak_rate": [0.6, 1.0],
        }
        keys = list(grid.keys())
        expected = [
            dict(zip(keys, combo))
            for combo in itertools.product(*grid.values())
        ]
        assert expand_grid(grid) == expected

    def test_config_ids_match_original(self):
        """Los config_id de expand_grid(dict) son idénticos a los del algoritmo original."""
        grid = {
            "spectral_radius": [0.7, 0.9],
            "input_scaling": [0.1, 0.2],
            "leak_rate": [1.0],
        }
        keys = list(grid.keys())
        original_ids = [
            make_config_id(dict(zip(keys, combo)))
            for combo in itertools.product(*grid.values())
        ]
        new_ids = [make_config_id(p) for p in expand_grid(grid)]
        assert new_ids == original_ids


class TestExpandGridList:
    """expand_grid con list[dict] → unión de cajas."""

    def test_two_disjoint_blocks(self):
        blocks = [
            {"spectral_radius": [0.7], "leak_rate": [1.0]},
            {"spectral_radius": [0.9], "leak_rate": [1.0]},
        ]
        points = expand_grid(blocks)
        assert len(points) == 2
        assert points[0] == {"spectral_radius": 0.7, "leak_rate": 1.0}
        assert points[1] == {"spectral_radius": 0.9, "leak_rate": 1.0}

    def test_overlapping_blocks_dedup(self):
        """Punto solapado aparece una sola vez; primera aparición prevalece."""
        blocks = [
            {"spectral_radius": [0.7, 0.9], "input_scaling": [0.1], "leak_rate": [1.0]},
            {"spectral_radius": [0.9, 1.1], "input_scaling": [0.1], "leak_rate": [1.0]},
        ]
        points = expand_grid(blocks)
        # (0.9, 0.1, 1.0) sólo una vez
        assert len(points) == 3
        rhos = [p["spectral_radius"] for p in points]
        assert rhos == [0.7, 0.9, 1.1]

    def test_dedup_preserves_first_occurrence_order(self):
        """La dedup mantiene el orden de primera aparición."""
        blocks = [
            {"x": [1.0, 2.0]},
            {"x": [2.0, 3.0]},
        ]
        points = expand_grid(blocks)
        assert [p["x"] for p in points] == [1.0, 2.0, 3.0]

    def test_config_id_identity_determines_dedup(self):
        """Dos puntos idénticos (mismo config_id) en bloques distintos → sólo uno."""
        pt = {"spectral_radius": 0.9, "input_scaling": 0.1, "leak_rate": 1.0}
        blocks = [{"spectral_radius": [0.9], "input_scaling": [0.1], "leak_rate": [1.0]},
                  {"spectral_radius": [0.9], "input_scaling": [0.1], "leak_rate": [1.0]}]
        points = expand_grid(blocks)
        assert len(points) == 1
        assert points[0] == pt

    def test_single_block_list_same_as_dict(self):
        """list con un solo bloque → idéntico a pasar el dict directamente."""
        grid_dict = {"spectral_radius": [0.7, 0.9], "leak_rate": [1.0]}
        grid_list = [grid_dict]
        assert expand_grid(grid_dict) == expand_grid(grid_list)

    def test_multi_axis_blocks_cross_product_per_block(self):
        """Cada bloque hace su propio cartesiano antes de unir."""
        blocks = [
            {"spectral_radius": [0.7, 0.9], "input_scaling": [0.1]},
            {"spectral_radius": [1.1], "input_scaling": [0.2, 0.3]},
        ]
        points = expand_grid(blocks)
        assert len(points) == 4
        block1 = [p for p in points if p["spectral_radius"] in (0.7, 0.9)]
        block2 = [p for p in points if p["spectral_radius"] == 1.1]
        assert len(block1) == 2
        assert len(block2) == 2


class TestExpandGridErrors:
    """Casos de error de expand_grid."""

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="vacío"):
            expand_grid([])

    def test_empty_axis_in_dict_raises(self):
        with pytest.raises(ValueError, match="vacío"):
            expand_grid({"spectral_radius": [], "input_scaling": [0.1]})

    def test_empty_axis_in_block_raises(self):
        with pytest.raises(ValueError, match="vacío"):
            expand_grid([{"spectral_radius": [0.7], "input_scaling": []}])

    def test_different_axes_raises(self):
        blocks = [
            {"spectral_radius": [0.7], "input_scaling": [0.1]},
            {"spectral_radius": [0.9], "leak_rate": [1.0]},
        ]
        with pytest.raises(ValueError, match="ejes distintos"):
            expand_grid(blocks)


class TestResolveGridSpec:
    """_resolve_grid_spec valida XOR y devuelve el spec correcto."""

    def test_grid_key_returned(self):
        cfg = {"grid": {"x": [1.0]}}
        spec = _resolve_grid_spec(cfg)
        assert spec == {"x": [1.0]}

    def test_grids_key_returned(self):
        blocks = [{"x": [1.0]}, {"x": [2.0]}]
        cfg = {"grids": blocks}
        spec = _resolve_grid_spec(cfg)
        assert spec is blocks

    def test_both_raises(self):
        cfg = {"grid": {"x": [1.0]}, "grids": [{"x": [1.0]}]}
        with pytest.raises(ValueError, match="exactamente uno"):
            _resolve_grid_spec(cfg)

    def test_neither_raises(self):
        cfg = {"sweep": {}}
        with pytest.raises(ValueError, match="exactamente uno"):
            _resolve_grid_spec(cfg)


@pytest.mark.parametrize(
    "grid_key, grid_spec",
    [
        ("grid", {"spectral_radius": [0.7], "input_scaling": [0.1], "ridge_param": [1e-6]}),
        ("grids", [{"spectral_radius": [0.7], "input_scaling": [0.1], "ridge_param": [1e-6]}]),
    ],
)
def test_sweep_runner_rejects_esn_candidate_ridge_param_with_readout_candidates(
    tmp_path,
    grid_key,
    grid_spec,
):
    cfg = {
        "sweep": {"name": "ridge_location", "output_dir": str(tmp_path), "seeds": [0]},
        "task": {
            "name": "narma10",
            "n_train": 20,
            "n_val": 10,
            "n_test": 10,
            "washout": 5,
        },
        "reservoir": {
            "type": "random_sparse",
            "N": 10,
            "sparsity": 0.9,
            "bias_scaling": 0.0,
        },
        grid_key: grid_spec,
        "readout": {"type": "ridge", "features": "states", "ridge_candidates": [1e-6]},
        "metrics": ["nmse"],
    }

    with pytest.raises(ValueError, match="readout\\.ridge_candidates"):
        SweepRunner(cfg)


# ---------------------------------------------------------------------------
# 2. Integración — invariante de alineación de config_ids
# ---------------------------------------------------------------------------

def _minimal_multitask_config(tmp_path: Path) -> dict[str, Any]:
    """Config mínima con grids: (2 bloques, 1 solapamiento → 3 puntos únicos)."""
    return {
        "sweep": {
            "name": "test_grid_union",
            "output_dir": str(tmp_path),
            "seeds": [0],
        },
        "reservoir": {
            "type": "random_sparse",
            "N": 30,
            "sparsity": 0.9,
            "bias_scaling": 0.0,
        },
        "grids": [
            {
                "spectral_radius": [0.7, 0.9],
                "input_scaling": [0.1],
                "leak_rate": [1.0],
            },
            {
                "spectral_radius": [0.9, 1.1],
                "input_scaling": [0.1],
                "leak_rate": [1.0],
            },
        ],
        "tasks": {
            "narma10": {
                "n_train": 400,
                "n_val": 100,
                "n_test": 100,
                "washout": 50,
                "state_policy": "reset",
            },
            "mackey_glass": {
                "n_train": 400,
                "n_val": 100,
                "n_test": 100,
                "washout": 50,
                "state_policy": "reset",
            },
            "memory_capacity": {
                "washout": 50,
                "input_length": 500,
                "fit_fraction": 0.5,
                "kmax": 20,
                "ridge_param": 1e-6,
            },
        },
        "readout": {
            "features": "states",
            "ridge_candidates": [1e-4, 1e-2],
        },
        "metrics": ["nmse"],
        "ranking": {
            "shortlist_top_n": 3,
            "narma10":        {"metric": "nmse",     "direction": "min"},
            "mackey_glass":   {"metric": "nmse",     "direction": "min"},
            "memory_capacity":{"metric": "mc_total", "direction": "max"},
        },
        "diagnostics": {"transient_kmax": 10},
    }


def test_integration_grids_alignment_invariant(tmp_path):
    """
    End-to-end con grids: mínimo:
    - n_configs == tamaño de la unión deduplicada (3)
    - config_ids de la fase MC == config_ids de NARMA/MG
    - el aggregator no lanza
    """
    from rc_lab.runners.multitask_sweep_runner import MultiTaskSweepRunner

    cfg = _minimal_multitask_config(tmp_path)
    runner = MultiTaskSweepRunner(cfg)
    summary = runner.run()

    # Tamaño correcto
    assert summary.n_configs == 3, f"Esperado 3, obtenido {summary.n_configs}"

    # Todos los config_ids presentes en el summary
    result_ids = {e.config_id for e in summary.configs}
    expected_points = expand_grid(cfg["grids"])
    expected_ids = {make_config_id(p) for p in expected_points}
    assert result_ids == expected_ids

    # Las tres tareas tienen exactamente los mismos config_ids (invariante de alineación)
    # Verificado implícitamente por el hecho de que el aggregator no lanzó.
    # Lo confirmamos también leyendo los runs de MC del disco.
    mc_dir = tmp_path / "mc"
    import json
    mc_ids = {
        json.loads(f.read_text())["config_id"]
        for f in mc_dir.glob("*.json")
    }
    assert mc_ids == expected_ids, (
        f"config_ids MC ≠ esperados: {mc_ids.symmetric_difference(expected_ids)}"
    )


def test_integration_grids_summary_grid_field(tmp_path):
    """El campo grid del summary almacena el grid_spec original (list[dict])."""
    from rc_lab.runners.multitask_sweep_runner import MultiTaskSweepRunner

    cfg = _minimal_multitask_config(tmp_path)
    summary = MultiTaskSweepRunner(cfg).run()

    assert isinstance(summary.grid, list)
    assert len(summary.grid) == 2


# ---------------------------------------------------------------------------
# 3. DesignComparisonRunner - passthrough grid/grids
# ---------------------------------------------------------------------------

def _minimal_design_config(tmp_path: Path) -> dict[str, Any]:
    """Config minima de design con grids: (2 bloques, 1 solapamiento -> 3 puntos)."""
    return {
        "sweep": {
            "name": "test_design_grid_union",
            "output_dir": str(tmp_path),
            "seeds": [0],
        },
        "designs": [
            {
                "name": "random_sparse_baseline",
                "reservoir": {
                    "type": "random_sparse",
                    "N": 30,
                    "sparsity": 0.9,
                    "bias_scaling": 0.0,
                },
            },
            {
                "name": "cycle_family",
                "reservoir": {
                    "type": "cycle",
                    "N": 30,
                    "cycle_weight": 1.0,
                    "bias_scaling": 0.0,
                },
            },
        ],
        "grids": [
            {
                "spectral_radius": [0.7, 0.9],
                "input_scaling": [0.1],
                "leak_rate": [1.0],
            },
            {
                "spectral_radius": [0.9, 1.1],
                "input_scaling": [0.1],
                "leak_rate": [1.0],
            },
        ],
        "tasks": {
            "narma10": {"enabled": True},
            "mackey_glass": {"enabled": True},
            "memory_capacity": {"enabled": True},
        },
        "readout": {
            "features": "states",
            "ridge_candidates": [1e-4],
        },
        "metrics": ["nmse"],
        "ranking": {
            "shortlist_top_n": 3,
            "narma10": {"metric": "nmse", "direction": "min"},
            "mackey_glass": {"metric": "nmse", "direction": "min"},
            "memory_capacity": {"metric": "mc_total", "direction": "max"},
        },
    }


def test_design_runner_grids_passthrough_aligns_families(monkeypatch, tmp_path):
    """
    Design con grids: propaga la misma union deduplicada a todas las familias.
    La comparacion no debe lanzar y cada summary ve n_configs == |union|.
    """
    import rc_lab.runners.design_comparison_runner as design_module
    from rc_lab.runners.multitask_sweep_runner import (
        MultiTaskConfigEntry,
        MultiTaskSweepSummary,
        RankingSpec,
    )

    seen_ids_by_design: dict[str, set[str]] = {}
    summaries_by_design = {}

    class FakeMultiTaskSweepRunner:
        def __init__(self, cfg: dict[str, Any]) -> None:
            self._cfg = cfg

        def run(self) -> MultiTaskSweepSummary:
            assert "grids" in self._cfg
            assert "grid" not in self._cfg

            design_name = Path(self._cfg["sweep"]["output_dir"]).name
            points = expand_grid(_resolve_grid_spec(self._cfg))
            entries: list[MultiTaskConfigEntry] = []
            design_offset = 0.0 if design_name == "random_sparse_baseline" else 0.01

            for idx, point in enumerate(points, start=1):
                cid = make_config_id(point)
                entries.append(
                    MultiTaskConfigEntry(
                        config_id=cid,
                        config_point=point,
                        n_seeds=len(self._cfg["sweep"]["seeds"]),
                        narma10_primary_metric="nmse",
                        narma10_val_mean=float(idx) + design_offset,
                        narma10_val_std=0.0,
                        narma10_test_mean=float(idx) + design_offset,
                        narma10_test_std=0.0,
                        mg_primary_metric="nmse",
                        mg_val_mean=float(idx) + design_offset,
                        mg_val_std=0.0,
                        mg_test_mean=float(idx) + design_offset,
                        mg_test_std=0.0,
                        mc_total_mean=float(10 - idx) - design_offset,
                        mc_total_std=0.0,
                        rank_narma10=idx,
                        rank_mg=idx,
                        rank_mc=idx,
                        aggregate_rank=float(idx),
                    )
                )

            summary = MultiTaskSweepSummary(
                sweep_name=self._cfg["sweep"]["name"],
                n_configs=len(entries),
                n_seeds=len(self._cfg["sweep"]["seeds"]),
                grid=_resolve_grid_spec(self._cfg),
                ranking_config={
                    "narma10": RankingSpec(metric="nmse", direction="min"),
                    "mackey_glass": RankingSpec(metric="nmse", direction="min"),
                    "memory_capacity": RankingSpec(metric="mc_total", direction="max"),
                },
                configs=entries,
                shortlist=[e.config_id for e in entries],
                shortlist_top_n=len(entries),
                timestamp="2024-01-01T00:00:00+00:00",
            )
            seen_ids_by_design[design_name] = {e.config_id for e in entries}
            summaries_by_design[design_name] = summary
            return summary

    monkeypatch.setattr(design_module, "MultiTaskSweepRunner", FakeMultiTaskSweepRunner)

    cfg = _minimal_design_config(tmp_path)
    expected_ids = {make_config_id(p) for p in expand_grid(cfg["grids"])}

    table = design_module.DesignComparisonRunner(cfg).run()

    assert set(seen_ids_by_design) == {"random_sparse_baseline", "cycle_family"}
    assert all(ids == expected_ids for ids in seen_ids_by_design.values())
    assert all(s.n_configs == len(expected_ids) for s in summaries_by_design.values())
    assert len(table) == len(cfg["designs"]) * len(expected_ids)


def test_design_runner_grid_dict_backward_compatible(tmp_path):
    """DesignComparisonRunner sigue aceptando grid: clasico y lo preserva."""
    from rc_lab.runners.design_comparison_runner import (
        DesignComparisonRunner,
        build_design_config,
    )

    cfg = _minimal_design_config(tmp_path)
    grid = {
        "spectral_radius": [0.7, 0.9],
        "input_scaling": [0.1],
        "leak_rate": [1.0],
    }
    cfg.pop("grids")
    cfg["grid"] = grid

    runner = DesignComparisonRunner(cfg)
    design_cfg = build_design_config(cfg, cfg["designs"][0])

    assert runner._grid_spec is grid
    assert design_cfg["grid"] is grid
    assert "grids" not in design_cfg


@pytest.mark.parametrize("case", ["both", "neither"])
def test_design_runner_grid_grids_xor_validation(case, tmp_path):
    """DesignComparisonRunner exige exactamente uno de grid/grids."""
    from rc_lab.runners.design_comparison_runner import DesignComparisonRunner

    cfg = _minimal_design_config(tmp_path)
    if case == "both":
        cfg["grid"] = {
            "spectral_radius": [0.7],
            "input_scaling": [0.1],
            "leak_rate": [1.0],
        }
    else:
        cfg.pop("grids")

    with pytest.raises(ValueError, match="exactamente uno"):
        DesignComparisonRunner(cfg)


# ---------------------------------------------------------------------------
# 4. Regression - pilot_multitask.yaml with classic grid:
# ---------------------------------------------------------------------------

def _load_pilot_multitask_yaml() -> dict[str, Any]:
    yaml_path = (
        Path(__file__).parent.parent
        / "configs" / "prelim_study" / "sweeps" / "pilot_multitask.yaml"
    )
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_regression_pilot_multitask_expands_12_points():
    """pilot_multitask.yaml (grid: clásico) expande exactamente 12 puntos."""
    cfg = _load_pilot_multitask_yaml()
    points = expand_grid(_resolve_grid_spec(cfg))
    assert len(points) == 12


def test_regression_pilot_multitask_config_ids_unchanged():
    """Los config_ids de pilot_multitask.yaml son idénticos al algoritmo original."""
    cfg = _load_pilot_multitask_yaml()
    grid = cfg["grid"]
    keys = list(grid.keys())
    original_ids = [
        make_config_id(dict(zip(keys, combo)))
        for combo in itertools.product(*grid.values())
    ]
    new_ids = [make_config_id(p) for p in expand_grid(_resolve_grid_spec(cfg))]
    assert new_ids == original_ids


def test_regression_grid_classic_accepted():
    """_validate_config acepta configs con grid: (retrocompatibilidad)."""
    from rc_lab.runners.multitask_sweep_runner import MultiTaskSweepRunner

    cfg = _load_pilot_multitask_yaml()
    cfg["sweep"]["output_dir"] = "/tmp/test_regression"  # no se usa, solo validate
    # No debería lanzar en construcción
    # (No llamamos a run() para no ejecutar el sweep completo)
    try:
        runner = MultiTaskSweepRunner.__new__(MultiTaskSweepRunner)
        runner._validate_config(cfg)
    except Exception as e:
        pytest.fail(f"_validate_config lanzó con grid: clásico: {e}")
