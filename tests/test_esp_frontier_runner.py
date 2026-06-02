"""
Tests de ESPFrontierRunner.

Cubre:
1. dry-run: cuenta de puntos correcta (|family|×|s_in|×|alpha|×|rho|×|seeds|).
2. rho sub-crítico → fraction_synchronized == 1.0.
3. rho_real reportado ≈ rho objetivo (tolerancia razonable).
4. Salida JSON-safe (sin NaN/inf crudos).
5. Summary contiene sigma_max y rho_real por fila.
6. Guardado incremental: cada punto genera un JSON en runs/.
7. final_slope_negative: helper funciona correctamente.
8. expand_points: orden determinista y completo.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from rc_lab.runners.esp_frontier_runner import (
    ESPFrontierRunner,
    _final_slope_negative,
    _subsample_curve,
)


# ---------------------------------------------------------------------------
# Config de test mínima (T pequeño para rapidez)
# ---------------------------------------------------------------------------

def _minimal_config(
    tmp_path: Path,
    rho_list: list[float] | None = None,
    sin_list: list[float] | None = None,
    alpha_list: list[float] | None = None,
    seeds: list[int] | None = None,
    families: list[dict] | None = None,
    T: int = 200,
    n_pairs: int = 3,
) -> dict:
    if rho_list is None:
        rho_list = [0.5]
    if sin_list is None:
        sin_list = [0.1]
    if alpha_list is None:
        alpha_list = [1.0]
    if seeds is None:
        seeds = [42, 123]
    if families is None:
        families = [
            {
                "name": "random_sparse",
                "type": "random_sparse",
                "N": 50,
                "sparsity": 0.9,
                "bias_scaling": 0.0,
            }
        ]
    return {
        "frontier": {
            "name": "test_frontier",
            "output_dir": str(tmp_path),
            "seeds": seeds,
        },
        "esp": {
            "T": T,
            "n_pairs": n_pairs,
            "eps": 1e-3,
        },
        "families": families,
        "grid": {
            "spectral_radius": rho_list,
            "input_scaling": sin_list,
            "leak_rate": alpha_list,
        },
        "diagnostics": {
            "transient_kmax": 10,
        },
    }


# ---------------------------------------------------------------------------
# Test 1: expand_points — cuenta correcta (dry-run)
# ---------------------------------------------------------------------------

def test_expand_points_count(tmp_path: Path) -> None:
    n_fam = 2
    n_sin = 3
    n_alpha = 2
    n_rho = 4
    n_seeds = 5
    families = [
        {"name": f"fam{i}", "type": "random_sparse", "N": 30,
         "sparsity": 0.9, "bias_scaling": 0.0}
        for i in range(n_fam)
    ]
    cfg = _minimal_config(
        tmp_path,
        rho_list=[0.3, 0.5, 0.7, 0.9],
        sin_list=[0.05, 0.1, 0.2],
        alpha_list=[0.8, 1.0],
        seeds=list(range(n_seeds)),
        families=families,
    )
    runner = ESPFrontierRunner(cfg)
    points = runner.expand_points()
    expected = n_fam * n_sin * n_alpha * n_rho * n_seeds
    assert len(points) == expected
    assert runner.n_points() == expected


# ---------------------------------------------------------------------------
# Test 2: rho sub-crítico → fraction_synchronized == 1.0
# ---------------------------------------------------------------------------

def test_subcritical_rho_full_sync(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path, rho_list=[0.5], T=300, n_pairs=5, seeds=[42])
    runner = ESPFrontierRunner(cfg)
    summary = runner.run()

    row = summary["rows"][0]
    assert row["fraction_synchronized_mean"] == pytest.approx(1.0, abs=0.0)


# ---------------------------------------------------------------------------
# Test 3: rho_real ≈ rho objetivo
# ---------------------------------------------------------------------------

def test_rho_real_close_to_target(tmp_path: Path) -> None:
    rho_target = 0.8
    cfg = _minimal_config(tmp_path, rho_list=[rho_target], seeds=[42])
    runner = ESPFrontierRunner(cfg)
    summary = runner.run()

    row = summary["rows"][0]
    assert abs(row["rho_real_mean"] - rho_target) < 0.05


# ---------------------------------------------------------------------------
# Test 4: JSON-safe — sin NaN/inf crudos en el summary
# ---------------------------------------------------------------------------

def test_summary_json_safe(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path, rho_list=[0.5, 1.2], seeds=[42])
    runner = ESPFrontierRunner(cfg)
    runner.run()

    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()

    with open(summary_path, encoding="utf-8") as f:
        text = f.read()

    # json.loads lanza si hay NaN/Infinity crudos en el texto
    data = json.loads(text)
    assert isinstance(data, dict)

    # Verificar recursivamente que no hay float NaN/inf
    def _check_no_bad_floats(obj: object) -> None:
        if isinstance(obj, float):
            assert math.isfinite(obj) or obj is None, f"Float no finito: {obj}"
        elif isinstance(obj, dict):
            for v in obj.values():
                _check_no_bad_floats(v)
        elif isinstance(obj, list):
            for v in obj:
                _check_no_bad_floats(v)

    _check_no_bad_floats(data)


# ---------------------------------------------------------------------------
# Test 5: Summary contiene sigma_max y rho_real
# ---------------------------------------------------------------------------

def test_summary_contains_sigma_and_rho_real(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path, rho_list=[0.6, 0.9], seeds=[42])
    runner = ESPFrontierRunner(cfg)
    summary = runner.run()

    assert "rows" in summary
    for row in summary["rows"]:
        assert "sigma_max_mean" in row, "Falta sigma_max_mean en fila del summary"
        assert "rho_real_mean" in row, "Falta rho_real_mean en fila del summary"
        assert isinstance(row["sigma_max_mean"], float)
        assert isinstance(row["rho_real_mean"], float)
        assert row["sigma_max_mean"] >= 0.0
        assert row["rho_real_mean"] >= 0.0


# ---------------------------------------------------------------------------
# Test 6: Guardado incremental — un JSON por punto en runs/
# ---------------------------------------------------------------------------

def test_incremental_save(tmp_path: Path) -> None:
    seeds = [42, 123]
    rho_list = [0.5, 0.7]
    cfg = _minimal_config(tmp_path, rho_list=rho_list, seeds=seeds)
    runner = ESPFrontierRunner(cfg)
    runner.run()

    runs_dir = tmp_path / "runs"
    assert runs_dir.exists()
    json_files = list(runs_dir.glob("*.json"))
    # 1 familia × 1 s_in × 1 alpha × 2 rho × 2 seeds = 4
    expected = 1 * 1 * 1 * len(rho_list) * len(seeds)
    assert len(json_files) == expected


# ---------------------------------------------------------------------------
# Test 7: _final_slope_negative helper
# ---------------------------------------------------------------------------

def test_final_slope_negative_decreasing() -> None:
    # Curva claramente decreciente al final
    curve = np.linspace(1.0, 0.01, 200)
    assert _final_slope_negative(curve) is True


def test_final_slope_negative_increasing() -> None:
    # Curva creciente al final (divergencia)
    curve = np.concatenate([np.linspace(1.0, 0.5, 100), np.linspace(0.5, 2.0, 100)])
    assert _final_slope_negative(curve) is False


# ---------------------------------------------------------------------------
# Test 8: _subsample_curve incluye siempre el último índice
# ---------------------------------------------------------------------------

def test_subsample_curve_includes_last() -> None:
    curve = np.arange(105, dtype=float)
    sub = _subsample_curve(curve, step=10)
    # Debe incluir el índice 104 (el último)
    assert sub[-1] == pytest.approx(104.0)
    # Y los primeros deben ser 0, 10, 20, ...
    assert sub[0] == pytest.approx(0.0)
    assert sub[1] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Test 9: Validación temprana — config incompleta lanza ValueError
# ---------------------------------------------------------------------------

def test_validation_missing_key(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path)
    del cfg["esp"]
    with pytest.raises(ValueError, match="esp"):
        ESPFrontierRunner(cfg)


def test_validation_empty_seeds(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path)
    cfg["frontier"]["seeds"] = []
    with pytest.raises(ValueError, match="seeds"):
        ESPFrontierRunner(cfg)


# ---------------------------------------------------------------------------
# Test 10: Determinismo — misma seed → mismo resultado
# ---------------------------------------------------------------------------

def test_determinism(tmp_path: Path) -> None:
    cfg1 = _minimal_config(tmp_path / "run1", rho_list=[0.7], seeds=[42])
    cfg2 = _minimal_config(tmp_path / "run2", rho_list=[0.7], seeds=[42])
    r1 = ESPFrontierRunner(cfg1).run()
    r2 = ESPFrontierRunner(cfg2).run()
    assert r1["rows"][0]["fraction_synchronized_mean"] == r2["rows"][0]["fraction_synchronized_mean"]
    assert r1["rows"][0]["sync_time_mean_mean"] == r2["rows"][0]["sync_time_mean_mean"]
