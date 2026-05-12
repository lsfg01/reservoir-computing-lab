"""
Tests para la feature nonnormal-transient-design.

Cubre:
- F.4: resolve_reservoir con type=nonnormal_chain y tipos existentes
"""

import pytest

from rc_lab.runners.runner import resolve_reservoir
from rc_lab.reservoirs.nonnormal_chain import NonnormalChainReservoir
from rc_lab.reservoirs.random_sparse import RandomSparseReservoir
from rc_lab.reservoirs.cycle import CycleReservoir
from rc_lab.reservoirs.cycle_jump import CycleJumpReservoir


# ---------------------------------------------------------------------------
# F.4 — Tests de resolve_reservoir
# ---------------------------------------------------------------------------

def test_resolve_reservoir_nonnormal_chain():
    """
    Validates: Requirements 8.1
    resolve_reservoir con type=nonnormal_chain devuelve NonnormalChainReservoir.
    """
    res = resolve_reservoir({
        "type": "nonnormal_chain",
        "spectral_radius": 0.9,
        "input_scaling": 0.1,
        "chain_strength": 0.3,
    })
    assert isinstance(res, NonnormalChainReservoir)


def test_resolve_reservoir_existing_types_unchanged():
    """
    Validates: Requirements 8.2
    Los tipos existentes (random_sparse, cycle, cycle_jump) siguen resolviendo
    correctamente tras añadir nonnormal_chain al registro.
    """
    # random_sparse
    res_rs = resolve_reservoir({
        "type": "random_sparse",
        "spectral_radius": 0.9,
        "input_scaling": 0.1,
        "sparsity": 0.9,
    })
    assert isinstance(res_rs, RandomSparseReservoir)

    # cycle
    res_cy = resolve_reservoir({
        "type": "cycle",
        "spectral_radius": 0.9,
        "input_scaling": 0.1,
        "cycle_weight": 1.0,
    })
    assert isinstance(res_cy, CycleReservoir)

    # cycle_jump
    res_cj = resolve_reservoir({
        "type": "cycle_jump",
        "spectral_radius": 0.9,
        "input_scaling": 0.1,
        "cycle_weight": 1.0,
        "jumps": [7],
        "jump_weight": 0.3,
    })
    assert isinstance(res_cj, CycleJumpReservoir)


import numpy as np
from rc_lab.reservoirs.diagnostics import compute_spectral_radius


# ---------------------------------------------------------------------------
# Parámetros comunes para tests de NonnormalChainReservoir
# ---------------------------------------------------------------------------

_N = 20
_N_INPUTS = 3
_SEED = 42
_SPECTRAL_RADIUS = 0.9
_CHAIN_STRENGTH = 0.3
_INPUT_SCALING = 0.1


# ---------------------------------------------------------------------------
# F.1 — Shapes correctas para NonnormalChainReservoir (Req 12.1 / 5.1)
# ---------------------------------------------------------------------------


def test_nonnormal_chain_shapes():
    """Req 12.1 / 5.1: NonnormalChainReservoir.build produce W(N,N), Win(N,n_inputs), bias(N,)."""
    res = NonnormalChainReservoir(
        spectral_radius=_SPECTRAL_RADIUS,
        input_scaling=_INPUT_SCALING,
    )
    mats = res.build(N=_N, n_inputs=_N_INPUTS, seed=_SEED)

    assert mats.W.shape == (_N, _N), (
        f"W.shape esperado ({_N},{_N}), obtenido {mats.W.shape}"
    )
    assert mats.Win.shape == (_N, _N_INPUTS), (
        f"Win.shape esperado ({_N},{_N_INPUTS}), obtenido {mats.Win.shape}"
    )
    assert mats.bias.shape == (_N,), (
        f"bias.shape esperado ({_N},), obtenido {mats.bias.shape}"
    )


# ---------------------------------------------------------------------------
# F.1 — Diagonal == spectral_radius (Req 5.1)
# ---------------------------------------------------------------------------


def test_nonnormal_chain_diagonal():
    """Req 5.1: np.diag(W) == spectral_radius para todos los elementos de la diagonal."""
    d = _SPECTRAL_RADIUS
    res = NonnormalChainReservoir(
        spectral_radius=d,
        input_scaling=_INPUT_SCALING,
        chain_strength=_CHAIN_STRENGTH,
    )
    mats = res.build(N=_N, n_inputs=1, seed=_SEED)

    np.testing.assert_allclose(
        np.diag(mats.W),
        d,
        atol=1e-12,
        err_msg=f"La diagonal de W debe ser exactamente {d}",
    )


# ---------------------------------------------------------------------------
# F.1 — W[i+1, i] == chain_strength para toda la subdiagonal (Req 5.2)
# ---------------------------------------------------------------------------


def test_nonnormal_chain_subdiagonal():
    """Req 5.2: W[i+1, i] == chain_strength para todo i in range(N-1)."""
    g = _CHAIN_STRENGTH
    res = NonnormalChainReservoir(
        spectral_radius=_SPECTRAL_RADIUS,
        input_scaling=_INPUT_SCALING,
        chain_strength=g,
    )
    mats = res.build(N=_N, n_inputs=1, seed=_SEED)

    for i in range(_N - 1):
        assert mats.W[i + 1, i] == pytest.approx(g), (
            f"W[{i+1}, {i}] = {mats.W[i+1, i]} != chain_strength = {g}"
        )


# ---------------------------------------------------------------------------
# F.1 — spectral_radius(W) ≈ spectral_radius sin reescalado (Req 5.4)
# ---------------------------------------------------------------------------


def test_nonnormal_chain_no_rescaling():
    """Req 5.4: compute_spectral_radius(W) ≈ spectral_radius con tolerancia 1e-6.

    Verifica que W NO se reescala tras la construcción: los autovalores son
    exactamente d = spectral_radius por construcción (matriz triangular inferior).
    """
    d = _SPECTRAL_RADIUS
    res = NonnormalChainReservoir(
        spectral_radius=d,
        input_scaling=_INPUT_SCALING,
        chain_strength=_CHAIN_STRENGTH,
    )
    mats = res.build(N=_N, n_inputs=1, seed=_SEED)

    rho = compute_spectral_radius(mats.W)
    assert abs(rho - d) < 1e-6, (
        f"compute_spectral_radius(W) = {rho} difiere de spectral_radius = {d} en más de 1e-6"
    )


# ---------------------------------------------------------------------------
# F.1 — Entradas fuera de diagonal y subdiagonal son cero (Req 5.3)
# ---------------------------------------------------------------------------


def test_nonnormal_chain_zero_offdiagonal():
    """Req 5.3: Todas las entradas fuera de la diagonal principal y la subdiagonal son cero."""
    res = NonnormalChainReservoir(
        spectral_radius=_SPECTRAL_RADIUS,
        input_scaling=_INPUT_SCALING,
        chain_strength=_CHAIN_STRENGTH,
    )
    mats = res.build(N=_N, n_inputs=1, seed=_SEED)
    W = mats.W

    for row in range(_N):
        for col in range(_N):
            # Diagonal principal: row == col
            # Subdiagonal: row == col + 1
            if row != col and row != col + 1:
                assert W[row, col] == 0.0, (
                    f"W[{row}, {col}] = {W[row, col]} debe ser 0 "
                    f"(fuera de diagonal y subdiagonal)"
                )


# ---------------------------------------------------------------------------
# F.2 — Tests de funciones de diagnóstico
# ---------------------------------------------------------------------------

import numpy as np
from rc_lab.reservoirs.diagnostics import (
    singular_values,
    transient_growth_curve,
    transient_growth_max,
    transient_growth_argmax,
    compute_spectral_radius,
    compute_spectral_norm,
)


def test_singular_values_diagonal():
    """
    Verifica que singular_values devuelve los valores correctos para una
    matriz diagonal conocida diag([3, 2, 1]).

    Validates: Requirements 12.2
    """
    W = np.diag([3.0, 2.0, 1.0])
    sv = singular_values(W)
    np.testing.assert_allclose(sv, [3.0, 2.0, 1.0], atol=1e-10)


def test_singular_values_descending_order():
    """
    Verifica que el array devuelto por singular_values está en orden descendente.

    Validates: Requirements 12.2
    """
    rng = np.random.default_rng(42)
    W = rng.standard_normal((10, 10))
    sv = singular_values(W)
    assert np.all(sv[:-1] >= sv[1:]), "Los valores singulares no están en orden descendente"


def test_transient_growth_curve_length():
    """
    Verifica que len(transient_growth_curve(W, kmax)) == kmax.

    Validates: Requirements 12.3, 3.1
    """
    W = 0.9 * np.eye(10)
    kmax = 30
    curve = transient_growth_curve(W, kmax=kmax)
    assert len(curve) == kmax


def test_transient_growth_argmax_range():
    """
    Verifica que transient_growth_argmax(W, kmax) devuelve un valor en [1, kmax].

    Validates: Requirements 12.4, 3.5
    """
    W = 0.9 * np.eye(10)
    kmax = 20
    argmax = transient_growth_argmax(W, kmax=kmax)
    assert 1 <= argmax <= kmax


def test_normal_diagonal_transient_growth():
    """
    Para W = rho*I, ||W^k||_2 = rho^k, que es máximo en k=1.
    Verifica que transient_growth_max ≈ rho.

    Validates: Requirements 3.4
    """
    rho = 0.9
    W = rho * np.eye(20)
    tgm = transient_growth_max(W, kmax=50)
    assert abs(tgm - rho) < 1e-10


def test_nonnormal_spectral_norm_exceeds_spectral_radius():
    """
    Para W = d*I + g*S con g=0.6 y N=50, verifica que sigma_max > rho.

    Validates: Requirements 3.1, 3.4
    """
    d = 0.9
    g = 0.6
    N = 50
    W = d * np.eye(N)
    for i in range(N - 1):
        W[i + 1, i] = g
    rho = compute_spectral_radius(W)
    sigma_max = compute_spectral_norm(W)
    assert sigma_max > rho


def test_transient_growth_curve_kmax_zero_raises():
    """
    Verifica que transient_growth_curve lanza ValueError cuando kmax=0.

    Validates: Requirements 12.5, 3.2
    """
    W = np.eye(5)
    with pytest.raises(ValueError):
        transient_growth_curve(W, kmax=0)


def test_transient_growth_curve_non_square_raises():
    """
    Verifica que transient_growth_curve lanza ValueError con una matriz no cuadrada.

    Validates: Requirements 12.6, 3.3
    """
    W = np.ones((3, 4))
    with pytest.raises(ValueError):
        transient_growth_curve(W, kmax=10)


# ---------------------------------------------------------------------------
# F.3 — Tests de tareas enabled/disabled
# ---------------------------------------------------------------------------

import dataclasses
from rc_lab.runners.multitask_sweep_runner import (
    MultiTaskAggregator,
    MCTaskSummary,
    MCConfigResult,
    RankingSpec,
)
from rc_lab.runners.sweep_runner import ConfigSummary, SweepSummary


def _make_config_summary(config_id: str, nmse_val: float, nmse_test: float) -> ConfigSummary:
    """Helper: construye un ConfigSummary mínimo con NMSE conocido."""
    return ConfigSummary(
        config_id=config_id,
        config_point={"spectral_radius": 0.9, "input_scaling": 0.1, "leak_rate": 1.0},
        n_seeds=3,
        val_mean={"nmse": nmse_val},
        val_std={"nmse": 0.01},
        test_mean={"nmse": nmse_test},
        test_std={"nmse": 0.01},
        best_ridge_mode=1e-6,
    )


def _make_sweep_summary(task_name: str, configs: list[ConfigSummary]) -> SweepSummary:
    """Helper: construye un SweepSummary mínimo."""
    return SweepSummary(
        sweep_name=f"test_{task_name}",
        n_configs=len(configs),
        n_seeds=configs[0].n_seeds if configs else 0,
        task_name=task_name,
        configs=configs,
        best_config_id=configs[0].config_id if configs else "",
        timestamp="2024-01-01T00:00:00+00:00",
    )


def _make_mc_summary(configs_data: list[tuple[str, float]]) -> MCTaskSummary:
    """Helper: construye un MCTaskSummary mínimo a partir de (config_id, mc_total) pairs."""
    configs = [
        MCConfigResult(
            config_id=cid,
            config_point={"spectral_radius": 0.9, "input_scaling": 0.1, "leak_rate": 1.0},
            n_seeds=3,
            mc_total_mean=mc_val,
            mc_total_std=0.5,
        )
        for cid, mc_val in configs_data
    ]
    return MCTaskSummary(
        sweep_name="test_mc",
        n_configs=len(configs),
        n_seeds=3,
        configs=configs,
        timestamp="2024-01-01T00:00:00+00:00",
    )


_RANKING_CONFIG = {
    "narma10": RankingSpec(metric="nmse", direction="min"),
    "mackey_glass": RankingSpec(metric="nmse", direction="min"),
    "memory_capacity": RankingSpec(metric="mc_total", direction="max"),
}


def test_backward_compat_no_enabled_field():
    """
    Validates: Requirements 12.7, 13.2, 13.3
    Una config sin campo `enabled` produce el mismo aggregate_rank que una
    config con todas las tareas marcadas como enabled=true.

    Verifica que MultiTaskAggregator sin enabled_tasks (comportamiento original)
    produce resultados idénticos a pasarle enabled_tasks con las tres tareas.
    """
    # Dos configs con valores conocidos
    narma10_configs = [
        _make_config_summary("cfg_A", nmse_val=0.10, nmse_test=0.11),
        _make_config_summary("cfg_B", nmse_val=0.20, nmse_test=0.21),
    ]
    mg_configs = [
        _make_config_summary("cfg_A", nmse_val=0.15, nmse_test=0.16),
        _make_config_summary("cfg_B", nmse_val=0.25, nmse_test=0.26),
    ]
    mc_data = [("cfg_A", 60.0), ("cfg_B", 40.0)]

    narma10_summary = _make_sweep_summary("narma10", narma10_configs)
    mg_summary = _make_sweep_summary("mackey_glass", mg_configs)
    mc_summary = _make_mc_summary(mc_data)

    # Aggregator sin enabled_tasks (comportamiento original — sin campo enabled)
    agg_no_enabled = MultiTaskAggregator(
        ranking_config=_RANKING_CONFIG,
        shortlist_top_n=2,
    )
    result_no_enabled = agg_no_enabled.aggregate(narma10_summary, mg_summary, mc_summary)

    # Aggregator con todas las tareas explícitamente habilitadas
    agg_all_enabled = MultiTaskAggregator(
        ranking_config=_RANKING_CONFIG,
        shortlist_top_n=2,
        enabled_tasks=["narma10", "mackey_glass", "memory_capacity"],
    )
    result_all_enabled = agg_all_enabled.aggregate(narma10_summary, mg_summary, mc_summary)

    # Los aggregate_ranks deben ser idénticos
    ranks_no_enabled = {e.config_id: e.aggregate_rank for e in result_no_enabled.configs}
    ranks_all_enabled = {e.config_id: e.aggregate_rank for e in result_all_enabled.configs}

    assert ranks_no_enabled.keys() == ranks_all_enabled.keys()
    for cid in ranks_no_enabled:
        assert ranks_no_enabled[cid] == pytest.approx(ranks_all_enabled[cid]), (
            f"aggregate_rank difiere para {cid}: "
            f"sin enabled={ranks_no_enabled[cid]}, "
            f"con enabled={ranks_all_enabled[cid]}"
        )


def test_aggregate_rank_only_active_tasks():
    """
    Validates: Requirements 1.3, 1.4
    Con narma10 y memory_capacity activos (mackey_glass deshabilitado),
    aggregate_rank = mean(rank_narma10, rank_mc), sin incluir rank_mg.
    """
    # Dos configs: cfg_A es mejor en narma10, cfg_B es mejor en mc
    narma10_configs = [
        _make_config_summary("cfg_A", nmse_val=0.10, nmse_test=0.11),  # rank 1 narma10
        _make_config_summary("cfg_B", nmse_val=0.20, nmse_test=0.21),  # rank 2 narma10
    ]
    mc_data = [
        ("cfg_A", 40.0),   # rank 2 mc
        ("cfg_B", 60.0),   # rank 1 mc
    ]

    narma10_summary = _make_sweep_summary("narma10", narma10_configs)
    mc_summary = _make_mc_summary(mc_data)

    # Solo narma10 y memory_capacity activos
    agg = MultiTaskAggregator(
        ranking_config=_RANKING_CONFIG,
        shortlist_top_n=2,
        enabled_tasks=["narma10", "memory_capacity"],
    )
    result = agg.aggregate(narma10_summary, mg_summary=None, mc_summary=mc_summary)

    entries_by_id = {e.config_id: e for e in result.configs}

    # cfg_A: rank_narma10=1, rank_mc=2 → aggregate_rank = (1+2)/2 = 1.5
    assert entries_by_id["cfg_A"].rank_narma10 == 1
    assert entries_by_id["cfg_A"].rank_mc == 2
    assert entries_by_id["cfg_A"].rank_mg is None
    assert entries_by_id["cfg_A"].aggregate_rank == pytest.approx(1.5)

    # cfg_B: rank_narma10=2, rank_mc=1 → aggregate_rank = (2+1)/2 = 1.5
    assert entries_by_id["cfg_B"].rank_narma10 == 2
    assert entries_by_id["cfg_B"].rank_mc == 1
    assert entries_by_id["cfg_B"].rank_mg is None
    assert entries_by_id["cfg_B"].aggregate_rank == pytest.approx(1.5)


def test_disabled_task_produces_none_summary():
    """
    Validates: Requirements 1.1, 1.2
    El summary de una tarea deshabilitada es None en el aggregator:
    los campos de mackey_glass son None cuando mg_summary=None y
    'mackey_glass' no está en enabled_tasks.
    """
    narma10_configs = [
        _make_config_summary("cfg_A", nmse_val=0.10, nmse_test=0.11),
        _make_config_summary("cfg_B", nmse_val=0.20, nmse_test=0.21),
    ]
    mc_data = [("cfg_A", 60.0), ("cfg_B", 40.0)]

    narma10_summary = _make_sweep_summary("narma10", narma10_configs)
    mc_summary = _make_mc_summary(mc_data)

    # mackey_glass deshabilitado: mg_summary=None, no en enabled_tasks
    agg = MultiTaskAggregator(
        ranking_config=_RANKING_CONFIG,
        shortlist_top_n=2,
        enabled_tasks=["narma10", "memory_capacity"],
    )
    result = agg.aggregate(narma10_summary, mg_summary=None, mc_summary=mc_summary)

    for entry in result.configs:
        # Todos los campos de mackey_glass deben ser None
        assert entry.mg_primary_metric is None, (
            f"mg_primary_metric debe ser None para tarea deshabilitada, "
            f"obtenido: {entry.mg_primary_metric!r}"
        )
        assert entry.mg_val_mean is None, (
            f"mg_val_mean debe ser None para tarea deshabilitada"
        )
        assert entry.mg_val_std is None
        assert entry.mg_test_mean is None
        assert entry.mg_test_std is None
        assert entry.rank_mg is None, (
            f"rank_mg debe ser None para tarea deshabilitada, "
            f"obtenido: {entry.rank_mg!r}"
        )


# ---------------------------------------------------------------------------
# F.2 (cont.) — Tests de reservoir_diagnostics() extendida
# ---------------------------------------------------------------------------

from rc_lab.reservoirs.diagnostics import reservoir_diagnostics

_EXPECTED_DIAG_KEYS = {
    "spectral_radius",
    "mean_abs_eigenvalue",
    "spectral_norm",
    "frobenius_norm",
    "density",
    "henrici_departure",
    "singular_value_max",
    "singular_value_min",
    "singular_value_mean",
    "singular_value_q90",
    "singular_condition_number",
    "transient_growth_max",
    "transient_growth_argmax",
}


def test_reservoir_diagnostics_keys():
    """
    Validates: Requirements 4.2
    reservoir_diagnostics() devuelve exactamente las 13 claves esperadas.
    """
    W = 0.9 * np.eye(10)
    result = reservoir_diagnostics(W)
    assert set(result.keys()) == _EXPECTED_DIAG_KEYS, (
        f"Claves inesperadas: {set(result.keys()) ^ _EXPECTED_DIAG_KEYS}"
    )
    assert len(result) == 13, (
        f"Se esperaban 13 claves, se obtuvieron {len(result)}"
    )


def test_reservoir_diagnostics_default_kmax():
    """
    Validates: Requirements 4.1, 4.3, 4.5
    Llamar a reservoir_diagnostics(W) sin transient_kmax produce el mismo
    resultado que llamar con transient_kmax=50.
    """
    rng = np.random.default_rng(0)
    N = 15
    W = rng.standard_normal((N, N)) * 0.1

    result_default = reservoir_diagnostics(W)
    result_explicit = reservoir_diagnostics(W, transient_kmax=50)

    assert result_default.keys() == result_explicit.keys()
    for key in result_default:
        val_default = result_default[key]
        val_explicit = result_explicit[key]
        # Handle inf values
        if val_default == float("inf") and val_explicit == float("inf"):
            continue
        assert val_default == pytest.approx(val_explicit, rel=1e-10), (
            f"Discrepancia en clave '{key}': default={val_default}, explicit={val_explicit}"
        )


def test_reservoir_diagnostics_no_recompute():
    """
    Validates: Requirements 4.4
    transient_growth_max y transient_growth_argmax son consistentes con
    transient_growth_curve calculada manualmente (misma W, mismo kmax).

    Verifica que reservoir_diagnostics no recomputa la curva de forma
    inconsistente: los valores de max y argmax deben coincidir con los
    obtenidos directamente de transient_growth_curve.
    """
    d = 0.9
    g = 0.3
    N = 20
    W = d * np.eye(N)
    for i in range(N - 1):
        W[i + 1, i] = g

    kmax = 50
    # Calcular la curva manualmente
    curve = transient_growth_curve(W, kmax=kmax)
    expected_max = float(np.max(curve))
    expected_argmax = int(np.argmax(curve)) + 1  # 1-indexed

    # Obtener los valores desde reservoir_diagnostics
    diag = reservoir_diagnostics(W, transient_kmax=kmax)

    assert diag["transient_growth_max"] == pytest.approx(expected_max, rel=1e-10), (
        f"transient_growth_max: esperado {expected_max}, obtenido {diag['transient_growth_max']}"
    )
    assert diag["transient_growth_argmax"] == expected_argmax, (
        f"transient_growth_argmax: esperado {expected_argmax}, obtenido {diag['transient_growth_argmax']}"
    )


# ---------------------------------------------------------------------------
# F.5 — Smoke test del experimento reducido (Req 12.8, 1.6, 1.7, 10.2, 10.3)
# ---------------------------------------------------------------------------

import json
import csv as csv_module
from rc_lab.runners.design_comparison_runner import DesignComparisonRunner


def test_smoke_nonnormal_comparison(tmp_path):
    """
    Validates: Requirements 12.8, 1.6, 1.7, 10.2, 10.3

    Smoke test del experimento reducido: 1 seed, grid mínimo, 2 diseños,
    mackey_glass deshabilitada. Verifica que DesignComparisonRunner produce
    comparison_summary.csv y comparison_summary.json sin errores, que el JSON
    contiene enabled_tasks=["narma10", "memory_capacity"] y que el CSV no
    contiene columnas mg_*.
    """
    cfg = {
        "sweep": {
            "name": "smoke_test",
            "output_dir": str(tmp_path / "smoke_test"),
            "seeds": [42],
        },
        "designs": [
            {
                "name": "random_sparse_baseline",
                "reservoir": {
                    "type": "random_sparse",
                    "N": 20,
                    "sparsity": 0.9,
                    "bias_scaling": 0.0,
                },
            },
            {
                "name": "nonnormal_chain_g0_3",
                "reservoir": {
                    "type": "nonnormal_chain",
                    "N": 20,
                    "chain_strength": 0.3,
                    "bias_scaling": 0.0,
                },
            },
        ],
        "grid": {
            "spectral_radius": [0.9],
            "input_scaling": [0.1],
            "leak_rate": [1.0],
        },
        "tasks": {
            "narma10": {
                "enabled": True,
                "n_train": 500,
                "n_val": 100,
                "n_test": 100,
                "washout": 50,
                "state_policy": "reset",
            },
            "mackey_glass": {
                "enabled": False,
                "n_train": 500,
                "n_val": 100,
                "n_test": 100,
                "washout": 50,
                "state_policy": "reset",
                "tau": 17,
                "dt": 0.1,
            },
            "memory_capacity": {
                "enabled": True,
                "washout": 50,
                "input_length": 500,
                "fit_fraction": 0.5,
                "kmax": 20,
                "ridge_param": 1e-6,
            },
        },
        "readout": {
            "type": "ridge",
            "features": "states",
            "ridge_candidates": [1e-6, 1e-4, 1e-2],
        },
        "metrics": ["nmse", "rmse"],
        "ranking": {
            "shortlist_top_n": 2,
            "narma10": {"metric": "nmse", "direction": "min"},
            "mackey_glass": {"metric": "nmse", "direction": "min"},
            "memory_capacity": {"metric": "mc_total", "direction": "max"},
        },
        "diagnostics": {"transient_kmax": 10},
    }

    # Ejecutar el runner — no debe lanzar excepciones
    runner = DesignComparisonRunner(cfg)
    table = runner.run()

    output_dir = tmp_path / "smoke_test"

    # Verificar que se producen los ficheros de salida
    csv_path = output_dir / "comparison_summary.csv"
    json_path = output_dir / "comparison_summary.json"
    assert csv_path.exists(), f"comparison_summary.csv no encontrado en {output_dir}"
    assert json_path.exists(), f"comparison_summary.json no encontrado en {output_dir}"

    # Verificar que el JSON contiene enabled_tasks = ["narma10", "memory_capacity"]
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    assert "enabled_tasks" in json_data, (
        "El JSON no contiene el campo 'enabled_tasks'"
    )
    assert json_data["enabled_tasks"] == ["narma10", "memory_capacity"], (
        f"enabled_tasks esperado ['narma10', 'memory_capacity'], "
        f"obtenido {json_data['enabled_tasks']!r}"
    )

    # Verificar que el CSV no contiene columnas mg_*
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv_module.DictReader(f)
        fieldnames = reader.fieldnames or []

    mg_columns_present = [col for col in fieldnames if col.startswith("mg_")]
    assert not mg_columns_present, (
        f"El CSV contiene columnas mg_* que deberían estar omitidas: {mg_columns_present}"
    )

    # Verificar que la tabla no está vacía
    assert len(table) > 0, "La tabla comparativa está vacía"
