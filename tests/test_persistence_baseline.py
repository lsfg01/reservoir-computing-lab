"""
Tests del baseline de persistencia como métrica de referencia por tarea.
"""

import numpy as np
import pytest

from rc_lab.metrics.persistence import (
    PersistenceResult,
    persistence_baseline,
    persistence_error,
)
from rc_lab.tasks.narma10 import Narma10Task
from rc_lab.tasks.mackey_glass import MackeyGlassTask
from rc_lab.tasks.delay_recall import DelayRecallTask


# ---------------------------------------------------------------------------
# Test 1 — Serie constante ⇒ error nulo
# ---------------------------------------------------------------------------

def test_persistence_constant_series_zero_error():
    """
    Persistencia sobre serie constante: ŷ(t) = y(t-1) = c, por lo que MSE = 0.
    NMSE no está definido (Var = 0) — se espera ValueError.
    """
    y = np.ones(100) * 3.7
    with pytest.raises(ValueError):
        persistence_error(y)


def test_persistence_nearly_constant_series_low_error():
    """
    Serie con varianza muy pequeña pero no nula: NMSE grande (señal casi plana,
    persistencia la sigue bien en MSE pero la norma es mínima).
    Solo verificamos que no hay crash y el resultado es finito.
    """
    rng = np.random.default_rng(0)
    y = np.ones(200) + rng.normal(0, 0.001, 200)
    result = persistence_error(y)
    assert np.isfinite(result.nmse)
    assert np.isfinite(result.rmse)
    assert np.isfinite(result.nrmse)


def test_persistence_known_formula():
    """
    Verifica la fórmula con valores exactos conocidos.
    y = [0, 1, 2], y_true = [1, 2], y_pred = [0, 1].
    MSE = ((1-0)^2 + (2-1)^2) / 2 = 1.0
    Var(y_true) = Var([1, 2]) = 0.25
    NMSE = 1.0 / 0.25 = 4.0; RMSE = 1.0; NRMSE = 2.0
    """
    y = np.array([0.0, 1.0, 2.0])
    result = persistence_error(y)
    assert result.nmse == pytest.approx(4.0, rel=1e-9)
    assert result.rmse == pytest.approx(1.0, rel=1e-9)
    assert result.nrmse == pytest.approx(2.0, rel=1e-9)
    assert result.n_samples == 2


# ---------------------------------------------------------------------------
# Test 2 — NARMA-10: NRMSE en el orden esperado (≈ 0.8)
# ---------------------------------------------------------------------------

def test_persistence_narma10_nrmse_order_of_magnitude():
    """
    Persistencia sobre NARMA-10 generado por el laboratorio.
    NRMSE esperado ≈ 0.8 (literatura RC). Verificamos orden de magnitud:
    debe estar en el intervalo (0.6, 1.0).
    """
    task = Narma10Task(state_policy="reset")
    data = task.generate(n_train=1000, n_val=500, n_test=500, washout=100, seed=42)

    result = persistence_baseline(data, task_name="narma10", split="test")
    assert result is not None
    assert 0.6 < result.nrmse < 1.0, (
        f"NRMSE de persistencia en NARMA-10 = {result.nrmse:.4f}, "
        f"esperado en (0.6, 1.0)"
    )


def test_persistence_narma10_val_and_test_both_work():
    """
    persistence_baseline funciona tanto en split='test' como en split='val'.
    """
    task = Narma10Task(state_policy="reset")
    data = task.generate(n_train=500, n_val=200, n_test=200, washout=50, seed=7)

    res_test = persistence_baseline(data, task_name="narma10", split="test")
    res_val = persistence_baseline(data, task_name="narma10", split="val")

    assert res_test is not None
    assert res_val is not None
    assert np.isfinite(res_test.nmse)
    assert np.isfinite(res_val.nmse)


# ---------------------------------------------------------------------------
# Test 3 — Alineamiento temporal correcto (sin fuga)
# ---------------------------------------------------------------------------

def test_persistence_temporal_alignment_no_leakage():
    """
    Verifica que la predicción usa y(t-1) y no y(t):
    construimos una serie donde solo la alineación correcta da NMSE < 1.

    Usamos y = [0, 1, 0, 1, ...] (alternante): la persistencia predice el
    opuesto del valor actual → MSE = 1.0 por muestra, Var = 0.25 → NMSE = 4.
    Si hubiera fuga (usar y(t)), NMSE sería 0.
    """
    T = 100
    y = np.array([float(i % 2) for i in range(T)])
    result = persistence_error(y)
    # Con y alternante: ŷ(t) = y(t-1) = 1 - y(t), el predictor es sistemáticamente
    # opuesto → NMSE >> 1. Si hubiera fuga (usar y(t)), NMSE sería 0.
    assert result.nmse > 1.0, (
        f"NMSE={result.nmse} debería ser >1 con alineación correcta (predictor opuesto)"
    )
    assert result.n_samples == T - 1


def test_persistence_n_samples_correct():
    """
    El número de muestras evaluadas es T-1 (se pierde el primer paso por el lag).
    """
    y = np.random.default_rng(0).uniform(0, 1, 50)
    result = persistence_error(y)
    assert result.n_samples == 49


# ---------------------------------------------------------------------------
# Test 4 — delay_recall: devuelve None (persistencia no aplica)
# ---------------------------------------------------------------------------

def test_persistence_delay_recall_returns_none():
    """
    Para delay_recall, persistence_baseline devuelve None porque la persistencia
    clásica no tiene semántica correcta sobre targets multisalida de retardos iid.
    La referencia implícita es corr² = 0 (suelo de la distribución iid).
    """
    task = DelayRecallTask(kmax=10)
    data = task.generate(n_train=300, n_val=100, n_test=100, washout=50, seed=0)

    result = persistence_baseline(data, task_name="delay_recall", split="test")
    assert result is None


def test_persistence_multisalida_raises():
    """
    persistence_error con array multisalida (más de una columna) lanza ValueError.
    """
    y = np.ones((50, 3))
    with pytest.raises(ValueError, match="escalar"):
        persistence_error(y)


# ---------------------------------------------------------------------------
# Validación de argumentos
# ---------------------------------------------------------------------------

def test_persistence_error_too_short_raises():
    with pytest.raises(ValueError, match="2 muestras"):
        persistence_error(np.array([1.0]))


def test_persistence_baseline_invalid_split_raises():
    task = Narma10Task()
    data = task.generate(n_train=100, n_val=50, n_test=50, washout=10, seed=0)
    with pytest.raises(ValueError, match="split"):
        persistence_baseline(data, task_name="narma10", split="train")


def test_persistence_baseline_val_missing_raises():
    """
    Pedir split='val' cuando n_val=0 lanza ValueError.
    """
    task = Narma10Task(state_policy="carryover")
    data = task.generate(n_train=100, n_val=0, n_test=50, washout=10, seed=0)
    with pytest.raises(ValueError, match="validación"):
        persistence_baseline(data, task_name="narma10", split="val")


# ---------------------------------------------------------------------------
# Determinismo
# ---------------------------------------------------------------------------

def test_persistence_deterministic():
    """
    Misma seed ⇒ mismo NMSE de persistencia.
    """
    task = Narma10Task(state_policy="reset")
    data1 = task.generate(n_train=300, n_val=100, n_test=100, washout=50, seed=99)
    data2 = task.generate(n_train=300, n_val=100, n_test=100, washout=50, seed=99)

    res1 = persistence_baseline(data1, task_name="narma10", split="test")
    res2 = persistence_baseline(data2, task_name="narma10", split="test")

    assert res1 is not None and res2 is not None
    assert res1.nmse == res2.nmse
    assert res1.rmse == res2.rmse


# ---------------------------------------------------------------------------
# Mackey-Glass (smoke test de comparabilidad)
# ---------------------------------------------------------------------------

def test_persistence_mackey_glass_finite():
    """
    Persistencia sobre Mackey-Glass devuelve valores finitos.
    """
    task = MackeyGlassTask(state_policy="reset")
    data = task.generate(n_train=500, n_val=200, n_test=200, washout=100, seed=0)

    result = persistence_baseline(data, task_name="mackey_glass", split="test")
    assert result is not None
    assert np.isfinite(result.nmse)
    assert np.isfinite(result.nrmse)
    assert result.nrmse > 0.0
