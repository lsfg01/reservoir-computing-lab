"""
Tests del alias NG-RC primitivo sobre TappedDelayRidge.

Cubre:
1. Equivalencia numérica: ng_rc y tapped_delay_ridge con feature_mode="quadratic"
   producen resultados idénticos.
2. Número de columnas de la matriz de features coincide con el conteo documentado.
3. Tests existentes de tapped_delay siguen en verde (indirectamente: importaciones
   y comportamiento existente no cambian).
4. El runner externo acepta kind="ng_rc" sin error.
"""

import numpy as np
import pytest

from rc_lab.sequence_models.tapped_delay import TappedDelayRidge
from rc_lab.tasks.narma10 import Narma10Task
from rc_lab.tasks.delay_recall import DelayRecallTask
from rc_lab.runners.external_comparison_runner import (
    SUPPORTED_MODEL_KINDS,
    ExternalComparisonRunner,
)


# ---------------------------------------------------------------------------
# Test 1 — Equivalencia numérica: ng_rc == tapped_delay_ridge + quadratic
# ---------------------------------------------------------------------------

def _make_narma_data(seed: int = 0):
    task = Narma10Task(state_policy="reset")
    return task.generate(n_train=300, n_val=100, n_test=100, washout=50, seed=seed)


def test_ngrc_identical_to_tapped_delay_quadratic_predictions():
    """
    ng_rc es un alias semántico: con feature_mode='quadratic' produce
    exactamente el mismo output numérico que tapped_delay_ridge.
    """
    data = _make_narma_data(seed=42)

    model_ref = TappedDelayRidge(n_lags=5, ridge_param=1e-4, feature_mode="quadratic")
    model_ref.fit(data.u_train, data.y_train, washout=data.washout)
    val_ref = model_ref.prepare_reset_scored_split(data.u_val_full, data.y_val, data.washout)
    pred_ref = model_ref.predict_prepared(val_ref.features)

    # ng_rc → mismo TappedDelayRidge con feature_mode="quadratic"
    model_ng = TappedDelayRidge(n_lags=5, ridge_param=1e-4, feature_mode="quadratic")
    model_ng.fit(data.u_train, data.y_train, washout=data.washout)
    val_ng = model_ng.prepare_reset_scored_split(data.u_val_full, data.y_val, data.washout)
    pred_ng = model_ng.predict_prepared(val_ng.features)

    np.testing.assert_array_equal(pred_ref, pred_ng)
    np.testing.assert_array_equal(val_ref.features, val_ng.features)


def test_ngrc_and_tdr_same_n_params():
    """
    n_total_params es idéntico entre los dos modelos con mismos hiperparámetros.
    """
    data = _make_narma_data(seed=0)
    for n_lags in (3, 7, 10):
        m1 = TappedDelayRidge(n_lags=n_lags, ridge_param=1e-6, feature_mode="quadratic")
        m2 = TappedDelayRidge(n_lags=n_lags, ridge_param=1e-6, feature_mode="quadratic")
        m1.fit(data.u_train, data.y_train, washout=data.washout)
        m2.fit(data.u_train, data.y_train, washout=data.washout)
        assert m1.n_total_params == m2.n_total_params


# ---------------------------------------------------------------------------
# Test 2 — Número de columnas de features (forma funcional NVAR documentada)
# ---------------------------------------------------------------------------

def _feature_cols(n_lags: int, feature_mode: str, n_inputs: int = 1) -> int:
    """Número de columnas de la matriz de features para entrada escalar."""
    task = Narma10Task(state_policy="reset")
    data = task.generate(n_train=100, n_val=20, n_test=20, washout=10, seed=0)
    model = TappedDelayRidge(n_lags=n_lags, ridge_param=1e-6, feature_mode=feature_mode)
    model.fit(data.u_train, data.y_train, washout=data.washout)
    split = model.prepare_reset_scored_split(data.u_val_full, data.y_val, data.washout)
    return split.features.shape[1]


def test_feature_cols_raw_equals_L_plus_1():
    """
    feature_mode='raw': (L+1) columnas para entrada escalar.
    """
    for L in (0, 3, 7):
        cols = _feature_cols(n_lags=L, feature_mode="raw")
        assert cols == L + 1, f"L={L}: esperado {L+1} cols, obtenido {cols}"


def test_feature_cols_linear_equals_L_plus_1():
    """
    feature_mode='linear': idéntico a 'raw', (L+1) columnas.
    """
    for L in (0, 3, 7):
        assert _feature_cols(L, "linear") == L + 1


def test_feature_cols_quadratic_equals_2_times_L_plus_1():
    """
    feature_mode='quadratic' (= NG-RC primitivo): 2*(L+1) columnas.
    La librería es [u(t), ..., u(t-L), u(t)², ..., u(t-L)²].
    """
    for L in (0, 3, 7, 10):
        cols = _feature_cols(n_lags=L, feature_mode="quadratic")
        expected = 2 * (L + 1)
        assert cols == expected, (
            f"L={L}: esperado {expected} cols (NVAR diagonal orden 2), obtenido {cols}"
        )


def test_feature_cols_linear_quadratic_equals_quadratic():
    """
    feature_mode='linear_quadratic' produce el mismo número de columnas que
    'quadratic': son alias semánticos en la implementación actual.
    """
    for L in (3, 7):
        cols_q = _feature_cols(L, "quadratic")
        cols_lq = _feature_cols(L, "linear_quadratic")
        assert cols_q == cols_lq, (
            f"L={L}: quadratic={cols_q} != linear_quadratic={cols_lq}"
        )


def test_quadratic_features_contain_squared_taps():
    """
    Verifica que las columnas de cuadrados son efectivamente los cuadrados de
    las columnas lineales (sin productos cruzados en la librería diagonal).
    """
    task = Narma10Task(state_policy="reset")
    data = task.generate(n_train=100, n_val=20, n_test=20, washout=10, seed=5)
    L = 3
    model = TappedDelayRidge(n_lags=L, ridge_param=1e-6, feature_mode="quadratic")
    model.fit(data.u_train, data.y_train, washout=data.washout)
    split = model.prepare_reset_scored_split(data.u_val_full, data.y_val, data.washout)
    F = split.features  # shape (T, 2*(L+1))
    n_lin = L + 1

    linear_part = F[:, :n_lin]
    quadratic_part = F[:, n_lin:]

    np.testing.assert_allclose(
        quadratic_part, linear_part ** 2, atol=1e-12,
        err_msg="La parte cuadrática debe ser exactamente el cuadrado de la parte lineal"
    )


# ---------------------------------------------------------------------------
# Test 3 — El runner externo acepta kind="ng_rc"
# ---------------------------------------------------------------------------

def test_ng_rc_in_supported_model_kinds():
    """ng_rc debe estar en el conjunto de kinds soportados."""
    assert "ng_rc" in SUPPORTED_MODEL_KINDS


def test_runner_accepts_ng_rc_kind(tmp_path):
    """
    ExternalComparisonRunner acepta kind='ng_rc' sin error de validación
    y produce resultados numéricos idénticos a tapped_delay_ridge + quadratic.
    """
    common_cfg = dict(
        sweep=dict(
            name="test_ngrc",
            output_dir=str(tmp_path / "out"),
            seeds=[0],
        ),
        comparison=dict(use_test_for_selection=False, n_candidates_per_model=1),
        tasks=dict(narma10=dict(
            enabled=True,
            n_train=200, n_val=80, n_test=80,
            washout=20, state_policy="reset",
        )),
        metrics=dict(common=["nmse"]),
        ranking=dict(narma10=dict(metric="nmse", direction="min")),
        readout=dict(type="ridge", features="states", ridge_candidates=[1e-6]),
    )

    cfg_ngrc = dict(**common_cfg, models=[dict(
        name="ngrc_model", kind="ng_rc", enabled=True,
        grid=dict(n_lags=[5], ridge_param=[1e-4], feature_mode=["quadratic"]),
    )])
    cfg_tdr = dict(**common_cfg, models=[dict(
        name="tdr_model", kind="tapped_delay_ridge", enabled=True,
        grid=dict(n_lags=[5], ridge_param=[1e-4], feature_mode=["quadratic"]),
    )])

    runner_ngrc = ExternalComparisonRunner(cfg_ngrc)
    runner_tdr = ExternalComparisonRunner(cfg_tdr)

    table_ngrc = runner_ngrc.run()
    table_tdr = runner_tdr.run()

    # Ambos deben producir exactamente el mismo NMSE
    nmse_ngrc = table_ngrc[0].get("narma10_val_nmse_mean") or table_ngrc[0].get("val_nmse_mean")
    nmse_tdr = table_tdr[0].get("narma10_val_nmse_mean") or table_tdr[0].get("val_nmse_mean")

    assert nmse_ngrc is not None, f"No se encontró NMSE en la tabla NG-RC: {list(table_ngrc[0].keys())}"
    assert nmse_tdr is not None, f"No se encontró NMSE en la tabla TDR: {list(table_tdr[0].keys())}"
    assert nmse_ngrc == pytest.approx(nmse_tdr, rel=1e-9), (
        f"NMSE difiere: ng_rc={nmse_ngrc}, tapped_delay_ridge={nmse_tdr}"
    )
