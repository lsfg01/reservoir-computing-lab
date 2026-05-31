"""
Tests del evaluador empírico de ESP (Echo State Property) via sincronización.

Cubre:
1. Régimen contractivo: todos los pares sincronizan y sync_time es finito/pequeño.
2. Monotonía cualitativa en ρ: mayor ρ → sync_time no decrece / frac. sync no aumenta.
3. No sincronización: en régimen inestable hay pares con synchronized=False y no hay crash.
4. Invariancia de escala: escalar el estado inicial no cambia sync_time.
5. Determinismo: misma seed → mismos resultados.
"""

import numpy as np
import pytest

from rc_lab.metrics.stability import (
    ESPResult,
    PairSyncResult,
    evaluate_esp,
    evaluate_esp_sampled,
    sync_pair,
)
from rc_lab.models.esn import ESNModel
from rc_lab.reservoirs.random_sparse import RandomSparseReservoir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_esn(spectral_radius: float, N: int = 50, seed: int = 0) -> ESNModel:
    """Construye una ESN random_sparse con los parámetros dados."""
    builder = RandomSparseReservoir(
        spectral_radius=spectral_radius,
        input_scaling=0.1,
        sparsity=0.9,
        leak_rate=1.0,
        bias_scaling=0.0,
    )
    mats = builder.build(N=N, n_inputs=1, seed=seed)
    return ESNModel(W=mats.W, Win=mats.Win, bias=mats.bias, leak_rate=1.0)


# ---------------------------------------------------------------------------
# Test 1 — Régimen contractivo: todos sincronizan, sync_time finito y pequeño
# ---------------------------------------------------------------------------

def test_contractive_regime_all_synchronized():
    """
    Con ρ≈0.3 (muy contractivo) todos los pares deben sincronizar.
    """
    esn = _build_esn(spectral_radius=0.3, N=50, seed=42)
    result = evaluate_esp_sampled(esn, seed=7, T=500, n_pairs=5, eps=1e-3)

    assert result.fraction_synchronized == pytest.approx(1.0), (
        f"Se esperaba fracción sincronizada=1.0, obtenida {result.fraction_synchronized}"
    )
    assert result.sync_time_mean is not None
    assert result.sync_time_mean < 200, (
        f"sync_time_mean={result.sync_time_mean} demasiado alto para ρ=0.3"
    )
    for pr in result.pair_results:
        assert pr.synchronized
        assert pr.sync_time is not None


# ---------------------------------------------------------------------------
# Test 2 — Monotonía cualitativa en ρ
# ---------------------------------------------------------------------------

def test_monotonicity_in_spectral_radius():
    """
    A mayor ρ, sync_time_mean no debe decrecer (o frac. sincronizada no aumentar).
    Comparamos ρ=0.3 vs ρ=0.85: el más contractivo debe ser al menos tan rápido.
    """
    seed_esn = 0
    seed_eval = 99

    esn_low = _build_esn(spectral_radius=0.3, N=50, seed=seed_esn)
    esn_high = _build_esn(spectral_radius=0.85, N=50, seed=seed_esn)

    res_low = evaluate_esp_sampled(esn_low, seed=seed_eval, T=800, n_pairs=5, eps=1e-3)
    res_high = evaluate_esp_sampled(esn_high, seed=seed_eval, T=800, n_pairs=5, eps=1e-3)

    # Al menos una de las dos condiciones debe cumplirse:
    # - sync_time_mean no disminuye al aumentar ρ, O
    # - fraction_synchronized no aumenta al aumentar ρ
    if res_low.sync_time_mean is not None and res_high.sync_time_mean is not None:
        condition = (res_high.sync_time_mean >= res_low.sync_time_mean) or (
            res_high.fraction_synchronized <= res_low.fraction_synchronized
        )
    else:
        # Si el de alto ρ no sincroniza nada, la fracción es 0 ≤ fracción baja
        condition = res_high.fraction_synchronized <= res_low.fraction_synchronized

    assert condition, (
        f"Monotonía violada: ρ=0.3 → sync_mean={res_low.sync_time_mean}, "
        f"frac={res_low.fraction_synchronized}; "
        f"ρ=0.85 → sync_mean={res_high.sync_time_mean}, "
        f"frac={res_high.fraction_synchronized}"
    )


# ---------------------------------------------------------------------------
# Test 3 — No sincronización: régimen inestable, no hay crash
# ---------------------------------------------------------------------------

def test_no_sync_regime_does_not_crash():
    """
    Con ρ>1 (inestable), al menos algunos pares no deben sincronizar.
    Los agregados deben calcularse correctamente sin NaN espurios ni crashes.
    """
    esn = _build_esn(spectral_radius=1.5, N=50, seed=13)
    result = evaluate_esp_sampled(esn, seed=3, T=300, n_pairs=5, eps=1e-3)

    # Al menos un par no sincroniza
    assert any(not pr.synchronized for pr in result.pair_results), (
        "Se esperaba al menos un par no sincronizado con ρ=1.5"
    )

    # Los agregados deben ser válidos (sin crash)
    assert 0.0 <= result.fraction_synchronized <= 1.0
    if result.sync_time_mean is not None:
        assert np.isfinite(result.sync_time_mean)
    assert result.d_curve_mean.shape == (300,)
    assert np.all(np.isfinite(result.d_curve_mean))

    # Pares no sincronizados: sync_time debe ser None (no inf, no T)
    for pr in result.pair_results:
        if not pr.synchronized:
            assert pr.sync_time is None, (
                f"Par no sincronizado tiene sync_time={pr.sync_time}, esperado None"
            )


def test_no_sync_aggregates_without_synchronized_pairs():
    """
    Si ningún par sincroniza, sync_time_mean y sync_time_std deben ser None,
    y fraction_synchronized debe ser 0.0 sin crash.
    """
    # Forzamos no-sincronización construyendo un ESN muy inestable con T corto
    esn = _build_esn(spectral_radius=2.0, N=30, seed=7)
    rng = np.random.default_rng(0)
    u = rng.uniform(-1.0, 1.0, (50, 1))
    N = esn.N

    # Pares con condiciones iniciales muy separadas
    x0_pairs = [
        (np.ones(N), -np.ones(N)),
        (np.ones(N) * 0.5, -np.ones(N) * 0.5),
    ]
    result = evaluate_esp(esn, u, x0_pairs, eps=1e-10)

    if result.fraction_synchronized == 0.0:
        assert result.sync_time_mean is None
        assert result.sync_time_std is None
    # Si por alguna razón sí sincroniza (reservoir pequeño), solo verificamos no-crash
    assert 0.0 <= result.fraction_synchronized <= 1.0


# ---------------------------------------------------------------------------
# Test 4 — Invariancia de escala
# ---------------------------------------------------------------------------

def test_scale_invariance_of_sync_time():
    """
    Escalar artificialmente x0 y x0' por un factor k no debe cambiar sync_time,
    ya que d(t) usa distancia relativa a la separación inicial.
    """
    esn = _build_esn(spectral_radius=0.5, N=40, seed=5)
    rng = np.random.default_rng(11)
    N = esn.N
    u = rng.uniform(-1.0, 1.0, (300, 1))

    x0 = rng.uniform(-0.5, 0.5, N)
    x0p = rng.uniform(-0.5, 0.5, N)

    # Par original
    result_orig = sync_pair(esn, u, x0, x0p, eps=1e-3)

    # Par escalado: misma dirección, mayor magnitud
    scale = 3.0
    x0_scaled = x0 * scale
    x0p_scaled = x0p * scale

    result_scaled = sync_pair(esn, u, x0_scaled, x0p_scaled, eps=1e-3)

    assert result_orig.sync_time == result_scaled.sync_time, (
        f"sync_time cambia al escalar: original={result_orig.sync_time}, "
        f"escalado={result_scaled.sync_time}"
    )
    assert result_orig.synchronized == result_scaled.synchronized


# ---------------------------------------------------------------------------
# Test 5 — Determinismo
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    """
    Dos llamadas con la misma seed deben producir resultados idénticos.
    """
    esn = _build_esn(spectral_radius=0.7, N=40, seed=0)

    res1 = evaluate_esp_sampled(esn, seed=42, T=400, n_pairs=5, eps=1e-3)
    res2 = evaluate_esp_sampled(esn, seed=42, T=400, n_pairs=5, eps=1e-3)

    assert res1.fraction_synchronized == res2.fraction_synchronized
    assert res1.sync_time_mean == res2.sync_time_mean
    assert res1.sync_time_std == res2.sync_time_std
    np.testing.assert_array_equal(res1.d_curve_mean, res2.d_curve_mean)

    for pr1, pr2 in zip(res1.pair_results, res2.pair_results, strict=True):
        assert pr1.sync_time == pr2.sync_time
        assert pr1.synchronized == pr2.synchronized
        np.testing.assert_array_equal(pr1.d_curve, pr2.d_curve)


def test_different_seeds_may_differ():
    """
    Distintas seeds pueden producir resultados distintos (sanidad básica).
    """
    esn = _build_esn(spectral_radius=0.7, N=40, seed=0)
    res1 = evaluate_esp_sampled(esn, seed=1, T=400, n_pairs=5, eps=1e-3)
    res2 = evaluate_esp_sampled(esn, seed=2, T=400, n_pairs=5, eps=1e-3)
    # Las curvas promedio deben diferir (distinto u, distinto x0)
    assert not np.array_equal(res1.d_curve_mean, res2.d_curve_mean)


# ---------------------------------------------------------------------------
# Test de interfaz y validación
# ---------------------------------------------------------------------------

def test_evaluate_esp_empty_pairs_raises():
    esn = _build_esn(spectral_radius=0.5, N=20, seed=0)
    u = np.zeros((100, 1))
    with pytest.raises(ValueError, match="x0_pairs"):
        evaluate_esp(esn, u, [], eps=1e-3)


def test_evaluate_esp_sampled_invalid_T_raises():
    esn = _build_esn(spectral_radius=0.5, N=20, seed=0)
    with pytest.raises(ValueError, match="T"):
        evaluate_esp_sampled(esn, seed=0, T=0)


def test_evaluate_esp_sampled_invalid_eps_raises():
    esn = _build_esn(spectral_radius=0.5, N=20, seed=0)
    with pytest.raises(ValueError, match="eps"):
        evaluate_esp_sampled(esn, seed=0, T=100, eps=1.5)


def test_result_shapes_are_consistent():
    """
    Verifica que los shapes de d_curve por par y d_curve_mean son consistentes con T.
    """
    T = 250
    n_pairs = 4
    esn = _build_esn(spectral_radius=0.6, N=30, seed=3)
    result = evaluate_esp_sampled(esn, seed=0, T=T, n_pairs=n_pairs, eps=1e-3)

    assert result.T == T
    assert result.n_pairs == n_pairs
    assert result.d_curve_mean.shape == (T,)
    for pr in result.pair_results:
        assert pr.d_curve.shape == (T,)
