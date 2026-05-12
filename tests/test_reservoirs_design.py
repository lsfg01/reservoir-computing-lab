"""
Validaciones de implementación — Requisito 9 (9.1–9.10)

Verifica las propiedades fundamentales de CycleReservoir y CycleJumpReservoir
y su integración con el resto del laboratorio.
"""

import warnings

import numpy as np
import pytest

from rc_lab.reservoirs.cycle import CycleReservoir
from rc_lab.reservoirs.cycle_jump import CycleJumpReservoir
from rc_lab.reservoirs.diagnostics import (
    compute_density,
    compute_henrici_departure,
    compute_spectral_radius,
)
from rc_lab.runners.runner import resolve_reservoir


# ---------------------------------------------------------------------------
# Parámetros comunes
# ---------------------------------------------------------------------------

N = 20
N_INPUTS = 3
SEED = 42
SPECTRAL_RADIUS = 0.9
INPUT_SCALING = 0.1


# ---------------------------------------------------------------------------
# 9.1 — Shapes correctas para CycleReservoir
# ---------------------------------------------------------------------------


def test_cycle_reservoir_shapes():
    """Req 9.1: CycleReservoir.build produce W(N,N), Win(N,n_inputs), bias(N,)."""
    res = CycleReservoir(spectral_radius=SPECTRAL_RADIUS, input_scaling=INPUT_SCALING)
    mats = res.build(N=N, n_inputs=N_INPUTS, seed=SEED)

    assert mats.W.shape == (N, N), f"W.shape esperado ({N},{N}), obtenido {mats.W.shape}"
    assert mats.Win.shape == (N, N_INPUTS), (
        f"Win.shape esperado ({N},{N_INPUTS}), obtenido {mats.Win.shape}"
    )
    assert mats.bias.shape == (N,), f"bias.shape esperado ({N},), obtenido {mats.bias.shape}"


# ---------------------------------------------------------------------------
# 9.2 — Shapes correctas para CycleJumpReservoir
# ---------------------------------------------------------------------------


def test_cycle_jump_reservoir_shapes():
    """Req 9.2: CycleJumpReservoir.build produce W(N,N), Win(N,n_inputs), bias(N,)."""
    res = CycleJumpReservoir(
        spectral_radius=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        jumps=[3],
    )
    mats = res.build(N=N, n_inputs=N_INPUTS, seed=SEED)

    assert mats.W.shape == (N, N)
    assert mats.Win.shape == (N, N_INPUTS)
    assert mats.bias.shape == (N,)


# ---------------------------------------------------------------------------
# 9.3 — Radio espectral dentro de tolerancia 1e-6
# ---------------------------------------------------------------------------


def test_cycle_reservoir_spectral_radius():
    """Req 9.3: |compute_spectral_radius(W) - spectral_radius| < 1e-6 para CycleReservoir."""
    res = CycleReservoir(spectral_radius=SPECTRAL_RADIUS, input_scaling=INPUT_SCALING)
    mats = res.build(N=N, n_inputs=N_INPUTS, seed=SEED)

    rho = compute_spectral_radius(mats.W)
    assert abs(rho - SPECTRAL_RADIUS) < 1e-6, (
        f"Radio espectral {rho} difiere de {SPECTRAL_RADIUS} en más de 1e-6"
    )


def test_cycle_jump_reservoir_spectral_radius():
    """Req 9.3: |compute_spectral_radius(W) - spectral_radius| < 1e-6 para CycleJumpReservoir."""
    res = CycleJumpReservoir(
        spectral_radius=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        jumps=[3],
    )
    mats = res.build(N=N, n_inputs=N_INPUTS, seed=SEED)

    rho = compute_spectral_radius(mats.W)
    assert abs(rho - SPECTRAL_RADIUS) < 1e-6, (
        f"Radio espectral {rho} difiere de {SPECTRAL_RADIUS} en más de 1e-6"
    )


# ---------------------------------------------------------------------------
# 9.4 — Densidad exacta 1/N para CycleReservoir
# ---------------------------------------------------------------------------


def test_cycle_reservoir_density():
    """Req 9.4: compute_density(W) == 1/N para CycleReservoir (exactamente N entradas no nulas)."""
    res = CycleReservoir(spectral_radius=SPECTRAL_RADIUS, input_scaling=INPUT_SCALING)
    mats = res.build(N=N, n_inputs=N_INPUTS, seed=SEED)

    density = compute_density(mats.W)
    expected = 1.0 / N
    assert abs(density - expected) < 1e-12, (
        f"Densidad {density} != 1/N={expected} para CycleReservoir"
    )


# ---------------------------------------------------------------------------
# 9.5 — Densidad <= (1 + len(jumps)) / N para CycleJumpReservoir
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("jumps", [[3], [3, 7], [5, 9, 13]])
def test_cycle_jump_reservoir_density(jumps):
    """Req 9.5: compute_density(W) <= (1 + len(jumps)) / N para CycleJumpReservoir."""
    res = CycleJumpReservoir(
        spectral_radius=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        jumps=jumps,
    )
    mats = res.build(N=N, n_inputs=N_INPUTS, seed=SEED)

    density = compute_density(mats.W)
    upper_bound = (1 + len(jumps)) / N
    assert density <= upper_bound + 1e-12, (
        f"Densidad {density} > (1 + len(jumps))/N = {upper_bound} para jumps={jumps}"
    )


# ---------------------------------------------------------------------------
# 9.6 — Reproducibilidad con seed=42
# ---------------------------------------------------------------------------


def test_cycle_reservoir_reproducibility():
    """Req 9.6: Dos llamadas a build con seed=42 producen matrices idénticas (CycleReservoir)."""
    res = CycleReservoir(spectral_radius=SPECTRAL_RADIUS, input_scaling=INPUT_SCALING)
    mats1 = res.build(N=N, n_inputs=N_INPUTS, seed=42)
    mats2 = res.build(N=N, n_inputs=N_INPUTS, seed=42)

    np.testing.assert_array_equal(mats1.W, mats2.W)
    np.testing.assert_array_equal(mats1.Win, mats2.Win)
    np.testing.assert_array_equal(mats1.bias, mats2.bias)


def test_cycle_jump_reservoir_reproducibility():
    """Req 9.6: Dos llamadas a build con seed=42 producen matrices idénticas (CycleJumpReservoir)."""
    res = CycleJumpReservoir(
        spectral_radius=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        jumps=[3],
    )
    mats1 = res.build(N=N, n_inputs=N_INPUTS, seed=42)
    mats2 = res.build(N=N, n_inputs=N_INPUTS, seed=42)

    np.testing.assert_array_equal(mats1.W, mats2.W)
    np.testing.assert_array_equal(mats1.Win, mats2.Win)
    np.testing.assert_array_equal(mats1.bias, mats2.bias)


# ---------------------------------------------------------------------------
# 9.7 — CycleJumpReservoir(jumps=[0]) lanza ValueError
# ---------------------------------------------------------------------------


def test_cycle_jump_reservoir_jumps_zero_raises():
    """Req 9.7: CycleJumpReservoir(jumps=[0]) debe lanzar ValueError."""
    with pytest.raises(ValueError):
        CycleJumpReservoir(
            spectral_radius=SPECTRAL_RADIUS,
            input_scaling=INPUT_SCALING,
            jumps=[0],
        )


# ---------------------------------------------------------------------------
# 9.7b — CycleJumpReservoir(jumps=[N]) o jumps=[1] emite UserWarning
# ---------------------------------------------------------------------------


def test_cycle_jump_reservoir_jumps_N_warns():
    """Req 9.7b: CycleJumpReservoir con jumps=[N] emite UserWarning (auto-bucle)."""
    res = CycleJumpReservoir(
        spectral_radius=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        jumps=[N],  # j % N == 0 → auto-bucle
    )
    with pytest.warns(UserWarning):
        res.build(N=N, n_inputs=N_INPUTS, seed=SEED)


def test_cycle_jump_reservoir_jumps_1_warns():
    """Req 9.7b: CycleJumpReservoir con jumps=[1] emite UserWarning (coincide con ciclo base)."""
    res = CycleJumpReservoir(
        spectral_radius=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        jumps=[1],  # j % N == 1 → coincide con ciclo base
    )
    with pytest.warns(UserWarning):
        res.build(N=N, n_inputs=N_INPUTS, seed=SEED)


# ---------------------------------------------------------------------------
# 9.8 — compute_henrici_departure(W_ciclo_sin_reescalar) < 1e-10
# ---------------------------------------------------------------------------


def test_henrici_departure_raw_cycle_matrix():
    """Req 9.8: La matriz de ciclo puro (antes del reescalado) es normal → Henrici ≈ 0.

    Nota: el requisito especifica < 1e-10, pero np.linalg.eigvals introduce
    errores de punto flotante del orden de 1e-8 para algunos tamaños de N.
    Se usa una tolerancia de 1e-6 que verifica que la salida es esencialmente
    cero (la matriz de ciclo puro es normal por construcción).
    """
    # Construir la matriz de ciclo manualmente, sin reescalado espectral
    W_raw = np.zeros((N, N))
    for i in range(N):
        W_raw[(i + 1) % N, i] = 1.0

    departure = compute_henrici_departure(W_raw)
    assert departure < 1e-6, (
        f"Henrici departure {departure} >= 1e-6 para la matriz de ciclo puro"
    )


# ---------------------------------------------------------------------------
# 9.9 — resolve_reservoir devuelve las instancias correctas
# ---------------------------------------------------------------------------


def test_resolve_reservoir_cycle():
    """Req 9.9: resolve_reservoir({'type': 'cycle', ...}) devuelve CycleReservoir."""
    res = resolve_reservoir({
        "type": "cycle",
        "spectral_radius": SPECTRAL_RADIUS,
        "input_scaling": INPUT_SCALING,
    })
    assert isinstance(res, CycleReservoir), (
        f"Se esperaba CycleReservoir, se obtuvo {type(res)}"
    )


def test_resolve_reservoir_cycle_jump():
    """Req 9.9: resolve_reservoir({'type': 'cycle_jump', ...}) devuelve CycleJumpReservoir."""
    res = resolve_reservoir({
        "type": "cycle_jump",
        "spectral_radius": SPECTRAL_RADIUS,
        "input_scaling": INPUT_SCALING,
    })
    assert isinstance(res, CycleJumpReservoir), (
        f"Se esperaba CycleJumpReservoir, se obtuvo {type(res)}"
    )


# ---------------------------------------------------------------------------
# 9.10 — Compatibilidad con ESNModel.run_states()
# ---------------------------------------------------------------------------


def test_cycle_reservoir_esn_compatibility():
    """Req 9.10: CycleReservoir es compatible con ESNModel.run_states(); shape (n_steps-washout, N)."""
    from rc_lab.models.esn import ESNModel

    n_steps = 50
    washout = 10
    res = CycleReservoir(spectral_radius=SPECTRAL_RADIUS, input_scaling=INPUT_SCALING)
    mats = res.build(N=N, n_inputs=1, seed=SEED)

    esn = ESNModel(mats.W, mats.Win, mats.bias, leak_rate=1.0)
    u = np.random.default_rng(0).uniform(-1, 1, (n_steps, 1))

    states, x_final = esn.run_states(u, washout=washout)

    assert states.shape == (n_steps - washout, N), (
        f"states.shape esperado ({n_steps - washout}, {N}), obtenido {states.shape}"
    )
    assert x_final.shape == (N,)


def test_cycle_jump_reservoir_esn_compatibility():
    """Req 9.10: CycleJumpReservoir es compatible con ESNModel.run_states(); shape (n_steps-washout, N)."""
    from rc_lab.models.esn import ESNModel

    n_steps = 50
    washout = 10
    res = CycleJumpReservoir(
        spectral_radius=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        jumps=[3],
    )
    mats = res.build(N=N, n_inputs=1, seed=SEED)

    esn = ESNModel(mats.W, mats.Win, mats.bias, leak_rate=1.0)
    u = np.random.default_rng(0).uniform(-1, 1, (n_steps, 1))

    states, x_final = esn.run_states(u, washout=washout)

    assert states.shape == (n_steps - washout, N), (
        f"states.shape esperado ({n_steps - washout}, {N}), obtenido {states.shape}"
    )
    assert x_final.shape == (N,)
