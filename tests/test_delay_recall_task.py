import numpy as np
import pytest

from rc_lab.evaluators.memory_capacity import MemoryCapacityEvaluator
from rc_lab.metrics.memory import memory_corr_total
from rc_lab.models.esn import ESNModel
from rc_lab.readouts.ridge import RidgeReadout
from rc_lab.reservoirs.random_sparse import RandomSparseReservoir
from rc_lab.tasks.delay_recall import DelayRecallTask, delay_recall_valid_start


def test_delay_recall_shapes_reset():
    task = DelayRecallTask(kmax=4, input_low=-1.0, input_high=1.0, state_policy="reset")
    data = task.generate(n_train=8, n_val=5, n_test=6, washout=4, seed=42)

    assert data.washout == 8
    assert data.u_train.shape == (16, 1)
    assert data.y_train.shape == (16, 4)
    assert data.u_val.shape == (5, 1)
    assert data.y_val.shape == (5, 4)
    assert data.u_test.shape == (6, 1)
    assert data.y_test.shape == (6, 4)
    assert data.u_val_full.shape == (13, 1)
    assert data.u_test_full.shape == (14, 1)


def test_delay_recall_targets_are_shifted():
    kmax = 3
    washout = 3
    task = DelayRecallTask(kmax=kmax, state_policy="reset")
    data = task.generate(n_train=6, n_val=0, n_test=4, washout=washout, seed=123)
    effective_washout = delay_recall_valid_start(washout, kmax)

    for t in range(effective_washout, effective_washout + 6):
        expected = [data.u_train[t - k, 0] for k in range(1, kmax + 1)]
        assert np.allclose(data.y_train[t], expected)

    for t in range(effective_washout, effective_washout + 4):
        expected = [data.u_test_full[t - k, 0] for k in range(1, kmax + 1)]
        assert np.allclose(data.y_test[t - effective_washout], expected)


def test_delay_recall_requires_washout_at_least_kmax():
    task = DelayRecallTask(kmax=5)
    with pytest.raises(ValueError, match="washout >= kmax"):
        task.generate(n_train=10, n_val=2, n_test=2, washout=4, seed=1)


def test_delay_recall_matches_memory_capacity_for_esn():
    seed = 123
    reservoir_seed = 7
    washout = 20
    kmax = 5
    input_length = 220
    ridge_param = 1e-6
    fit_fraction = 0.5

    matrices = RandomSparseReservoir(
        spectral_radius=0.8,
        input_scaling=0.5,
        sparsity=0.7,
        bias_scaling=0.0,
    ).build(N=20, n_inputs=1, seed=reservoir_seed)
    esn = ESNModel(matrices.W, matrices.Win, matrices.bias, leak_rate=1.0)

    mc = MemoryCapacityEvaluator(
        washout=washout,
        input_length=input_length,
        fit_fraction=fit_fraction,
        kmax=kmax,
        ridge_param=ridge_param,
    ).evaluate_details(esn, seed=seed)

    task = DelayRecallTask(kmax=kmax, state_policy="carryover")
    data = task.generate(
        n_train=mc.fit_samples,
        n_val=mc.eval_samples,
        n_test=0,
        washout=washout,
        seed=seed,
    )
    assert data.washout == washout + kmax

    X_train, x_end = esn.run_states(data.u_train, washout=data.washout)
    Y_train = data.y_train[data.washout:]
    readout = RidgeReadout(ridge_param=ridge_param)
    readout.fit(X_train, Y_train)

    X_val, _ = esn.run_states(data.u_val, washout=0, x0=x_end)
    y_pred = readout.predict(X_val)

    assert memory_corr_total(data.y_val, y_pred) == pytest.approx(mc.mc_total, rel=1e-10, abs=1e-10)
