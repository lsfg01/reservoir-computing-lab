import numpy as np
import pytest

from rc_lab.metrics.memory import (
    corr2_by_delay,
    memory_corr_total,
    memory_eff_total,
    nmse_by_delay,
)


def test_perfect_prediction_has_full_memory_totals():
    rng = np.random.default_rng(42)
    y_true = rng.normal(size=(200, 5))
    y_pred = y_true.copy()

    assert memory_corr_total(y_true, y_pred) == pytest.approx(5.0)
    assert memory_eff_total(y_true, y_pred) == pytest.approx(5.0)
    assert np.allclose(corr2_by_delay(y_true, y_pred), np.ones(5))
    assert np.allclose(nmse_by_delay(y_true, y_pred), np.zeros(5))


def test_memory_metrics_handle_zero_variance_without_crashing():
    y_true = np.ones((20, 3))
    y_pred = np.zeros((20, 3))

    corr = corr2_by_delay(y_true, y_pred)
    nmse = nmse_by_delay(y_true, y_pred)

    assert np.allclose(corr, np.zeros(3))
    assert np.isinf(nmse).all()
    assert memory_corr_total(y_true, y_pred) == pytest.approx(0.0)
    assert memory_eff_total(y_true, y_pred) == pytest.approx(0.0)

