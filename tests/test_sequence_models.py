import numpy as np
import pytest

from rc_lab.sequence_models.tapped_delay import TappedDelayRidge
from rc_lab.tasks.delay_recall import DelayRecallTask


def test_tapped_delay_ridge_fit_predict_shape_and_params():
    task = DelayRecallTask(kmax=2)
    data = task.generate(n_train=30, n_val=10, n_test=10, washout=3, seed=42)

    model = TappedDelayRidge(n_lags=2, ridge_param=1e-6, feature_mode="raw")
    model.fit(data.u_train, data.y_train, washout=data.washout)
    val_split = model.prepare_reset_scored_split(data.u_val_full, data.y_val, data.washout)
    pred = model.predict_prepared(val_split.features)

    assert pred.shape == data.y_val.shape
    assert model.n_total_params == 6
    assert model.n_trainable_params == 6


def test_simple_rnn_smoke_when_torch_is_available():
    pytest.importorskip("torch")
    from rc_lab.sequence_models.torch_models import fit_torch_sequence_model

    task = DelayRecallTask(kmax=2)
    data = task.generate(n_train=20, n_val=8, n_test=8, washout=2, seed=7)

    result = fit_torch_sequence_model(
        kind="torch_simple_rnn",
        task_data=data,
        config_point={
            "hidden_size": 4,
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "num_layers": 1,
            "max_epochs": 2,
            "patience": 1,
        },
        metrics=["memory_corr_total"],
        seed=7,
        device="cpu",
        evaluate_test=True,
    )

    assert result["y_pred_val"].shape == data.y_val.shape
    assert result["y_pred_test"].shape == data.y_test.shape
    assert result["metadata"]["epochs_ran"] >= 1
    assert result["metadata"]["n_total_params"] > 0

