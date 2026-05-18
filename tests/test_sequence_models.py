import numpy as np
import pytest

from rc_lab.sequence_models.tapped_delay import TappedDelayRidge
from rc_lab.tasks.base import TaskData
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
            "grad_clip": 1.0,
            "training_mode": "windowed",
            "bptt_length": 8,
            "batch_size": 2,
            "window_stride": 4,
            "window_washout": 2,
            "normalize_inputs": True,
            "normalize_targets": True,
        },
        metrics=["memory_corr_total", "rmse"],
        seed=7,
        device="cpu",
        evaluate_test=True,
        task_name="delay_recall",
        task_cfg={"kmax": 2},
    )

    assert result["y_pred_val"].shape == data.y_val.shape
    assert result["y_pred_test"].shape == data.y_test.shape
    assert result["metadata"]["epochs_ran"] >= 1
    assert result["metadata"]["n_total_params"] > 0
    assert result["metadata"]["status"] == "ok"
    assert result["metadata"]["training_mode"] == "windowed"
    assert result["metadata"]["normalized_inputs"] is True
    assert result["metadata"]["normalized_targets"] is True
    assert result["metadata"]["grad_clip"] == pytest.approx(1.0)
    assert "fit_s" in result["timing"]
    assert "final_test_s" in result["timing"]


def test_torch_normalization_uses_train_only_and_inverse_transform():
    from rc_lab.sequence_models.torch_models import _apply_normalization, _fit_normalization

    data = TaskData(
        u_train=np.array([[0.0], [2.0], [4.0], [6.0]]),
        y_train=np.array([[10.0], [12.0], [14.0], [16.0]]),
        u_val=np.array([[100.0]]),
        y_val=np.array([[50.0]]),
        u_test=np.array([[200.0]]),
        y_test=np.array([[80.0]]),
        washout=1,
        state_policy="carryover",
    )

    norm = _fit_normalization(data, washout=data.washout, normalize_inputs=True, normalize_targets=True)
    normalized = _apply_normalization(data, norm)

    assert norm.input_mean.tolist() == pytest.approx([3.0])
    assert norm.target_mean.tolist() == pytest.approx([14.0])
    assert normalized.u_val[0, 0] == pytest.approx((100.0 - 3.0) / norm.input_std[0])
    assert norm.inverse_y(normalized.y_val)[0, 0] == pytest.approx(50.0)


def test_torch_metrics_are_reported_on_original_scale_when_normalized():
    pytest.importorskip("torch")
    from rc_lab.metrics.error import rmse
    from rc_lab.sequence_models.torch_models import fit_torch_sequence_model

    task = DelayRecallTask(kmax=1)
    data = task.generate(n_train=18, n_val=6, n_test=6, washout=1, seed=11)

    result = fit_torch_sequence_model(
        kind="torch_lstm",
        task_data=data,
        config_point={
            "hidden_size": 3,
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "num_layers": 1,
            "dropout": 0.0,
            "max_epochs": 1,
            "patience": 1,
            "normalize_inputs": True,
            "normalize_targets": True,
        },
        metrics=["rmse"],
        seed=11,
        device="cpu",
        evaluate_test=False,
        task_name="delay_recall",
        task_cfg={"kmax": 1},
    )

    assert result["metadata"]["status"] == "ok"
    assert result["val_metrics"]["rmse"] == pytest.approx(rmse(data.y_val, result["y_pred_val"]))


def test_windowed_dataset_shapes():
    from rc_lab.sequence_models.torch_models import _build_windowed_arrays

    task = DelayRecallTask(kmax=2)
    data = task.generate(n_train=20, n_val=5, n_test=5, washout=3, seed=5)
    u_windows, y_windows, masks = _build_windowed_arrays(
        data,
        bptt_length=8,
        window_stride=4,
        window_washout=2,
    )

    assert u_windows.ndim == 3
    assert u_windows.shape[1:] == (8, 1)
    assert y_windows.shape[1:] == (8, 2)
    assert masks.shape == (u_windows.shape[0], 8, 1)
    assert np.all(masks.sum(axis=(1, 2)) > 0)


def test_torch_non_finite_loss_is_reported():
    pytest.importorskip("torch")
    from rc_lab.sequence_models.torch_models import fit_torch_sequence_model

    task = DelayRecallTask(kmax=1)
    data = task.generate(n_train=12, n_val=4, n_test=4, washout=1, seed=13)
    data.u_train[2, 0] = np.nan

    result = fit_torch_sequence_model(
        kind="torch_simple_rnn",
        task_data=data,
        config_point={
            "hidden_size": 3,
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "num_layers": 1,
            "max_epochs": 2,
            "patience": 1,
        },
        metrics=["nmse", "memory_corr_total"],
        seed=13,
        device="cpu",
        evaluate_test=True,
    )

    assert result["metadata"]["status"] == "failed_non_finite_loss"
    assert result["test_metrics"] == {}
    assert np.isposinf(result["val_metrics"]["nmse"])
    assert np.isneginf(result["val_metrics"]["memory_corr_total"])


def test_torch_timeout_is_reported():
    pytest.importorskip("torch")
    from rc_lab.sequence_models.torch_models import fit_torch_sequence_model

    task = DelayRecallTask(kmax=1)
    data = task.generate(n_train=14, n_val=4, n_test=4, washout=1, seed=17)

    result = fit_torch_sequence_model(
        kind="torch_simple_rnn",
        task_data=data,
        config_point={
            "hidden_size": 3,
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "num_layers": 1,
            "max_epochs": 5,
            "patience": 5,
            "max_train_seconds_per_run": 0.0,
        },
        metrics=["nmse"],
        seed=17,
        device="cpu",
        evaluate_test=True,
    )

    assert result["metadata"]["status"] == "timeout"
    assert result["test_metrics"] == {}
    assert np.isposinf(result["val_metrics"]["nmse"])
