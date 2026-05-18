from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from rc_lab.tasks.base import TaskData
from rc_lab.utils.timing import timer


_MAXIMIZED_METRICS = {
    "corr2_by_delay",
    "memory_corr_total",
    "memory_eff_total",
    "max_delay_corr_above_threshold",
}


def _require_torch() -> Any:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for kind='torch_simple_rnn' and kind='torch_lstm'. "
            "Install torch in the active environment to run these models."
        ) from exc
    return torch, nn


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def set_torch_seed(seed: int) -> None:
    torch, _ = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_torch_sequence_model(
    kind: str,
    task_data: TaskData,
    config_point: dict[str, Any],
    metrics: list[str],
    seed: int,
    device: str = "cpu",
    evaluate_test: bool = False,
    task_name: str | None = None,
    task_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Train a SimpleRNN/LSTM and evaluate validation/test in the original scale.

    The legacy full-sequence protocol remains the default. New external configs
    can opt into windowed training with truncated BPTT via training_mode.
    """
    from rc_lab.sequence_models.training import compute_metrics

    torch, nn = _require_torch()
    set_torch_seed(seed)
    requested_device = device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    input_size = int(task_data.u_train.shape[1])
    output_size = int(task_data.y_train.shape[1])
    hidden_size = int(config_point.get("hidden_size", 32))
    num_layers = int(config_point.get("num_layers", 1))
    dropout = float(config_point.get("dropout", 0.0))
    dropout_eff = dropout if num_layers > 1 else 0.0
    max_epochs = int(config_point.get("max_epochs", 100))
    patience = int(config_point.get("patience", 10))
    grad_clip = config_point.get("grad_clip")
    grad_clip_value = None if grad_clip is None else float(grad_clip)
    training_mode = str(config_point.get("training_mode", "full_sequence"))
    if training_mode not in {"full_sequence", "windowed"}:
        raise ValueError("training_mode must be 'full_sequence' or 'windowed'")
    normalize_inputs = bool(config_point.get("normalize_inputs", False))
    normalize_targets = bool(config_point.get("normalize_targets", False))
    max_train_seconds = config_point.get("max_train_seconds_per_run")
    max_train_seconds = None if max_train_seconds is None else float(max_train_seconds)

    washout = int(task_data.washout)
    normalization = _fit_normalization(
        task_data=task_data,
        washout=washout,
        normalize_inputs=normalize_inputs,
        normalize_targets=normalize_targets,
    )
    model_data = _apply_normalization(task_data, normalization)

    model = _SequenceRegressor(
        kind=kind,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout_eff,
        nn=nn,
    ).to(dev)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config_point.get("learning_rate", 1e-3)),
        weight_decay=float(config_point.get("weight_decay", 0.0)),
    )

    val_inputs, val_targets = _validation_tensors(model_data, torch, dev)

    bptt_length = None
    batch_size = None
    window_stride = None
    window_washout = None
    window_count = None
    train_u = None
    train_y = None
    window_u = None
    window_y = None
    window_mask = None

    if training_mode == "windowed":
        bptt_length = int(config_point.get("bptt_length", min(100, len(model_data.u_train))))
        bptt_length = max(1, min(bptt_length, len(model_data.u_train)))
        batch_size = max(1, int(config_point.get("batch_size", 32)))
        window_stride = max(1, int(config_point.get("window_stride", max(1, bptt_length // 2))))
        window_washout = _resolve_window_washout(
            raw=config_point.get("window_washout", "auto"),
            task_name=task_name,
            task_cfg=task_cfg or {},
            bptt_length=bptt_length,
            output_size=output_size,
        )
        u_windows, y_windows, masks = _build_windowed_arrays(
            model_data,
            bptt_length=bptt_length,
            window_stride=window_stride,
            window_washout=window_washout,
        )
        window_count = int(u_windows.shape[0])
        if window_count:
            window_u = torch.as_tensor(u_windows, dtype=torch.float32, device=dev)
            window_y = torch.as_tensor(y_windows, dtype=torch.float32, device=dev)
            window_mask = torch.as_tensor(masks, dtype=torch.float32, device=dev)
    else:
        train_u = _tensor(model_data.u_train, torch, dev)
        train_y = _tensor(model_data.y_train[washout:], torch, dev)

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_ran = 0
    final_train_loss = None
    status = "ok"

    timing: dict[str, float] = {}
    with timer() as t_train:
        train_start = time.perf_counter()
        if training_mode == "windowed" and not window_count:
            status = "failed_no_train_windows"
        else:
            for epoch in range(1, max_epochs + 1):
                model.train()
                optimizer.zero_grad()

                if training_mode == "windowed":
                    assert window_u is not None
                    assert window_y is not None
                    assert window_mask is not None
                    epoch_loss, status = _run_windowed_epoch(
                        model=model,
                        optimizer=optimizer,
                        window_u=window_u,
                        window_y=window_y,
                        window_mask=window_mask,
                        batch_size=int(batch_size),
                        grad_clip=grad_clip_value,
                        torch=torch,
                        nn=nn,
                    )
                    final_train_loss = epoch_loss
                    if status != "ok":
                        epochs_ran = epoch
                        break
                else:
                    assert train_u is not None
                    assert train_y is not None
                    pred_train, _ = model(train_u)
                    loss = nn.functional.mse_loss(pred_train[:, washout:, :], train_y)
                    final_train_loss = float(loss.detach().cpu().item())
                    if not torch.isfinite(loss):
                        status = "failed_non_finite_loss"
                        epochs_ran = epoch
                        break
                    loss.backward()
                    if grad_clip_value is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    pred_val = _predict_val(model, model_data, val_inputs, torch, dev)
                    val_loss_t = nn.functional.mse_loss(pred_val, val_targets)
                    val_loss = float(val_loss_t.detach().cpu().item())
                if not torch.isfinite(val_loss_t) or not math.isfinite(val_loss):
                    status = "failed_non_finite_loss"
                    epochs_ran = epoch
                    break

                epochs_ran = epoch
                if val_loss < best_val - 1e-12:
                    best_val = val_loss
                    best_epoch = epoch
                    best_state = deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break

                if (
                    max_train_seconds is not None
                    and time.perf_counter() - train_start > max_train_seconds
                ):
                    status = "timeout"
                    break
    timing["train_s"] = t_train["elapsed"]
    timing["fit_s"] = timing["train_s"]

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata = _metadata(
        status=status,
        epochs_ran=epochs_ran,
        best_epoch=best_epoch,
        max_epochs=max_epochs,
        best_val=best_val,
        final_train_loss=final_train_loss,
        dev=dev,
        requested_device=requested_device,
        n_total=n_total,
        n_trainable=n_trainable,
        grad_clip=grad_clip_value,
        training_mode=training_mode,
        bptt_length=bptt_length,
        batch_size=batch_size,
        window_stride=window_stride,
        window_washout=window_washout,
        window_count=window_count,
        normalization=normalization,
    )

    if status != "ok":
        timing["eval_s"] = 0.0
        timing["validation_s"] = 0.0
        timing["tuning_s"] = timing["fit_s"]
        timing["total_s"] = timing["train_s"]
        return {
            "val_metrics": _bad_metrics(metrics),
            "test_metrics": {},
            "timing": timing,
            "metadata": metadata,
            "y_pred_val": None,
            "y_pred_test": None,
        }

    model.load_state_dict(best_state)
    model.eval()

    y_pred_test = None
    test_metrics = {}
    with torch.no_grad():
        with timer() as t_val_eval:
            pred_val_t = _predict_val(model, model_data, val_inputs, torch, dev)
            y_pred_val_norm = pred_val_t.squeeze(0).detach().cpu().numpy()
            y_val_norm = val_targets.squeeze(0).detach().cpu().numpy()
            y_pred_val = normalization.inverse_y(y_pred_val_norm)
            y_val = normalization.inverse_y(y_val_norm)

        if evaluate_test:
            with timer() as t_test:
                pred_test_t, y_test_t = _predict_test(model, model_data, torch, dev)
                y_pred_test_norm = pred_test_t.squeeze(0).detach().cpu().numpy()
                y_test_norm = y_test_t.squeeze(0).detach().cpu().numpy()
                y_pred_test = normalization.inverse_y(y_pred_test_norm)
                y_test = normalization.inverse_y(y_test_norm)
                test_metrics = compute_metrics(y_test, y_pred_test, metrics)
            timing["final_test_s"] = t_test["elapsed"]
            timing["test_s"] = timing["final_test_s"]
            timing["selected_total_s"] = timing["fit_s"] + timing["final_test_s"]
    timing["validation_s"] = t_val_eval["elapsed"]
    timing["tuning_s"] = timing["fit_s"] + timing["validation_s"]
    timing["eval_s"] = timing["validation_s"] + timing.get("final_test_s", 0.0)
    timing["total_s"] = timing["train_s"] + timing["eval_s"]

    return {
        "val_metrics": compute_metrics(y_val, y_pred_val, metrics),
        "test_metrics": test_metrics,
        "timing": timing,
        "metadata": metadata,
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
    }


class _SequenceRegressor:
    def __new__(
        cls,
        kind: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float,
        nn: Any,
    ) -> Any:
        class SequenceRegressor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if kind == "torch_simple_rnn":
                    self.recurrent = nn.RNN(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        batch_first=True,
                        nonlinearity="tanh",
                    )
                elif kind == "torch_lstm":
                    self.recurrent = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        batch_first=True,
                    )
                else:
                    raise ValueError(f"Unsupported torch model kind: {kind!r}")
                self.readout = nn.Linear(hidden_size, output_size)

            def forward(self, x: Any, h0: Any = None) -> tuple[Any, Any]:
                out, h = self.recurrent(x, h0)
                return self.readout(out), h

        return SequenceRegressor()


@dataclass
class _Normalization:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    normalize_inputs: bool
    normalize_targets: bool

    def transform_u(self, value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if not self.normalize_inputs:
            return arr.copy()
        return (arr - self.input_mean) / self.input_std

    def transform_y(self, value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if not self.normalize_targets:
            return arr.copy()
        return (arr - self.target_mean) / self.target_std

    def inverse_y(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if not self.normalize_targets:
            return arr
        return arr * self.target_std + self.target_mean


def _fit_normalization(
    task_data: TaskData,
    washout: int,
    normalize_inputs: bool,
    normalize_targets: bool,
) -> _Normalization:
    input_mean = np.mean(task_data.u_train, axis=0)
    input_std = _safe_std(task_data.u_train)
    target_train = task_data.y_train[washout:] if washout < len(task_data.y_train) else task_data.y_train
    target_mean = np.mean(target_train, axis=0)
    target_std = _safe_std(target_train)
    return _Normalization(
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        normalize_inputs=normalize_inputs,
        normalize_targets=normalize_targets,
    )


def _apply_normalization(task_data: TaskData, normalization: _Normalization) -> TaskData:
    return TaskData(
        u_train=normalization.transform_u(task_data.u_train),
        y_train=normalization.transform_y(task_data.y_train),
        u_val=normalization.transform_u(task_data.u_val),
        y_val=normalization.transform_y(task_data.y_val),
        u_test=normalization.transform_u(task_data.u_test),
        y_test=normalization.transform_y(task_data.y_test),
        washout=task_data.washout,
        state_policy=task_data.state_policy,
        u_val_full=normalization.transform_u(task_data.u_val_full),
        u_test_full=normalization.transform_u(task_data.u_test_full),
    )


def _safe_std(value: np.ndarray) -> np.ndarray:
    std = np.std(value, axis=0)
    return np.where(std == 0.0, 1.0, std)


def _tensor(array: np.ndarray, torch: Any, device: Any) -> Any:
    return torch.as_tensor(array, dtype=torch.float32, device=device).unsqueeze(0)


def _validation_tensors(task_data: TaskData, torch: Any, device: Any) -> tuple[Any, Any]:
    if task_data.y_val is None or task_data.u_val is None:
        raise ValueError("Validation split is required for external comparison")
    if task_data.u_val_full is not None:
        return (
            _tensor(task_data.u_val_full, torch, device),
            _tensor(task_data.y_val, torch, device),
        )
    return (
        _tensor(task_data.u_val, torch, device),
        _tensor(task_data.y_val, torch, device),
    )


def _predict_val(model: Any, task_data: TaskData, val_inputs: Any, torch: Any, device: Any) -> Any:
    if task_data.u_val_full is not None:
        pred, _ = model(val_inputs)
        return pred[:, task_data.washout :, :]

    train_inputs = _tensor(task_data.u_train, torch, device)
    _, h_train = model(train_inputs)
    pred, _ = model(val_inputs, h_train)
    return pred


def _predict_test(model: Any, task_data: TaskData, torch: Any, device: Any) -> tuple[Any, Any]:
    if task_data.u_test_full is not None:
        test_inputs = _tensor(task_data.u_test_full, torch, device)
        pred, _ = model(test_inputs)
        return pred[:, task_data.washout :, :], _tensor(task_data.y_test, torch, device)

    train_inputs = _tensor(task_data.u_train, torch, device)
    _, h_train = model(train_inputs)
    if task_data.u_val is not None:
        val_inputs = _tensor(task_data.u_val, torch, device)
        _, h_val = model(val_inputs, h_train)
    else:
        h_val = h_train
    test_inputs = _tensor(task_data.u_test, torch, device)
    pred, _ = model(test_inputs, h_val)
    return pred, _tensor(task_data.y_test, torch, device)


def _resolve_window_washout(
    raw: Any,
    task_name: str | None,
    task_cfg: dict[str, Any],
    bptt_length: int,
    output_size: int,
) -> int:
    if raw is None or raw == "auto":
        if task_name == "delay_recall":
            value = min(int(task_cfg.get("kmax", output_size)), bptt_length // 2)
        elif task_name == "narma10":
            value = int(task_cfg.get("window_washout", 10))
        elif task_name == "mackey_glass":
            value = int(task_cfg.get("window_washout", 20))
        else:
            value = min(10, bptt_length // 2)
    else:
        value = int(raw)
    return max(0, min(value, max(0, bptt_length - 1)))


def _build_windowed_arrays(
    task_data: TaskData,
    bptt_length: int,
    window_stride: int,
    window_washout: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bptt_length <= 0:
        raise ValueError("bptt_length must be > 0")
    if window_stride <= 0:
        raise ValueError("window_stride must be > 0")

    total = len(task_data.u_train)
    bptt = min(int(bptt_length), total)
    starts = list(range(0, total - bptt + 1, int(window_stride)))
    last_start = total - bptt
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    elif not starts:
        starts = [0]

    u_windows: list[np.ndarray] = []
    y_windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for start in starts:
        score_start = max(int(window_washout), int(task_data.washout) - start, 0)
        if score_start >= bptt:
            continue
        stop = start + bptt
        mask = np.zeros((bptt, 1), dtype=float)
        mask[score_start:] = 1.0
        u_windows.append(task_data.u_train[start:stop])
        y_windows.append(task_data.y_train[start:stop])
        masks.append(mask)

    if not u_windows:
        return (
            np.empty((0, bptt, task_data.u_train.shape[1]), dtype=float),
            np.empty((0, bptt, task_data.y_train.shape[1]), dtype=float),
            np.empty((0, bptt, 1), dtype=float),
        )
    return (
        np.stack(u_windows, axis=0),
        np.stack(y_windows, axis=0),
        np.stack(masks, axis=0),
    )


def _run_windowed_epoch(
    model: Any,
    optimizer: Any,
    window_u: Any,
    window_y: Any,
    window_mask: Any,
    batch_size: int,
    grad_clip: float | None,
    torch: Any,
    nn: Any,
) -> tuple[float | None, str]:
    n_windows = int(window_u.shape[0])
    order = torch.randperm(n_windows, device=window_u.device)
    loss_sum = 0.0
    weight_sum = 0
    for start in range(0, n_windows, batch_size):
        idx = order[start : start + batch_size]
        optimizer.zero_grad()
        pred, _ = model(window_u[idx])
        mask = window_mask[idx]
        denom = mask.sum() * pred.shape[-1]
        loss = (((pred - window_y[idx]) ** 2) * mask).sum() / denom
        if not torch.isfinite(loss):
            return float(loss.detach().cpu().item()), "failed_non_finite_loss"
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_n = int(idx.numel())
        loss_sum += float(loss.detach().cpu().item()) * batch_n
        weight_sum += batch_n
    return loss_sum / max(weight_sum, 1), "ok"


def _bad_metrics(metric_names: list[str]) -> dict[str, float]:
    return {
        name: float("-inf") if name in _MAXIMIZED_METRICS else float("inf")
        for name in metric_names
    }


def _metadata(
    status: str,
    epochs_ran: int,
    best_epoch: int,
    max_epochs: int,
    best_val: float,
    final_train_loss: float | None,
    dev: Any,
    requested_device: str,
    n_total: int,
    n_trainable: int,
    grad_clip: float | None,
    training_mode: str,
    bptt_length: int | None,
    batch_size: int | None,
    window_stride: int | None,
    window_washout: int | None,
    window_count: int | None,
    normalization: _Normalization,
) -> dict[str, Any]:
    return {
        "status": status,
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch),
        "early_stopped": status == "ok" and epochs_ran < max_epochs,
        "final_train_loss": final_train_loss,
        "best_val_loss": best_val,
        "grad_clip": grad_clip,
        "training_mode": training_mode,
        "bptt_length": bptt_length,
        "batch_size": batch_size,
        "window_stride": window_stride,
        "window_washout": window_washout,
        "window_count": window_count,
        "validation_mode": "full_sequence",
        "normalized_inputs": normalization.normalize_inputs,
        "normalized_targets": normalization.normalize_targets,
        "input_mean": normalization.input_mean.tolist(),
        "input_std": normalization.input_std.tolist(),
        "target_mean": normalization.target_mean.tolist(),
        "target_std": normalization.target_std.tolist(),
        "device": str(dev),
        "requested_device": requested_device,
        "n_total_params": int(n_total),
        "n_trainable_params": int(n_trainable),
    }
