from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from rc_lab.tasks.base import TaskData
from rc_lab.utils.timing import timer


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
) -> dict[str, Any]:
    """
    Train a SimpleRNN/LSTM on one full sequence and evaluate validation/test.

    Test predictions are computed only when evaluate_test=True.
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

    max_epochs = int(config_point.get("max_epochs", 100))
    patience = int(config_point.get("patience", 10))
    washout = int(task_data.washout)
    train_u = _tensor(task_data.u_train, torch, dev)
    train_y = _tensor(task_data.y_train[washout:], torch, dev)
    val_inputs, val_targets = _validation_tensors(task_data, torch, dev)

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_ran = 0

    timing: dict[str, float] = {}
    with timer() as t_train:
        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred_train, _ = model(train_u)
            loss = nn.functional.mse_loss(pred_train[:, washout:, :], train_y)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_val = _predict_val(model, task_data, val_inputs, torch, dev)
                val_loss = nn.functional.mse_loss(pred_val, val_targets).item()

            epochs_ran = epoch
            if val_loss < best_val - 1e-12:
                best_val = float(val_loss)
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
    timing["train_s"] = t_train["elapsed"]

    model.load_state_dict(best_state)
    model.eval()

    with timer() as t_eval:
        with torch.no_grad():
            pred_val_t = _predict_val(model, task_data, val_inputs, torch, dev)
            y_pred_val = pred_val_t.squeeze(0).detach().cpu().numpy()
            y_val = val_targets.squeeze(0).detach().cpu().numpy()

            if evaluate_test:
                pred_test_t, y_test_t = _predict_test(model, task_data, torch, dev)
                y_pred_test = pred_test_t.squeeze(0).detach().cpu().numpy()
                y_test = y_test_t.squeeze(0).detach().cpu().numpy()
            else:
                y_pred_test = None
                y_test = None
    timing["eval_s"] = t_eval["elapsed"]
    timing["total_s"] = timing["train_s"] + timing["eval_s"]

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "val_metrics": compute_metrics(y_val, y_pred_val, metrics),
        "test_metrics": compute_metrics(y_test, y_pred_test, metrics) if evaluate_test and y_test is not None and y_pred_test is not None else {},
        "timing": timing,
        "metadata": {
            "epochs_ran": epochs_ran,
            "best_epoch": best_epoch,
            "early_stopped": epochs_ran < max_epochs,
            "device": str(dev),
            "requested_device": requested_device,
            "n_total_params": int(n_total),
            "n_trainable_params": int(n_trainable),
            "best_val_loss": best_val,
        },
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

