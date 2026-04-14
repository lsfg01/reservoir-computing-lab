import numpy as np

from rc_lab.tasks.base import BaseTask, TaskData


def generate_mackey_glass(
    T: int,
    seed: int,
    tau: int = 17,
    dt: float = 0.1,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera la serie de Mackey-Glass mediante integración numérica (RK4).

        dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

    La tarea de predicción es: dado x(t), predecir x(t+1).

    Devuelve (u, y) con shapes (T, 1), donde u = x(t) e y = x(t+1).
    """
    # Número de pasos de historia necesarios
    history_len = int(tau / dt) + 1
    total_steps = T + 1  # +1 para poder construir y = x(t+1)

    # Inicializar con valor constante 0.9 (condición inicial estándar)
    x = np.full(history_len + total_steps, 0.9)

    def mg_deriv(x_now: float, x_delayed: float) -> float:
        return beta * x_delayed / (1.0 + x_delayed ** n) - gamma * x_now

    # Integración RK4
    for i in range(history_len, history_len + total_steps):
        x_now = x[i - 1]
        x_del = x[i - 1 - history_len + 1]  # x(t - tau)

        k1 = mg_deriv(x_now, x_del)
        k2 = mg_deriv(x_now + 0.5 * dt * k1, x_del)
        k3 = mg_deriv(x_now + 0.5 * dt * k2, x_del)
        k4 = mg_deriv(x_now + dt * k3, x_del)

        x[i] = x_now + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    series = x[history_len : history_len + total_steps]
    u = series[:-1].reshape(-1, 1)   # x(t)
    y = series[1:].reshape(-1, 1)    # x(t+1)
    return u, y


class MackeyGlassTask(BaseTask):
    """Tarea Mackey-Glass: predecir x(t+1) dado x(t)."""

    def __init__(self, tau: int = 17, dt: float = 0.1) -> None:
        self._tau = tau
        self._dt = dt

    @property
    def name(self) -> str:
        return "mackey_glass"

    @property
    def primary_metric(self) -> str:
        return "nmse"

    def generate(self, n_train: int, n_val: int, n_test: int, washout: int, seed: int) -> TaskData:
        total = n_train + n_val + n_test
        u, y = generate_mackey_glass(total, seed=seed, tau=self._tau, dt=self._dt)

        u_train = u[:n_train]
        y_train = y[:n_train]

        if n_val > 0:
            u_val: np.ndarray | None = u[n_train : n_train + n_val]
            y_val: np.ndarray | None = y[n_train : n_train + n_val]
        else:
            u_val, y_val = None, None

        u_test = u[n_train + n_val :]
        y_test = y[n_train + n_val :]

        return TaskData(
            u_train=u_train,
            y_train=y_train,
            u_val=u_val,
            y_val=y_val,
            u_test=u_test,
            y_test=y_test,
            washout=washout,
        )
