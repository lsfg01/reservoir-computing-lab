import numpy as np

from rc_lab.tasks.base import BaseTask, TaskData


def generate_narma10(
    T: int,
    seed: int,
    amplitude: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera la serie NARMA de orden 10.

    Ecuación de recurrencia:
        y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-k) for k in 0..9)
                + 1.5*u(t-9)*u(t) + 0.1

    donde u(t) ~ U(0, amplitude).

    Devuelve (u, y) con shapes (T, 1).
    Los primeros 10 pasos se usan como calentamiento interno.
    """
    rng = np.random.default_rng(seed)
    # Generamos T + 10 pasos para descartar el calentamiento
    total = T + 10
    u_full = rng.uniform(0, amplitude, total)
    y_full = np.zeros(total)

    for t in range(10, total - 1):
        y_full[t + 1] = (
            0.3 * y_full[t]
            + 0.05 * y_full[t] * np.sum(y_full[t - 9 : t + 1])
            + 1.5 * u_full[t - 9] * u_full[t]
            + 0.1
        )

    u = u_full[10:].reshape(-1, 1)
    y = y_full[10:].reshape(-1, 1)
    return u, y


class Narma10Task(BaseTask):
    """Tarea NARMA-10: predecir y(t) dado u(t)."""

    @property
    def name(self) -> str:
        return "narma10"

    @property
    def primary_metric(self) -> str:
        return "nmse"

    def generate(self, n_train: int, n_test: int, washout: int, seed: int) -> TaskData:
        total = n_train + n_test
        u, y = generate_narma10(total, seed=seed)

        u_train = u[:n_train]
        y_train = y[:n_train]
        u_test = u[n_train:]
        y_test = y[n_train:]

        return TaskData(
            u_train=u_train,
            y_train=y_train,
            u_test=u_test,
            y_test=y_test,
            washout=washout,
        )
