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
    Los primeros 10 pasos se usan como calentamiento interno de la recurrencia.
    """
    rng = np.random.default_rng(seed)
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
    """
    Tarea NARMA-10: predecir y(t) dado u(t).

    Protocolo de splits
    -------------------
    state_policy="reset" (por defecto):
        La serie se organiza como:
            [washout][n_train][washout][n_val][washout][n_test]
        Cada split tiene su propio bloque de warmup de longitud `washout`.
        u_val_full y u_test_full transportan los bloques completos con warmup.
        u_val, y_val, u_test, y_test son los arrays puntuados (post-warmup).

    state_policy="carryover":
        La serie se organiza como:
            [washout][n_train][n_val][n_test]
        El estado del reservoir se propaga entre splits contiguos.
        u_val_full y u_test_full son None.
    """

    def __init__(self, state_policy: str = "reset") -> None:
        if state_policy not in ("reset", "carryover"):
            raise ValueError(f"state_policy inválido: {state_policy!r}")
        self._state_policy = state_policy

    @property
    def name(self) -> str:
        return "narma10"

    @property
    def primary_metric(self) -> str:
        return "nmse"

    def generate(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        washout: int,
        seed: int,
    ) -> TaskData:
        if self._state_policy == "reset":
            return self._generate_reset(n_train, n_val, n_test, washout, seed)
        else:
            return self._generate_carryover(n_train, n_val, n_test, washout, seed)

    def _generate_reset(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        washout: int,
        seed: int,
    ) -> TaskData:
        """
        Protocolo reset: cada split tiene warmup propio de longitud `washout`.

        Serie total generada:
            [washout][n_train][washout][n_val][washout][n_test]
        """
        n_val_blocks = 1 if n_val > 0 else 0
        total = (
            washout + n_train
            + n_val_blocks * (washout + n_val)
            + washout + n_test
        )
        u, y = generate_narma10(total, seed=seed)

        # Extraer bloques
        i = 0

        u_train = u[i : i + washout + n_train]
        y_train = y[i : i + washout + n_train]
        i += washout + n_train

        if n_val > 0:
            u_val_full = u[i : i + washout + n_val]
            y_val_full = y[i : i + washout + n_val]
            u_val: np.ndarray | None = u_val_full[washout:]
            y_val: np.ndarray | None = y_val_full[washout:]
            i += washout + n_val
        else:
            u_val_full = None
            u_val, y_val = None, None

        u_test_full = u[i : i + washout + n_test]
        y_test_full = y[i : i + washout + n_test]
        u_test = u_test_full[washout:]
        y_test = y_test_full[washout:]

        return TaskData(
            u_train=u_train,
            y_train=y_train,
            u_val=u_val,
            y_val=y_val,
            u_test=u_test,
            y_test=y_test,
            washout=washout,
            state_policy="reset",
            u_val_full=u_val_full,
            u_test_full=u_test_full,
        )

    def _generate_carryover(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        washout: int,
        seed: int,
    ) -> TaskData:
        """
        Protocolo carryover: serie continua, estado propagado entre splits.

        Serie total generada:
            [washout][n_train][n_val][n_test]
        """
        total = washout + n_train + n_val + n_test
        u, y = generate_narma10(total, seed=seed)

        i = 0
        u_train = u[i : i + washout + n_train]
        y_train = y[i : i + washout + n_train]
        i += washout + n_train

        if n_val > 0:
            u_val: np.ndarray | None = u[i : i + n_val]
            y_val: np.ndarray | None = y[i : i + n_val]
            i += n_val
        else:
            u_val, y_val = None, None

        u_test = u[i : i + n_test]
        y_test = y[i : i + n_test]

        return TaskData(
            u_train=u_train,
            y_train=y_train,
            u_val=u_val,
            y_val=y_val,
            u_test=u_test,
            y_test=y_test,
            washout=washout,
            state_policy="carryover",
            u_val_full=None,
            u_test_full=None,
        )
