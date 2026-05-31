"""
Baseline trivial de persistencia como métrica de referencia por tarea.

La persistencia predice que el siguiente valor es igual al último observado:
    ŷ(t) = y(t−1)

Sirve como línea de flotación: el éxito real en una tarea de predicción no es
NMSE < 1, sino batir la persistencia. Para series con autocorrelación positiva,
el NRMSE de persistencia puede caer significativamente por debajo de 1.

Valores de referencia conocidos (literatura RC):
    NARMA-10  NRMSE ≈ 0.83   (depende de longitud y amplitud de la señal)
    Santa-Fe  NRMSE ≈ 0.97

Alineamiento temporal
---------------------
Para tareas escalares de predicción (narma10, mackey_glass):
    y_true  = y[1:]   (valores desde el paso 1 en adelante)
    y_pred  = y[:-1]  (valor anterior, el "persistido")
El alineamiento usa estrictamente y(t-1); no hay fuga de información futura.

Para delay_recall (target multisalida de retardos u(t-k)):
    La persistencia clásica no aplica. El target en la columna k es u(t-k),
    un valor pasado de la entrada, no de la salida del modelo. "Persistir"
    significaría predecir u(t-k) ≈ u(t-k-1), que es un retardo de retardo y
    no es la referencia semántica correcta. Además, para iid uniforme,
    corr(u(t-k), u(t-k-1)) = 0, así que el suelo de corr² = 0 ya es el
    baseline implícito. Por eso persistence_baseline devuelve None para
    delay_recall, documentando explícitamente que la referencia correcta
    es corr² = 0 (suelo de la distribución iid).

Este módulo es puramente funcional: no modifica runners, rankings ni R_agg.
Los valores de referencia se adjuntan como contexto a las tablas de resultados.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rc_lab.metrics.error import nmse, rmse
from rc_lab.tasks.base import TaskData


@dataclass
class PersistenceResult:
    """Error de persistencia sobre un bloque de evaluación."""
    nmse: float
    rmse: float
    nrmse: float   # sqrt(nmse), análogo al NRMSE de la literatura RC
    n_samples: int


def persistence_error(y: np.ndarray) -> PersistenceResult:
    """
    Calcula el error de persistencia sobre una serie escalar.

    Predicción: ŷ(t) = y(t-1), evaluada sobre el par (y[1:], y[:-1]).

    Parámetros
    ----------
    y : array 1D o (T, 1) con la serie temporal de referencia

    Devuelve
    --------
    PersistenceResult con nmse, rmse y nrmse = sqrt(nmse).

    Raises
    ------
    ValueError si y tiene menos de 2 muestras o varianza cero.
    ValueError si y no es escalar (más de una columna).
    """
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] != 1:
            raise ValueError(
                f"persistence_error espera una serie escalar; "
                f"recibida shape {arr.shape}. "
                f"Para delay_recall (multisalida) usa persistence_baseline con task_name='delay_recall'."
            )
        arr = arr[:, 0]
    elif arr.ndim != 1:
        raise ValueError(f"persistence_error espera un array 1D o (T,1), recibido shape {arr.shape}")

    if arr.shape[0] < 2:
        raise ValueError(f"Se necesitan al menos 2 muestras; recibidas {arr.shape[0]}")

    y_true = arr[1:]
    y_pred = arr[:-1]

    nmse_val = nmse(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    nrmse_val = float(np.sqrt(nmse_val))

    return PersistenceResult(
        nmse=nmse_val,
        rmse=rmse_val,
        nrmse=nrmse_val,
        n_samples=len(y_true),
    )


def persistence_baseline(
    task_data: TaskData,
    task_name: str,
    split: str = "test",
) -> PersistenceResult | None:
    """
    Calcula el baseline de persistencia para una tarea a partir de su TaskData.

    Parámetros
    ----------
    task_data : datos generados por la tarea (TaskData)
    task_name : nombre de la tarea ('narma10', 'mackey_glass', 'delay_recall')
    split     : bloque sobre el que calcular ('test' o 'val')

    Devuelve
    --------
    PersistenceResult para tareas escalares de predicción (narma10, mackey_glass).
    None para delay_recall, donde la referencia semántica correcta es corr² = 0
    (suelo de la distribución iid uniforme); ver docstring del módulo.

    Raises
    ------
    ValueError para split inválido o split 'val' sin bloque de validación.
    """
    if split not in ("test", "val"):
        raise ValueError(f"split debe ser 'test' o 'val'; recibido {split!r}")

    if task_name == "delay_recall":
        return None

    if split == "test":
        y = task_data.y_test
    else:
        if task_data.y_val is None:
            raise ValueError("El bloque de validación no existe (n_val == 0)")
        y = task_data.y_val

    return persistence_error(y)
