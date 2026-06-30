# Parámetros entrenables de la comparación externa final

Fecha de inspección: 2026-06-30.

Ejecución auditada: `external_final_esn_vs_rnn_lstm`.

Fuentes principales:

- configuración: `configs/final_campaign/external/external_final_esn_vs_rnn_lstm.yaml`;
- resultados agregados: `comparison_summary.csv` y `comparison_summary.json`;
- resultados por modelo y tarea: `<modelo>/<tarea>/summary.json`;
- resultados por semilla: `<modelo>/<tarea>/runs/<config_id>_seed<seed>.json`;
- implementaciones bajo `src/rc_lab/`.

## 1. Conclusión ejecutiva

La ejecución contiene ocho familias de modelos, seis candidatos por familia,
tres tareas y siete semillas. Por tanto, el barrido de validación tiene
`8 × 6 × 3 × 7 = 1008` combinaciones modelo-candidato-tarea-semilla, además de
las reevaluaciones finales de los candidatos seleccionados. En las ESN, cada
combinación prueba internamente los seis valores de regularización ridge; todos
ellos tienen el mismo número de coeficientes.

El tamaño entrenable de una instancia depende de la tarea porque Delay Recall
produce 100 salidas, mientras que NARMA-10 y Mackey–Glass producen una salida:

| Modelo o variante | Candidatos | Delay Recall (100 salidas) | NARMA-10 (1 salida) | Mackey–Glass (1 salida) | Media guardada en el CSV raíz |
|---|---:|---:|---:|---:|---:|
| `random_sparse_baseline`, N=100 | 6 | 10 000 | 100 | 100 | 3 400 |
| `cycle_scr`, N=100 | 6 | 10 000 | 100 | 100 | 3 400 |
| `cycle_jump_j7`, N=100 | 6 | 10 000 | 100 | 100 | 3 400 |
| `nonnormal_chain_g0_3`, N=100 | 6 | 10 000 | 100 | 100 | 3 400 |
| `multiscale_three_random`, N=100 | 6 | 10 000 | 100 | 100 | 3 400 |
| Simple RNN, H=64 | 3 learning rates | 10 788 | 4 353 | 4 353 | 6 498 |
| Simple RNN, H=128 | 3 learning rates | 29 668 | 16 897 | 16 897 | 21 154 |
| LSTM, H=64 | 3 learning rates | 23 652 | 17 217 | 17 217 | 19 362 |
| LSTM, H=128 | 3 learning rates | 79 972 | 67 201 | 67 201 | 71 458 |
| Tapped-delay, L=10, raw | 1 | 1 100 | 11 | 11 | 374 |
| Tapped-delay, L=10, quadratic | 1 | 2 200 | 22 | 22 | 748 |
| Tapped-delay, L=25, raw | 1 | 2 600 | 26 | 26 | 884 |
| Tapped-delay, L=25, quadratic | 1 | 5 200 | 52 | 52 | 1 768 |
| Tapped-delay, L=50, raw | 1 | 5 100 | 51 | 51 | 1 734 |
| Tapped-delay, L=50, quadratic | 1 | 10 200 | 102 | 102 | 3 468 |

La columna `n_trainable_params_mean` de `comparison_summary.csv` **no describe
una arquitectura concreta**. El runner calcula la media de los conteos de las
tres tareas (`external_comparison_runner.py:809-856`). Por ejemplo:

`(10 788 + 4 353 + 4 353) / 3 = 6 498`.

No existe, por tanto, una RNN H=64 instanciada con 6 498 parámetros en esta
campaña: sus instancias tienen 10 788 o 4 353 según la tarea.

## 2. Criterio de conteo

Se cuenta como parámetro entrenable cada escalar perteneciente a un coeficiente
aprendido durante el ajuste del modelo y conservado para inferencia:

- en las ESN, únicamente los coeficientes de `Wout`;
- en Simple RNN y LSTM, todos los tensores recurrentes y el readout lineal,
  porque todos tienen `requires_grad=True`;
- en TappedDelayRidge, todos los coeficientes de su regresión ridge.

No se cuentan:

- `W`, `Win` ni el vector de bias del reservoir ESN: se generan a partir de la
  semilla y permanecen congelados;
- el estado oculto de una RNN/LSTM o de una ESN: es activación, no parámetro;
- los estados internos de Adam;
- medias y desviaciones de normalización: son estadísticos del conjunto de
  entrenamiento, no tensores optimizados;
- hiperparámetros seleccionados con validación (`ridge_param`, learning rate,
  H, número de lags, radio espectral, input scaling o leak rate). La validación
  decide entre candidatos, pero esos escalares no son coeficientes aprendidos
  del modelo.

Este criterio recoge la intención de contar el readout de una ESN y todos los
pesos actualizados por gradiente de una RNN. En particular, seleccionar
`ridge_param` no añade un parámetro: cambia la solución de los coeficientes de
`Wout`, cuyo número no cambia.

## 3. Dimensiones impuestas por las tareas

Todas las tareas usan entrada escalar, por lo que `I=1`.

- Delay Recall construye
  `y(t)=[u(t-1), ..., u(t-100)]`; por ello `O=100`
  (`delay_recall.py:13-26` y `kmax: 100` en la configuración).
- NARMA-10 devuelve `y` con shape `(T, 1)` (`narma10.py:36-37`); `O=1`.
- Mackey–Glass devuelve pares escalares con shape `(T, 1)`
  (`mackey_glass.py:170-171`); `O=1`.

Los tamaños de train, validación y test, el washout, BPTT, batch size, número de
épocas, paciencia y número de semillas no alteran el número de parámetros.

## 4. Derivación por implementación

### 4.1. Las cinco ESN

La dinámica de `ESNModel` usa matrices ya construidas y no contiene ningún
procedimiento de actualización de `W`, `Win` o bias (`models/esn.py:4-61`).
El runner genera estados y ajusta un `RidgeReadout`
(`sweep_runner.py:386-406`).

La campaña fija:

- `N=100` para las cinco familias;
- `readout.features: states`;
- `Ridge(..., fit_intercept=False)`.

Por tanto, la matriz de diseño tiene `D=N=100` columnas y:

`P_ESN = D × O = 100 × O`.

Resultado:

- Delay Recall: `100 × 100 = 10 000`;
- NARMA-10: `100 × 1 = 100`;
- Mackey–Glass: `100 × 1 = 100`.

El runner usa exactamente la misma fórmula en
`external_comparison_runner.py:681-697`. Los metadatos por tarea guardan
10 000, 100 y 100 para los seis candidatos de cada una de las cinco familias.

La topología sólo cambia cómo se producen los 100 estados congelados:

| Modelo | Implementación del reservoir fijo | Consecuencia para el conteo |
|---|---|---|
| `random_sparse_baseline` | `RandomSparseReservoir`: W aleatoria con sparsity=0.9, reescalada por radio espectral; Win aleatoria (`random_sparse.py:11-68`). | Sólo se entrena el readout: 100×O. |
| `cycle_scr` | `CycleReservoir`: ciclo dirigido con 100 conexiones recurrentes antes del reescalado (`cycle.py:6-100`). | Sólo se entrena el readout: 100×O. |
| `cycle_jump_j7` | `CycleJumpReservoir`: ciclo más saltos j=7 de peso 0.3 (`cycle_jump.py:20-153`). | Sólo se entrena el readout: 100×O. |
| `nonnormal_chain_g0_3` | `NonnormalChainReservoir`: `W=ρI+0.3S`, con shift unidireccional (`nonnormal_chain.py:6-115`). | Sólo se entrena el readout: 100×O. |
| `multiscale_three_random` | `MultiScaleReservoir`: tres bloques aleatorios 30/30/40 y acoplamiento fijo (`multiscale.py:14-110`, `147-183`). | Sólo se entrena el readout: 100×O. |

El campo `n_total_params_mean=13 600` de los resultados ESN no debe confundirse
con el conteo entrenable. El runner contabiliza además, como almacenamiento
fijo, `W` (10 000), `Win` (100) y bias (100): 10 200 escalares no entrenables.
La media total resulta `10 200 + 3 400 = 13 600`.

### 4.2. Simple RNN

La implementación es una `torch.nn.RNN` de una capa, entrada escalar y
no linealidad tanh, seguida por `Linear(H,O)`
(`torch_models.py:335-371`). El código cuenta todos los `numel()` con
`requires_grad=True` (`torch_models.py:253-254`).

Para una capa:

- `weight_ih_l0`: `H × I = H`;
- `weight_hh_l0`: `H × H = H²`;
- `bias_ih_l0`: `H`;
- `bias_hh_l0`: `H`;
- peso del readout: `O × H`;
- bias del readout: `O`.

Con `I=1`:

`P_RNN(H,O) = H² + 3H + O(H+1)`.

| H | O | W_ih | W_hh | biases RNN | peso readout | bias readout | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 64 | 4 096 | 128 | 64 | 1 | 4 353 |
| 64 | 100 | 64 | 4 096 | 128 | 6 400 | 100 | 10 788 |
| 128 | 1 | 128 | 16 384 | 256 | 128 | 1 | 16 897 |
| 128 | 100 | 128 | 16 384 | 256 | 12 800 | 100 | 29 668 |

Las tres learning rates de cada H cambian el entrenamiento, no la arquitectura.

### 4.3. LSTM

La implementación usa una `torch.nn.LSTM` de una capa seguida por la misma
`Linear(H,O)` (`torch_models.py:359-371`). PyTorch agrupa los parámetros de las
cuatro puertas; por eso:

- `weight_ih_l0`: `4H × I`;
- `weight_hh_l0`: `4H × H`;
- `bias_ih_l0`: `4H`;
- `bias_hh_l0`: `4H`;
- readout: `O × H + O`.

Con `I=1`:

`P_LSTM(H,O) = 4H² + 12H + O(H+1)`.

| H | O | W_ih | W_hh | biases LSTM | peso readout | bias readout | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 256 | 16 384 | 512 | 64 | 1 | 17 217 |
| 64 | 100 | 256 | 16 384 | 512 | 6 400 | 100 | 23 652 |
| 128 | 1 | 512 | 65 536 | 1 024 | 128 | 1 | 67 201 |
| 128 | 100 | 512 | 65 536 | 1 024 | 12 800 | 100 | 79 972 |

Dropout es 0.0 y, en cualquier caso, no introduciría parámetros. Las tres
learning rates de cada H tampoco cambian el conteo.

### 4.4. TappedDelayRidge

`TappedDelayRidge` construye retardos desde `u(t)` hasta `u(t-L)`, ambos
incluidos (`tapped_delay.py:19-66`, `158-171`):

- modo raw: `D=L+1`;
- modo quadratic: concatena los mismos términos y sus cuadrados individuales,
  sin productos cruzados; `D=2(L+1)`.

El readout ridge usa `fit_intercept=False`, así que no hay término constante
adicional (`readouts/ridge.py:58`). La propiedad de la clase devuelve
directamente:

`P_TDR = D × O`

(`tapped_delay.py:174-181`).

| L | Modo | D | O=1 | O=100 |
|---:|---|---:|---:|---:|
| 10 | raw | 11 | 11 | 1 100 |
| 10 | quadratic | 22 | 22 | 2 200 |
| 25 | raw | 26 | 26 | 2 600 |
| 25 | quadratic | 52 | 52 | 5 200 |
| 50 | raw | 51 | 51 | 5 100 |
| 50 | quadratic | 102 | 102 | 10 200 |

## 5. Los 48 candidatos concretos

### 5.1. Candidatos ESN

Todos los candidatos siguientes tienen el vector de conteos
`[Delay Recall, NARMA-10, Mackey–Glass] = [10 000, 100, 100]`. Los cambios en
ρ, escala de entrada y α modifican la dinámica fija, no el tamaño de `Wout`.
El identificador sólo codifica el punto candidato, por lo que un mismo
`config_id` puede aparecer bajo distintas familias sin representar el mismo
reservoir.

| Modelo | config_id | ρ | s_in | α | Parámetros entrenables por tarea |
|---|---|---:|---:|---:|---|
| `random_sparse_baseline` | `9992b49ddf96` | 0.95 | 0.05 | 1.0 | 10 000 / 100 / 100 |
| `random_sparse_baseline` | `0e41bf66b8bb` | 0.95 | 0.10 | 0.9 | 10 000 / 100 / 100 |
| `random_sparse_baseline` | `4d2160951482` | 1.00 | 0.20 | 0.9 | 10 000 / 100 / 100 |
| `random_sparse_baseline` | `4e018c35c4e2` | 1.00 | 0.40 | 0.9 | 10 000 / 100 / 100 |
| `random_sparse_baseline` | `156770123aaf` | 1.20 | 1.00 | 0.3 | 10 000 / 100 / 100 |
| `random_sparse_baseline` | `b07b757cf213` | 1.40 | 1.50 | 0.3 | 10 000 / 100 / 100 |
| `cycle_scr` | `9992b49ddf96` | 0.95 | 0.05 | 1.0 | 10 000 / 100 / 100 |
| `cycle_scr` | `92bfaaea8b8b` | 1.00 | 0.05 | 1.0 | 10 000 / 100 / 100 |
| `cycle_scr` | `259248d07a08` | 0.95 | 0.20 | 0.9 | 10 000 / 100 / 100 |
| `cycle_scr` | `4d2160951482` | 1.00 | 0.20 | 0.9 | 10 000 / 100 / 100 |
| `cycle_scr` | `5d1a27f2c344` | 1.00 | 1.00 | 0.3 | 10 000 / 100 / 100 |
| `cycle_scr` | `b07b757cf213` | 1.40 | 1.50 | 0.3 | 10 000 / 100 / 100 |
| `cycle_jump_j7` | `9992b49ddf96` | 0.95 | 0.05 | 1.0 | 10 000 / 100 / 100 |
| `cycle_jump_j7` | `96af7170b8b0` | 1.00 | 0.10 | 0.9 | 10 000 / 100 / 100 |
| `cycle_jump_j7` | `4d2160951482` | 1.00 | 0.20 | 0.9 | 10 000 / 100 / 100 |
| `cycle_jump_j7` | `4e018c35c4e2` | 1.00 | 0.40 | 0.9 | 10 000 / 100 / 100 |
| `cycle_jump_j7` | `156770123aaf` | 1.20 | 1.00 | 0.3 | 10 000 / 100 / 100 |
| `cycle_jump_j7` | `b07b757cf213` | 1.40 | 1.50 | 0.3 | 10 000 / 100 / 100 |
| `nonnormal_chain_g0_3` | `9e51842d34ab` | 0.95 | 0.20 | 1.0 | 10 000 / 100 / 100 |
| `nonnormal_chain_g0_3` | `0e41bf66b8bb` | 0.95 | 0.10 | 0.9 | 10 000 / 100 / 100 |
| `nonnormal_chain_g0_3` | `259248d07a08` | 0.95 | 0.20 | 0.9 | 10 000 / 100 / 100 |
| `nonnormal_chain_g0_3` | `4d2160951482` | 1.00 | 0.20 | 0.9 | 10 000 / 100 / 100 |
| `nonnormal_chain_g0_3` | `4e018c35c4e2` | 1.00 | 0.40 | 0.9 | 10 000 / 100 / 100 |
| `nonnormal_chain_g0_3` | `4df7ca5b3a5c` | 1.00 | 0.40 | 0.5 | 10 000 / 100 / 100 |
| `multiscale_three_random` | `9992b49ddf96` | 0.95 | 0.05 | 1.0 | 10 000 / 100 / 100 |
| `multiscale_three_random` | `259248d07a08` | 0.95 | 0.20 | 0.9 | 10 000 / 100 / 100 |
| `multiscale_three_random` | `4d2160951482` | 1.00 | 0.20 | 0.9 | 10 000 / 100 / 100 |
| `multiscale_three_random` | `4e018c35c4e2` | 1.00 | 0.40 | 0.9 | 10 000 / 100 / 100 |
| `multiscale_three_random` | `156770123aaf` | 1.20 | 1.00 | 0.3 | 10 000 / 100 / 100 |
| `multiscale_three_random` | `059a4ddb8c9d` | 1.40 | 1.00 | 0.3 | 10 000 / 100 / 100 |

### 5.2. Candidatos Simple RNN y LSTM

| Modelo | config_id | H | learning rate | Delay Recall | NARMA-10 | Mackey–Glass | Media del CSV |
|---|---|---:|---:|---:|---:|---:|---:|
| Simple RNN | `b760a145a7c4` | 64 | 0.001 | 10 788 | 4 353 | 4 353 | 6 498 |
| Simple RNN | `8e28073ddfcb` | 64 | 0.003 | 10 788 | 4 353 | 4 353 | 6 498 |
| Simple RNN | `ad4f2296d003` | 64 | 0.010 | 10 788 | 4 353 | 4 353 | 6 498 |
| Simple RNN | `e8737e669707` | 128 | 0.001 | 29 668 | 16 897 | 16 897 | 21 154 |
| Simple RNN | `b7104fa59923` | 128 | 0.003 | 29 668 | 16 897 | 16 897 | 21 154 |
| Simple RNN | `084341ab599f` | 128 | 0.010 | 29 668 | 16 897 | 16 897 | 21 154 |
| LSTM | `d59c1b882400` | 64 | 0.001 | 23 652 | 17 217 | 17 217 | 19 362 |
| LSTM | `6c1421851c33` | 64 | 0.003 | 23 652 | 17 217 | 17 217 | 19 362 |
| LSTM | `58e109cc028f` | 64 | 0.010 | 23 652 | 17 217 | 17 217 | 19 362 |
| LSTM | `fc91c7c7fdb4` | 128 | 0.001 | 79 972 | 67 201 | 67 201 | 71 458 |
| LSTM | `c7d9c0de5940` | 128 | 0.003 | 79 972 | 67 201 | 67 201 | 71 458 |
| LSTM | `2fd4392c51ef` | 128 | 0.010 | 79 972 | 67 201 | 67 201 | 71 458 |

Los demás hiperparámetros de estos doce candidatos son comunes: una capa,
BPTT=200, batch size=32, weight decay=0, readout lineal, entrenamiento windowed
y normalización de entradas y objetivos.

### 5.3. Candidatos TappedDelayRidge

| config_id | L | Modo | D | Delay Recall | NARMA-10 | Mackey–Glass | Media del CSV |
|---|---:|---|---:|---:|---:|---:|---:|
| `e5146fb1df11` | 10 | raw | 11 | 1 100 | 11 | 11 | 374 |
| `85b8bdf21c0a` | 10 | quadratic | 22 | 2 200 | 22 | 22 | 748 |
| `6c6534af441c` | 25 | raw | 26 | 2 600 | 26 | 26 | 884 |
| `f81eab77cd40` | 25 | quadratic | 52 | 5 200 | 52 | 52 | 1 768 |
| `a6aa1bed1edb` | 50 | raw | 51 | 5 100 | 51 | 51 | 1 734 |
| `9a9199a75441` | 50 | quadratic | 102 | 10 200 | 102 | 102 | 3 468 |

## 6. Los ocho candidatos agregados finalmente seleccionados

`comparison_summary.json > best_by_model` selecciona un candidato agregado por
familia. Como cada tarea entrena una instancia con su propio número de salidas,
incluso este “mejor modelo” sigue teniendo un tamaño por tarea:

| Modelo | config_id seleccionado | Variante | Delay Recall | NARMA-10 / Mackey–Glass | Media guardada |
|---|---|---|---:|---:|---:|
| `random_sparse_baseline` | `0e41bf66b8bb` | ρ=0.95, s_in=0.1, α=0.9 | 10 000 | 100 | 3 400 |
| `cycle_scr` | `9992b49ddf96` | ρ=0.95, s_in=0.05, α=1.0 | 10 000 | 100 | 3 400 |
| `cycle_jump_j7` | `96af7170b8b0` | ρ=1.0, s_in=0.1, α=0.9 | 10 000 | 100 | 3 400 |
| `nonnormal_chain_g0_3` | `9e51842d34ab` | ρ=0.95, s_in=0.2, α=1.0 | 10 000 | 100 | 3 400 |
| `multiscale_three_random` | `4d2160951482` | ρ=1.0, s_in=0.2, α=0.9 | 10 000 | 100 | 3 400 |
| `simple_rnn` | `8e28073ddfcb` | H=64, lr=0.003 | 10 788 | 4 353 | 6 498 |
| `lstm` | `6c1421851c33` | H=64, lr=0.003 | 23 652 | 17 217 | 19 362 |
| `tapped_delay_ridge` | `9a9199a75441` | L=50, quadratic | 10 200 | 102 | 3 468 |

## 7. Comprobaciones realizadas

1. Se inspeccionaron las 48 filas de `comparison_summary.csv`; cada familia
   contiene exactamente seis candidatos.
2. Se inspeccionó `metadata_mean.n_trainable_params` en los resúmenes JSON de
   cada modelo y tarea. Los 144 pares modelo-candidato-tarea coinciden con las
   tablas anteriores.
3. Para RNN y LSTM se instanciaron las ocho combinaciones
   `(kind, H, O)` con la clase real y se sumó `numel()` por tensor. Los totales
   y desgloses coinciden exactamente con los JSON por semilla.
4. Para ESN y TappedDelayRidge se contrastó la dimensión real de sus matrices
   de diseño y el uso de `fit_intercept=False`.
5. Todas las semillas de una misma combinación reportan el mismo conteo; la
   semilla cambia los valores aprendidos, no el número de coeficientes.

## 8. Recomendación para tablas y figuras

Para comparaciones por tarea se debe usar el conteo específico de esa tarea,
no `n_trainable_params_mean`:

- Delay Recall: `delay_recall_n_trainable_params`;
- NARMA-10: `narma10_n_trainable_params`;
- Mackey–Glass: `mg_n_trainable_params`.

La media del CSV raíz sólo es válida como un resumen artificial de coste entre
tareas. En especial, usar 3 400 para una ESN, 6 498 para una RNN H=64 o 19 362
para una LSTM H=64 en una figura de una tarea concreta atribuye al modelo un
tamaño que ninguna instancia de esa tarea tuvo.

Los CSV/JSON, rankings, tabla maestra, figura de coste y paquetes ZIP de esta
ejecución fueron actualizados el 2026-06-30 para propagar los seis campos
específicos (`total` y `trainable` por tarea). Los campos históricos
`n_total_params_mean` y `n_trainable_params_mean` se conservan únicamente por
compatibilidad y mantienen explícitamente su semántica de media entre tareas.
