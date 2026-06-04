# CAMPAIGN_DESIGN.md — Decisiones de la campaña experimental final

Documento de estado vivo. Consolida las decisiones tomadas para la campaña
final del TFG de Reservoir Computing. Sirve como punto de partida para retomar
el trabajo (incluido en un chat nuevo): pégalo junto al  y se tiene el
contexto completo sin depender de memoria de conversación.

Leer el  primero (arquitectura de cuatro pasos). Este documento asume
ese flujo y fija los parámetros concretos de la campaña.

---

## Origen: análisis crítico previo

Tras revisar código + memoria + papers, se identificaron debilidades
experimentales a corregir ANTES de lanzar la campaña. Las estructurales ya se
corrigieron (Fase 1, ver abajo). Las de análisis se harán post-campaña (Fase 3).
Estado: listos para lanzar la campaña.

---

## Fase 1 — Enhancements de código (COMPLETADA)

Módulos nuevos aislados, cada uno verificado por separado, sin tocar el núcleo:

1. **Evaluador ESP** (`metrics/stability.py`) ✓
   - `sync_pair` / `evaluate_esp` / `evaluate_esp_sampled`.
   - Washout empírico = tiempo de sincronización de dos trayectorias bajo la
     misma entrada desde x0 distintos. Distancia RELATIVA d(t)=‖Δx(t)‖/‖Δx(0)‖
     (comparable entre familias de distinta escala), umbral ε=1e-3.
   - 5 pares por defecto (configurable), input U(-1,1), sentinela explícito +
     flag `synchronized` si no converge. Sin perturbación local (futuro).
   - 11 tests en verde.

2. **Base case de persistencia** (`metrics/persistence.py`) ✓
   - Métrica de REFERENCIA por tarea (no compite en ranking).
   - `persistence_error(y)` reutiliza error.py; `persistence_baseline(task_data,
     task_name, split)`. NARMA-10 verificado NRMSE≈0.83.
   - delay_recall devuelve None (persistencia clásica no aplica; referencia es
     corr²=0). 14 tests en verde.

3. **NG-RC primitivo** (`sequence_models/tapped_delay.py` + alias en runner) ✓
   - `feature_mode="quadratic"` documentado como NVAR de orden 2 DIAGONAL:
     2(L+1) columnas, lineales + cuadrados, SIN productos cruzados ni constante.
   - Alias `ng_rc` en `external_comparison_runner` apunta al mismo cómputo
     (equivalencia numérica exacta verificada). 9 tests en verde.
   - Brecha hacia Gauthier completo documentada (cruzados, orden>2, constante).

4. **Sanity check de redes** (notebook exploratorio, NO va al repo) ✓ — ver abajo.

---

## Hallazgos del sanity check (entrenamiento RNN/LSTM)

Notebook en GPU, 1 semilla, sin tope, lr×hidden×layers, 3 tareas:
- **Status 24/24 `ok`** en las tres tareas: el entrenamiento es legítimo, ningún
  colapso. Blinda la campaña: no se infra-entrena por error.
- **lr importa y varía por tarea/modelo** (óptimos en 1e-2 y 3e-3, casi nunca
  1e-3 solo) → lr ENTRA al grid de candidatos.
- **delay_recall + simple_rnn**: no agota aprendizaje ni a 600 ni a 800 épocas
  (best_epoch≈591 a 600). Sigue mejorando — es un corte por presupuesto, no por
  convergencia. Decisión: ver "cap de épocas".
- **layers=2 ayuda al RNN en delay_recall** (~18% sobre L=1), pero se fija L=1
  por COMPARABILIDAD con el reservoir de capa única. Decisión deliberada, a
  declarar en la memoria como conservadora a favor del competidor.
- **MG "sorprendentemente bueno"** (LSTM NMSE~6e-4): NO es artefacto. El horizonte
  es real y duro (≈4.9·τ). La mejora vs preliminar se debe a entrenar sin tope
  (la preliminar tenía tope 120s). Confirma que las redes estaban infra-entrenadas
  por tiempo y que con entrenamiento completo mejoran — hallazgo legítimo.

---

## Decisiones de la campaña (CERRADAS)

### Principio rector de honestidad
La tesis NO es "la ESN gana a todo". Es "la ESN es competitiva siendo mucho más
barata e interpretable". Por tanto:
- NO recortar entrenamiento de las redes para que gane la ESN. Una LSTM que gana
  MG pero tarda 100× más y es opaca ES el resultado, no la derrota.
- El tiempo es una MÉTRICA reportada, no un sesgo a esconder.
- Declarar explícitamente: (a) L=1 en redes por comparabilidad; (b) el sweep de
  selección de región es un barrido extra que solo la ESN tiene (es selección de
  hiperparámetros, análoga a elegir arquitectura de red); (c) reportar tipo de
  parada (early_stopped vs truncado a cap) por candidato.

### Cap de épocas (redes externas)
- **800 épocas, patience ~50**, declarado como límite de presupuesto UNIFORME
  (no como convergencia, porque el RNN-delay_recall no converge ni a 800).
- Reportar status de parada por candidato en los resultados. Transparencia
  convierte el corte en limitación declarada, no en sesgo oculto.
- (Se descartó 500: bajarlo DESPUÉS de ver que 600/800 no agotan habría sido un
  recorte que perjudica selectivamente al competidor.)

### Grid de candidatos externos (torch) — 6 candidatos
```yaml
learning_rate: [1e-2, 3e-3, 1e-3]   # la perilla decisiva (sanity)
hidden_size:   [64, 128]
# num_layers: 1 (fijo, comparabilidad — declarado)
# weight_decay: 0.0 (fijo; se soltó del grid para bajar de 12 a 6)
# bptt_length: 200 para las 3 tareas, vía lista de UN valor [200] en el grid
#   (no multiplica candidatos; evita tocar el runner para bptt por-tarea)
# max_epochs: 800, patience: ~50
```
Nota: el config preliminar tenía 12 candidatos (hidden×weight_decay×bptt) MAL
asignados — gastaba 3 niveles en bptt (poco influyente) y 0 en lr (decisivo).
Rebalanceado a lr×hidden.

### Semillas
- **7 semillas** en toda la campaña (externa y baseline), por coherencia.
- Externa (torch, CPU): 6 cand × 7 semillas × 3 tareas × 2 modelos torch ≈ **2.5h**
  (cota superior, medida con candidato caro hidden=128). Una pasada, modo
  antibloqueo de pantalla. El runner externo guarda cada run incrementalmente,
  así que una interrupción no pierde lo computado.
- Baseline/design (ESN, numpy): barato (~1h con diagnósticos en design).
- Prioridad si hubiera que recortar: semillas sobre candidatos (la varianza
  entre semillas alimenta Friedman/Nemenyi de Fase 3). Nunca por debajo de 5.

### N (tamaño de reservoir)
- **N=100 fijo** en capa de design y comparación externa (comparabilidad
  estructural y con el dimensionado de las redes).
- Verificación de escalado N∈{100,300,500}: solo COMPROBACIÓN de que la región
  robusta generaliza, NO condición de operación. Va a la memoria como argumento
  de robustez.

### Sweep de selección de región (Paso 1)
- Grid AMPLIO motivado por teoría (no "por barrer"):
  - ρ denso cerca de 1 y cruzando la frontera ESP (la teoría de memoria predice
    ρ→1; cruzar ρ=1 es experimento, no error). Ej. {0.6,0.7,0.8,0.9,1.0,1.1} o
    más denso cerca de 1.
  - s_in de casi-lineal a no-lineal (las tareas piden regímenes distintos).
  - leak/α moderado.
- Solo `random_sparse`, N=100, tareas narma10/mackey_glass/memory_capacity.
- Salida: `shortlist_top_n` por aggregate_rank = **región robusta**.
- Criterio de región robusta (a declarar en memoria): conjunto de configs en el
  mejor decil de aggregate_rank QUE ADEMÁS superan el base case de persistencia
  en las tres tareas. Robustez = MESETA amplia y bien localizada (ρ cerca de 1,
  s_in moderado), no un pico aislado. Una meseta evidencia robustez estructural;
  un pico evidenciaría fragilidad (lo contrario de la tesis).
- Visualización: heatmaps 2D (cortes ρ×s_in a leak fijo), no volumen 3D.

### Capa de design (Paso 3 — la central)
- Grid ESTRECHO = región destilada del sweep (ej. 18 puntos
  ρ{0.8,0.9,1.0}×s_in{0.10,0.15,0.20}×leak{0.9,1.0}).
- Familias: random_sparse_baseline, cycle_jump_j7, nonnormal_chain (g0_1, g0_3),
  multiscale. Todas N=100, alineadas por config_id.
- **Diagnósticos viven aquí**, sobre los puntos del grid de design (y para los
  más caros como ‖W^k‖, basta con los mejores candidatos de cada familia).
- Tarea de memoria: memory_capacity clásica (kmax≈200). MC con kmax alineado a N
  (no kmax=200 con N=100, que diluye; usar ≈N o reportar hasta N).

### Estudio de frontera ESP (Paso 0)
- Barrido de ρ (denso, ~15-20 valores, 1-2 familias, N=100) midiendo washout
  empírico → curva washout(ρ), localiza dónde diverge. Coste: minutos.
- Es RESULTADO PREVIO que justifica el rango de ρ del sweep (cap. 6 memoria).
- El mismo evaluador ESP se usa luego como diagnóstico puntual por config (washout
  del seleccionado, junto a ρ/Henrici/Gk), para comparar familias a igual ρ.

### Tareas — parametrización fija (debe ser IDÉNTICA en sanity/benchmark/campaña)
- **mackey_glass**: tau=17, dt=0.1, beta=0.2, gamma=0.1, n=10,
  initial_history=random_uniform [1.1,1.3], discard_transient=1000,
  sample_stride=10, prediction_horizon=84 (≈4.9·τ, horizonte largo real).
- **narma10**: estándar; base case persistencia ≈0.83 NRMSE (listón de éxito).
- **delay_recall** (solo externa): kmax=100, input U(-1,1). bptt≥kmax obligatorio
  (si kmax sube, bptt sube o la tarea es imposible para la red por construcción).

---

## Estado de campaña — Fase 2 (actualizado)

### Hecho
- **Reorganización de repo** ejecutada: configs/ y results/ bajo `prelim_study/`
  (sweeps, designs, external, experiments, components/{reservoirs,tasks,evaluators})
  y `final_campaign/` (frontier). Runners leen `output_dir` del config; reorg fue
  solo mover + reescribir `output_dir`. Fix aplicado: test_external_comparison_runner
  apuntaba a la ruta antigua del config externo.
- **Paso 0 — Frontera de estabilidad (ESP)**: módulo aislado `esp_frontier_runner`
  (envoltura de `evaluate_esp_sampled` + diagnósticos W) verificado y fiel.
  - Decisiones cerradas: kmax MC = N = 100 (opción 120 con corte en 100); arrancar
    Fase 0 aislada antes de fijar el grid del sweep; eps=1e-3.
  - Corrida v1 (T=4000, α=1): frontera nítida en ρ=1 a drive bajo. **Marco corregido:
    NO anclar parámetros al lab (todo lo previo = MVP) y NO imponer ρ≤1.**
  - Corrida dilución (T=8000, 3780 pts: ρ×s_in×α×seeds): summary.json/csv en
    `results/final_campaign/frontier/esp_frontier_dilution/`. Añadidos al runner:
    diagnóstico de amplitud (`saturation_mean` = ⟨|x|⟩, ¡es amplitud, no saturación!)
    + `saturation_frac` (|x|>0.99) + `nonsync_fraction_descending` (truncamiento).
- **Hallazgos clave** (material cap. análisis):
  - Frontera empírica se desplaza > ρ=1 con s_in (hasta ~1.5 a s_in≥0.8); robusto a α.
    σmax<1 (≈ρ 0.52) muy conservadora. Dilución genuina, sin saturación dura
    (sat_frac máx ≈0.013; amplitud media máx ≈0.56).
  - **Memoria ≠ ρ alto**: washout explota en ρ→1 SOLO a drive bajo; a drive alto el
    olvido es rápido aunque ρ=1.3–1.5. Memoria y no-linealidad viven en esquinas opuestas.
  - Leak: alarga washout y atenúa amplitud (desacopla escala temporal/amplitud del
    drive). Su frontera-localización NO es resoluble a T=8000 (truncamiento a α bajo).
  - washout=1000 (transitorio descartado del lab) validado para α∈[0.5,1] (sync ≤~436
    en ρ=1); solo se rompe con α≤0.3.
- **Visualización**: módulo reutilizable `src/rc_lab/viz/` (style, primitives, io,
  frontier, tables) + `scripts/plot_frontier.py` (bootstrap csv↔json). Deps añadidas:
  matplotlib + pandas. `io` lee CSV (interfaz tabular común a todos los runners).
  Figuras F1–F7 en `results/final_campaign/frontier/figs/`.
- **Memoria**: borrador LaTeX de la sección de frontera (`frontera_estabilidad.tex`,
  fuera del repo).

### Regiones candidatas para el sweep (fijadas, R³; α≈1 salvo R4)
- **R1 contractiva ref.**: ρ∈[0.6,0.9], s_in∈[0.05,0.2] — olvido rápido, casi lineal.
- **R2 cresta memoria**: ρ∈[0.9,1.0], s_in∈[0.05,0.2] — washout largo, baja no-lin.
- **R3 no-lineal extendida**: ρ∈[1.0,1.4], s_in∈[0.4,1.5] — ESP válida >1, alta
  amplitud, olvido rápido; apuesta = tareas de mezcla no lineal, no memoria.
- **R4 no-lineal con fuga**: α∈[0.3,0.5] + s_in alto — no-linealidad de R3 con más
  memoria y menos amplitud.

### Pendiente / próxima sesión
- **Cerrar regiones del sweep con las figuras delante** (R1–R4 ya casi fijas;
  decidir interiores exactos + puntos de control cruzando la frontera).
- **Redactar el config del sweep** (Paso 1) que barra esas subregiones (uniones en
  R³, no una caja). Recordar kmax MC = 100; tareas multitask.
- Cosméticos abiertos de figuras (PATCH 2, no bloqueante): frontera-escalera en
  F1–F3, mallado categórico (anchos iguales, ticks centrados), F7 (frontera = curvas
  por α + R1–R3 planas en α=1), F5 (polilínea continua sólido↔censurado). Ajustar
  caption F7 en el .tex tras rehacerla.
- Secundario (opcional, redondear cap. dilución): frontera del leak a T escalado
  ~1/α si se quiere resolver su localización.


### Limitaciones
- NARMA10 estándar inestable para L largos; train acotado a 2400/600/1000 (total 7000) para mantener finitas las 7 semillas; fix de fondo = NARMA10 acotado con tanh, bugfix futuro

---

## Fase 3 — Análisis estadístico (POST-campaña, pendiente)

Stack mínimo (marco de Demšar 2006, no paramétrico sobre rankings):
- **Friedman por tarea** (ómnibus sobre semillas) → si rechaza, post-hoc.
- **Nemenyi** (todos contra todos) o **Bonferroni-Dunn** (todos contra baseline)
  + **diagrama CD** por tarea.
- **aggregate_rank** como síntesis DESCRIPTIVA entre tareas (con T=3 tareas la
  inferencia entre-tareas es subdimensionada; declararlo). Probar también el
  Friedman agregado: si cycle_jump no domina ninguna tarea pero gana aggregate,
  es el patrón "mejor compromiso" y Friedman agregado lo refleja (probablemente
  no rechace con T=3, lo cual APOYA la narrativa de compromiso).
- **Tamaño de efecto** (probabilidad de superioridad / Wilcoxon pareado) junto a
  p-valores. Corrección por comparaciones múltiples (Nemenyi ya la incorpora).
- Validable con datos sintéticos (reproducir ejemplo de Demšar) mientras corre
  la campaña.

---

## Orden de ejecución de la campaña

1. Estudio de frontera ESP (Paso 0) → fija rango de ρ.
2. Sweep de selección de región (Paso 1) → región robusta (shortlist).
3. Verificación de escalado (Paso 2, opcional) → robustez de la región.
4. Capa de design (Paso 3) → familias + baseline + diagnósticos, N=100.
5. Comparación externa (Paso 4) → vs RNN/LSTM/NG-RC, ~2.5h CPU.
6. Análisis estadístico (Fase 3) → Friedman/Nemenyi/CD.
