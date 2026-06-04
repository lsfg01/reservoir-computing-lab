"""
run_multitask_sweep.py — Barrido multi-tarea del baseline ESN.

Evalúa cada configuración del reservoir en NARMA-10, Mackey-Glass y Memory
Capacity, agrega los resultados por seeds y produce un ranking inter-tarea
con una shortlist de configuraciones candidatas.

Uso:
    python scripts/run_multitask_sweep.py --config configs/sweeps/pilot_multitask.yaml
    python scripts/run_multitask_sweep.py --config configs/sweeps/pilot_multitask.yaml --dry-run
"""

import argparse
import sys
from pathlib import Path

# Asegurar que src/ está en el path cuando se ejecuta como script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rc_lab.runners.sweep_runner import _resolve_grid_spec, expand_grid
from rc_lab.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Barrido multi-tarea ESN: NARMA-10, Mackey-Glass y Memory Capacity"
    )
    parser.add_argument(
        "--config", required=True,
        help="Ruta al YAML de configuración del barrido multi-tarea",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mostrar tamaño del grid sin ejecutar ninguna corrida",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # --- Dry-run ---
    if args.dry_run:
        grid_points = expand_grid(_resolve_grid_spec(config))
        seeds = config["sweep"]["seeds"]
        n_tasks = 3
        print(f"Configuraciones: {len(grid_points)}")
        print(f"Seeds:           {len(seeds)}")
        print(f"Total corridas:  {len(grid_points)} x {len(seeds)} x {n_tasks} tareas = "
              f"{len(grid_points) * len(seeds) * n_tasks}")
        return

    # --- Ejecución completa ---
    from rc_lab.runners.multitask_sweep_runner import MultiTaskSweepRunner

    runner = MultiTaskSweepRunner(config)
    summary = runner.run()

    # --- Tabla de resultados ---
    n10_metric = summary.configs[0].narma10_primary_metric if summary.configs else "nmse"
    mg_metric  = summary.configs[0].mg_primary_metric      if summary.configs else "nmse"

    col_w = 14
    print(f"\n{'config_id':<{col_w}}  {'agg_rank':>8}  "
          f"{'n10_val_' + n10_metric:>14}  "
          f"{'mg_val_' + mg_metric:>13}  "
          f"{'mc_total':>10}  "
          f"{'r_n10':>5}  {'r_mg':>5}  {'r_mc':>5}")
    print("-" * 90)

    for entry in summary.configs:
        shortlist_marker = " *" if entry.config_id in summary.shortlist else ""
        print(
            f"{entry.config_id:<{col_w}}  "
            f"{entry.aggregate_rank:>8.2f}  "
            f"{entry.narma10_val_mean:>14.4f}  "
            f"{entry.mg_val_mean:>13.4f}  "
            f"{entry.mc_total_mean:>10.2f}  "
            f"{entry.rank_narma10:>5}  "
            f"{entry.rank_mg:>5}  "
            f"{entry.rank_mc:>5}"
            f"{shortlist_marker}"
        )

    print(f"\nShortlist (top-{summary.shortlist_top_n} por aggregate_rank):")
    for cid in summary.shortlist:
        entry = next(e for e in summary.configs if e.config_id == cid)
        print(f"  {cid}  agg_rank={entry.aggregate_rank:.2f}  "
              f"n10={entry.narma10_val_mean:.4f}  "
              f"mg={entry.mg_val_mean:.4f}  "
              f"mc={entry.mc_total_mean:.2f}")

    print(f"\nResultados en: {config['sweep']['output_dir']}")


if __name__ == "__main__":
    main()
