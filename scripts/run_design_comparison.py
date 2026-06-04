"""
run_design_comparison.py — Comparación controlada entre familias de reservoir.

Ejecuta el experimento de comparación de diseños definido en un YAML de
configuración, instanciando ``DesignComparisonRunner`` y produciendo la tabla
comparativa ``comparison_summary.csv`` en el directorio de salida.

Uso:
    python scripts/run_design_comparison.py --config configs/designs/cycle_comparison.yaml
    python scripts/run_design_comparison.py --config configs/designs/cycle_comparison.yaml --dry-run
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
        description="Comparación de diseños de reservoir: cycle vs random_sparse baseline"
    )
    parser.add_argument(
        "--config", required=True,
        help="Ruta al YAML de configuración del experimento de comparación",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mostrar tamano del grid de diseno sin ejecutar ninguna corrida",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    from rc_lab.runners.design_comparison_runner import DesignComparisonRunner

    runner = DesignComparisonRunner(config)
    if args.dry_run:
        grid_points = expand_grid(_resolve_grid_spec(config))
        seeds = config["sweep"]["seeds"]
        designs = config["designs"]
        enabled_tasks = [
            task_name
            for task_name in ("narma10", "mackey_glass", "memory_capacity")
            if config["tasks"].get(task_name, {}).get("enabled", True)
        ]

        print(f"Configuraciones por diseno: {len(grid_points)}")
        print(f"Disenos:                    {len(designs)}")
        print(f"Seeds:                      {len(seeds)}")
        print(f"Tareas activas:             {len(enabled_tasks)} ({', '.join(enabled_tasks)})")
        print(
            f"Total corridas:             {len(grid_points)} x {len(designs)} "
            f"x {len(seeds)} x {len(enabled_tasks)} tareas = "
            f"{len(grid_points) * len(designs) * len(seeds) * len(enabled_tasks)}"
        )
        return

    runner.run()

    output_dir = Path(config["sweep"]["output_dir"])
    csv_path = output_dir / "comparison_summary.csv"
    print(f"\nComparación completada. Resultados en: {csv_path}")


if __name__ == "__main__":
    main()
