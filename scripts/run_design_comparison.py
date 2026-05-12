"""
run_design_comparison.py — Comparación controlada entre familias de reservoir.

Ejecuta el experimento de comparación de diseños definido en un YAML de
configuración, instanciando ``DesignComparisonRunner`` y produciendo la tabla
comparativa ``comparison_summary.csv`` en el directorio de salida.

Uso:
    python scripts/run_design_comparison.py --config configs/designs/cycle_comparison.yaml
"""

import argparse
import sys
from pathlib import Path

# Asegurar que src/ está en el path cuando se ejecuta como script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rc_lab.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparación de diseños de reservoir: cycle vs random_sparse baseline"
    )
    parser.add_argument(
        "--config", required=True,
        help="Ruta al YAML de configuración del experimento de comparación",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    from rc_lab.runners.design_comparison_runner import DesignComparisonRunner

    runner = DesignComparisonRunner(config)
    runner.run()

    output_dir = Path(config["sweep"]["output_dir"])
    csv_path = output_dir / "comparison_summary.csv"
    print(f"\nComparación completada. Resultados en: {csv_path}")


if __name__ == "__main__":
    main()
