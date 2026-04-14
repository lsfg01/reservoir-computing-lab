"""
Punto de entrada principal del laboratorio ESN — baseline clásico.

Uso:
    uv run scripts/run_baseline.py
    uv run scripts/run_baseline.py --config configs/experiments/baseline_narma10.yaml
"""
import argparse
import sys
from pathlib import Path

# Asegurar que src/ está en el path cuando se ejecuta directamente
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rc_lab.runners.runner import ExperimentRunner
from rc_lab.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecutar baseline ESN")
    parser.add_argument(
        "--config",
        default="configs/experiments/baseline_narma10.yaml",
        help="Ruta al archivo de configuración del experimento",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config["experiment"]["name"]
    seeds = config["experiment"]["seeds"]

    print(f"Experimento : {exp_name}")
    print(f"Seeds       : {seeds}")
    print(f"Reservoir   : {config['reservoir']['type']}  N={config['reservoir']['N']}")
    print(f"Task        : {config['task']['name']}")
    print("-" * 50)

    runner = ExperimentRunner(config)
    results = runner.run_multi_seed(seeds)

    print(f"\n{'seed':>6}  {'nmse':>10}  {'rmse':>10}  {'train_s':>8}  {'total_s':>8}")
    print("-" * 50)
    for r in results:
        print(
            f"{r.seed:>6}  "
            f"{r.metrics.get('nmse', float('nan')):>10.4f}  "
            f"{r.metrics.get('rmse', float('nan')):>10.4f}  "
            f"{r.timing['train_s']:>8.3f}  "
            f"{r.timing['total_s']:>8.3f}"
        )

    output_dir = config["experiment"].get("output_dir", "results")
    print(f"\nResultados guardados en: {output_dir}/")


if __name__ == "__main__":
    main()
