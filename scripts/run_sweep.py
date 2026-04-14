"""
run_sweep.py — Punto de entrada para barridos de hiperparámetros del baseline ESN.

Uso:
    python scripts/run_sweep.py --config configs/sweeps/pilot_narma10.yaml
    python scripts/run_sweep.py --config configs/sweeps/pilot_narma10.yaml --dry-run
"""

import argparse
import itertools
import sys
from pathlib import Path

# Asegurar que src/ está en el path cuando se ejecuta como script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rc_lab.utils.io import load_config


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Barrido de hiperparámetros ESN baseline")
    parser.add_argument("--config", required=True, help="Ruta al YAML de configuración del sweep")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar tamaño del grid sin ejecutar")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        grid = expand_grid(config["grid"])
        seeds = config["sweep"]["seeds"]
        print(f"Configuraciones: {len(grid)}")
        print(f"Seeds:           {len(seeds)}")
        print(f"Total corridas:  {len(grid) * len(seeds)}")
        return

    from rc_lab.runners.sweep_runner import SweepRunner

    runner = SweepRunner(config)
    summary = runner.run()

    # Tabla resumen
    print(f"\n{'config_id':<14}  {'val_nmse':>10}  {'±':>6}  {'test_nmse':>10}  {'±':>6}")
    print("-" * 56)
    for cfg in sorted(summary.configs, key=lambda c: c.val_mean.get("nmse", float("inf"))):
        val_m = cfg.val_mean.get("nmse", float("nan"))
        val_s = cfg.val_std.get("nmse", float("nan"))
        tst_m = cfg.test_mean.get("nmse", float("nan"))
        tst_s = cfg.test_std.get("nmse", float("nan"))
        marker = " *" if cfg.config_id == summary.best_config_id else ""
        print(f"{cfg.config_id:<14}  {val_m:>10.4f}  {val_s:>6.4f}  {tst_m:>10.4f}  {tst_s:>6.4f}{marker}")

    print(f"\nMejor config (por val): {summary.best_config_id}")
    print(f"Resultados en:          {config['sweep']['output_dir']}")


if __name__ == "__main__":
    main()
