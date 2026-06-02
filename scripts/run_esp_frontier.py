"""
run_esp_frontier.py — Estudio de frontera ESP (Paso 0 de la campaña).

Uso:
    uv run python scripts/run_esp_frontier.py --config configs/frontier/esp_frontier_v1.yaml
    uv run python scripts/run_esp_frontier.py --config configs/frontier/esp_frontier_v1.yaml --dry-run
"""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rc_lab.utils.io import load_config


def _n_points(config: dict) -> int:
    families = config["families"]
    grid = config["grid"]
    seeds = config["frontier"]["seeds"]
    return (
        len(families)
        * len(grid["input_scaling"])
        * len(grid["leak_rate"])
        * len(grid["spectral_radius"])
        * len(seeds)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estudio de frontera ESP para reservoirs")
    parser.add_argument("--config", required=True, help="Ruta al YAML de configuración")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar número de puntos sin ejecutar")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        families = config["families"]
        grid = config["grid"]
        seeds = config["frontier"]["seeds"]
        n_fam = len(families)
        n_sin = len(grid["input_scaling"])
        n_alpha = len(grid["leak_rate"])
        n_rho = len(grid["spectral_radius"])
        n_seeds = len(seeds)
        total = n_fam * n_sin * n_alpha * n_rho * n_seeds
        print(f"Familias:          {n_fam}  ({[f['name'] for f in families]})")
        print(f"input_scaling:     {n_sin}  {grid['input_scaling']}")
        print(f"leak_rate:         {n_alpha}  {grid['leak_rate']}")
        print(f"spectral_radius:   {n_rho}  {grid['spectral_radius']}")
        print(f"Seeds:             {n_seeds}")
        print(f"Total puntos:      {total}  (= {n_fam}×{n_sin}×{n_alpha}×{n_rho}×{n_seeds})")
        return

    from rc_lab.runners.esp_frontier_runner import ESPFrontierRunner

    runner = ESPFrontierRunner(config)
    summary = runner.run()

    # Tabla resumen: rho -> washout_mean, fraction_sync_mean, sigma_max
    print(f"\n{'rho_target':>10}  {'frac_sync':>10}  {'sync_time':>10}  {'sigma_max':>10}  {'rho_real':>10}  family")
    print("-" * 72)
    for row in summary["rows"]:
        frac = row["fraction_synchronized_mean"]
        st = row["sync_time_mean_mean"]
        st_str = f"{st:10.1f}" if st is not None else f"{'N/A':>10}"
        sigma = row["sigma_max_mean"]
        rho_r = row["rho_real_mean"]
        print(
            f"{row['rho_target']:>10.3f}  {frac:>10.3f}  {st_str}  {sigma:>10.4f}  {rho_r:>10.4f}  {row['family_name']}"
        )

    print(f"\nResultados en: {config['frontier']['output_dir']}")


if __name__ == "__main__":
    main()
