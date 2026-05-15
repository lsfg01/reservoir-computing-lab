"""
Run an external comparison between ESN/RC reservoirs and sequence models.

Usage:
    uv run python scripts/run_external_comparison.py --config configs/external/esn_vs_rnn_lstm.yaml
    uv run python scripts/run_external_comparison.py --config configs/external/esn_vs_rnn_lstm.yaml --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rc_lab.runners.external_comparison_runner import ExternalComparisonRunner
from rc_lab.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="External ESN vs RNN/LSTM/NARX comparison")
    parser.add_argument("--config", required=True, help="Path to the external comparison YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate and expand grids without training")
    args = parser.parse_args()

    config = load_config(args.config)
    runner = ExternalComparisonRunner(config)
    if args.dry_run:
        runner.dry_run()
        return

    runner.run()
    output_dir = Path(config["sweep"]["output_dir"])
    print(f"\nExternal comparison completed. Results in: {output_dir / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()

