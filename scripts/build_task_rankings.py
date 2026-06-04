"""
build_task_rankings.py -- Post-hoc task-wise global rankings.

Usage:
    python scripts/build_task_rankings.py \
      --comparison-json results/final_campaign/design/comparison_summary.json \
      --top-n 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rc_lab.analysis.task_rankings import load_comparison_summary, save_task_rankings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build post-hoc global rankings per task from comparison_summary.json",
    )
    parser.add_argument(
        "--comparison-json",
        required=True,
        help="Path to comparison_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the comparison JSON parent directory.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top rows per task to keep in task_rankings.json",
    )
    args = parser.parse_args()

    comparison_json = Path(args.comparison_json)
    output_dir = Path(args.output_dir) if args.output_dir else comparison_json.parent

    payload = load_comparison_summary(comparison_json)
    paths = save_task_rankings(payload, output_dir, top_n=args.top_n)

    print("Task rankings written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

