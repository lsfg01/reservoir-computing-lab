"""Build final-campaign tables and figures from persisted sweep/design results.

Example:
    python scripts/build_campaign_artifacts.py \
        --sweep-dir results/final_campaign/sweep \
        --design-dir results/final_campaign/design \
        --out-dir results/final_campaign/artifacts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rc_lab.viz.campaign import build_campaign_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build manuscript-ready final-campaign artifacts from summary CSVs.",
    )
    parser.add_argument("--sweep-dir", required=True, help="Directory containing sweep summary.csv")
    parser.add_argument("--design-dir", required=True, help="Directory containing design comparison_summary.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for tables and figures")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg", "png"],
        help="Figure formats to write, e.g. pdf svg png",
    )
    args = parser.parse_args()

    outputs = build_campaign_artifacts(
        sweep_dir=args.sweep_dir,
        design_dir=args.design_dir,
        out_dir=args.out_dir,
        formats=tuple(args.formats),
    )

    print("Campaign artifacts written:")
    for group, value in outputs.items():
        print(f"  {group}:")
        if isinstance(value, dict):
            for name, path in value.items():
                print(f"    {name}: {path}")
        else:
            for path in value:
                print(f"    {path}")


if __name__ == "__main__":
    main()
