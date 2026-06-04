"""
plot_frontier.py — Generate ESP frontier figures (F1–F8).

Usage:
    uv run python scripts/plot_frontier.py
    uv run python scripts/plot_frontier.py --summary path/to/summary.csv
    uv run python scripts/plot_frontier.py --summary path/to/summary.csv \\
        --out results/figs/ --runs-dir path/to/runs/ --regions regions.json

Bootstrap: if --summary (*.csv) is missing but a sibling summary.json exists,
the CSV is regenerated once from the JSON rows and the script continues.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_DEFAULT_SUMMARY = (
    "results/final_campaign/frontier/esp_frontier_dilution/summary.csv"
)
_DEFAULT_OUT = "results/final_campaign/frontier/figs/"


def _bootstrap_csv(csv_path: Path) -> bool:
    """
    If csv_path is absent but a sibling summary.json exists, regenerate the CSV.
    Returns True if CSV is now available.
    """
    if csv_path.exists():
        return True
    json_path = csv_path.with_suffix(".json")
    if not json_path.exists():
        return False

    import pandas as pd

    print(f"[bootstrap] {csv_path.name} not found; generating from {json_path.name} …")
    with open(json_path, encoding="utf-8") as f:
        summary = json.load(f)
    rows = summary.get("rows", [])
    if not rows:
        print("[bootstrap] JSON has no 'rows'; cannot generate CSV.")
        return False
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[bootstrap] wrote {csv_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ESP frontier figures")
    parser.add_argument(
        "--summary",
        default=_DEFAULT_SUMMARY,
        help=f"Path to summary CSV (default: {_DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        help=f"Output directory for figures (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Directory of per-point JSON files (needed for F8 only)",
    )
    parser.add_argument(
        "--regions",
        default=None,
        help="JSON file with candidate regions list (optional; uses CANDIDATE_REGIONS if omitted)",
    )
    args = parser.parse_args()

    csv_path = Path(args.summary)
    if not _bootstrap_csv(csv_path):
        print(f"ERROR: summary CSV not found and cannot bootstrap from JSON: {csv_path}")
        sys.exit(1)

    regions = None
    if args.regions is not None:
        with open(args.regions, encoding="utf-8") as f:
            regions = json.load(f)

    from rc_lab.viz.frontier import CANDIDATE_REGIONS, build_frontier_figures

    kwargs: dict = {
        "summary_csv": csv_path,
        "out_dir": args.out,
    }
    if args.runs_dir is not None:
        kwargs["runs_dir"] = args.runs_dir
    if regions is not None:
        kwargs["regions"] = regions
    else:
        kwargs["regions"] = CANDIDATE_REGIONS

    paths = build_frontier_figures(**kwargs)

    print(f"\nFiguras generadas ({len(paths)}):")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
