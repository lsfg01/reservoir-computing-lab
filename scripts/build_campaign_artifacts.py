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
    parser.add_argument("--external-dir", default=None, help="Directory containing external comparison_summary.csv")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg", "png"],
        help="Figure formats to write, e.g. pdf svg png",
    )
    parser.add_argument(
        "--run-diagnostic",
        action="store_true",
        default=False,
        help="Re-run selected torch candidates with record_history=True to produce "
             "F_external_training_curves (§5). Requires --external-dir.",
    )
    parser.add_argument(
        "--diagnostic-seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Seeds for the torch diagnostic re-run.",
    )
    parser.add_argument(
        "--diagnostic-device",
        default="cpu",
        help="PyTorch device for the diagnostic re-run (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--kmax",
        type=int,
        default=100,
        help="kmax used in the delay_recall task (controls ESN readout size, default=100).",
    )
    parser.add_argument(
        "--esn-n",
        type=int,
        default=100,
        dest="esn_n",
        help="ESN reservoir size N (from reservoir config, default=100). "
             "Used to compute per-task readout parameter counts without inversion.",
    )
    args = parser.parse_args()

    if args.run_diagnostic and args.external_dir is None:
        parser.error("--run-diagnostic requires --external-dir")

    outputs = build_campaign_artifacts(
        sweep_dir=args.sweep_dir,
        design_dir=args.design_dir,
        out_dir=args.out_dir,
        formats=tuple(args.formats),
        external_dir=args.external_dir,
        kmax=args.kmax,
        esn_N=args.esn_n,
        run_diagnostic=args.run_diagnostic,
        diagnostic_seeds=tuple(args.diagnostic_seeds),
        diagnostic_device=args.diagnostic_device,
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
