#!/usr/bin/env python3
"""Plot dataset action quality PCA panels from static_analysis exports.

Usage:
    python experiments/action_cloud/static_analysis/plot_cloud_quality.py --all
    python experiments/action_cloud/static_analysis/plot_cloud_quality.py --slug mix_divpen_b5_beta0.05_ideal_init
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STATIC_DIR))

from cloud_quality_common import plot_cloud_quality_panels

EXPORT_ROOT = Path(__file__).resolve().parent / "exports"
FIG_ROOT = Path(__file__).resolve().parent / "figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--export_dir", type=str, default=str(EXPORT_ROOT))
    parser.add_argument("--figure_dir", type=str, default=str(FIG_ROOT))
    args = parser.parse_args()

    export_root = Path(args.export_dir)
    fig_root = Path(args.figure_dir)

    if args.all:
        slugs = sorted(p.name for p in export_root.iterdir() if p.is_dir())
    elif args.slug:
        slugs = [args.slug]
    else:
        slugs = sorted(p.name for p in export_root.iterdir() if (p / "dataset_cloud.npz").exists())
        if not slugs:
            raise SystemExit("No exports found. Run extract_cloud_quality.py first.")

    for slug in slugs:
        export_path = export_root / slug / "dataset_cloud.npz"
        if not export_path.exists():
            print(f"SKIP {slug}: missing {export_path}")
            continue
        out_dir = fig_root / slug
        panels, reward = plot_cloud_quality_panels(export_path, out_dir)
        print(f"Saved {panels}")
        print(f"Saved {reward}")


if __name__ == "__main__":
    main()
