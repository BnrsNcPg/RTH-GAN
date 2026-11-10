"""Create a new source-preserving Landsat JPEG intensity dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.landsat_jpeg_preparation import prepare_landsat_jpeg
from experiment_utils import read_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Scene or explicit grid-window CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory outside source raster directories")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset/landsat_jpeg_preparation.yaml"))
    args = parser.parse_args()
    report = prepare_landsat_jpeg(args.manifest, args.output_dir, read_yaml(args.config), configuration_path=args.config)
    print(json.dumps({"status": report["status"], "accepted_counts": report["accepted_counts"],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
