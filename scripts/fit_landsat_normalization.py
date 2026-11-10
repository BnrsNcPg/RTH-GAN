"""Fit quantitative normalization on training scenes without modifying sources."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.landsat import fit_landsat_normalization
from experiment_utils import read_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--quality-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing statistics: {args.output}")
    statistics = fit_landsat_normalization(args.manifest, read_yaml(args.quality_config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved training-subset normalization: {args.output}")


if __name__ == "__main__":
    main()
