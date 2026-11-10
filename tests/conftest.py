from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data.manifest import MANIFEST_COLUMNS, save_manifest


@pytest.fixture
def paired_manifest(tmp_path):
    visible_dir, infrared_dir = tmp_path / "visible", tmp_path / "infrared"
    visible_dir.mkdir()
    infrared_dir.mkdir()
    grid = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    visible = np.stack([grid, grid, grid], axis=-1)
    Image.fromarray(visible, mode="RGB").save(visible_dir / "0001.png")
    Image.fromarray(grid, mode="L").save(infrared_dir / "0001.png")
    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update({
        "pair_id": "aviid1_000001", "dataset": "AVIID", "subset": "AVIID1",
        "visible_path": str(visible_dir / "0001.png"), "infrared_path": str(infrared_dir / "0001.png"),
        "visible_filename": "0001.png", "infrared_filename": "0001.png",
        "visible_width": 32, "visible_height": 32, "infrared_width": 32, "infrared_height": 32,
        "visible_channels": 3, "infrared_channels": 1, "infrared_is_replicated_gray": False,
        "controlled_split": "train", "pairing_method": "test", "registration_status": "PASS",
    })
    path = tmp_path / "datasets" / "manifests" / "all_pairs.csv"
    save_manifest(pd.DataFrame([row]), path)
    return path

