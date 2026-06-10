"""Load scalar features from melody-features .npz archives."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .catalog import parse_filename
from .schema import scalar_feature_keys

DEFAULT_FEATURES_DIR = Path(
    os.environ.get("BIMMUDA_FEATURES_DIR", "/Users/davidwhyatt/features")
)


def load_npz(path: Path | str) -> dict[str, float]:
    """Load scalar features from a single .npz file."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        features = {}
        for key in scalar_feature_keys(list(data.files)):
            value = data[key]
            if value.shape != ():
                continue
            features[key] = float(value.item())
        return features


def load_features_dir(
    directory: Path | str | None = None,
    *,
    pattern: str = "*.npz",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Load all scalar features in a directory into one DataFrame."""
    directory = Path(directory or DEFAULT_FEATURES_DIR)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")

    rows = []
    iterator = tqdm(files, desc="Loading features") if show_progress else files
    for path in iterator:
        meta = parse_filename(path.stem)
        row = {
            "file": path.name,
            **meta,
            **load_npz(path),
        }
        rows.append(row)

    return pd.DataFrame(rows)
