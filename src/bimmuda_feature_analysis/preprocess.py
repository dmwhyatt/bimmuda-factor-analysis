"""Numeric preprocessing aligned with Style-Classification-Analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .cluster import feature_columns
from .metadata import METADATA_COLUMNS

EXPORT_META_COLUMNS = ("file", "song_id", "artist", "title", *METADATA_COLUMNS)


def export_raw_features(df: pd.DataFrame, output_path: Path) -> Path:
    """Write metadata + unscaled numeric features for R scoring scripts."""
    columns = feature_columns(df)
    if not columns:
        raise ValueError("No feature columns found in DataFrame")

    matrix = df[columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    meta_cols = [col for col in EXPORT_META_COLUMNS if col in df.columns]
    export = pd.concat(
        [df[meta_cols].reset_index(drop=True), matrix.reset_index(drop=True)],
        axis=1,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_path, index=False)
    return output_path


def prepare_efa_matrix(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    drop_zero_variance: bool = True,
) -> tuple[pd.DataFrame, list[str], StandardScaler]:
    """Build a z-scored feature matrix for EFA.

    Matches ``feature_selection.prepare_numeric_feature_matrix`` and
    ``factor_logistic.R``: inf → NA → 0, drop zero-variance columns, scale.
    """
    columns = columns or feature_columns(df)
    if not columns:
        raise ValueError("No feature columns found in DataFrame")

    matrix = df[columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if drop_zero_variance:
        stds = matrix.std()
        kept = stds.index[(stds > 0) & stds.notna()].tolist()
        dropped = len(columns) - len(kept)
        if dropped:
            matrix = matrix[kept]
        columns = kept

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(matrix),
        columns=columns,
        index=matrix.index,
    )
    return scaled, columns, scaler
