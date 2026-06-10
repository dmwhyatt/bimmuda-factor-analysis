"""Join BiMMuDa chart metadata (year, position, genre) onto feature tables."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

DEFAULT_METADATA_PATH = Path(
    os.environ.get(
        "BIMMUDA_METADATA_CSV",
        "/Users/davidwhyatt/AMT-evaluation/data/bimmuda/metadata/bimmuda_per_song_metadata.csv",
    )
)

METADATA_COLUMNS = (
    "chart_year",
    "chart_position",
    "decade",
    "genre_broad_1",
    "genre_broad_2",
)


def normalize_song_key(artist: str, title: str) -> str:
    """Build a stable join key from artist and title labels."""

    def _norm(text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", str(text).lower().strip())
        return re.sub(r"\s+", " ", cleaned).strip()

    return f"{_norm(artist)}|||{_norm(title)}"


def load_chart_metadata(path: Path | str | None = None) -> pd.DataFrame:
    """Load BiMMuDa per-song metadata and normalise column names."""
    path = Path(path or DEFAULT_METADATA_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"Chart metadata not found: {path}")

    raw = pd.read_csv(path)
    renamed = raw.rename(
        columns={
            "Title": "title",
            "Artist": "artist",
            "Year": "chart_year",
            "Position": "chart_position",
            "Genre (Broad 1)": "genre_broad_1",
            "Genre (Broad 2)": "genre_broad_2",
        }
    )
    renamed["song_key"] = renamed.apply(
        lambda row: normalize_song_key(row["artist"], row["title"]),
        axis=1,
    )
    renamed["decade"] = (renamed["chart_year"] // 10) * 10
    return renamed.drop_duplicates("song_key")


def attach_chart_metadata(
    df: pd.DataFrame,
    path: Path | str | None = None,
    *,
    require_match: bool = True,
) -> pd.DataFrame:
    """Attach chart year and decade columns by matching artist/title."""
    metadata = load_chart_metadata(path)
    result = df.copy()
    result["song_key"] = result.apply(
        lambda row: normalize_song_key(row["artist"], row["title"]),
        axis=1,
    )
    join_cols = ["song_key", *METADATA_COLUMNS]
    result = result.merge(metadata[join_cols], on="song_key", how="left")
    result = result.drop(columns="song_key")

    missing = result["chart_year"].isna().sum()
    if missing and require_match:
        sample = result.loc[result["chart_year"].isna(), ["artist", "title"]].head(5)
        raise ValueError(
            f"Chart metadata missing for {missing} songs. Examples:\n{sample.to_string(index=False)}"
        )
    return result
