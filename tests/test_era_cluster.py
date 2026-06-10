import pandas as pd

from bimmuda_feature_analysis.cluster import (
    cluster_by_decade,
    cluster_songs_within_eras,
    era_cluster_summary,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file": [f"s{i}.npz" for i in range(6)],
            "song_id": [f"s{i}" for i in range(6)],
            "artist": [f"A{i}" for i in range(6)],
            "title": [f"T{i}" for i in range(6)],
            "chart_year": [1980, 1981, 1982, 1990, 1991, 1992],
            "decade": [1980, 1980, 1980, 1990, 1990, 1990],
            "absolute_pitch.mean_pitch": [60.0, 62.0, 61.0, 60.5, 62.5, 61.5],
            "timing.npvi": [0.4, 0.42, 0.39, 0.41, 0.43, 0.38],
            "tonality.tonalness": [0.8, 0.75, 0.82, 0.79, 0.76, 0.81],
        }
    )


def test_cluster_by_decade():
    result = cluster_by_decade(_toy_df())
    assert set(result["cluster"]) == {1980, 1990}
    assert result.attrs["clustering_method"] == "decade"


def test_cluster_songs_within_eras():
    result = cluster_songs_within_eras(_toy_df(), n_clusters=2, min_era_size=3)
    assert "cluster" in result.columns
    assert "era" in result.columns
    assert set(result["era"]) == {1980, 1990}
    assert result.attrs["clustering_method"] == "within_era"


def test_era_cluster_summary():
    clustered = cluster_songs_within_eras(_toy_df(), n_clusters=2, min_era_size=3)
    summary = era_cluster_summary(clustered)
    assert {"era", "cluster", "n_songs"}.issubset(summary.columns)
    assert summary["n_songs"].sum() == len(clustered)
