import numpy as np
import pandas as pd
import pytest

from bimmuda_feature_analysis.cluster import (
    cluster_songs,
    feature_columns,
    prepare_feature_matrix,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file": ["a.npz", "b.npz", "c.npz", "d.npz"],
            "song_id": ["a", "b", "c", "d"],
            "artist": ["A1", "A2", "A3", "A4"],
            "title": ["T1", "T2", "T3", "T4"],
            "absolute_pitch.mean_pitch": [60.0, 62.0, 61.0, 90.0],
            "timing.npvi": [0.4, 0.42, 0.39, 0.1],
            "tonality.tonalness": [0.8, 0.75, 0.82, 0.2],
        }
    )


def test_feature_columns():
    cols = feature_columns(_toy_df())
    assert cols == [
        "absolute_pitch.mean_pitch",
        "timing.npvi",
        "tonality.tonalness",
    ]


def test_prepare_feature_matrix_shape():
    matrix, columns, _ = prepare_feature_matrix(_toy_df())
    assert matrix.shape == (4, 3)
    assert len(columns) == 3
    assert np.isfinite(matrix).all()


def test_cluster_songs_assigns_labels():
    result = cluster_songs(_toy_df(), n_clusters=2, random_state=0)
    assert "cluster" in result.columns
    assert "pca_x" in result.columns
    assert set(result["cluster"]) <= {0, 1}
    assert "silhouette" in result.attrs
