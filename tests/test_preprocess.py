import pandas as pd
import pytest

from bimmuda_feature_analysis.preprocess import prepare_efa_matrix


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file": ["a.npz", "b.npz", "c.npz"],
            "song_id": ["a", "b", "c"],
            "artist": ["A1", "A2", "A3"],
            "title": ["T1", "T2", "T3"],
            "absolute_pitch.mean_pitch": [60.0, 62.0, 61.0],
            "timing.npvi": [0.4, 0.42, 0.39],
            "tonality.tonalness": [0.8, 0.75, 0.82],
            "constant.feature": [1.0, 1.0, 1.0],
        }
    )


def test_prepare_efa_matrix_zero_fill_and_scale():
    matrix, columns, _ = prepare_efa_matrix(_toy_df())
    assert "constant.feature" not in columns
    assert len(columns) == 3
    assert matrix.shape == (3, 3)
    assert matrix.mean().abs().max() < 1e-10


def test_prepare_efa_matrix_handles_nan():
    df = _toy_df()
    df.loc[0, "timing.npvi"] = float("inf")
    matrix, columns, _ = prepare_efa_matrix(df)
    assert matrix["timing.npvi"].notna().all()
