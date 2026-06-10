from pathlib import Path

import numpy as np
import pytest

from bimmuda_feature_analysis import load_features_dir, load_npz, parse_filename


FEATURES_DIR = Path("/Users/davidwhyatt/features")


@pytest.fixture(scope="module")
def sample_npz() -> Path:
    files = sorted(FEATURES_DIR.glob("*.npz"))
    if not files:
        pytest.skip("features directory not available")
    return files[0]


def test_parse_filename():
    meta = parse_filename("Adele_-_Easy_On_Me")
    assert meta["artist"] == "Adele"
    assert meta["title"] == "Easy On Me"
    assert meta["song_id"] == "Adele_-_Easy_On_Me"


def test_parse_filename_with_feat():
    meta = parse_filename("24kGoldn_feat._Iann_Dior_-_Mood")
    assert "24kGoldn" in meta["artist"]
    assert meta["title"] == "Mood"


def test_load_npz_returns_scalars_only(sample_npz: Path):
    features = load_npz(sample_npz)
    assert features
    assert all(isinstance(v, float) for v in features.values())
    assert "absolute_pitch.mean_pitch" in features


@pytest.mark.slow
def test_load_features_dir_shape():
    df = load_features_dir(FEATURES_DIR, show_progress=False)
    assert len(df) == 366
    assert "artist" in df.columns
    assert "timing.npvi" in df.columns
    assert df["timing.npvi"].notna().any()
