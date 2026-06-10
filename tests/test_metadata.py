from pathlib import Path

import pandas as pd
import pytest

from bimmuda_feature_analysis.metadata import (
    attach_chart_metadata,
    load_chart_metadata,
    normalize_song_key,
)


@pytest.fixture()
def metadata_csv(tmp_path: Path) -> Path:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "Title,Artist,Year,Position,Genre (Broad 1),Genre (Broad 2)\n"
        "Easy On Me,Adele,2021,1,Pop,N/A\n"
        "Mood,24kGoldn feat. Iann Dior,2020,5,Pop,N/A\n"
    )
    return path


def test_normalize_song_key():
    assert normalize_song_key("Adele", "Easy On Me") == "adele|||easy on me"


def test_load_chart_metadata(metadata_csv: Path):
    meta = load_chart_metadata(metadata_csv)
    assert list(meta["chart_year"]) == [2021, 2020]
    assert list(meta["decade"]) == [2020, 2020]


def test_attach_chart_metadata(metadata_csv: Path):
    df = pd.DataFrame(
        {
            "artist": ["Adele", "24kGoldn feat. Iann Dior", "Unknown"],
            "title": ["Easy On Me", "Mood", "Song"],
        }
    )
    with pytest.raises(ValueError, match="Chart metadata missing"):
        attach_chart_metadata(df, metadata_csv)

    result = attach_chart_metadata(df.iloc[:2], metadata_csv)
    assert result["chart_year"].tolist() == [2021, 2020]
    assert result["decade"].tolist() == [2020, 2020]
