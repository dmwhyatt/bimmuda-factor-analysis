from pathlib import Path

from bimmuda_feature_analysis.billboard_download import (
    audio_filename,
    build_search_query,
    load_download_manifest,
    save_download_manifest,
)
from bimmuda_feature_analysis.billboard_wiki import ChartEntry


def test_audio_filename_is_unique_and_safe():
    entry = ChartEntry(2025, 9, 'APT.', 'Rosé and Bruno Mars')
    assert audio_filename(entry) == "2025_009 - Rosé and Bruno Mars - APT..mp3"


def test_build_search_query():
    entry = ChartEntry(2020, 1, "Blinding Lights", "The Weeknd")
    assert build_search_query(entry) == "The Weeknd Blinding Lights official audio"


def test_download_manifest_roundtrip(tmp_path: Path):
    from bimmuda_feature_analysis.billboard_download import DownloadRecord

    records = {
        "2020_001": DownloadRecord(
            song_id="2020_001",
            chart_year=2020,
            chart_position=1,
            title="Blinding Lights",
            artist="The Weeknd",
            search_query="The Weeknd Blinding Lights official audio",
            status="done",
            audio_file="2020_001 - The Weeknd - Blinding Lights.mp3",
            youtube_id="abc123",
            youtube_title="Blinding Lights",
        )
    }
    path = tmp_path / "download_manifest.json"
    save_download_manifest(path, records)
    loaded = load_download_manifest(path)
    assert loaded["2020_001"].status == "done"
    assert loaded["2020_001"].youtube_id == "abc123"
