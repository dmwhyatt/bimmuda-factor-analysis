"""Download chart song audio with yt-dlp."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .billboard_wiki import ChartEntry

DownloadStatus = Literal["pending", "done", "failed", "skipped"]


@dataclass
class DownloadRecord:
    song_id: str
    chart_year: int
    chart_position: int
    title: str
    artist: str
    search_query: str
    status: DownloadStatus = "pending"
    audio_file: str | None = None
    youtube_id: str | None = None
    youtube_title: str | None = None
    error: str | None = None


def audio_filename(entry: ChartEntry, ext: str = "mp3") -> str:
    """Build a unique, pipeline-friendly audio filename."""
    safe_artist = _sanitize_filename_part(entry.artist)
    safe_title = _sanitize_filename_part(entry.title)
    return f"{entry.song_id} - {safe_artist} - {safe_title}.{ext}"


def _sanitize_filename_part(text: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "unknown"


def build_search_query(entry: ChartEntry) -> str:
    return f"{entry.artist} {entry.title} official audio"


def _yt_dlp_command() -> list[str]:
    """Prefer the yt-dlp tied to the active Python environment."""
    for candidate in (
        Path(sys.executable).parent / "yt-dlp",
        Path(sys.prefix) / "bin" / "yt-dlp",
    ):
        if candidate.is_file():
            return [str(candidate)]
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise FileNotFoundError(
            "yt-dlp not found. Install with: pip install -e '.[billboard]'"
        ) from exc
    return [sys.executable, "-m", "yt_dlp"]


def download_entry(
    entry: ChartEntry,
    audio_dir: Path,
    *,
    force: bool = False,
    sleep_seconds: float = 1.0,
) -> DownloadRecord:
    """Search YouTube and download one song as mp3."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_name = audio_filename(entry, ext="mp3")
    output_path = audio_dir / output_name
    record = DownloadRecord(
        song_id=entry.song_id,
        chart_year=entry.chart_year,
        chart_position=entry.chart_position,
        title=entry.title,
        artist=entry.artist,
        search_query=build_search_query(entry),
    )

    if output_path.is_file() and not force:
        record.status = "skipped"
        record.audio_file = output_name
        return record

    search_url = f"ytsearch1:{record.search_query}"
    temp_stem = audio_dir / f".{entry.song_id}.download"
    cmd = [
        *_yt_dlp_command(),
        search_url,
        "--no-playlist",
        "--ignore-errors",
        "--no-overwrites",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "--output",
        str(temp_stem) + ".%(ext)s",
        "--print",
        "after_move:%(id)s\t%(title)s",
        "--print",
        "filepath",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        downloaded = _find_downloaded_file(audio_dir, entry.song_id, stdout)

        if result.returncode != 0 or downloaded is None:
            record.status = "failed"
            record.error = stderr or stdout or f"yt-dlp exited {result.returncode}"
            _cleanup_temp_files(audio_dir, entry.song_id)
            return record

        final_path = output_path
        if downloaded != final_path:
            if final_path.exists():
                final_path.unlink()
            downloaded.rename(final_path)

        record.status = "done"
        record.audio_file = output_name
        for line in stdout.splitlines():
            if "\t" in line and not line.startswith("/") and not line.endswith(".mp3"):
                youtube_id, youtube_title = line.split("\t", 1)
                record.youtube_id = youtube_id.strip()
                record.youtube_title = youtube_title.strip()
                break
        return record

    except subprocess.TimeoutExpired:
        record.status = "failed"
        record.error = "yt-dlp timed out after 300s"
        _cleanup_temp_files(audio_dir, entry.song_id)
        return record
    except OSError as exc:
        record.status = "failed"
        record.error = str(exc)
        _cleanup_temp_files(audio_dir, entry.song_id)
        return record


def _find_downloaded_file(
    audio_dir: Path,
    song_id: str,
    stdout: str = "",
) -> Path | None:
    for line in stdout.splitlines():
        candidate = Path(line.strip())
        if candidate.is_file() and candidate.suffix.lower() in {
            ".mp3",
            ".m4a",
            ".webm",
            ".opus",
            ".wav",
        }:
            return candidate

    candidates = sorted(audio_dir.glob(f".{song_id}.download.*"))
    if not candidates:
        candidates = sorted(audio_dir.glob(f"{song_id}*"))
    for path in candidates:
        if path.suffix.lower() in {".mp3", ".m4a", ".webm", ".opus", ".wav"}:
            return path
    return None


def _cleanup_temp_files(audio_dir: Path, song_id: str) -> None:
    for path in audio_dir.glob(f".{song_id}.download.*"):
        path.unlink(missing_ok=True)


def load_download_manifest(path: Path) -> dict[str, DownloadRecord]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    records: dict[str, DownloadRecord] = {}
    for item in raw.get("items", []):
        record = DownloadRecord(**item)
        records[record.song_id] = record
    return records


def save_download_manifest(path: Path, records: dict[str, DownloadRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "items": [asdict(record) for record in sorted(records.values(), key=lambda r: r.song_id)],
        "summary": _summarize_records(records),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summarize_records(records: dict[str, DownloadRecord]) -> dict[str, int]:
    summary = {"pending": 0, "done": 0, "failed": 0, "skipped": 0}
    for record in records.values():
        summary[record.status] = summary.get(record.status, 0) + 1
    return summary
