"""Build a Billboard Hot 100 audio dataset from Wikipedia + yt-dlp."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .billboard_download import (
    DownloadRecord,
    download_entry,
    load_download_manifest,
    save_download_manifest,
)
from .billboard_wiki import ChartEntry, scrape_years

DEFAULT_DATA_ROOT = Path("data/billboard_hot100")
METADATA_FILENAME = "billboard_hot100_metadata.csv"
MANIFEST_FILENAME = "download_manifest.json"


def entries_to_metadata_df(entries: list[ChartEntry]) -> pd.DataFrame:
    rows = [
        {
            "Title": entry.title,
            "Artist": entry.artist,
            "Year": entry.chart_year,
            "Position": entry.chart_position,
            "song_id": entry.song_id,
            "decade": (entry.chart_year // 10) * 10,
        }
        for entry in entries
    ]
    return pd.DataFrame(rows)


def scrape_command(
    *,
    start_year: int,
    end_year: int,
    output_dir: Path,
    require_full_chart: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    entries = scrape_years(
        start_year,
        end_year,
        require_full_chart=require_full_chart,
    )
    df = entries_to_metadata_df(entries)
    metadata_path = metadata_dir / METADATA_FILENAME
    df.to_csv(metadata_path, index=False)
    print(f"Scraped {len(df)} songs ({start_year}-{end_year})")
    print(f"Metadata: {metadata_path}")
    return df


def download_command(
    *,
    output_dir: Path,
    metadata_path: Path | None = None,
    limit: int | None = None,
    force: bool = False,
    sleep_seconds: float = 1.0,
) -> dict[str, DownloadRecord]:
    metadata_path = metadata_path or (output_dir / "metadata" / METADATA_FILENAME)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}. Run `scrape` first."
        )

    audio_dir = output_dir / "audio"
    manifest_path = output_dir / "metadata" / MANIFEST_FILENAME
    records = load_download_manifest(manifest_path)

    df = pd.read_csv(metadata_path)
    entries = [
        ChartEntry(
            chart_year=int(row["Year"]),
            chart_position=int(row["Position"]),
            title=str(row["Title"]),
            artist=str(row["Artist"]),
        )
        for _, row in df.iterrows()
    ]
    if limit is not None:
        entries = entries[:limit]

    iterator = tqdm(entries, desc="Downloading audio")
    for entry in iterator:
        iterator.set_postfix_str(entry.song_id)
        if (
            not force
            and entry.song_id in records
            and records[entry.song_id].status in {"done", "skipped"}
        ):
            continue
        record = download_entry(
            entry,
            audio_dir,
            force=force,
            sleep_seconds=sleep_seconds,
        )
        records[entry.song_id] = record
        save_download_manifest(manifest_path, records)

    summary = {status: 0 for status in ("done", "skipped", "failed", "pending")}
    for record in records.values():
        summary[record.status] = summary.get(record.status, 0) + 1
    print(f"Audio directory: {audio_dir}")
    print(f"Download manifest: {manifest_path}")
    print(f"Summary: {summary}")
    return records


def build_command(
    *,
    start_year: int,
    end_year: int,
    output_dir: Path,
    limit: int | None = None,
    force: bool = False,
    sleep_seconds: float = 1.0,
    require_full_chart: bool = False,
) -> None:
    scrape_command(
        start_year=start_year,
        end_year=end_year,
        output_dir=output_dir,
        require_full_chart=require_full_chart,
    )
    download_command(
        output_dir=output_dir,
        limit=limit,
        force=force,
        sleep_seconds=sleep_seconds,
    )


def _default_years() -> tuple[int, int]:
    return 2016, 2025


def main(argv: list[str] | None = None) -> None:
    start_default, end_default = _default_years()
    parser = argparse.ArgumentParser(
        description=(
            "Build a Billboard Year-End Hot 100 dataset: scrape Wikipedia, "
            "download audio with yt-dlp for the audio-symbolic-pipeline."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape_parser = subparsers.add_parser(
        "scrape", help="Scrape Wikipedia year-end charts to metadata CSV"
    )
    scrape_parser.add_argument("--start-year", type=int, default=start_default)
    scrape_parser.add_argument("--end-year", type=int, default=end_default)
    scrape_parser.add_argument(
        "--require-full-chart",
        action="store_true",
        help="Fail if any year has fewer than 100 rows",
    )

    download_parser = subparsers.add_parser(
        "download", help="Download audio for scraped metadata via yt-dlp"
    )
    download_parser.add_argument("--metadata", type=Path, default=None)
    download_parser.add_argument("--limit", type=int, default=None)
    download_parser.add_argument("--force", action="store_true")
    download_parser.add_argument("--sleep", type=float, default=1.0)

    build_parser = subparsers.add_parser("build", help="Scrape then download")
    build_parser.add_argument("--start-year", type=int, default=start_default)
    build_parser.add_argument("--end-year", type=int, default=end_default)
    build_parser.add_argument("--limit", type=int, default=None)
    build_parser.add_argument("--force", action="store_true")
    build_parser.add_argument("--sleep", type=float, default=1.0)
    build_parser.add_argument("--require-full-chart", action="store_true")

    args = parser.parse_args(argv)
    output_dir: Path = args.output_dir

    if args.command == "scrape":
        scrape_command(
            start_year=args.start_year,
            end_year=args.end_year,
            output_dir=output_dir,
            require_full_chart=args.require_full_chart,
        )
    elif args.command == "download":
        download_command(
            output_dir=output_dir,
            metadata_path=args.metadata,
            limit=args.limit,
            force=args.force,
            sleep_seconds=args.sleep,
        )
    elif args.command == "build":
        build_command(
            start_year=args.start_year,
            end_year=args.end_year,
            output_dir=output_dir,
            limit=args.limit,
            force=args.force,
            sleep_seconds=args.sleep,
            require_full_chart=args.require_full_chart,
        )
