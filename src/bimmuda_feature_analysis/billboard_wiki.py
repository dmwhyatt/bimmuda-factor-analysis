"""Scrape Billboard Year-End Hot 100 lists from Wikipedia."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser

WIKI_BASE = "https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{year}"
USER_AGENT = "bimmuda-billboard-dataset/0.1 (research; contact: dmwhyatt)"


@dataclass(frozen=True)
class ChartEntry:
    """One row from a year-end Hot 100 table."""

    chart_year: int
    chart_position: int
    title: str
    artist: str

    @property
    def song_id(self) -> str:
        return f"{self.chart_year}_{self.chart_position:03d}"


class _WikiTableParser(HTMLParser):
    """Extract (position, title, artist) rows from the year-end wikitable."""

    def __init__(self) -> None:
        super().__init__()
        self._in_wikitable = False
        self._in_row = False
        self._in_cell = False
        self._cell_text: list[str] = []
        self._current_row: list[str] = []
        self._active_rowspan: dict[int, tuple[str, int]] = {}
        self.rows: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "table" and "wikitable" in (attr_map.get("class") or ""):
            self._in_wikitable = True
            return
        if not self._in_wikitable:
            return
        if tag == "tr":
            self._in_row = True
            self._current_row = []
        elif tag == "td" and self._in_row:
            self._in_cell = True
            self._cell_text = []
            self._pending_rowspan = int(attr_map.get("rowspan") or "1")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._in_wikitable:
            self._in_wikitable = False
            return
        if not self._in_wikitable:
            return
        if tag == "td" and self._in_cell:
            text = _normalize_cell_text("".join(self._cell_text))
            col_index = self._next_column_index()
            self._current_row.append(text)
            if self._pending_rowspan > 1:
                self._active_rowspan[col_index] = (text, self._pending_rowspan - 1)
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            self._fill_rowspan_cells()
            if len(self._current_row) >= 3:
                position, title, artist = self._current_row[:3]
                if position.isdigit():
                    self.rows.append((position, title, artist))
            self._in_row = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text.append(data)

    def _next_column_index(self) -> int:
        return len(self._current_row)

    def _fill_rowspan_cells(self) -> None:
        col_index = len(self._current_row)
        while col_index in self._active_rowspan:
            text, remaining = self._active_rowspan[col_index]
            self._current_row.append(text)
            if remaining <= 1:
                del self._active_rowspan[col_index]
            else:
                self._active_rowspan[col_index] = (text, remaining - 1)
            col_index = len(self._current_row)


def _normalize_cell_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = text.replace('"', "").replace("“", "").replace("”", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wiki_url_for_year(year: int) -> str:
    return WIKI_BASE.format(year=year)


def fetch_wiki_html(year: int, *, timeout: float = 30.0) -> str:
    url = wiki_url_for_year(year)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Wikipedia request failed for {year}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Wikipedia request failed for {year}: {exc.reason}") from exc


def parse_year_end_table(html: str, year: int) -> list[ChartEntry]:
    parser = _WikiTableParser()
    parser.feed(html)
    if not parser.rows:
        raise ValueError(f"No chart rows found for {year}")

    entries: list[ChartEntry] = []
    seen_positions: set[int] = set()
    for position_text, title, artist in parser.rows:
        position = int(position_text)
        if position in seen_positions:
            continue
        seen_positions.add(position)
        entries.append(
            ChartEntry(
                chart_year=year,
                chart_position=position,
                title=title,
                artist=artist,
            )
        )

    entries.sort(key=lambda entry: entry.chart_position)
    return entries


def scrape_year(year: int) -> list[ChartEntry]:
    """Download and parse one Billboard year-end Hot 100 page."""
    html = fetch_wiki_html(year)
    return parse_year_end_table(html, year)


def scrape_years(
    start_year: int,
    end_year: int,
    *,
    require_full_chart: bool = False,
) -> list[ChartEntry]:
    """Scrape consecutive chart years (inclusive)."""
    if start_year > end_year:
        raise ValueError(f"start_year ({start_year}) must be <= end_year ({end_year})")

    all_entries: list[ChartEntry] = []
    for year in range(start_year, end_year + 1):
        entries = scrape_year(year)
        if require_full_chart and len(entries) != 100:
            missing = sorted(
                set(range(1, 101)) - {entry.chart_position for entry in entries}
            )
            raise ValueError(
                f"Expected 100 songs for {year}, found {len(entries)}. "
                f"Missing positions: {missing}"
            )
        all_entries.extend(entries)
    return all_entries
