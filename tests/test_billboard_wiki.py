from bimmuda_feature_analysis.billboard_wiki import (
    ChartEntry,
    parse_year_end_table,
    wiki_url_for_year,
)

SAMPLE_HTML = """
<table class="wikitable sortable">
<tr><th>No.</th><th>Title</th><th>Artist(s)</th></tr>
<tr>
<td scope="row">1
</td>
<td>"<a href="/wiki/Die_with_a_Smile">Die with a Smile</a>"</td>
<td><a href="/wiki/Lady_Gaga">Lady Gaga</a> and <a href="/wiki/Bruno_Mars">Bruno Mars</a>
</td></tr>
<tr>
<td scope="row">2
</td>
<td>"<a href="/wiki/Luther_(song)">Luther</a>"</td>
<td><a href="/wiki/Kendrick_Lamar">Kendrick Lamar</a> and <a href="/wiki/SZA">SZA</a>
</td></tr>
</table>
"""

LEGACY_HTML = """
<table class="wikitable">
<tr><th>#</th><th>Single</th><th>Artist</th></tr>
<tr>
<td>1</td>
<td>"<a href="/wiki/Blinding_Lights">Blinding Lights</a>"</td>
<td><a href="/wiki/The_Weeknd">The Weeknd</a>
</td></tr>
<tr>
<td>2</td>
<td>"<a href="/wiki/Circles_(Post_Malone_song)">Circles</a>"</td>
<td><a href="/wiki/Post_Malone">Post Malone</a>
</td></tr>
</table>
"""


def test_wiki_url_for_year():
    assert wiki_url_for_year(2025).endswith("Billboard_Year-End_Hot_100_singles_of_2025")


def test_parse_modern_wikitable():
    entries = parse_year_end_table(SAMPLE_HTML, 2025)
    assert len(entries) == 2
    assert entries[0] == ChartEntry(2025, 1, "Die with a Smile", "Lady Gaga and Bruno Mars")
    assert entries[1].title == "Luther"
    assert entries[1].song_id == "2025_002"


def test_parse_legacy_wikitable():
    entries = parse_year_end_table(LEGACY_HTML, 2020)
    assert len(entries) == 2
    assert entries[0].artist == "The Weeknd"


def test_chart_entry_song_id():
    entry = ChartEntry(2019, 42, "Old Town Road", "Lil Nas X")
    assert entry.song_id == "2019_042"


ROWSPAN_HTML = """
<table class="wikitable">
<tr><th>#</th><th>Title</th><th>Artist</th></tr>
<tr>
<td>1</td>
<td>"Love Yourself"</td>
<td rowspan="2">Justin Bieber</td>
</tr>
<tr>
<td>2</td>
<td>"Sorry"</td>
</tr>
</table>
"""


def test_parse_rowspan_artist_cells():
    entries = parse_year_end_table(ROWSPAN_HTML, 2016)
    assert len(entries) == 2
    assert entries[0].title == "Love Yourself"
    assert entries[1] == ChartEntry(2016, 2, "Sorry", "Justin Bieber")
