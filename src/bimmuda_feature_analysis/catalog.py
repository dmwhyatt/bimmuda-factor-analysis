"""Filename parsing for feature archives."""

from __future__ import annotations

import re


def parse_filename(stem: str) -> dict[str, str]:
    """Parse ``Artist_-_Title`` stem into metadata fields."""
    if "_-_" not in stem:
        return {"song_id": stem, "artist": stem, "title": ""}

    artist_part, title_part = stem.split("_-_", 1)
    return {
        "song_id": stem,
        "artist": _decode_token(artist_part),
        "title": _decode_token(title_part),
    }


def _decode_token(token: str) -> str:
    """Convert filename token back to a readable label."""
    text = token.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text
