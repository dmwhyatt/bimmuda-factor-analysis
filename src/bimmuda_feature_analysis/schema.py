"""Feature schema for melody-features .npz files."""

METADATA_KEYS = frozenset(
    {
        "features_json",
        "melody_num",
        "rosvot_note_durs",
        "rosvot_pitches",
    }
)

FEATURE_CATEGORIES = (
    "absolute_pitch",
    "pitch_class",
    "pitch_interval",
    "contour",
    "timing",
    "inter_onset_interval",
    "tonality",
    "metre",
    "expectation",
    "complexity",
    "lexical_diversity",
    "corpus",
    "idyom",
)


def scalar_feature_keys(npz_keys: list[str]) -> list[str]:
    """Return sorted scalar feature keys, excluding metadata arrays."""
    return sorted(k for k in npz_keys if k not in METADATA_KEYS)
