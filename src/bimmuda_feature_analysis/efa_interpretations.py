"""Provisional labels for Billboard-native EFA factors."""

from __future__ import annotations

import pandas as pd

BILLBOARD_FACTOR_LABELS: dict[str, str] = {
    "F1": "Long varied durations",
    "F2": "Distinct scalar regularity",
    "F3": "Sparse rest-heavy phrasing",
    "F4": "Wide interval variability",
    "F5": "Diverse interval vocabulary",
    "F6": "Stable in-scale motion",
    "F7": "Chromatic stepwise motion",
    "F8": "Dominant metric pulse",
    "F9": "Mobile pitch variety",
    "F10": "Strong secondary pulse",
    "F11": "Higher register",
    "F12": "Distributed pitch focus",
    "F13": "Uneven long rhythms",
    "F14": "Corpus-weighted vocabulary",
    "F15": "Strong primary pulse",
    "F16": "Wide rhythmic span",
    "F17": "Pitch-class center",
    "F18": "Primary pulse dominance",
    "F19": "Tonal expectation tension",
    "F20": "Upward melodic direction",
}

FACTOR_TREND_COLORS = (
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#B279A2",
    "#EECA3B",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#499894",
    "#86BCB6",
    "#D37295",
    "#FABFD2",
    "#A0CBE8",
    "#D4A6C8",
)


def panel_ylim(
    y,
    sem,
    *,
    padding_frac: float = 0.22,
    min_padding: float = 0.18,
    min_span: float = 0.85,
) -> tuple[float, float]:
    """Y limits from decade means ± SEM so outliers do not dominate panel space."""
    import numpy as np

    y_arr = np.asarray(y, dtype=float)
    sem_arr = np.asarray(sem, dtype=float)
    band_lo = float(np.min(y_arr - sem_arr))
    band_hi = float(np.max(y_arr + sem_arr))
    span = band_hi - band_lo
    pad = max(padding_frac * span, min_padding, 0.35 * float(np.max(sem_arr)))

    y_lo = band_lo - pad
    y_hi = band_hi + pad

    if y_hi - y_lo < min_span:
        mid = (band_hi + band_lo) / 2
        half = min_span / 2
        y_lo, y_hi = mid - half, mid + half

    if y_lo < 0 < y_hi:
        return y_lo, y_hi

    margin = 0.12 * (y_hi - y_lo)
    if y_lo > 0 and y_lo < margin:
        y_lo = -margin
    if y_hi < 0 and abs(y_hi) < margin:
        y_hi = margin
    return y_lo, y_hi


def factor_display_name(factor: str, *, labels: dict[str, str] | None = None) -> str:
    """Return ``F1 — Label`` for plots and tables."""
    labels = labels or BILLBOARD_FACTOR_LABELS
    label = labels.get(factor)
    if label:
        return f"{factor} — {label}"
    return factor


def factor_label_only(factor: str, *, labels: dict[str, str] | None = None) -> str:
    """Human-readable factor label without the ``F# —`` prefix."""
    labels = labels or BILLBOARD_FACTOR_LABELS
    return labels.get(factor, factor)


def factor_stats_caption(
    factor: str,
    *,
    year_by_factor: dict | None = None,
    kruskal_by_factor: dict | None = None,
) -> str:
    """Compact Spearman / Kruskal-Wallis caption for factor trend panels."""
    parts: list[str] = []
    if year_by_factor and factor in year_by_factor:
        rho = year_by_factor[factor]["spearman_rho"]
        p_year = year_by_factor[factor]["p_value"]
        parts.append(f"ρ = {rho:+.2f} (p = {p_year:.1e})")
    if kruskal_by_factor and factor in kruskal_by_factor:
        p_decade = kruskal_by_factor[factor]["p_value"]
        parts.append(f"KW p = {p_decade:.1e}")
    return " · ".join(parts)


def loading_based_factor_labels(
    top_loadings: pd.DataFrame,
    *,
    n_features: int = 2,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build short factor names from top loading features."""
    overrides = overrides or BILLBOARD_FACTOR_LABELS
    labels: dict[str, str] = {}
    for factor, group in top_loadings.groupby("factor"):
        if factor in overrides:
            labels[factor] = overrides[factor]
            continue
        features = group.sort_values("rank").head(n_features)["feature"].tolist()
        short = ", ".join(f.split(".")[-1].replace("_", " ") for f in features)
        labels[factor] = short
    return labels
