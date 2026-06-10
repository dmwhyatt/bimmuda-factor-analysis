"""Temporal trends of EFA factor scores across chart years."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, spearmanr

from .efa import EfaResult, factor_score_columns
from .efa_interpretations import (
    BILLBOARD_FACTOR_LABELS,
    FACTOR_TREND_COLORS,
    factor_display_name,
    factor_label_only,
    factor_stats_caption,
    loading_based_factor_labels,
    panel_ylim,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"


def publish_github_pages(
    output_dir: Path | str,
    *,
    scores: pd.DataFrame,
    decade_stats: pd.DataFrame,
    kruskal_stats: pd.DataFrame | None = None,
    year_stats: pd.DataFrame | None = None,
    factor_labels: dict[str, str] | None = None,
    factor_cols: list[str] | None = None,
) -> Path:
    """Write GitHub Pages site to ``docs/`` (index + parallel scree)."""
    from .efa_plots import plot_factor_trends_dashboard_interactive

    output_dir = Path(output_dir)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / ".nojekyll").touch()

    plot_factor_trends_dashboard_interactive(
        scores,
        decade_stats,
        DOCS_DIR / "index.html",
        kruskal_stats=kruskal_stats,
        year_stats=year_stats,
        factor_labels=factor_labels,
        factor_cols=factor_cols,
        scree_page_href="scree.html",
        data_dir=output_dir,
    )

    scree_src = output_dir / "efa_parallel_scree.html"
    if scree_src.is_file():
        scree_html = scree_src.read_text(encoding="utf-8")
        back_link = (
            '<p style="margin:12px 24px 0;font-family:system-ui,sans-serif;">'
            '<a href="index.html#efa">← Back to factor trends report</a></p>'
        )
        if "Back to factor trends report" not in scree_html:
            scree_html = scree_html.replace("<body>", f"<body>\n  {back_link}", 1)
        (DOCS_DIR / "scree.html").write_text(scree_html, encoding="utf-8")

    return DOCS_DIR / "index.html"


def _decade_sem(decade_stats: pd.DataFrame, factor_col: str) -> pd.Series:
    std_col = f"{factor_col}_std"
    return decade_stats[std_col] / np.sqrt(decade_stats["n_songs"].clip(lower=1))


def _attach_factor_labels(
    stats: pd.DataFrame,
    labels: dict[str, str] | None,
) -> pd.DataFrame:
    if not labels:
        return stats
    out = stats.copy()
    out["factor_name"] = out["factor"].map(labels)
    return out


def _lighten(color: str, *, amount: float = 0.75) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def plot_factor_decade_trend_panels(
    scores: pd.DataFrame,
    decade_stats: pd.DataFrame,
    *,
    output_path: Path,
    kruskal_stats: pd.DataFrame | None = None,
    year_stats: pd.DataFrame | None = None,
    factor_labels: dict[str, str] | None = None,
    factor_cols: list[str] | None = None,
) -> Path:
    """Small-multiples chart of decade mean factor scores with SEM error bars."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    factor_cols = factor_cols or factor_score_columns(scores)
    n_factors = len(factor_cols)
    n_cols = 5 if n_factors > 8 else 4
    n_rows = int(np.ceil(n_factors / n_cols))

    kruskal_by_factor = {}
    if kruskal_stats is not None:
        kruskal_by_factor = kruskal_stats.set_index("factor").to_dict("index")
    year_by_factor = {}
    if year_stats is not None:
        year_by_factor = year_stats.set_index("factor").to_dict("index")

    sns.set_theme(style="white", context="notebook", font_scale=0.92)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.55 * n_cols, 2.85 * n_rows),
        sharex=True,
    )
    fig.patch.set_facecolor("#f4f5f7")
    axes = np.atleast_1d(axes).flatten()
    series = decade_stats.sort_values("decade")
    x = series["decade"].to_numpy()

    for index, factor_col in enumerate(factor_cols):
        ax = axes[index]
        color = FACTOR_TREND_COLORS[index % len(FACTOR_TREND_COLORS)]
        light = _lighten(color, amount=0.86)
        mean_col = f"{factor_col}_mean"
        y = series[mean_col].to_numpy()
        sem = _decade_sem(series, factor_col).to_numpy()

        ax.set_facecolor("#ffffff")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#d8dde6")
        ax.spines["bottom"].set_color("#d8dde6")

        ax.fill_between(x, y - sem, y + sem, color=light, alpha=0.95, zorder=1)
        ax.plot(
            x,
            y,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.8,
            markeredgecolor=color,
            zorder=2,
        )
        ax.axhline(0, color="#d0d4dc", linewidth=0.8, zorder=0)

        y_lo, y_hi = panel_ylim(y, sem)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(x.min() - 6, x.max() + 6)
        ax.margins(x=0)

        label = factor_label_only(factor_col, labels=factor_labels)
        ax.text(
            0.02,
            1.02,
            factor_col,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        ax.text(
            0.02,
            0.96,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color="#4a4f5c",
            clip_on=False,
        )
        stats = factor_stats_caption(
            factor_col,
            year_by_factor=year_by_factor,
            kruskal_by_factor=kruskal_by_factor,
        )
        if stats:
            ax.text(
                0.98,
                0.04,
                stats,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.2,
                color="#6b7280",
            )

        if index % n_cols == 0:
            ax.set_ylabel("Score", color="#6b7280", fontsize=8.5)
        else:
            ax.set_yticklabels([])
        ax.tick_params(colors="#6b7280", labelsize=8)
        ax.grid(axis="y", alpha=0.28, color="#e5e7eb")

    for ax in axes[n_factors:]:
        ax.set_visible(False)

    for ax in axes[-n_cols:]:
        ax.set_xlabel("Decade", color="#6b7280", fontsize=8.5)

    fig.suptitle(
        "BiMMuDa Billboard melodies — retained EFA factors by decade (mean ± SEM)",
        fontsize=13,
        fontweight="bold",
        color="#1f2937",
        y=0.995,
    )
    fig.text(
        0.5,
        0.985,
        "Open efa_factor_trends_dashboard.html for an interactive gallery with per-song detail",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#6b7280",
    )
    fig.subplots_adjust(top=0.90, hspace=0.52, wspace=0.28, left=0.07, right=0.98, bottom=0.06)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_factor_trends_overview(
    decade_stats: pd.DataFrame,
    *,
    output_path: Path,
    factor_labels: dict[str, str] | None = None,
) -> Path:
    """Overlay z-scored decade means for all factors on one chart."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    factor_cols = [
        col.removesuffix("_mean")
        for col in decade_stats.columns
        if col.endswith("_mean")
    ]
    decades = decade_stats.sort_values("decade")["decade"]

    plt.figure(figsize=(11, 6))
    palette = sns.color_palette("tab10", n_colors=len(factor_cols))

    for color, factor_col in zip(palette, factor_cols):
        means = decade_stats.sort_values("decade")[f"{factor_col}_mean"]
        normalized = (means - means.mean()) / (means.std() or 1.0)
        label = factor_display_name(factor_col, labels=factor_labels)
        plt.plot(decades, normalized, marker="o", linewidth=2, label=label, color=color)

    plt.axhline(0, color="0.7", linewidth=0.8)
    plt.title("Normalized decade trajectories (all EFA factors)")
    plt.xlabel("Chart decade")
    plt.ylabel("Z-scored decade mean (within factor)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    return output_path


def _factor_name(scores: pd.DataFrame, factor_col: str, mask: pd.Series) -> str | None:
    name_col = f"{factor_col}_name"
    if name_col not in scores.columns:
        return None
    named = scores.loc[mask, name_col].dropna()
    if named.empty:
        return None
    return str(named.iloc[0])


def factor_trend_stats(scores: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each factor score with chart year."""
    if "chart_year" not in scores.columns:
        raise ValueError("chart_year column required on factor scores")

    rows = []
    years = pd.to_numeric(scores["chart_year"], errors="coerce")
    for col in factor_score_columns(scores):
        values = pd.to_numeric(scores[col], errors="coerce")
        mask = years.notna() & values.notna()
        rho, p_value = spearmanr(years[mask], values[mask])
        rows.append(
            {
                "factor": col,
                "spearman_rho": float(rho),
                "p_value": float(p_value),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("factor")


def factor_decade_kruskal_stats(scores: pd.DataFrame) -> pd.DataFrame:
    """Kruskal-Wallis test of each factor score across chart decades."""
    if "decade" not in scores.columns:
        raise ValueError("decade column required on factor scores")

    decade = pd.to_numeric(scores["decade"], errors="coerce")
    rows = []
    for col in factor_score_columns(scores):
        values = pd.to_numeric(scores[col], errors="coerce")
        mask = decade.notna() & values.notna()
        groups = [
            values[mask & (decade == label)].values
            for label in sorted(decade[mask].unique())
        ]
        groups = [group for group in groups if len(group) > 0]

        if len(groups) < 2:
            h_statistic, p_value = float("nan"), float("nan")
        else:
            h_statistic, p_value = kruskal(*groups)

        rows.append(
            {
                "factor": col,
                "factor_name": _factor_name(scores, col, mask),
                "kruskal_h": float(h_statistic),
                "p_value": float(p_value),
                "n_songs": int(mask.sum()),
                "n_decades": len(groups),
            }
        )
    return pd.DataFrame(rows).sort_values("factor")


def factor_trends_by_decade(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of factor scores by chart decade."""
    if "decade" not in scores.columns:
        raise ValueError("decade column required on factor scores")

    factor_cols = factor_score_columns(scores)
    grouped = scores.groupby("decade")[factor_cols]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std().add_suffix("_std")
    counts = scores.groupby("decade").size().rename("n_songs")
    return pd.concat([counts, means, stds], axis=1).reset_index().sort_values("decade")


def plot_factor_timelines(
    scores: pd.DataFrame,
    output_dir: Path,
    *,
    filename_prefix: str = "efa_timeline",
) -> None:
    """Write one scatter+regression plot per factor (chart year vs score)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = scores.copy()
    plot_df["chart_year"] = pd.to_numeric(plot_df["chart_year"], errors="coerce")
    hue_col = "decade" if "decade" in plot_df.columns else None

    for col in factor_score_columns(scores):
        name_col = f"{col}_name"
        label = plot_df[name_col].iloc[0] if name_col in plot_df.columns else col
        plt.figure(figsize=(9, 5))
        if hue_col:
            sns.scatterplot(
                data=plot_df,
                x="chart_year",
                y=col,
                hue=hue_col,
                palette="viridis",
                alpha=0.65,
                s=40,
            )
            sns.regplot(
                data=plot_df,
                x="chart_year",
                y=col,
                scatter=False,
                color="black",
                line_kws={"linewidth": 1.5},
            )
        else:
            sns.regplot(
                data=plot_df,
                x="chart_year",
                y=col,
                scatter_kws={"alpha": 0.55, "s": 35},
                line_kws={"color": "crimson"},
            )
        plt.title(f"{label} vs chart year")
        plt.xlabel("Chart year")
        plt.ylabel(f"{col} score")
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_prefix}_{col}.png", dpi=150)
        plt.close()


def plot_factor_decade_means(
    decade_summary: pd.DataFrame,
    output_path: Path,
    *,
    factor_labels: dict[str, str] | None = None,
) -> None:
    """Heatmap of mean factor scores by decade."""
    mean_cols = [col for col in decade_summary.columns if col.endswith("_mean")]
    heatmap_df = decade_summary.set_index("decade")[mean_cols]
    heatmap_df.columns = [col.removesuffix("_mean") for col in heatmap_df.columns]
    if factor_labels:
        heatmap_df = heatmap_df.rename(columns=factor_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_df, cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.title("Mean factor scores by chart decade")
    plt.xlabel("Factor")
    plt.ylabel("Decade")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_factor_trend_outputs(
    scores: pd.DataFrame,
    output_dir: Path,
    *,
    prefix: str = "efa",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Write decade Kruskal-Wallis tests, summaries, and timeline plots."""
    output_dir = Path(output_dir)
    kruskal_stats = factor_decade_kruskal_stats(scores)
    decade_stats = factor_trends_by_decade(scores)
    year_stats = None
    factor_labels = None
    if prefix == "efa":
        top_loadings_path = output_dir / f"{prefix}_top_loadings.csv"
        if top_loadings_path.is_file():
            factor_labels = loading_based_factor_labels(pd.read_csv(top_loadings_path))
            kruskal_stats = _attach_factor_labels(kruskal_stats, factor_labels)
    if "chart_year" in scores.columns:
        year_stats = factor_trend_stats(scores)
        year_stats.to_csv(output_dir / f"{prefix}_trends_by_year.csv", index=False)
    kruskal_stats.to_csv(output_dir / f"{prefix}_decade_kruskal.csv", index=False)
    decade_stats.to_csv(output_dir / f"{prefix}_trends_by_decade.csv", index=False)
    plot_factor_timelines(
        scores,
        output_dir / f"{prefix}_timelines",
        filename_prefix=f"{prefix}_timeline",
    )
    plot_factor_decade_means(
        decade_stats,
        output_dir / f"{prefix}_decade_means_heatmap.png",
        factor_labels=factor_labels,
    )
    if prefix == "efa":
        from .efa import retained_parallel_factors
        from .efa_plots import plot_factor_trends_dashboard_interactive

        parallel_path = output_dir / f"{prefix}_parallel_analysis.csv"
        retained = None
        if parallel_path.is_file():
            retained = retained_parallel_factors(pd.read_csv(parallel_path))
        if retained:
            retained_set = set(retained)
            kruskal_stats = kruskal_stats.loc[kruskal_stats["factor"].isin(retained_set)]
            if year_stats is not None:
                year_stats = year_stats.loc[year_stats["factor"].isin(retained_set)]

        plot_factor_decade_trend_panels(
            scores,
            decade_stats,
            output_path=output_dir / f"{prefix}_factor_trends_dashboard.png",
            kruskal_stats=kruskal_stats,
            year_stats=year_stats,
            factor_labels=factor_labels,
            factor_cols=retained,
        )
        plot_factor_trends_overview(
            decade_stats,
            output_path=output_dir / f"{prefix}_factor_trends_overview.png",
            factor_labels=factor_labels,
        )
        plot_factor_trends_dashboard_interactive(
            scores,
            decade_stats,
            output_path=output_dir / f"{prefix}_factor_trends_dashboard.html",
            kruskal_stats=kruskal_stats,
            year_stats=year_stats,
            factor_labels=factor_labels,
            factor_cols=retained,
            data_dir=output_dir,
        )
        publish_github_pages(
            output_dir,
            scores=scores,
            decade_stats=decade_stats,
            kruskal_stats=kruskal_stats,
            year_stats=year_stats,
            factor_labels=factor_labels,
            factor_cols=retained,
        )
    return kruskal_stats, decade_stats, year_stats


def evaluate_factor_trends(
    result: EfaResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Write trend tables and plots for a completed EFA result."""
    return write_factor_trend_outputs(result.scores, result.output_dir, prefix="efa")
