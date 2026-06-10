"""Interactive Plotly plots for EFA diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .efa_interpretations import (
    FACTOR_TREND_COLORS,
    factor_label_only,
    factor_stats_caption,
    panel_ylim,
)


def has_uncertainty_columns(parallel: pd.DataFrame) -> bool:
    required = {"obs_lo", "obs_hi", "sim_lo", "sim_hi"}
    return required.issubset(parallel.columns)


def suggest_observed_elbow(
    parallel: pd.DataFrame,
    *,
    max_factor: int = 30,
) -> int:
    """Estimate scree elbow on observed eigenvalues (max distance from chord)."""
    sub = parallel.loc[parallel["factor"] <= max_factor].copy()
    if len(sub) < 3:
        return int(sub["factor"].iloc[0])

    x = sub["factor"].to_numpy(dtype=float)
    y = sub["observed"].to_numpy(dtype=float)
    x_norm = (x - x[0]) / (x[-1] - x[0])
    y_norm = (y - y[0]) / (y[-1] - y[0])

    dx = x_norm[-1] - x_norm[0]
    dy = y_norm[-1] - y_norm[0]
    denom = np.hypot(dx, dy) or 1.0
    distances = np.abs(dy * (x_norm - x_norm[0]) - dx * (y_norm - y_norm[0])) / denom
    return int(sub.iloc[int(np.argmax(distances))]["factor"])


def _hex_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _add_ribbon(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    *,
    y_lo: str,
    y_hi: str,
    name: str,
    color: str,
    alpha: float = 0.18,
) -> None:
    x = plot_df["factor"]
    y_upper = plot_df[y_hi]
    y_lower = plot_df[y_lo]
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill="toself",
            fillcolor=_hex_rgba(color, alpha),
            line={"width": 0},
            name=name,
            hoverinfo="skip",
            legendgroup=name,
            showlegend=True,
        )
    )


def plot_parallel_scree_interactive(
    parallel: pd.DataFrame,
    output_path: Path | str,
    *,
    n_factors_used: int | None = None,
    max_factor: int = 40,
) -> Path:
    """Write an interactive parallel-analysis scree plot (Plotly HTML)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = parallel.loc[parallel["factor"] <= max_factor].copy()
    parallel_suggest = int(plot_df.loc[plot_df["retain"], "factor"].count())
    elbow = suggest_observed_elbow(plot_df, max_factor=max_factor)
    with_uncertainty = has_uncertainty_columns(plot_df)

    fig = go.Figure()

    if with_uncertainty:
        _add_ribbon(
            fig,
            plot_df,
            y_lo="sim_lo",
            y_hi="sim_hi",
            name="Simulated 95% envelope",
            color="#d62728",
        )
        _add_ribbon(
            fig,
            plot_df,
            y_lo="obs_lo",
            y_hi="obs_hi",
            name="Observed 95% bootstrap CI",
            color="#1f77b4",
        )

    if with_uncertainty:
        obs_custom = np.stack(
            [
                plot_df["obs_lo"],
                plot_df["obs_hi"],
                plot_df["simulated"],
                plot_df["retain"].map({True: "retain", False: "drop"}),
            ],
            axis=-1,
        )
        obs_hover = (
            "Factor %{x}<br>"
            "Observed: %{y:.3f}<br>"
            "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>"
            "Simulated mean: %{customdata[2]:.3f}<br>"
            "Parallel analysis: %{customdata[3]}<extra></extra>"
        )
        sim_custom = np.stack([plot_df["sim_lo"], plot_df["sim_hi"]], axis=-1)
        sim_hover = (
            "Factor %{x}<br>"
            "Simulated mean: %{y:.3f}<br>"
            "95% envelope: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"
        )
    else:
        obs_custom = np.stack(
            [
                plot_df["simulated"],
                plot_df["retain"].map({True: "retain", False: "drop"}),
            ],
            axis=-1,
        )
        obs_hover = (
            "Factor %{x}<br>"
            "Observed: %{y:.3f}<br>"
            "Simulated: %{customdata[0]:.3f}<br>"
            "Parallel analysis: %{customdata[1]}<extra></extra>"
        )
        sim_custom = None
        sim_hover = "Factor %{x}<br>Simulated: %{y:.3f}<extra></extra>"

    fig.add_trace(
        go.Scatter(
            x=plot_df["factor"],
            y=plot_df["observed"],
            mode="lines+markers",
            name="Observed (FA eigenvalues)",
            line={"color": "#1f77b4", "width": 2.5},
            marker={"size": 7},
            customdata=obs_custom,
            hovertemplate=obs_hover,
            legendgroup="observed",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["factor"],
            y=plot_df["simulated"],
            mode="lines+markers",
            name="Simulated mean (random data)",
            line={"color": "#d62728", "width": 2, "dash": "dash"},
            marker={"size": 6},
            customdata=sim_custom,
            hovertemplate=sim_hover,
            legendgroup="simulated",
        )
    )

    shapes = []
    annotations = []

    def _vline(x: float, color: str, dash: str, label: str, y: float) -> None:
        shapes.append(
            {
                "type": "line",
                "x0": x,
                "x1": x,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": color, "width": 2, "dash": dash},
            }
        )
        annotations.append(
            {
                "x": x,
                "y": y,
                "yref": "paper",
                "text": label,
                "showarrow": False,
                "font": {"color": color, "size": 11},
            }
        )

    _vline(
        parallel_suggest + 0.5,
        "#2ca02c",
        "dot",
        f"Parallel analysis: k={parallel_suggest}",
        1.02,
    )
    _vline(elbow + 0.5, "#ff7f0e", "dot", f"Observed elbow: k={elbow}", 0.96)

    if n_factors_used is not None:
        _vline(
            n_factors_used + 0.5,
            "#9467bd",
            "dashdot",
            f"Extracted: k={n_factors_used}",
            0.90,
        )

    subtitle = (
        "Blue band = bootstrap 95% CI on observed; red band = simulation 95% envelope"
        if with_uncertainty
        else "Run bimmuda-efa with --refresh-uncertainty for CI bands"
    )

    fig.update_layout(
        title={
            "text": "Parallel analysis scree plot (FA eigenvalues)<br>"
            f"<sup>{subtitle}</sup>",
            "x": 0.02,
            "xanchor": "left",
        },
        xaxis={
            "title": "Factor number",
            "dtick": 1,
            "range": [0.5, plot_df["factor"].max() + 0.5],
        },
        yaxis={"title": "Eigenvalue"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.08, "x": 0},
        hovermode="x unified",
        shapes=shapes,
        annotations=annotations,
        template="plotly_white",
        height=600,
        width=960,
        margin={"t": 110},
    )

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "toImageButtonOptions": {"format": "png", "filename": "efa_parallel_scree"},
        },
    )
    return output_path


def _decade_sem(decade_stats: pd.DataFrame, factor_col: str) -> pd.Series:
    std_col = f"{factor_col}_std"
    return decade_stats[std_col] / np.sqrt(decade_stats["n_songs"].clip(lower=1))


def _single_factor_trend_figure(
    scores: pd.DataFrame,
    decade_stats: pd.DataFrame,
    factor_col: str,
    *,
    color: str,
    factor_labels: dict[str, str] | None = None,
    kruskal_by_factor: dict | None = None,
    year_by_factor: dict | None = None,
) -> go.Figure:
    """One large interactive chart for a single retained factor."""
    plot_df = scores.copy()
    plot_df["decade"] = pd.to_numeric(plot_df["decade"], errors="coerce")
    series = decade_stats.sort_values("decade")
    mean_col = f"{factor_col}_mean"
    sem = _decade_sem(series, factor_col)
    y_lo, y_hi = panel_ylim(series[mean_col].to_numpy(), sem.to_numpy())

    label = factor_label_only(factor_col, labels=factor_labels)
    stats = factor_stats_caption(
        factor_col,
        year_by_factor=year_by_factor,
        kruskal_by_factor=kruskal_by_factor,
    )
    subtitle = stats or "Decade means with SEM; points are individual songs"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["decade"],
            y=plot_df[factor_col],
            mode="markers",
            name="Songs",
            marker={
                "size": 7,
                "color": plot_df["decade"],
                "colorscale": "Viridis",
                "opacity": 0.35,
                "line": {"width": 0},
            },
            customdata=np.stack(
                [
                    plot_df.get("title", pd.Series([""] * len(plot_df))).fillna(""),
                    plot_df.get("artist", pd.Series([""] * len(plot_df))).fillna(""),
                    plot_df.get("chart_year", pd.Series([""] * len(plot_df))).fillna(""),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]} · %{customdata[2]}<br>"
                "Decade=%{x}<br>Score=%{y:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["decade"],
            y=series[mean_col],
            mode="lines+markers",
            name="Decade mean",
            line={"color": color, "width": 3},
            marker={
                "size": 11,
                "color": "white",
                "line": {"color": color, "width": 2.5},
            },
            error_y={
                "type": "data",
                "array": sem,
                "visible": True,
                "thickness": 1.5,
                "color": color,
            },
            hovertemplate=(
                "Decade=%{x}<br>Mean=%{y:.2f}<br>SEM=%{error_y.array:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line={"color": "#d1d5db", "width": 1})

    fig.update_layout(
        title={
            "text": f"{factor_col} — {label}<br><sup>{subtitle}</sup>",
            "x": 0.02,
            "xanchor": "left",
        },
        template="plotly_white",
        height=560,
        margin={"t": 90, "r": 24, "b": 56, "l": 64},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="closest",
        xaxis={
            "title": "Chart decade",
            "dtick": 10,
            "range": [series["decade"].min() - 6, series["decade"].max() + 6],
            "gridcolor": "#eef0f3",
        },
        yaxis={
            "title": "Factor score",
            "range": [y_lo, y_hi],
            "gridcolor": "#eef0f3",
            "zeroline": False,
        },
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f8f9fb",
    )
    return fig


def _resolve_variance_pct(
    variance_pct: float | None,
    data_dir: Path | str | None,
) -> float | None:
    """Read cumulative SS-loadings variance from ``efa_variance.csv`` when needed."""
    if variance_pct is not None:
        return variance_pct
    if data_dir is None:
        return None
    variance_path = Path(data_dir) / "efa_variance.csv"
    if not variance_path.is_file():
        return None
    variance_df = pd.read_csv(variance_path)
    if "cum_var_pct" not in variance_df.columns or not len(variance_df):
        return None
    return float(variance_df["cum_var_pct"].iloc[-1])


def _build_loadings_by_factor(
    loadings_path: Path | str,
    factor_cols: list[str],
) -> dict[str, list[dict[str, float | str]]]:
    """Build sortable per-factor loading rows (column order in CSV maps to F1…Fn)."""
    path = Path(loadings_path)
    if not path.is_file():
        return {}

    df = pd.read_csv(path)
    if "feature" not in df.columns:
        return {}

    loading_cols = [col for col in df.columns if col != "feature"]
    rows_by_factor: dict[str, list[dict[str, float | str]]] = {}
    for index, factor in enumerate(factor_cols):
        if index >= len(loading_cols):
            break
        col = loading_cols[index]
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        entries: list[dict[str, float | str]] = []
        for feature, loading in zip(df["feature"], values):
            val = float(loading)
            entries.append(
                {
                    "feature": str(feature),
                    "loading": round(val, 4),
                    "abs_loading": round(abs(val), 4),
                }
            )
        entries.sort(key=lambda row: row["abs_loading"], reverse=True)
        rows_by_factor[factor] = entries
    return rows_by_factor


def _report_summary_stats(
    scores: pd.DataFrame,
    factor_cols: list[str],
    *,
    n_features: int = 220,
    variance_pct: float | None = None,
) -> dict[str, str | int]:
    """Build headline numbers for the static report sections."""
    stats: dict[str, str | int] = {
        "n_songs": len(scores),
        "n_factors": len(factor_cols),
        "n_features": n_features,
        "n_decades": int(scores["decade"].nunique()) if "decade" in scores.columns else 0,
    }
    if "chart_year" in scores.columns:
        years = pd.to_numeric(scores["chart_year"], errors="coerce").dropna()
        if len(years):
            stats["year_min"] = int(years.min())
            stats["year_max"] = int(years.max())
        else:
            stats["year_min"] = stats["year_max"] = "—"
    else:
        stats["year_min"] = stats["year_max"] = "—"
    stats["variance_pct"] = f"{variance_pct:.1f}" if variance_pct is not None else "—"
    return stats


def _trend_report_extras(
    factor_cols: list[str],
    kruskal_stats: pd.DataFrame | None,
) -> dict[str, int]:
    """Count factors with significant decade trends for the report hero."""
    if kruskal_stats is None or not factor_cols:
        return {"n_sig_trends": 0}

    merged = kruskal_stats.loc[kruskal_stats["factor"].isin(factor_cols)].copy()
    n_sig = int((merged["p_value"] < 0.05).sum())
    return {"n_sig_trends": n_sig}


_GALLERY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="description" content="BiMMuDa Factor Analysis — exploratory factor analysis of Billboard vocal melodies with interactive decade trends, loadings, and methods." />
  <title>BiMMuDa Factor Analysis</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #f4f5f7;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #e5e7eb;
      --accent: #4c78a8;
      --accent-soft: #e8eef6;
      --max-width: 960px;
      --page-width: min(1360px, calc(100vw - 48px));
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
      line-height: 1.6;
    }}
    a {{ color: var(--accent); }}
    .site-header {{
      position: sticky;
      top: 0;
      z-index: 20;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(8px);
    }}
    .site-header-inner {{
      max-width: var(--page-width);
      margin: 0 auto;
      padding: 14px 24px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .site-title {{
      margin: 0;
      font-size: 1.05rem;
      font-weight: 650;
    }}
    .site-nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px 16px;
      font-size: 0.88rem;
    }}
    .site-nav a {{
      text-decoration: none;
      color: var(--muted);
      font-weight: 500;
    }}
    .site-nav a:hover {{ color: var(--accent); }}
    .page {{
      max-width: var(--page-width);
      margin: 0 auto;
      padding: 0 24px 48px;
    }}
    .trends-section {{
      padding-top: 16px;
      margin-top: 0;
    }}
    .trends-section h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.45rem, 2.8vw, 1.85rem);
      line-height: 1.25;
      letter-spacing: -0.02em;
    }}
    .hero-lede {{
      margin: 0;
      color: var(--text);
      font-size: 0.98rem;
      line-height: 1.55;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
    }}
    .stat-value {{
      display: block;
      font-size: 1.2rem;
      font-weight: 700;
      color: var(--accent);
      line-height: 1.2;
    }}
    .stat-label {{
      display: block;
      margin-top: 2px;
      font-size: 0.68rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      line-height: 1.3;
    }}
    .methods-lead {{
      margin-top: 48px;
      padding-top: 32px;
      border-top: 2px solid var(--border);
    }}
    .methods-lead h2 {{
      margin: 0 0 8px;
      font-size: 1.2rem;
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .methods-lead p {{
      margin: 0 0 28px;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .pipeline-credits {{
      margin: 14px 0 0;
      font-size: 0.9rem;
      color: var(--muted);
    }}
    section {{
      margin-top: 36px;
    }}
    section h2 {{
      margin: 0 0 12px;
      font-size: 1.35rem;
      letter-spacing: -0.01em;
    }}
    section p, section li {{
      color: #374151;
    }}
    .card p, .methods-lead p, .card li {{
      max-width: 72ch;
    }}
    section ul {{
      padding-left: 1.25rem;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 22px 24px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .pipeline {{
      display: flex;
      flex-wrap: wrap;
      align-items: stretch;
      gap: 8px;
      margin: 20px 0 8px;
    }}
    .pipeline-step {{
      flex: 1 1 120px;
      min-width: 120px;
      background: var(--accent-soft);
      border: 1px solid #c9d8ea;
      border-radius: 10px;
      padding: 12px 14px;
      font-size: 0.88rem;
    }}
    .pipeline-step strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 0.82rem;
      color: var(--accent);
    }}
    .pipeline-arrow {{
      align-self: center;
      color: var(--muted);
      font-size: 1.1rem;
      flex: 0 0 auto;
    }}
    .callout {{
      margin-top: 16px;
      padding: 12px 14px;
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
      border-radius: 0 8px 8px 0;
      font-size: 0.92rem;
    }}
    .gallery-shell {{
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      background: var(--panel);
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }}
    .gallery-header {{
      padding: 10px 16px;
      border-bottom: 1px solid var(--border);
      background: #fafbfc;
    }}
    .gallery-header-row {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    .gallery-hint {{
      margin: 0;
      color: var(--muted);
      font-size: 0.84rem;
      line-height: 1.4;
      flex: 1 1 240px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 0;
    }}
    .toolbar label {{
      font-size: 0.85rem;
      color: var(--muted);
    }}
    .toolbar select, .toolbar button {{
      font: inherit;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      padding: 7px 12px;
      color: var(--text);
      cursor: pointer;
    }}
    .toolbar button:hover {{ background: #f9fafb; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(220px, 280px) minmax(0, 1fr);
      min-height: 620px;
    }}
    .factor-nav {{
      border-right: 1px solid var(--border);
      background: #fafbfc;
      overflow-y: auto;
      padding: 12px;
      max-height: 720px;
    }}
    .factor-item {{
      display: block;
      width: 100%;
      text-align: left;
      border: 1px solid transparent;
      border-left: 4px solid transparent;
      border-radius: 10px;
      background: transparent;
      padding: 10px 12px;
      margin-bottom: 6px;
      cursor: pointer;
      color: inherit;
      font: inherit;
    }}
    .factor-item:hover {{ background: rgba(255,255,255,0.85); }}
    .factor-item.active {{
      background: var(--panel);
      border-color: var(--border);
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}
    .factor-item.hidden {{ display: none; }}
    .factor-id {{
      font-weight: 700;
      font-size: 0.92rem;
    }}
    .factor-label {{
      display: block;
      margin-top: 2px;
      font-size: 0.78rem;
      color: var(--muted);
      line-height: 1.35;
    }}
    .factor-meta {{
      display: block;
      margin-top: 4px;
      font-size: 0.72rem;
      color: #9ca3af;
    }}
    .badge {{
      display: inline-block;
      margin-left: 6px;
      padding: 1px 6px;
      border-radius: 999px;
      font-size: 0.68rem;
      font-weight: 600;
      vertical-align: middle;
    }}
    .badge.sig {{ background: #dcfce7; color: #166534; }}
    .badge.flat {{ background: #f3f4f6; color: #6b7280; }}
    .gallery-main {{ padding: 16px 18px 24px; }}
    #chart {{ min-height: 580px; }}
    .loadings-toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      align-items: center;
      margin: 16px 0 12px;
      font-size: 0.88rem;
    }}
    .loadings-toolbar label {{ color: var(--muted); }}
    .loadings-toolbar select,
    .loadings-toolbar input[type="search"] {{
      font: inherit;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 7px 10px;
      background: var(--panel);
      color: var(--text);
    }}
    .loadings-toolbar input[type="search"] {{ min-width: 200px; flex: 1 1 180px; }}
    .loadings-toolbar input[type="range"] {{ width: 120px; vertical-align: middle; }}
    .loadings-wrap {{
      max-height: 420px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 10px;
    }}
    .loadings-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
    }}
    .loadings-table th {{
      position: sticky;
      top: 0;
      background: #f3f4f6;
      text-align: left;
      padding: 8px 12px;
      border-bottom: 1px solid var(--border);
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }}
    .loadings-table th:hover {{ background: #e5e7eb; }}
    .loadings-table td {{
      padding: 7px 12px;
      border-bottom: 1px solid #f0f1f3;
      font-variant-numeric: tabular-nums;
    }}
    .loadings-table tr:hover td {{ background: #fafbfc; }}
    .loadings-table .num {{ text-align: right; }}
    .loadings-table .pos {{ color: #166534; }}
    .loadings-table .neg {{ color: #b91c1c; }}
    .loadings-caption {{
      margin: 0 0 4px;
      font-size: 0.9rem;
      color: var(--muted);
    }}
    .site-footer {{
      max-width: var(--page-width);
      margin: 0 auto;
      padding: 24px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.85rem;
    }}
    @media (max-width: 960px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .factor-nav {{
        max-height: 220px;
        border-right: none;
        border-bottom: 1px solid var(--border);
      }}
      .pipeline-arrow {{ display: none; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header class="site-header">
    <div class="site-header-inner">
      <p class="site-title">BiMMuDa Factor Analysis</p>
      <nav class="site-nav" aria-label="Page sections">
        <a href="#trends">Trends</a>
        <a href="#pipeline">Pipeline</a>
        <a href="#efa">EFA</a>
        <a href="#loadings">Loadings</a>
      </nav>
    </div>
  </header>

  <div class="page">
    <section id="trends" class="trends-section">
      <h1>Billboard top 5 melodies change with time</h1>
      <p class="hero-lede">
        Using <a href="https://github.com/madelinehamilton/BiMMuDa">BiMMuDa</a>, we measured the extent to which vocal melodies change over the course of
        ~80 years. By comparing trends in high-level musical structures, we can begin to understand
        the musical styles of each decade using empirical methods.
      </p>
      <div class="stats">
        <div class="stat"><span class="stat-value">{n_sig_trends}/{n_factors}</span><span class="stat-label">Factors with decade trends</span></div>
        <div class="stat"><span class="stat-value">{n_songs}</span><span class="stat-label">Songs</span></div>
        <div class="stat"><span class="stat-value">{n_decades}</span><span class="stat-label">Chart decades</span></div>
        <div class="stat"><span class="stat-value">{variance_pct}%</span><span class="stat-label">Variance explained</span></div>
      </div>
      <div class="gallery-shell">
        <div class="gallery-header">
          <div class="gallery-header-row">
            <p class="gallery-hint">{subtitle}</p>
            <div class="toolbar">
              <label for="filter">Show</label>
              <select id="filter">
                <option value="all">All retained factors</option>
                <option value="sig">Significant decade trends (KW p &lt; 0.05)</option>
              </select>
              <button id="prev" type="button">← Previous</button>
              <button id="next" type="button">Next →</button>
            </div>
          </div>
        </div>
        <div class="layout">
          <nav class="factor-nav" id="nav" aria-label="Factor list"></nav>
          <div class="gallery-main"><div id="chart"></div></div>
        </div>
      </div>
    </section>

    <div class="methods-lead">
      <h2>Methods</h2>
      <p>How audio became melody features, and how factors were extracted and tested for temporal change.</p>
    </div>

    <section id="pipeline">
      <h2>From audio to melody features</h2>
      <div class="card">
        <div class="pipeline" role="list" aria-label="Audio to features pipeline">
          <div class="pipeline-step" role="listitem">
            <strong>1. Chart recording</strong>
            <a href="https://github.com/madelinehamilton/BiMMuDa">BiMMuDa</a> Billboard sample
          </div>
          <span class="pipeline-arrow" aria-hidden="true">→</span>
          <div class="pipeline-step" role="listitem">
            <strong>2. Vocal separation</strong>
            <a href="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21">Music-Source-Separation-Training</a> v1.0.21
          </div>
          <span class="pipeline-arrow" aria-hidden="true">→</span>
          <div class="pipeline-step" role="listitem">
            <strong>3. Transcription</strong>
            ROSVOT → MIDI-like notes
          </div>
          <span class="pipeline-arrow" aria-hidden="true">→</span>
          <div class="pipeline-step" role="listitem">
            <strong>4. Features</strong>
            220 scalar melody descriptors
          </div>
        </div>
        <p class="pipeline-credits">
          Descriptors via the
          <a href="https://github.com/dmwhyatt/audio-symbolic-pipeline">audio-symbolic-pipeline</a>
          and <a href="https://github.com/dmwhyatt/melody-features">melody-features</a>:
          vocals isolated with
          <a href="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21">Music-Source-Separation-Training v1.0.21</a>
          (BS-RoFormer / Mel-Band RoFormer models), transcribed with <a href="https://github.com/RickyL-2000/ROSVOT">ROSVOT</a>, then
          summarised as <code>.npz</code> feature files joined to
          <a href="https://github.com/madelinehamilton/BiMMuDa">BiMMuDa</a> chart metadata.
        </p>
      </div>
    </section>

    <section id="efa">
      <h2>Exploratory factor analysis</h2>
      <div class="card">
        <p>
          With 220 correlated melody descriptors, EFA finds a smaller set of latent
          <em>style factors</em> that summarise shared variance across songs. The workflow
          follows
          <a href="https://github.com/dmwhyatt/Style-Classification-Analysis">Style-Classification-Analysis</a>:
        </p>
        <ul>
          <li><strong>Preprocessing</strong> — Replace infinities and missing values, drop zero-variance columns, then z-score each feature.</li>
          <li><strong>Factor retention</strong> — Horn&rsquo;s parallel analysis (100 random-data simulations). A factor is kept when its observed eigenvalue exceeds the simulated 95th percentile (parallel-analysis null).</li>
          <li><strong>Extraction</strong> — Principal axis factoring (PAF) in R <code>psych</code>.</li>
          <li><strong>Rotation</strong> — Promax (oblique), allowing factors to correlate.</li>
          <li><strong>Scoring</strong> — Regression factor scores per song for interpretation and trend analysis.</li>
          <li><strong>Temporal evaluation</strong> — Kruskal&ndash;Wallis <em>H</em> tests factor scores across chart decades; Spearman &rho; vs chart year is reported as supplementary.</li>
        </ul>
        <p>
          Parallel analysis retained <strong>{n_factors} factors</strong> from {n_features} melody
          features (~<strong>{variance_pct}%</strong> common variance). Provisional factor names
          come from the strongest loadings — use the
          <a href="#loadings">loadings table</a> to reproduce those interpretations.
        </p>
        {scree_link_block}
      </div>
    </section>

    <section id="loadings">
      <h2>Factor loadings</h2>
      <div class="card">
        <p class="loadings-caption">
          Promax-rotated loadings for every melody feature. Select a factor, filter by minimum
          |loading|, or search feature names — the same matrix used to label factors in the gallery.
        </p>
        <div class="loadings-toolbar">
          <label for="loadings-factor">Factor</label>
          <select id="loadings-factor"></select>
          <label for="loadings-threshold">Min |loading|</label>
          <input type="range" id="loadings-threshold" min="0" max="0.8" step="0.05" value="0.25" />
          <span id="loadings-threshold-val">0.25</span>
          <input type="search" id="loadings-search" placeholder="Filter features…" aria-label="Filter features" />
        </div>
        <div class="loadings-wrap">
          <table class="loadings-table" id="loadings-table">
            <thead>
              <tr>
                <th data-sort="rank">#</th>
                <th data-sort="feature">Feature</th>
                <th data-sort="loading" class="num">Loading</th>
                <th data-sort="abs" class="num">|Loading|</th>
              </tr>
            </thead>
            <tbody id="loadings-body"></tbody>
          </table>
        </div>
      </div>
    </section>
  </div>

  <footer class="site-footer">
    Dataset:
    <a href="https://github.com/madelinehamilton/BiMMuDa">BiMMuDa</a>
    · Analysis:
    <a href="https://github.com/dmwhyatt/bimmuda-feature-analysis">bimmuda-feature-analysis</a>
    · Pipeline:
    <a href="https://github.com/dmwhyatt/audio-symbolic-pipeline">audio-symbolic-pipeline</a>
    · Separation:
    <a href="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21">Music-Source-Separation-Training v1.0.21</a>
    · Transcription:
    <a href="https://github.com/RickyL-2000/ROSVOT">ROSVOT</a>
    · Features:
    <a href="https://github.com/dmwhyatt/melody-features">melody-features</a>
  </footer>
  <script>
    const LOADINGS = {loadings_json};
    const LOADING_LABELS = {loadings_labels_json};
    const FACTORS = {factors_json};
    const PLOTS = {plots_json};
    let loadingsSort = {{ key: "abs", asc: false }};

    (function initLoadings() {{
      const factorSelect = document.getElementById("loadings-factor");
      const threshold = document.getElementById("loadings-threshold");
      const thresholdVal = document.getElementById("loadings-threshold-val");
      const search = document.getElementById("loadings-search");
      const tbody = document.getElementById("loadings-body");
      const factorIds = Object.keys(LOADINGS);
      if (!factorIds.length) {{
        document.getElementById("loadings").style.display = "none";
        return;
      }}
      factorIds.forEach(id => {{
        const opt = document.createElement("option");
        opt.value = id;
        const label = LOADING_LABELS[id] || id;
        opt.textContent = label === id ? id : id + " — " + label;
        factorSelect.appendChild(opt);
      }});

      function renderLoadings() {{
        const factor = factorSelect.value;
        const minAbs = parseFloat(threshold.value);
        const q = (search.value || "").trim().toLowerCase();
        thresholdVal.textContent = minAbs.toFixed(2);
        let rows = (LOADINGS[factor] || []).filter(r =>
          r.abs_loading >= minAbs && (!q || r.feature.toLowerCase().includes(q))
        );
        const {{ key, asc }} = loadingsSort;
        rows = rows.slice().sort((a, b) => {{
          let av, bv;
          if (key === "feature") {{ av = a.feature; bv = b.feature; return asc ? av.localeCompare(bv) : bv.localeCompare(av); }}
          if (key === "loading") {{ av = a.loading; bv = b.loading; }}
          else if (key === "abs") {{ av = a.abs_loading; bv = b.abs_loading; }}
          else {{ av = 0; bv = 0; }}
          return asc ? av - bv : bv - av;
        }});
        tbody.innerHTML = rows.map((r, i) =>
          "<tr><td>" + (i + 1) + "</td><td><code>" + r.feature + "</code></td>" +
          '<td class="num ' + (r.loading >= 0 ? "pos" : "neg") + '">' + r.loading.toFixed(3) + "</td>" +
          '<td class="num">' + r.abs_loading.toFixed(3) + "</td></tr>"
        ).join("");
      }}

      factorSelect.addEventListener("change", renderLoadings);
      threshold.addEventListener("input", renderLoadings);
      search.addEventListener("input", renderLoadings);
      document.querySelectorAll(".loadings-table th").forEach(th => {{
        th.addEventListener("click", () => {{
          const key = th.dataset.sort;
          if (loadingsSort.key === key) loadingsSort.asc = !loadingsSort.asc;
          else loadingsSort = {{ key, asc: key === "feature" }};
          renderLoadings();
        }});
      }});
      renderLoadings();
    }})();

    let visible = [];
    let index = 0;

    function kwSig(f) {{ return f.kw_p != null && f.kw_p < 0.05; }}

    function rebuildVisible() {{
      const mode = document.getElementById("filter").value;
      visible = FACTORS.filter(f => mode === "all" || kwSig(f));
      document.querySelectorAll(".factor-item").forEach(el => {{
        const show = mode === "all" || el.dataset.sig === "1";
        el.classList.toggle("hidden", !show);
      }});
      if (!visible.length) return;
      if (index >= visible.length) index = visible.length - 1;
      showFactor(visible[index].id, false);
    }}

    function showFactor(id, scrollNav = true) {{
      const pos = visible.findIndex(f => f.id === id);
      if (pos < 0) return;
      index = pos;
      const payload = PLOTS[id];
      Plotly.react("chart", payload.data, payload.layout, {{responsive: true}});
      document.querySelectorAll(".factor-item").forEach(el => {{
        el.classList.toggle("active", el.dataset.id === id);
      }});
      if (scrollNav) {{
        const active = document.querySelector('.factor-item[data-id="' + id + '"]');
        if (active) active.scrollIntoView({{block: "nearest"}});
      }}
    }}

    function step(delta) {{
      if (!visible.length) return;
      index = (index + delta + visible.length) % visible.length;
      showFactor(visible[index].id);
    }}

    const nav = document.getElementById("nav");
    FACTORS.forEach(f => {{
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "factor-item";
      btn.dataset.id = f.id;
      btn.dataset.sig = kwSig(f) ? "1" : "0";
      btn.style.borderLeftColor = f.color;
      btn.innerHTML =
        '<span class="factor-id">' + f.id +
        '<span class="badge ' + (kwSig(f) ? 'sig' : 'flat') + '">' +
        (kwSig(f) ? 'decade trend' : 'stable') + '</span></span>' +
        '<span class="factor-label">' + f.label + '</span>' +
        '<span class="factor-meta">' + (f.stats || '') + '</span>';
      btn.addEventListener("click", () => showFactor(f.id));
      nav.appendChild(btn);
    }});

    document.getElementById("filter").addEventListener("change", rebuildVisible);
    document.getElementById("prev").addEventListener("click", () => step(-1));
    document.getElementById("next").addEventListener("click", () => step(1));
    document.addEventListener("keydown", e => {{
      if (e.key === "ArrowLeft") step(-1);
      if (e.key === "ArrowRight") step(1);
    }});

    visible = FACTORS.slice();
    showFactor(FACTORS[0].id, false);
  </script>
</body>
</html>
"""


def plot_factor_trends_dashboard_interactive(
    scores: pd.DataFrame,
    decade_stats: pd.DataFrame,
    output_path: Path | str,
    *,
    kruskal_stats: pd.DataFrame | None = None,
    year_stats: pd.DataFrame | None = None,
    factor_labels: dict[str, str] | None = None,
    factor_cols: list[str] | None = None,
    variance_pct: float | None = None,
    n_features: int = 220,
    scree_page_href: str | None = None,
    data_dir: Path | str | None = None,
) -> Path:
    """Interactive browsable gallery of EFA factor decade trends (Plotly HTML)."""
    from .efa import factor_score_columns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir) if data_dir is not None else output_path.parent

    factor_cols = factor_cols or factor_score_columns(scores)
    kruskal_by_factor = {}
    if kruskal_stats is not None:
        kruskal_by_factor = kruskal_stats.set_index("factor").to_dict("index")
    year_by_factor = {}
    if year_stats is not None:
        year_by_factor = year_stats.set_index("factor").to_dict("index")

    variance_pct = _resolve_variance_pct(variance_pct, data_dir)

    loadings_by_factor = _build_loadings_by_factor(
        data_dir / "efa_loadings.csv",
        factor_cols,
    )
    loadings_labels = {
        factor: factor_label_only(factor, labels=factor_labels)
        for factor in factor_cols
    }

    report_stats = _report_summary_stats(
        scores,
        factor_cols,
        n_features=n_features,
        variance_pct=variance_pct,
    )
    trend_extras = _trend_report_extras(factor_cols, kruskal_stats)

    scree_link_block = ""
    if scree_page_href:
        scree_link_block = (
            '<div class="callout">'
            f'<a href="{scree_page_href}">View the interactive parallel scree plot</a>'
            " — observed vs simulated eigenvalues with 95% confidence bands."
            "</div>"
        )

    factors_meta = []
    plots: dict[str, dict] = {}
    for index, factor_col in enumerate(factor_cols):
        color = FACTOR_TREND_COLORS[index % len(FACTOR_TREND_COLORS)]
        fig = _single_factor_trend_figure(
            scores,
            decade_stats,
            factor_col,
            color=color,
            factor_labels=factor_labels,
            kruskal_by_factor=kruskal_by_factor,
            year_by_factor=year_by_factor,
        )
        payload = json.loads(fig.to_json())
        plots[factor_col] = {"data": payload["data"], "layout": payload["layout"]}
        kw_p = None
        if factor_col in kruskal_by_factor:
            kw_p = kruskal_by_factor[factor_col].get("p_value")
        factors_meta.append(
            {
                "id": factor_col,
                "label": factor_label_only(factor_col, labels=factor_labels),
                "color": color,
                "kw_p": kw_p,
                "stats": factor_stats_caption(
                    factor_col,
                    year_by_factor=year_by_factor,
                    kruskal_by_factor=kruskal_by_factor,
                ),
            }
        )

    html = _GALLERY_HTML.format(
        subtitle=(
            "F1–F{count}: decade means ± SEM, songs on hover. Filter significant trends or use ← →."
        ).format(count=len(factor_cols)),
        factors_json=json.dumps(factors_meta),
        plots_json=json.dumps(plots),
        loadings_json=json.dumps(loadings_by_factor),
        loadings_labels_json=json.dumps(loadings_labels),
        scree_link_block=scree_link_block,
        **report_stats,
        **trend_extras,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
