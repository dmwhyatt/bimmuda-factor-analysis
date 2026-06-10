from pathlib import Path

import pandas as pd

from bimmuda_feature_analysis.efa_plots import (
    _build_loadings_by_factor,
    _resolve_variance_pct,
    plot_factor_trends_dashboard_interactive,
    plot_parallel_scree_interactive,
    suggest_observed_elbow,
)


def test_suggest_observed_elbow():
    parallel = pd.DataFrame(
        {
            "factor": [1, 2, 3, 4, 5],
            "observed": [10.0, 5.0, 3.0, 2.5, 2.4],
            "simulated": [1.0, 1.0, 1.0, 1.0, 1.0],
            "retain": [True, True, True, True, True],
        }
    )
    elbow = suggest_observed_elbow(parallel, max_factor=5)
    assert elbow in {2, 3}


def test_plot_factor_trends_gallery(tmp_path: Path):
    scores = pd.DataFrame(
        {
            "decade": [1980, 1980, 1990, 1990, 2000, 2000],
            "chart_year": [1981, 1985, 1991, 1995, 2001, 2005],
            "title": ["A", "B", "C", "D", "E", "F"],
            "artist": ["X"] * 6,
            "F1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "F2": [1.0, 1.0, -1.0, -1.0, 0.5, 0.5],
        }
    )
    decade_stats = pd.DataFrame(
        {
            "decade": [1980, 1990, 2000],
            "n_songs": [2, 2, 2],
            "F1_mean": [0.5, 2.5, 4.5],
            "F1_std": [0.5, 0.5, 0.5],
            "F2_mean": [1.0, -1.0, 0.5],
            "F2_std": [0.0, 0.0, 0.0],
        }
    )
    kruskal = pd.DataFrame(
        {
            "factor": ["F1", "F2"],
            "p_value": [0.01, 0.2],
            "kruskal_h": [10.0, 1.0],
        }
    )
    out = plot_factor_trends_dashboard_interactive(
        scores,
        decade_stats,
        tmp_path / "gallery.html",
        kruskal_stats=kruskal,
        factor_cols=["F1", "F2"],
    )
    html = out.read_text(encoding="utf-8")
    assert out.is_file()
    assert "billboard top 5 melodies change with time" in html.lower()
    assert 'id="trends"' in html
    assert "from audio to melody features" in html.lower()
    assert "exploratory factor analysis" in html.lower()
    assert "id=\"loadings\"" in html
    assert "LOADINGS" in html
    assert "Plotly.react" in html
    assert '"F1"' in html
    assert '"F2"' in html


def test_resolve_variance_pct(tmp_path: Path):
    variance = tmp_path / "efa_variance.csv"
    variance.write_text("factor,cum_var_pct\nF1,14.78\nF2,22.51\n", encoding="utf-8")
    assert _resolve_variance_pct(None, tmp_path) == 22.51


def test_build_loadings_by_factor(tmp_path: Path):
    loadings = tmp_path / "efa_loadings.csv"
    loadings.write_text(
        "feature,PA1,PA2\n"
        "timing.mean_duration,0.9,0.1\n"
        "timing.ioi_range,0.2,0.8\n",
        encoding="utf-8",
    )
    by_factor = _build_loadings_by_factor(loadings, ["F1", "F2"])
    assert by_factor["F1"][0]["feature"] == "timing.mean_duration"
    assert by_factor["F2"][0]["feature"] == "timing.ioi_range"


def test_plot_parallel_scree_interactive(tmp_path: Path):
    parallel = pd.DataFrame(
        {
            "factor": [1, 2, 3],
            "observed": [5.0, 2.0, 1.5],
            "obs_lo": [4.5, 1.8, 1.3],
            "obs_hi": [5.5, 2.2, 1.7],
            "simulated": [1.2, 1.1, 1.0],
            "sim_lo": [1.0, 0.95, 0.9],
            "sim_hi": [1.4, 1.25, 1.1],
            "retain": [True, True, False],
        }
    )
    out = plot_parallel_scree_interactive(parallel, tmp_path / "scree.html", n_factors_used=2)
    assert out.is_file()
    assert "plotly" in out.read_text(encoding="utf-8").lower()
