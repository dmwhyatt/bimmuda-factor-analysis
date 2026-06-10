import numpy as np
import pandas as pd

from bimmuda_feature_analysis.efa import retained_parallel_factors
from bimmuda_feature_analysis.efa_interpretations import panel_ylim
from bimmuda_feature_analysis.efa_trends import (
    factor_decade_kruskal_stats,
    factor_trend_stats,
    factor_trends_by_decade,
)


def test_factor_trend_stats():
    scores = pd.DataFrame(
        {
            "chart_year": list(range(2000, 2010)),
            "decade": [2000] * 10,
            "F1": np.linspace(-1, 1, 10),
            "F2": np.random.default_rng(0).normal(size=10),
        }
    )
    stats = factor_trend_stats(scores)
    assert set(stats["factor"]) == {"F1", "F2"}
    f1 = stats.loc[stats["factor"] == "F1", "spearman_rho"].iloc[0]
    assert f1 > 0.9


def test_factor_trends_by_decade():
    scores = pd.DataFrame(
        {
            "decade": [1980, 1980, 1990, 1990],
            "F1": [0.0, 2.0, 4.0, 6.0],
        }
    )
    summary = factor_trends_by_decade(scores)
    assert summary.loc[summary["decade"] == 1980, "F1_mean"].iloc[0] == 1.0
    assert summary.loc[summary["decade"] == 1990, "F1_mean"].iloc[0] == 5.0


def test_factor_decade_kruskal_stats():
    scores = pd.DataFrame(
        {
            "decade": [1980, 1980, 1980, 1990, 1990, 1990],
            "chart_year": [1981, 1982, 1983, 1991, 1992, 1993],
            "F1": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            "F2": np.random.default_rng(1).normal(size=6),
        }
    )
    stats = factor_decade_kruskal_stats(scores)
    assert set(stats["factor"]) == {"F1", "F2"}
    f1_p = stats.loc[stats["factor"] == "F1", "p_value"].iloc[0]
    assert f1_p < 0.05
    assert stats.loc[stats["factor"] == "F1", "n_decades"].iloc[0] == 2


def test_panel_ylim_ignores_outlier_scale():
    y = np.array([0.1, 0.0, -0.1, -0.2])
    sem = np.array([0.05, 0.05, 0.05, 0.05])
    y_lo, y_hi = panel_ylim(y, sem)
    assert y_hi - y_lo < 2.0
    assert y_lo < 0 < y_hi


def test_retained_parallel_factors():
    parallel = pd.DataFrame(
        {
            "factor": [1, 2, 3],
            "retain": [True, True, False],
        }
    )
    assert retained_parallel_factors(parallel) == ["F1", "F2"]
