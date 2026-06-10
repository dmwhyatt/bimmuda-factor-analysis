"""Load and analyse scalar melody feature datasets."""

from .catalog import parse_filename
from .cluster import (
    cluster_by_decade,
    cluster_songs,
    cluster_songs_within_eras,
    era_cluster_summary,
    feature_columns,
    prepare_feature_matrix,
)
from .efa import EfaResult, load_efa_outputs, run_efa, summarize_top_loadings
from .efa_trends import (
    evaluate_factor_trends,
    factor_decade_kruskal_stats,
    factor_trend_stats,
    factor_trends_by_decade,
    write_factor_trend_outputs,
)
from .loader import load_features_dir, load_npz
from .metadata import attach_chart_metadata, load_chart_metadata, normalize_song_key
from .preprocess import prepare_efa_matrix
from .schema import FEATURE_CATEGORIES, METADATA_KEYS, scalar_feature_keys
from .style_factors import (
    StyleFactorResult,
    evaluate_style_factor_trends,
    load_style_factor_outputs,
    run_style_factor_scoring,
)

__all__ = [
    "FEATURE_CATEGORIES",
    "METADATA_KEYS",
    "EfaResult",
    "StyleFactorResult",
    "attach_chart_metadata",
    "cluster_by_decade",
    "cluster_songs",
    "cluster_songs_within_eras",
    "era_cluster_summary",
    "evaluate_factor_trends",
    "evaluate_style_factor_trends",
    "feature_columns",
    "factor_decade_kruskal_stats",
    "factor_trend_stats",
    "factor_trends_by_decade",
    "load_chart_metadata",
    "load_efa_outputs",
    "load_style_factor_outputs",
    "load_features_dir",
    "load_npz",
    "normalize_song_key",
    "parse_filename",
    "prepare_efa_matrix",
    "prepare_feature_matrix",
    "run_efa",
    "run_style_factor_scoring",
    "scalar_feature_keys",
    "summarize_top_loadings",
    "write_factor_trend_outputs",
]
