"""Clustering utilities for scalar melody feature tables."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .loader import DEFAULT_FEATURES_DIR, load_features_dir
from .metadata import METADATA_COLUMNS, attach_chart_metadata
from .schema import FEATURE_CATEGORIES

IDENTIFIER_COLUMNS = (
    "file",
    "song_id",
    "artist",
    "title",
    "cluster",
    "era",
    "song_key",
    *METADATA_COLUMNS,
)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return scalar feature column names from a loaded features table."""
    return [
        col
        for col in df.columns
        if col not in IDENTIFIER_COLUMNS and "." in col
    ]


def prepare_feature_matrix(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str], StandardScaler]:
    """Scale feature columns for clustering."""
    columns = columns or feature_columns(df)
    if not columns:
        raise ValueError("No feature columns found in DataFrame")

    matrix = df[columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    imputed = SimpleImputer(strategy="median").fit_transform(matrix)

    # Drop constant columns — they carry no clustering signal and break scaling.
    selector = VarianceThreshold(threshold=0.0)
    filtered = selector.fit_transform(imputed)
    kept = selector.get_support()
    used_columns = [col for col, keep in zip(columns, kept) if keep]

    scaled = StandardScaler().fit_transform(filtered)
    return scaled, used_columns, StandardScaler()


def choose_cluster_count(
    matrix: np.ndarray,
    *,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Score k-means models across a range of cluster counts."""
    rows = []
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(matrix)
        score = silhouette_score(matrix, labels)
        rows.append({"n_clusters": k, "silhouette": score, "inertia": model.inertia_})
    return pd.DataFrame(rows)


def cluster_songs(
    df: pd.DataFrame,
    *,
    n_clusters: int = 6,
    random_state: int = 42,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Assign k-means cluster labels and 2D PCA coordinates."""
    matrix, used_columns, _ = prepare_feature_matrix(df, columns=columns)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(matrix)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(matrix)

    result = df.copy()
    result["cluster"] = labels
    result["pca_x"] = coords[:, 0]
    result["pca_y"] = coords[:, 1]
    result.attrs["feature_columns"] = used_columns
    result.attrs["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    result.attrs["silhouette"] = float(silhouette_score(matrix, labels))
    return result


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise cluster sizes and representative songs."""
    grouped = (
        df.groupby("cluster", as_index=False)
        .agg(
            n_songs=("song_id", "count"),
            sample_artists=("artist", lambda s: ", ".join(sorted(set(s))[:3])),
        )
        .sort_values("cluster")
    )
    return grouped


def cluster_by_decade(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each song to a decade cluster from its chart year."""
    if "decade" not in df.columns:
        raise ValueError("decade column missing — call attach_chart_metadata() first")

    result = df.copy()
    result["cluster"] = result["decade"].astype(int)
    result["era"] = result["decade"].astype(int)
    result.attrs["clustering_method"] = "decade"
    return result


def cluster_songs_within_eras(
    df: pd.DataFrame,
    *,
    n_clusters: int = 2,
    era_column: str = "decade",
    min_era_size: int = 8,
    random_state: int = 42,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Run k-means separately within each chart-era group."""
    if era_column not in df.columns:
        raise ValueError(f"{era_column!r} missing — call attach_chart_metadata() first")

    result = df.copy()
    result["cluster"] = pd.NA
    result["era"] = result[era_column]
    silhouettes: dict[int, float] = {}

    for era, group in result.groupby(era_column, sort=True):
        era_label = int(era)
        if len(group) < min_era_size:
            result.loc[group.index, "cluster"] = 0
            continue

        era_clusters = min(n_clusters, len(group))
        if era_clusters < 2:
            result.loc[group.index, "cluster"] = 0
            continue

        clustered = cluster_songs(
            group,
            n_clusters=era_clusters,
            random_state=random_state,
            columns=columns,
        )
        result.loc[group.index, "cluster"] = clustered["cluster"].astype(int)
        silhouettes[era_label] = clustered.attrs["silhouette"]

    matrix, used_columns, _ = prepare_feature_matrix(result, columns=columns)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(matrix)
    result["pca_x"] = coords[:, 0]
    result["pca_y"] = coords[:, 1]
    result["cluster"] = result["cluster"].astype(int)
    result.attrs["feature_columns"] = used_columns
    result.attrs["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    result.attrs["clustering_method"] = "within_era"
    result.attrs["era_silhouettes"] = silhouettes
    return result


def era_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise melody clusters within each chart decade."""
    if "era" not in df.columns:
        raise ValueError("era column missing")

    grouped = (
        df.groupby(["era", "cluster"], as_index=False)
        .agg(
            n_songs=("song_id", "count"),
            year_min=("chart_year", "min"),
            year_max=("chart_year", "max"),
            sample_artists=("artist", lambda s: ", ".join(sorted(set(s))[:3])),
        )
        .sort_values(["era", "cluster"])
    )
    return grouped


def plot_clusters(
    df: pd.DataFrame,
    output_path: Path,
    *,
    hue: str = "cluster",
    title: str | None = None,
) -> None:
    """Write a PCA scatter plot coloured by cluster or era."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="pca_x",
        y="pca_y",
        hue=hue,
        palette="tab10",
        s=35,
        alpha=0.85,
        legend=True,
    )
    plt.title(title or "Billboard vocal melodies — k-means clusters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_clusters_by_chart_year(df: pd.DataFrame, output_path: Path) -> None:
    """Write a timeline scatter: chart year vs PC1, coloured by decade."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    sns.scatterplot(
        data=df,
        x="chart_year",
        y="pca_x",
        hue="decade",
        palette="viridis",
        s=40,
        alpha=0.85,
        legend=True,
    )
    plt.title("Billboard vocal melodies by chart year (PC1 projection)")
    plt.xlabel("Chart year")
    plt.ylabel("PC1")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_category_means_by_cluster(df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap of mean z-scored features by category and cluster."""
    feature_cols = feature_columns(df)
    category_map = {}
    for col in feature_cols:
        category = col.split(".", 1)[0]
        category_map.setdefault(category, []).append(col)

    rows = []
    for cluster, group in df.groupby("cluster"):
        for category in FEATURE_CATEGORIES:
            cols = category_map.get(category, [])
            if not cols:
                continue
            values = group[cols].apply(pd.to_numeric, errors="coerce")
            rows.append(
                {
                    "cluster": cluster,
                    "category": category,
                    "mean": values.to_numpy().mean(),
                }
            )

    heatmap_df = pd.DataFrame(rows).pivot(index="category", columns="cluster", values="mean")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(heatmap_df, cmap="vlag", center=0, annot=False)
    plt.title("Mean feature values by category and cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cluster scalar melody features")
    parser.add_argument(
        "features_dir",
        nargs="?",
        default=str(DEFAULT_FEATURES_DIR),
        help="Directory containing .npz feature files",
    )
    parser.add_argument("--clusters", type=int, default=6, help="Number of k-means clusters")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for CSV and plot outputs",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Also write silhouette scores for k=2..10",
    )
    parser.add_argument(
        "--by-decade",
        action="store_true",
        help="Cluster by chart decade instead of melody features",
    )
    parser.add_argument(
        "--within-decade",
        action="store_true",
        help="Run k-means within each chart decade (requires chart metadata)",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to bimmuda_per_song_metadata.csv (for --by-decade / --within-decade)",
    )
    args = parser.parse_args(argv)

    df = load_features_dir(args.features_dir, show_progress=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.by_decade or args.within_decade:
        df = attach_chart_metadata(df, path=args.metadata)

    if args.by_decade:
        clustered = cluster_by_decade(df)
        matrix, _, _ = prepare_feature_matrix(clustered)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(matrix)
        clustered["pca_x"] = coords[:, 0]
        clustered["pca_y"] = coords[:, 1]
        summary = (
            clustered.groupby("cluster", as_index=False)
            .agg(
                n_songs=("song_id", "count"),
                year_min=("chart_year", "min"),
                year_max=("chart_year", "max"),
                sample_artists=("artist", lambda s: ", ".join(sorted(set(s))[:3])),
            )
            .sort_values("cluster")
        )
        clustered.drop(columns=["pca_x", "pca_y"], errors="ignore").to_csv(
            output_dir / "clusters_by_decade.csv",
            index=False,
        )
        summary.to_csv(output_dir / "clusters_by_decade_summary.csv", index=False)
        plot_clusters(
            clustered,
            output_dir / "clusters_by_decade_pca.png",
            hue="decade",
            title="Billboard vocal melodies by chart decade (PCA projection)",
        )
        plot_clusters_by_chart_year(clustered, output_dir / "clusters_by_chart_year.png")
        print(f"Loaded {len(df)} songs with chart metadata")
        print("Clustered by chart decade:")
        print(summary.to_string(index=False))
        print(f"Wrote outputs to {output_dir.resolve()}")
        return

    matrix, _, _ = prepare_feature_matrix(df)
    if args.sweep:
        sweep = choose_cluster_count(matrix)
        sweep.to_csv(output_dir / "cluster_sweep.csv", index=False)

    if args.within_decade:
        clustered = cluster_songs_within_eras(df, n_clusters=args.clusters)
        summary = era_cluster_summary(clustered)
        clustered.drop(columns=["pca_x", "pca_y"], errors="ignore").to_csv(
            output_dir / "clusters_within_decade.csv",
            index=False,
        )
        summary.to_csv(output_dir / "clusters_within_decade_summary.csv", index=False)
        plot_clusters(
            clustered,
            output_dir / "clusters_within_decade_pca.png",
            hue="era",
            title=f"Melody clusters within chart decades (k={args.clusters} per era)",
        )
        plot_clusters_by_chart_year(clustered, output_dir / "clusters_within_decade_timeline.png")
        print(f"Loaded {len(df)} songs with chart metadata")
        print(f"Melody clustering within decades (k={args.clusters} where n≥8):")
        print(summary.to_string(index=False))
        if clustered.attrs.get("era_silhouettes"):
            print("Per-era silhouettes:", clustered.attrs["era_silhouettes"])
        print(f"Wrote outputs to {output_dir.resolve()}")
        return

    clustered = cluster_songs(df, n_clusters=args.clusters)
    clustered.drop(columns=["pca_x", "pca_y"], errors="ignore").to_csv(
        output_dir / "clusters.csv",
        index=False,
    )
    summary = cluster_summary(clustered)
    summary.to_csv(output_dir / "cluster_summary.csv", index=False)

    plot_clusters(clustered, output_dir / "clusters_pca.png")
    plot_category_means_by_cluster(clustered, output_dir / "clusters_by_category.png")

    silhouette = clustered.attrs.get("silhouette")
    print(f"Loaded {len(df)} songs with {len(feature_columns(df))} scalar features")
    print(f"Clustered into {args.clusters} groups (silhouette={silhouette:.3f})")
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {output_dir.resolve()}")
