"""Apply Style-Classification EFA loadings to Billboard melodies and track trends."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .efa_trends import write_factor_trend_outputs
from .metadata import attach_chart_metadata
from .preprocess import export_raw_features

DEFAULT_STYLE_PROJECT = Path(
    os.environ.get(
        "STYLE_CLASSIFICATION_DIR",
        "/Users/davidwhyatt/Style-Classification-Analysis",
    )
)
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "style_factor_scores.R"


@dataclass
class StyleFactorResult:
    """Outputs from scoring Billboard with Style-Classification factors."""

    scores: pd.DataFrame
    top_loadings: pd.DataFrame
    loadings: pd.DataFrame
    variance: pd.DataFrame
    alignment: pd.DataFrame
    output_dir: Path
    style_project_dir: Path


def run_style_factor_scoring(
    df: pd.DataFrame,
    *,
    output_dir: Path | str = "outputs",
    style_project_dir: Path | str | None = None,
    metadata_path: Path | str | None = None,
    rscript_path: str | None = None,
    refresh_model: bool = False,
) -> StyleFactorResult:
    """Score Billboard melodies using Style-Classification EFA loadings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    style_dir = Path(style_project_dir or DEFAULT_STYLE_PROJECT)

    if not style_dir.is_dir():
        raise FileNotFoundError(
            f"Style-Classification project not found: {style_dir}\n"
            "Set STYLE_CLASSIFICATION_DIR to the project root."
        )

    if "chart_year" not in df.columns:
        df = attach_chart_metadata(df, path=metadata_path)

    features_csv = output_dir / "billboard_features_style_factors.csv"
    export_raw_features(df, features_csv)

    model_path = output_dir / "style_factor_model.rds"
    if refresh_model and model_path.is_file():
        model_path.unlink()

    rscript = rscript_path or shutil.which("Rscript")
    if not rscript:
        raise RuntimeError("Rscript not found on PATH; install R to score Style factors.")

    cmd = [rscript, str(SCRIPT_PATH), str(style_dir), str(features_csv), str(output_dir)]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Style factor scoring script failed:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip())

    return load_style_factor_outputs(output_dir, style_project_dir=style_dir)


def load_style_factor_outputs(
    output_dir: Path | str,
    *,
    style_project_dir: Path | str | None = None,
) -> StyleFactorResult:
    """Load CSV outputs written by ``style_factor_scores.R``."""
    output_dir = Path(output_dir)
    return StyleFactorResult(
        scores=pd.read_csv(output_dir / "style_factor_scores.csv"),
        top_loadings=pd.read_csv(output_dir / "style_factor_top_loadings.csv"),
        loadings=pd.read_csv(output_dir / "style_factor_loadings.csv"),
        variance=pd.read_csv(output_dir / "style_factor_variance.csv"),
        alignment=pd.read_csv(output_dir / "style_factor_alignment.csv"),
        output_dir=output_dir,
        style_project_dir=Path(style_project_dir or DEFAULT_STYLE_PROJECT),
    )


def print_style_factor_hints(result: StyleFactorResult, *, top_n: int = 5) -> None:
    """Print Style-Classification factor names and top loadings."""
    align = result.alignment.iloc[0]
    print(
        f"Scored {int(align['n_billboard_songs'])} Billboard songs using "
        f"{int(align['n_style_features'])} Style-Classification features "
        f"(trained on {int(align['n_style_training_songs'])} folk melodies)\n"
    )
    if int(align["n_missing_features_imputed"]) > 0:
        print(
            f"  {int(align['n_missing_features_imputed'])} features missing from "
            "Billboard export were imputed at 0 before z-scoring.\n"
        )

    for _, row in result.variance.iterrows():
        print(
            f"{row['factor']} ({row['factor_name']}): "
            f"{row['prop_var_pct']:.1f}% SS loadings"
        )
    print()

    for factor, group in result.top_loadings.groupby("factor"):
        rows = group.sort_values("rank").head(top_n)
        name = rows["factor_name"].iloc[0]
        features = ", ".join(rows["feature"].tolist())
        print(f"{factor} {name}: {features}")


def evaluate_style_factor_trends(
    result: StyleFactorResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Write chart-decade trend tables and plots for Style-Classification factor scores."""
    return write_factor_trend_outputs(
        result.scores,
        result.output_dir,
        prefix="style_factor",
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .loader import DEFAULT_FEATURES_DIR, load_features_dir

    parser = argparse.ArgumentParser(
        description=(
            "Score Billboard melodies with Style-Classification EFA factors "
            "and track scores over chart years"
        ),
    )
    parser.add_argument(
        "features_dir",
        nargs="?",
        default=str(DEFAULT_FEATURES_DIR),
        help="Directory containing .npz feature files",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for CSV/plot outputs",
    )
    parser.add_argument(
        "--style-dir",
        default=str(DEFAULT_STYLE_PROJECT),
        help="Path to Style-Classification-Analysis project",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to bimmuda_per_song_metadata.csv",
    )
    parser.add_argument(
        "--refresh-model",
        action="store_true",
        help="Re-fit Style-Classification EFA instead of using cached model.rds",
    )
    parser.add_argument(
        "--no-trends",
        action="store_true",
        help="Skip chart-year trend analysis",
    )
    args = parser.parse_args(argv)

    df = load_features_dir(args.features_dir, show_progress=True)
    result = run_style_factor_scoring(
        df,
        output_dir=args.output_dir,
        style_project_dir=args.style_dir,
        metadata_path=args.metadata,
        refresh_model=args.refresh_model,
    )
    print_style_factor_hints(result)

    if not args.no_trends:
        kruskal_stats, decade_stats, _year_stats = evaluate_style_factor_trends(result)
        print("\nStyle factor vs chart decade (Kruskal-Wallis):")
        print(kruskal_stats.to_string(index=False))
        print(f"\nWrote trend outputs to {Path(args.output_dir).resolve()}")

    print(f"\nStyle factor outputs in {Path(args.output_dir).resolve()}")
