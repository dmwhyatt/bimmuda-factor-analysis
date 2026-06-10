"""Exploratory factor analysis for Billboard melody features."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .metadata import METADATA_COLUMNS, attach_chart_metadata
from .efa_plots import has_uncertainty_columns, plot_parallel_scree_interactive
from .efa_interpretations import loading_based_factor_labels
from .preprocess import prepare_efa_matrix

EFA_META_COLUMNS = (
    "file",
    "song_id",
    "artist",
    "title",
    *METADATA_COLUMNS,
)
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "factor_billboard.R"


def refresh_parallel_uncertainty(
    output_dir: Path | str,
    *,
    features_csv: Path | str | None = None,
    n_iter: int = 100,
    max_factors: int = 20,
    rscript_path: str | None = None,
) -> pd.DataFrame:
    """Recompute parallel analysis CIs without refitting the full EFA model."""
    output_dir = Path(output_dir)
    features_csv = Path(
        features_csv or output_dir / "billboard_features_efa.csv"
    )
    if not features_csv.is_file():
        raise FileNotFoundError(
            f"Missing {features_csv}; run bimmuda-efa first to export features."
        )

    rscript = rscript_path or shutil.which("Rscript")
    if not rscript:
        raise RuntimeError("Rscript not found on PATH; install R to compute uncertainty.")

    cmd = [
        rscript,
        str(SCRIPT_PATH),
        str(features_csv),
        str(output_dir),
        "",
        str(n_iter),
        str(max_factors),
        "uncertainty-only",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Parallel uncertainty script failed:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    return pd.read_csv(output_dir / "efa_parallel_analysis.csv")


def retained_parallel_factors(parallel_analysis: pd.DataFrame) -> list[str]:
    """Factor names whose observed eigenvalues exceed the parallel-analysis null."""
    retained = parallel_analysis.loc[parallel_analysis["retain"], "factor"].astype(int)
    return [f"F{number}" for number in sorted(retained)]


@dataclass
class EfaResult:
    """Outputs from a completed EFA run."""

    n_factors: int
    parallel_suggestion: int
    scores: pd.DataFrame
    top_loadings: pd.DataFrame
    loadings: pd.DataFrame
    variance: pd.DataFrame
    parallel_analysis: pd.DataFrame
    output_dir: Path


def export_retained_factor_interpretations(result: EfaResult) -> pd.DataFrame:
    """Write interpretations for all parallel-analysis retained factors."""
    retained = retained_parallel_factors(result.parallel_analysis)
    labels = loading_based_factor_labels(result.top_loadings)
    rows = []
    for factor in retained:
        top = result.top_loadings.loc[result.top_loadings["factor"] == factor].sort_values("rank")
        rows.append(
            {
                "factor": factor,
                "factor_name": labels.get(factor, factor),
                "top_features": "; ".join(top.head(5)["feature"].tolist()),
                "ss_loading_pct": result.variance.loc[
                    result.variance["factor"] == factor, "prop_var_pct"
                ].iloc[0]
                if factor in result.variance["factor"].values
                else None,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(result.output_dir / "efa_retained_factor_interpretations.csv", index=False)
    return table


def export_efa_features(df: pd.DataFrame, output_path: Path) -> Path:
    """Write a CSV of metadata + scaled-ready numeric features for R/psych."""
    scaled, columns, _ = prepare_efa_matrix(df)
    meta_cols = [col for col in EFA_META_COLUMNS if col in df.columns]
    export = pd.concat([df[meta_cols].reset_index(drop=True), scaled.reset_index(drop=True)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_path, index=False)
    return output_path


def run_efa(
    df: pd.DataFrame,
    *,
    output_dir: Path | str = "outputs",
    n_factors: int | None = None,
    n_iter: int = 100,
    max_factors: int = 20,
    metadata_path: Path | str | None = None,
    rscript_path: str | None = None,
) -> EfaResult:
    """Run EFA via ``scripts/factor_billboard.R`` (psych, promax, parallel analysis)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "chart_year" not in df.columns:
        df = attach_chart_metadata(df, path=metadata_path)

    features_csv = output_dir / "billboard_features_efa.csv"
    export_efa_features(df, features_csv)

    rscript = rscript_path or shutil.which("Rscript")
    if not rscript:
        raise RuntimeError("Rscript not found on PATH; install R to run EFA.")

    cmd = [
        rscript,
        str(SCRIPT_PATH),
        str(features_csv),
        str(output_dir),
        "" if n_factors is None else str(n_factors),
        str(n_iter),
        str(max_factors),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "EFA R script failed:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip())

    result = load_efa_outputs(output_dir)
    export_retained_factor_interpretations(result)
    scree_path = plot_parallel_scree_interactive(
        result.parallel_analysis,
        output_dir / "efa_parallel_scree.html",
        n_factors_used=result.n_factors,
    )
    print(f"Wrote interactive scree plot to {scree_path.resolve()}")
    return result


def load_efa_outputs(output_dir: Path | str) -> EfaResult:
    """Load CSV outputs written by ``factor_billboard.R``."""
    output_dir = Path(output_dir)
    parallel = pd.read_csv(output_dir / "efa_parallel_analysis.csv")
    parallel_suggestion = int(parallel.loc[parallel["retain"], "factor"].count())

    variance = pd.read_csv(output_dir / "efa_variance.csv")
    scores = pd.read_csv(output_dir / "efa_factor_scores.csv")
    top_loadings = pd.read_csv(output_dir / "efa_top_loadings.csv")
    loadings = pd.read_csv(output_dir / "efa_loadings.csv")

    return EfaResult(
        n_factors=len(variance),
        parallel_suggestion=parallel_suggestion,
        scores=scores,
        top_loadings=top_loadings,
        loadings=loadings,
        variance=variance,
        parallel_analysis=parallel,
        output_dir=output_dir,
    )


def factor_score_columns(scores: pd.DataFrame) -> list[str]:
    """Return factor score column names (F1, F2, ...)."""
    return [col for col in scores.columns if col.startswith("F") and col[1:].isdigit()]


def summarize_top_loadings(top_loadings: pd.DataFrame, *, n: int = 5) -> dict[str, list[str]]:
    """Group top loading feature names by factor for quick interpretation."""
    summary: dict[str, list[str]] = {}
    for factor, group in top_loadings.groupby("factor"):
        rows = group.sort_values("rank").head(n)
        summary[str(factor)] = rows["feature"].tolist()
    return summary


def print_interpretation_hints(result: EfaResult, *, top_n: int = 5) -> None:
    """Print top loadings per factor to stdout."""
    hints = summarize_top_loadings(result.top_loadings, n=top_n)
    print(f"Parallel analysis suggests {result.parallel_suggestion} factors")
    print(
        f"Extracted {result.n_factors} factors "
        f"({result.variance['cum_var_pct'].iloc[-1]:.1f}% cumulative SS loadings)\n"
    )
    for factor, features in hints.items():
        print(f"{factor}: {', '.join(features)}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .loader import DEFAULT_FEATURES_DIR, load_features_dir
    from .efa_trends import evaluate_factor_trends

    parser = argparse.ArgumentParser(
        description="Exploratory factor analysis on Billboard melody features",
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
        help="Directory for EFA CSV/plot outputs",
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=None,
        help="Number of factors (default: parallel analysis suggestion)",
    )
    parser.add_argument(
        "--parallel-iter",
        type=int,
        default=100,
        help="Iterations for Horn's parallel analysis",
    )
    parser.add_argument(
        "--max-factors",
        type=int,
        default=20,
        help="Cap on factor count when parallel analysis over-suggests (default 20)",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to bimmuda_per_song_metadata.csv",
    )
    parser.add_argument(
        "--no-trends",
        action="store_true",
        help="Skip chart-year trend analysis",
    )
    parser.add_argument(
        "--scree-only",
        action="store_true",
        help="Regenerate interactive scree plot from existing efa_parallel_analysis.csv",
    )
    parser.add_argument(
        "--refresh-uncertainty",
        action="store_true",
        help="Recompute bootstrap/simulation bands for the scree plot (~2–5 min)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)

    if args.scree_only or args.refresh_uncertainty:
        if args.refresh_uncertainty or not (
            output_dir / "efa_parallel_analysis.csv"
        ).is_file() or not has_uncertainty_columns(
            pd.read_csv(output_dir / "efa_parallel_analysis.csv", nrows=1)
        ):
            refresh_parallel_uncertainty(
                output_dir,
                n_iter=args.parallel_iter,
                max_factors=args.max_factors,
            )

    if args.scree_only:
        parallel = pd.read_csv(output_dir / "efa_parallel_analysis.csv")
        n_used = None
        variance_path = output_dir / "efa_variance.csv"
        if variance_path.is_file():
            n_used = len(pd.read_csv(variance_path))
        path = plot_parallel_scree_interactive(
            parallel,
            output_dir / "efa_parallel_scree.html",
            n_factors_used=n_used,
        )
        print(f"Wrote interactive scree plot to {path.resolve()}")
        return

    if args.refresh_uncertainty:
        parallel = pd.read_csv(output_dir / "efa_parallel_analysis.csv")
        path = plot_parallel_scree_interactive(
            parallel,
            output_dir / "efa_parallel_scree.html",
        )
        print(f"Wrote interactive scree plot to {path.resolve()}")
        return

    df = load_features_dir(args.features_dir, show_progress=True)
    result = run_efa(
        df,
        output_dir=args.output_dir,
        n_factors=args.factors,
        n_iter=args.parallel_iter,
        max_factors=args.max_factors,
        metadata_path=args.metadata,
    )
    print_interpretation_hints(result)

    if not args.no_trends:
        kruskal_stats, decade_stats, _year_stats = evaluate_factor_trends(result)
        print("\nFactor vs chart decade (Kruskal-Wallis):")
        print(kruskal_stats.to_string(index=False))
        print(f"\nWrote trend outputs to {Path(args.output_dir).resolve()}")

    print(f"\nEFA outputs in {Path(args.output_dir).resolve()}")
