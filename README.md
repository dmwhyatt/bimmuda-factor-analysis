# bimmuda-feature-analysis

EFA of melody features in [BiMMuDa](https://github.com/madelinehamilton/BiMMuDa) — Billboard Hot 100 **top 5 per chart year** (366 songs × 220 scalar features from [melody-features](https://github.com/dmwhyatt/melody-features)).

We fit a **Billboard-native** factor model (parallel analysis → promax EFA via R `psych`, following [Style-Classification-Analysis](https://github.com/dmwhyatt/Style-Classification-Analysis)), name factors from their loadings, and test whether factor scores **change over chart decades**. Clustering scripts are leftover from earlier exploration.

## Install

```bash
git clone https://github.com/dmwhyatt/bimmuda-factor-analysis.git
cd bimmuda-factor-analysis
pip install -e ".[dev]"
```

Python 3.9+. EFA requires R with `psych`, `tidyverse`, and `jsonlite`. Point commands at a directory of `.npz` feature files (optional env: `BIMMUDA_FEATURES_DIR`, `BIMMUDA_METADATA_CSV`).

## Run

```bash
bimmuda-efa /path/to/features
```

Main result: **`outputs/efa_factor_trends_dashboard.html`** — interactive gallery of retained factors (F1–F20) with decade means and per-song hover. Also writes scree plot, loadings, factor scores, and Kruskal–Wallis tests to `outputs/`.

## How it works

1. **Parallel analysis** — retain factors whose eigenvalues exceed a random-data null (~20 for this corpus).
2. **Promax EFA** — extract factors, score each song, interpret from top loadings (`efa_retained_factor_interpretations.csv`).
3. **Trends** — decade means ± SEM; Kruskal–Wallis by decade (`efa_decade_kruskal.csv`).

## Optional

Included in the repository are some early versions of analysis that ultimately didn't prove to be as promising.

**Style-Classification factors** — score BiMMuDa with the Essen/China/Europe loading matrix instead of fitting new factors (`bimmuda-style-factors`; needs `STYLE_CLASSIFICATION_DIR`).

**Clustering** — earlier attempt at k-means on raw features (`bimmuda-cluster`).

## License

MIT
