#!/usr/bin/env Rscript
# Apply Style-Classification-Analysis EFA loadings to Billboard melodies.
# Fits (or loads cached) the Essen/China/Europe model from factor_logistic.R,
# then computes regression factor scores on Billboard z-scored features.

suppressPackageStartupMessages({
  library(tidyverse)
  library(psych)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop(
    "Usage: style_factor_scores.R STYLE_PROJECT_DIR BILLBOARD_FEATURES_CSV OUTPUT_DIR\n",
    "  STYLE_PROJECT_DIR  path to Style-Classification-Analysis\n",
    "  BILLBOARD_FEATURES_CSV  metadata + numeric features (from bimmuda export)\n",
    "  OUTPUT_DIR  directory for CSV outputs"
  )
}

STYLE_DIR <- args[[1]]
BILLBOARD_CSV <- args[[2]]
OUTPUT_DIR <- args[[3]]
N_FACTORS <- 8L
TOP_LOADINGS_N <- 10L

if (!dir.exists(STYLE_DIR)) {
  stop("Style-Classification project not found: ", STYLE_DIR)
}
if (!file.exists(BILLBOARD_CSV)) {
  stop("Missing Billboard features file: ", BILLBOARD_CSV)
}
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

MODEL_PATH <- file.path(OUTPUT_DIR, "style_factor_model.rds")
LOADINGS_PATH <- file.path(OUTPUT_DIR, "style_factor_loadings.csv")

factor_names <- c(
  "1. Long Pulses",
  "2. Metric Strength",
  "3. Scalic Ascent & Narrow Rhythm",
  "4. Interval Variability",
  "5. Busy Stepwise Melody",
  "6. Corpus Familiar & Long",
  "7. Dual-Pulse Rhythm",
  "8. High Pitch Height"
)

basename_no_ext <- function(x) {
  tools::file_path_sans_ext(basename(as.character(x)))
}

fit_style_model <- function() {
  features_csv <- file.path(STYLE_DIR, "essen_china_europe_features.csv")
  if (!file.exists(features_csv)) {
    stop(
      "Missing ", features_csv, "\n",
      "Run logistic.py in Style-Classification-Analysis first."
    )
  }

  features <- read_csv(features_csv, show_col_types = FALSE)

  pearce_txt <- file.path(STYLE_DIR, "pearce_default_idyom_basenames.txt")
  if (file.exists(pearce_txt)) {
    pearce_bases <- readLines(pearce_txt, warn = FALSE)
    pearce_bases <- pearce_bases[nzchar(pearce_bases)]
    n_before <- nrow(features)
    features <- features %>%
      filter(!tolower(basename_no_ext(melody_id)) %in% pearce_bases)
    n_drop <- n_before - nrow(features)
    if (n_drop > 0) {
      message("Excluded ", n_drop, " melody row(s) overlapping pearce_default_idyom.")
    }
  }

  features_numeric <- features %>%
    select(where(is.numeric)) %>%
    select(-melody_num)

  features_numeric <- features_numeric %>%
    mutate(across(everything(), ~ replace(.x, is.infinite(.x), NA_real_))) %>%
    mutate(across(everything(), ~ replace_na(.x, 0)))

  x_raw <- features_numeric
  if (nrow(x_raw) == 0) {
    stop("No rows remain in Style-Classification feature matrix.")
  }

  variances <- sapply(x_raw, var)
  zero_var_cols <- names(variances[variances == 0 | is.na(variances)])
  x_raw <- x_raw %>% select(-any_of(zero_var_cols))
  scaled_mat <- scale(x_raw)
  x_scaled <- as.data.frame(scaled_mat)

  message(
    "Fitting Style-Classification EFA on ",
    nrow(x_scaled), " melodies, ",
    ncol(x_scaled), " features..."
  )
  fit <- fa(x_scaled, nfactors = N_FACTORS, rotate = "promax", fm = "pa")

  list(
    fit = fit,
    feature_names = colnames(x_scaled),
    feature_means = attr(scaled_mat, "scaled:center"),
    feature_sds = attr(scaled_mat, "scaled:scale"),
    factor_names = factor_names[seq_len(N_FACTORS)],
    n_training_rows = nrow(x_scaled),
    n_training_features = ncol(x_scaled)
  )
}

load_or_fit_model <- function() {
  if (file.exists(MODEL_PATH)) {
    message("Loading cached Style-Classification model from ", MODEL_PATH)
    return(readRDS(MODEL_PATH))
  }

  model <- fit_style_model()
  saveRDS(model, MODEL_PATH)
  message("Cached Style-Classification model to ", MODEL_PATH)
  model
}

export_loadings <- function(model) {
  loadings_mat <- as.matrix(model$fit$loadings[, , drop = FALSE])
  rownames(loadings_mat) <- model$feature_names
  loadings_df <- as.data.frame(loadings_mat) %>%
    rownames_to_column("feature") %>%
    mutate(across(starts_with("PA"), ~ round(.x, 6)))

  write_csv(loadings_df, LOADINGS_PATH)

  ss_loadings <- colSums(loadings_mat^2, na.rm = TRUE)
  prop_var <- ss_loadings / nrow(loadings_mat)
  cum_var <- cumsum(prop_var)
  var_table <- tibble(
    factor = paste0("F", seq_len(N_FACTORS)),
    factor_name = model$factor_names,
    ss_loading = round(ss_loadings, 4),
    prop_var_pct = round(100 * prop_var, 2),
    cum_var_pct = round(100 * cum_var, 2)
  )
  write_csv(var_table, file.path(OUTPUT_DIR, "style_factor_variance.csv"))

  top_parts <- list()
  for (i in seq_len(N_FACTORS)) {
    col_name <- colnames(loadings_mat)[i]
    v <- loadings_mat[, i]
    top_idx <- order(abs(v), decreasing = TRUE)[seq_len(min(TOP_LOADINGS_N, length(v)))]
    top_parts[[i]] <- tibble(
      factor = paste0("F", i),
      factor_name = model$factor_names[i],
      rank = seq_along(top_idx),
      feature = rownames(loadings_mat)[top_idx],
      loading = unname(v[top_idx]),
      abs_loading = abs(unname(v[top_idx]))
    )
  }
  top_loadings <- bind_rows(top_parts)
  write_csv(top_loadings, file.path(OUTPUT_DIR, "style_factor_top_loadings.csv"))
  top_loadings
}

align_billboard_matrix <- function(features, model) {
  feature_names <- model$feature_names
  meta_cols <- c(
    "file", "song_id", "artist", "title",
    "chart_year", "chart_position", "decade",
    "genre_broad_1", "genre_broad_2"
  )

  meta_present <- intersect(meta_cols, names(features))
  features_meta <- features %>% select(all_of(meta_present))

  features_numeric <- features %>%
    select(where(is.numeric)) %>%
    select(-any_of(meta_cols))

  features_numeric <- features_numeric %>%
    mutate(across(everything(), ~ replace(.x, is.infinite(.x), NA_real_))) %>%
    mutate(across(everything(), ~ replace_na(.x, 0)))

  missing_feats <- setdiff(feature_names, names(features_numeric))
  if (length(missing_feats) > 0) {
    message(
      length(missing_feats),
      " Style-Classification feature(s) missing from Billboard export; ",
      "imputing Essen training means before z-scoring."
    )
    for (feat in missing_feats) {
      features_numeric[[feat]] <- model$feature_means[[feat]]
    }
  }

  x_raw <- features_numeric %>% select(all_of(feature_names))

  x_scaled <- as.data.frame(
    scale(
      x_raw,
      center = model$feature_means[feature_names],
      scale = model$feature_sds[feature_names]
    )
  )

  list(meta = features_meta, x_scaled = x_scaled, missing_features = missing_feats)
}

model <- load_or_fit_model()
top_loadings <- export_loadings(model)

billboard <- read_csv(BILLBOARD_CSV, show_col_types = FALSE)
aligned <- align_billboard_matrix(billboard, model)

score_result <- factor.scores(
  aligned$x_scaled,
  model$fit,
  method = "regression"
)
factor_scores <- as.data.frame(score_result$scores)
colnames(factor_scores) <- paste0("F", seq_len(ncol(factor_scores)))

name_map <- setNames(model$factor_names, paste0("F", seq_len(N_FACTORS)))
factor_name_cols <- paste0(colnames(factor_scores), "_name")
for (col in colnames(factor_scores)) {
  factor_scores[[paste0(col, "_name")]] <- name_map[[col]]
}

scores_df <- bind_cols(aligned$meta, factor_scores)
write_csv(scores_df, file.path(OUTPUT_DIR, "style_factor_scores.csv"))

alignment_summary <- tibble(
  n_billboard_songs = nrow(scores_df),
  n_style_training_songs = model$n_training_rows,
  n_style_features = model$n_training_features,
  n_missing_features_imputed = length(aligned$missing_features)
)
write_csv(alignment_summary, file.path(OUTPUT_DIR, "style_factor_alignment.csv"))
if (length(aligned$missing_features) > 0) {
  write_csv(
    tibble(feature = aligned$missing_features),
    file.path(OUTPUT_DIR, "style_factor_missing_features.csv")
  )
}

cat("\nStyle-Classification factor scores for Billboard\n")
cat("  Songs scored:", nrow(scores_df), "\n")
cat("  Features in model:", model$n_training_features, "\n")
cat("  Missing features imputed:", length(aligned$missing_features), "\n\n")
cat("Factor names:\n")
for (i in seq_len(N_FACTORS)) {
  cat(sprintf("  F%d: %s\n", i, model$factor_names[i]))
}
cat("\nWrote:\n")
cat(" ", file.path(OUTPUT_DIR, "style_factor_scores.csv"), "\n")
cat(" ", LOADINGS_PATH, "\n")
cat(" ", file.path(OUTPUT_DIR, "style_factor_top_loadings.csv"), "\n")
cat(" ", file.path(OUTPUT_DIR, "style_factor_variance.csv"), "\n")
