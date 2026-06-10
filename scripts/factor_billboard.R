#!/usr/bin/env Rscript
# Exploratory factor analysis for Billboard melody features.
# Adapted from Style-Classification-Analysis/factor_logistic.R (psych EFA only).

suppressPackageStartupMessages({
  library(tidyverse)
  library(psych)
})

args <- commandArgs(trailingOnly = TRUE)
FEATURES_CSV <- if (length(args) >= 1) args[[1]] else "outputs/billboard_features_efa.csv"
OUTPUT_DIR <- if (length(args) >= 2) args[[2]] else "outputs"
N_FACTORS_ARG <- if (length(args) >= 3 && nzchar(args[[3]])) as.integer(args[[3]]) else NA_integer_
N_ITER <- if (length(args) >= 4) as.integer(args[[4]]) else 100L
MAX_FACTORS <- if (length(args) >= 5 && nzchar(args[[5]])) as.integer(args[[5]]) else 8L
TOP_LOADINGS_N <- 10L

if (!file.exists(FEATURES_CSV)) {
  stop("Missing features file: ", FEATURES_CSV)
}
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

meta_cols <- c(
  "file", "song_id", "artist", "title",
  "chart_year", "chart_position", "decade",
  "genre_broad_1", "genre_broad_2"
)

features <- read_csv(FEATURES_CSV, show_col_types = FALSE)
meta_present <- intersect(meta_cols, names(features))
features_meta <- features %>% select(all_of(meta_present))

features_numeric <- features %>%
  select(where(is.numeric)) %>%
  select(-any_of(meta_cols))

features_numeric <- features_numeric %>%
  mutate(across(everything(), ~ replace(.x, is.infinite(.x), NA_real_))) %>%
  mutate(across(everything(), ~ replace_na(.x, 0)))

x_raw <- features_numeric
if (nrow(x_raw) == 0) {
  stop("No rows remain in feature matrix.")
}

variances <- sapply(x_raw, var)
zero_var_cols <- names(variances[variances == 0 | is.na(variances)])
x_raw <- x_raw %>% select(-any_of(zero_var_cols))
x_scaled <- as.data.frame(scale(x_raw))

pa_mat <- as.matrix(x_scaled)
pa_result <- fa.parallel(pa_mat, fa = "fa", plot = FALSE, n.iter = N_ITER)

fa_eigenvalues <- function(m) {
  n_factors <- ncol(m)
  pa <- tryCatch(
    suppressWarnings(
      fa.parallel(m, fa = "fa", plot = FALSE, n.iter = 1, SMC = FALSE)
    ),
    error = function(e) NULL
  )
  if (is.null(pa)) {
    return(rep(NA_real_, n_factors))
  }
  ev <- pa$fa.values
  if (length(ev) < n_factors) {
    ev <- c(ev, rep(NA_real_, n_factors - length(ev)))
  }
  ev
}

n_ev <- length(pa_result$fa.values)
n_obs <- nrow(x_scaled)
p_vars <- ncol(x_scaled)
N_BOOT <- 100L
N_SIM_UNC <- max(N_ITER, 100L)

set.seed(42)
cat("Computing simulation envelope (", N_SIM_UNC, " iterations)...\n", sep = "")
sim_mat <- matrix(NA_real_, N_SIM_UNC, n_ev)
for (i in seq_len(N_SIM_UNC)) {
  rnd <- scale(matrix(rnorm(n_obs * p_vars), n_obs, p_vars))
  sim_mat[i, ] <- fa_eigenvalues(as.data.frame(rnd))
}

cat("Computing bootstrap CIs on observed eigenvalues (", N_BOOT, " resamples)...\n", sep = "")
boot_mat <- matrix(NA_real_, N_BOOT, n_ev)
for (b in seq_len(N_BOOT)) {
  idx <- sample.int(n_obs, n_obs, replace = TRUE)
  boot_mat[b, ] <- fa_eigenvalues(x_scaled[idx, , drop = FALSE])
}

quantile_row <- function(x, prob) {
  as.numeric(stats::quantile(x, probs = prob, na.rm = TRUE, names = FALSE))
}

parallel_df <- tibble(
  factor = seq_len(n_ev),
  observed = pa_result$fa.values,
  obs_lo = apply(boot_mat, 2, quantile_row, prob = 0.025),
  obs_hi = apply(boot_mat, 2, quantile_row, prob = 0.975),
  simulated = pa_result$fa.sim,
  sim_lo = apply(sim_mat, 2, quantile_row, prob = 0.025),
  sim_hi = apply(sim_mat, 2, quantile_row, prob = 0.975),
  retain = pa_result$fa.values > pa_result$fa.sim,
)
parallel_suggest <- sum(parallel_df$retain)
write_csv(parallel_df, file.path(OUTPUT_DIR, "efa_parallel_analysis.csv"))

UNCERTAINTY_ONLY <- length(args) >= 6 && identical(args[[6]], "uncertainty-only")

if (!is.na(N_FACTORS_ARG)) {
  N_FACTORS <- N_FACTORS_ARG
} else {
  N_FACTORS <- min(parallel_suggest, MAX_FACTORS)
  if (parallel_suggest > MAX_FACTORS) {
    message(
      "Parallel analysis suggested ", parallel_suggest,
      " factors; using ", N_FACTORS,
      " (cap=", MAX_FACTORS, "). Inspect efa_parallel_scree.pdf."
    )
  }
}
if (N_FACTORS < 1) {
  stop("Parallel analysis suggested zero factors; pass --factors explicitly.")
}

scree_df <- parallel_df %>%
  transmute(
    Factor = factor,
    Observed = observed,
    Obs_lo = obs_lo,
    Obs_hi = obs_hi,
    Simulated = simulated,
    Sim_lo = sim_lo,
    Sim_hi = sim_hi,
  )

p_scree <- ggplot(scree_df, aes(x = Factor)) +
  geom_ribbon(aes(ymin = Sim_lo, ymax = Sim_hi), fill = "#d62728", alpha = 0.15) +
  geom_ribbon(aes(ymin = Obs_lo, ymax = Obs_hi), fill = "#1f77b4", alpha = 0.15) +
  geom_line(aes(y = Observed, color = "Observed eigenvalues"), linewidth = 1) +
  geom_point(aes(y = Observed, color = "Observed eigenvalues"), size = 1.8) +
  geom_line(aes(y = Simulated, color = "Simulated mean"), linewidth = 1, linetype = "dashed") +
  labs(
    x = "Factor Number",
    y = "Eigenvalue",
    color = NULL,
    caption = "Shaded bands: 95% bootstrap CI (observed) and 95% simulation envelope (null)",
  ) +
  scale_color_manual(
    values = c(
      "Observed eigenvalues" = "#1f77b4",
      "Simulated mean" = "#d62728"
    )
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(OUTPUT_DIR, "efa_parallel_scree.pdf"),
  plot = p_scree,
  width = 8,
  height = 6,
)

if (UNCERTAINTY_ONLY) {
  cat("Wrote parallel analysis with uncertainty to", normalizePath(OUTPUT_DIR), "\n")
  quit(status = 0)
}

cat("Rows used:", nrow(x_scaled), "\n")
cat("Numeric features used:", ncol(x_scaled), "\n")
cat("Parallel analysis suggestion:", parallel_suggest, "\n")
cat("Factors extracted:", N_FACTORS, "\n\n")

fit <- fa(x_scaled, nfactors = N_FACTORS, rotate = "promax", fm = "pa")
loadings_mat <- as.matrix(fit$loadings[, ])
ss_loadings <- colSums(loadings_mat^2, na.rm = TRUE)
prop_var <- ss_loadings / nrow(loadings_mat)
cum_var <- cumsum(prop_var)

write_csv(
  tibble(
    factor = paste0("F", seq_len(N_FACTORS)),
    ss_loading = round(ss_loadings, 4),
    prop_var_pct = round(100 * prop_var, 2),
    cum_var_pct = round(100 * cum_var, 2),
  ),
  file.path(OUTPUT_DIR, "efa_variance.csv")
)

rownames(loadings_mat) <- colnames(x_scaled)
write_csv(
  as.data.frame(loadings_mat) %>%
    rownames_to_column("feature"),
  file.path(OUTPUT_DIR, "efa_loadings.csv"),
)

loading_top_list <- list()
for (j in seq_len(N_FACTORS)) {
  lj <- loadings_mat[, j, drop = TRUE]
  ord <- order(abs(lj), decreasing = TRUE)
  top_idx <- head(ord, TOP_LOADINGS_N)
  for (r in seq_along(top_idx)) {
    ii <- top_idx[r]
    loading_top_list[[length(loading_top_list) + 1]] <- tibble(
      factor = paste0("F", j),
      factor_index = j,
      rank = r,
      feature = rownames(loadings_mat)[ii],
      loading = unname(lj[ii]),
      abs_loading = abs(unname(lj[ii])),
    )
  }
}
top_loadings_df <- bind_rows(loading_top_list)
write_csv(top_loadings_df, file.path(OUTPUT_DIR, "efa_top_loadings.csv"))

score_result <- factor.scores(x_scaled, fit, method = "regression")
factor_scores <- as.data.frame(score_result$scores)
colnames(factor_scores) <- paste0("F", seq_len(ncol(factor_scores)))

scores_df <- bind_cols(features_meta, factor_scores)
write_csv(scores_df, file.path(OUTPUT_DIR, "efa_factor_scores.csv"))

cat(
  "Cumulative variance explained by", N_FACTORS, "factors:",
  sprintf("%.2f%%", 100 * cum_var[length(cum_var)]), "\n"
)
cat("Wrote EFA outputs to", normalizePath(OUTPUT_DIR), "\n")
