# Wild cluster bootstrap inference for Table B (temperature x shock effects).
#
# Run from the repo root:
#
#     Rscript code/analysis/updates/one-off/wild_cluster_bootstrap_table_b.R
#
# Needs fixest, nanoparquet and fwildclusterboot. fwildclusterboot is no longer
# on CRAN; install it from r-universe:
#
#     install.packages("fwildclusterboot",
#       repos = c("https://s3alfisc.r-universe.dev", "https://cloud.r-project.org"))
#
# Writes output/tables/wild_cluster_bootstrap_table_b.tex: for each Table B
# column (7-day mean temperature), the differential-heat coefficient with its
# kabupaten-clustered SE and p-value, and the wild cluster bootstrap p-value
# and 95% CI (restricted WCR, Rademacher weights, 9,999 draws, kabupaten
# clusters).
#
# fwildclusterboot projects out one fixed effect and keeps the rest as dummy
# columns. We project out kecamatan (nested in kabupaten clusters) everywhere
# except the individual-FE column, where we project out pidlink (20k levels)
# and keep kecamatan as dummies. pidlink is not nested in kabupaten for people
# who changed kabupaten between waves.
#
# Formulas mirror code/library/specs.py and
# code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; keep them
# in sync. Singleton fixed effects are dropped first, as pyfixest does.

suppressPackageStartupMessages({
  library(fixest)
  library(fwildclusterboot)
  library(nanoparquet)
})

ANALYSIS_INPUT <- "data/generated/30_analysis_table_input.parquet"
TABLE_FILE <- "output/tables/wild_cluster_bootstrap_table_b.tex"
CLUSTER <- "kabupaten_full_code"
HEAT <- "tmean_7d"
BOOTSTRAP_DRAWS <- 9999
SEED <- 20261120
CONTROLS <- paste(
  "age + female + edu_yrs_nullable + married + widowed + divorced",
  "+ ethnicity_nullable + religion_nullable"
)
FE <- "month_year + gadm_fullcode"
FE_PANEL <- "month_year + gadm_fullcode + pidlink + ifls5"
# Bootstrap version of the panel model: project out pidlink and enter kecamatan
# and month-year as factor regressors. With person FE, most kecamatan dummies
# are collinear (only cross-wave movers vary), and fwildclusterboot cannot drop
# collinear FE dummies itself; fixest drops them when they are regressors.
# ifls5 is absorbed by month-year (each month-year falls in one wave).
FE_PANEL_BOOT <- "factor(gadm_fullcode) + factor(month_year) | pidlink"

load_analysis_data <- function() {
  df <- as.data.frame(read_parquet(ANALYSIS_INPUT))
  df$ifls5 <- as.integer(df$wave == "IFLS5")
  # Single month-by-year FE column (pyfixest's month*year); fwildclusterboot
  # cannot parse fixest's month^year syntax.
  df$month_year <- paste(df$year, df$month, sep = "-")
  # fwildclusterboot evaluates the FE part of the formula as R arithmetic when
  # building its model frame, which errors on character columns. Integer codes
  # avoid that; fixest treats FE variables as categories either way.
  for (fe in c("month_year", "gadm_fullcode", "pidlink")) {
    df[[fe]] <- as.integer(factor(df[[fe]]))
  }
  df
}

# Identification (as in Table B): kecamatan and month-by-year fixed effects
# (plus individual FE in column 2), so heat effects come from 7-day weather
# deviations around local and calendar norms. The differential term assumes
# those deviations are as good as random with respect to shock status within
# kecamatan and month-year.
triple_diff <- function(group, post) {
  sprintf("cesd_z ~ %s * %s * %s + %s | %s", group, post, HEAT, CONTROLS, FE)
}

triple_diff_panel <- function(group, post, fe = FE_PANEL) {
  sprintf(
    "cesd_z ~ %3$s + %1$s:%2$s:%3$s + %1$s:%3$s + %2$s:%3$s + %1$s:%2$s + %4$s %5$s %6$s",
    group, post, HEAT, CONTROLS, if (fe == FE_PANEL) "|" else "+", fe
  )
}

table_b_specs <- function(df) {
  wave5 <- df[df$ifls5 == 1, ]
  list(
    list(label = "Palm Shock", data = df, projected_fe = "gadm_fullcode",
         formula = triple_diff("palm_farmer_hh_ifls4", "ifls5"),
         param = "palm_farmer_hh_ifls4:ifls5:tmean_7d"),
    list(label = "\\shortstack{Palm Shock\\\\Panel}", data = df, projected_fe = "pidlink",
         formula = triple_diff_panel("palm_farmer_hh_ifls4", "ifls5"),
         boot_formula = triple_diff_panel("palm_farmer_hh_ifls4", "ifls5", FE_PANEL_BOOT),
         param = "palm_farmer_hh_ifls4:ifls5:tmean_7d"),
    list(label = "\\shortstack{Palm Shock\\\\Price Drop}", data = df, projected_fe = "gadm_fullcode",
         formula = triple_diff("palm_price_gap_z", "ifls5"),
         param = "palm_price_gap_z:ifls5:tmean_7d"),
    list(label = "Urban Vehicle Owners", data = wave5, projected_fe = "gadm_fullcode",
         formula = triple_diff("urban_vehicle_hh_ifls4", "post_subsidy"),
         param = "urban_vehicle_hh_ifls4:post_subsidy:tmean_7d"),
    list(label = "\\shortstack{Urban Vehicle Owners\\\\No Cash Transfer}", data = wave5,
         projected_fe = "gadm_fullcode",
         formula = triple_diff("urban_vehicle_transfer_nonrecipient_ifls4", "post_subsidy"),
         param = "urban_vehicle_transfer_nonrecipient_ifls4:post_subsidy:tmean_7d"),
    list(label = "\\shortstack{Urban Vehicle Owners\\\\Cash Transfer}", data = wave5,
         projected_fe = "gadm_fullcode",
         formula = triple_diff("urban_vehicle_transfer_recipient_ifls4", "post_subsidy"),
         param = "urban_vehicle_transfer_recipient_ifls4:post_subsidy:tmean_7d"),
    list(label = "Job Loss", data = df, projected_fe = "gadm_fullcode",
         formula = sprintf("cesd_z ~ job_loss_180d * %s + %s | %s", HEAT, CONTROLS, FE),
         param = "job_loss_180d:tmean_7d")
  )
}

# fixest may reorder the parts of an interaction name (a:b:c -> c:a:b).
coefficient_name <- function(fit, param) {
  parts <- function(name) sort(strsplit(name, ":", fixed = TRUE)[[1]])
  matches <- Filter(function(name) identical(parts(name), parts(param)), names(coef(fit)))
  stopifnot(length(matches) == 1)
  matches
}

# Fit the Table B model, then refit on exactly its estimation sample (after NA
# and singleton drops) so fwildclusterboot sees the same rows. The refit uses
# the bootstrap formula where one is given (panel column) and must reproduce
# the Table B coefficient. Reported coefficients, SEs and p-values come from
# the Table B model.
fit_models <- function(spec) {
  table_b <- feols(as.formula(spec$formula), data = spec$data, fixef.rm = "singleton",
                   cluster = CLUSTER, notes = FALSE, warn = FALSE)
  estimation_sample <<- spec$data[obs(table_b), ]
  boot_formula <- if (is.null(spec$boot_formula)) spec$formula else spec$boot_formula
  boot <- feols(as.formula(boot_formula), data = estimation_sample, cluster = CLUSTER,
                notes = FALSE, warn = FALSE)
  table_b_term <- coefficient_name(table_b, spec$param)
  boot_term <- coefficient_name(boot, spec$param)
  stopifnot(isTRUE(all.equal(coef(boot)[[boot_term]], coef(table_b)[[table_b_term]],
                             tolerance = 1e-6)))
  list(table_b = table_b, table_b_term = table_b_term, boot = boot, boot_term = boot_term)
}

bootstrap_column <- function(spec) {
  models <- fit_models(spec)
  set.seed(SEED)
  dqrng::dqset.seed(SEED)
  boot <- boottest(
    models$boot,
    param = models$boot_term,
    clustid = CLUSTER,
    fe = spec$projected_fe,
    B = BOOTSTRAP_DRAWS,
    type = "rademacher",
    impose_null = TRUE
  )
  ci <- confint(boot)
  term <- models$table_b_term
  data.frame(
    label = spec$label,
    b = unname(coef(models$table_b)[term]),
    se = unname(se(models$table_b)[term]),
    p = unname(pvalue(models$table_b)[term]),
    wcb_p = pval(boot),
    wcb_ci_low = ci[1],
    wcb_ci_high = ci[2],
    n = nobs(models$table_b),
    clusters = length(unique(estimation_sample[[CLUSTER]]))
  )
}

stars <- function(p) {
  ifelse(p < 0.01, "***", ifelse(p < 0.05, "**", ifelse(p < 0.10, "*", "")))
}

latex_row <- function(label, cells) {
  paste0(label, " & ", paste(cells, collapse = " & "), " \\\\")
}

build_table <- function(results) {
  n_cols <- nrow(results)
  header <- c(
    sprintf("\\begin{tabular}{@{}l%s}", strrep("c", n_cols)),
    "\\toprule",
    sprintf(" & \\multicolumn{%d}{c}{CES-D z-score} \\\\", n_cols),
    sprintf("\\cmidrule(lr){2-%d}", n_cols + 1),
    latex_row("", results$label),
    latex_row("", sprintf("(%d)", seq_len(n_cols))),
    "\\midrule\\addlinespace[2.5pt]"
  )
  body <- c(
    latex_row("Differential Impact of Heat", sprintf("%.3f%s", results$b, stars(results$p))),
    latex_row("", sprintf("(%.3f)", results$se)),
    latex_row("\\quad Clustered $p$", sprintf("[%.3f]", results$p)),
    latex_row("\\quad Wild bootstrap $p$", sprintf("[%.3f]", results$wcb_p)),
    latex_row("\\quad Wild bootstrap 95\\% CI",
              sprintf("[%.3f, %.3f]", results$wcb_ci_low, results$wcb_ci_high)),
    "\\midrule\\addlinespace[2.5pt]",
    latex_row("Observations", format(results$n, big.mark = ",")),
    latex_row("Kabupaten clusters", results$clusters),
    latex_row("Kecamatan", rep("x", n_cols)),
    latex_row("Month-Year", rep("x", n_cols)),
    latex_row("Individual", c("-", "x", rep("-", n_cols - 2))),
    "\\bottomrule",
    "\\end{tabular}"
  )
  paste(c(header, body), collapse = "\n")
}

main <- function() {
  specs <- table_b_specs(load_analysis_data())
  results <- do.call(rbind, lapply(specs, function(spec) {
    started <- Sys.time()
    row <- bootstrap_column(spec)
    message(sprintf("%-60s %.1fs", spec$param, as.numeric(Sys.time() - started, units = "secs")))
    row
  }))
  dir.create(dirname(TABLE_FILE), recursive = TRUE, showWarnings = FALSE)
  writeLines(build_table(results), TABLE_FILE)
  print(results[, setdiff(names(results), "label")], digits = 4, row.names = FALSE)
}

main()
