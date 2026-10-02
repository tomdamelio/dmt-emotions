# Heteroscedastic LME robustness analysis
# Compares standard LME vs heteroscedastic LME (varIdent by State)
# for SMNA, HR, and RVT
#
# Run from project root:
#   Rscript scripts/heteroscedastic_lme.R

library(nlme)
`%||%` <- function(a, b) if (is.null(a)) b else a

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE <- normalizePath(getwd(), mustWork = FALSE)   # run from the repository root
SMNA_CSV <- file.path(BASE, "results/eda/smna/smna_auc_long_data_z.csv")
HR_CSV   <- file.path(BASE, "results/ecg/hr/hr_minute_long_data_z.csv")
RVT_CSV  <- file.path(BASE, "results/resp/rvt/resp_rvt_minute_long_data_z.csv")
OUT_DIR  <- file.path(BASE, "results/robustness")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ─── Helper: fit both models and extract fixed effects ────────────────────────
fit_models <- function(df, outcome_col, label) {

  if ("Scale" %in% names(df)) df <- df[df$Scale == "z", ]
  df$State <- factor(df$State, levels = c("RS", "DMT"))
  df$Dose  <- factor(df$Dose,  levels = c("Low", "High"))
  df[[outcome_col]] <- as.numeric(df[[outcome_col]])
  df$State_n <- as.numeric(df$State == "DMT")
  df$Dose_n  <- as.numeric(df$Dose  == "High")
  df$StateXDose_n <- df$State_n * df$Dose_n

  formula_fixed <- as.formula(
    paste0(outcome_col,
           " ~ State * Dose + window_c + State:window_c + Dose:window_c")
  )

  cat("\n", strrep("=", 60), "\n")
  cat("MODALITY:", label, "\n")
  cat("N rows:", nrow(df), "| Subjects:", length(unique(df$subject)), "\n")
  cat("States:", paste(levels(df$State), collapse = ", "), "\n")
  cat(strrep("=", 60), "\n")

  # ── Standard LME ──
  cat("\n--- Standard LME (homoscedastic) ---\n")
  m_std <- tryCatch(
    lme(formula_fixed,
        random = ~ 0 + State_n + Dose_n + StateXDose_n | subject,
        data   = df,
        method = "REML",
        control = lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200)),
    error = function(e) { cat("Standard LME failed:", conditionMessage(e), "\n"); NULL }
  )
  if (!is.null(m_std)) print(summary(m_std)$tTable)

  # ── Heteroscedastic LME ──
  cat("\n--- Heteroscedastic LME (varIdent by State) ---\n")
  m_het <- tryCatch(
    lme(formula_fixed,
        random  = ~ 0 + State_n + Dose_n + StateXDose_n | subject,
        weights = varIdent(form = ~ 1 | State),
        data    = df,
        method  = "REML",
        control = lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200)),
    error = function(e) { cat("Heteroscedastic LME failed:", conditionMessage(e), "\n"); NULL }
  )
  if (!is.null(m_het)) {
    print(summary(m_het)$tTable)
    cat("\nVariance structure (sigma ratio DMT/RS):\n")
    print(coef(m_het$modelStruct$varStruct, unconstrained = FALSE, allCoef = TRUE))
  }

  # ── Likelihood Ratio Test (refit with ML for LRT) ──
  if (!is.null(m_std) && !is.null(m_het)) {
    cat("\n--- Likelihood Ratio Test (ML, not REML) ---\n")
    m_std_ml <- update(m_std, method = "ML")
    m_het_ml <- update(m_het, method = "ML")
    print(anova(m_std_ml, m_het_ml))
  }

  # ── Save fixed-effects comparison to CSV ──
  results_list <- list()
  for (mod_name in c("standard", "heteroscedastic")) {
    m <- if (mod_name == "standard") m_std else m_het
    if (is.null(m)) next
    tt <- as.data.frame(summary(m)$tTable)
    tt$term     <- rownames(tt)
    tt$model    <- mod_name
    tt$modality <- label
    rownames(tt) <- NULL
    results_list[[mod_name]] <- tt
  }
  if (length(results_list) > 0) {
    out <- do.call(rbind, results_list)
    csv_path <- file.path(OUT_DIR, paste0("lme_", tolower(label), "_comparison.csv"))
    write.csv(out, csv_path, row.names = FALSE)
    cat("\nSaved:", csv_path, "\n")
  }

  invisible(list(standard = m_std, heteroscedastic = m_het))
}

# ─── Run for each modality ────────────────────────────────────────────────────
smna_df <- read.csv(SMNA_CSV, stringsAsFactors = FALSE)
hr_df   <- read.csv(HR_CSV,   stringsAsFactors = FALSE)
rvt_df  <- read.csv(RVT_CSV,  stringsAsFactors = FALSE)

fit_models(smna_df, "AUC",     "SMNA")
fit_models(hr_df,   "HR",      "HR")
fit_models(rvt_df,  "RSP_RVT", "RVT")

cat("\n\nDone. Results saved to:", OUT_DIR, "\n")
