################################################################################
# AI/ML in Biomarker Discovery — Week 5 Lab
# Title:   Cross-Platform Harmonization, ML Classification & External Validation
# Disease: Alzheimer's Disease | Biomarker: miRNA
# Audience: Wet-lab biologists — Weeks 1–4 pipelines assumed complete
#
# Learning Goals for This Script:
#   1. Load and inspect both training (GSE120584) and validation (GSE46579) datasets
#   2. Harmonize miRNA names across miRBase versions using miRBaseConverter
#   3. Find and report the feature intersection between two platforms
#   4. Apply per-dataset z-score standardization before cross-dataset testing
#   5. Build ML classifiers entirely in R: Random Forest and LASSO logistic regression
#   6. Compute SHAP feature importance using the fastshap package
#   7. Compare AUC values between training CV and external validation using DeLong's test
#   8. Produce calibration plots to assess probability accuracy
#   9. Compile a complete model results summary table
#  10. Save all output objects and report session information
#
# Datasets:
#   GSE120584 — Serum small RNA-seq, 3 groups: AD / MCI / Control   [TRAINING]
#   GSE46579  — Whole blood Affymetrix microarray, AD / Control      [VALIDATION]
#
# Run each section with Ctrl+Enter (Windows/Linux) or Cmd+Enter (Mac).
# Sections are designed to run in order; each builds on the previous.
################################################################################


# ==============================================================================
# SECTION 1: Load Packages
# ==============================================================================
# If any library() call fails, install the missing package and re-run.
#
# Bioconductor packages:
#   BiocManager::install("miRBaseConverter")
#
# CRAN packages:
#   install.packages(c("caret", "randomForest", "glmnet", "fastshap", "pROC",
#                      "ggplot2", "dplyr", "readr", "tidyr"))
#
# Note: caret will prompt to install additional "suggested" packages — allow it.
# Note: fastshap requires R >= 4.0. Install with: install.packages("fastshap")

suppressPackageStartupMessages({
  # Bioconductor
  library(miRBaseConverter)   # miRNA name versioning across miRBase releases

  # CRAN — ML
  library(caret)              # unified CV framework
  library(randomForest)       # Random Forest classifier
  library(glmnet)             # LASSO / ridge / elastic-net

  # CRAN — model interpretation
  library(fastshap)           # SHAP values for any black-box model

  # CRAN — statistics and evaluation
  library(pROC)               # ROC curves, AUC, DeLong's test

  # CRAN — data manipulation and plotting
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
})

# Install miRBaseConverter if not already present
if (!requireNamespace("miRBaseConverter", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
  BiocManager::install("miRBaseConverter")
  library(miRBaseConverter)
}

cat("All packages loaded.\n")

# Consistent colour palette for disease groups (matches Week 2–4 scripts)
GROUP_COLOURS <- c(
  "Control"                   = "#4575B4",   # blue
  "Mild Cognitive Impairment" = "#FEE090",   # amber
  "Alzheimer's Disease"       = "#D73027"    # red
)

# Ensure output directories exist
for (d in c("data/processed", "results/Week5", "qc_reports")) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}
cat("Output directories confirmed.\n\n")


# ==============================================================================
# SECTION 2: Load Preprocessed Datasets
# ==============================================================================
# These .rds files were created at the end of the Week 2 pipeline.
# GSE120584: RNA-seq (VST-transformed), 3 groups, serum
# GSE46579:  Microarray (RMA-normalized), 2 groups, whole blood
#
# Expected object dimensions:
#   expr_gse120584: ~500 miRNAs × 148 samples (after QC filtering)
#   expr_gse46579:  ~1700 probe sets × 65 samples (before miRNA-level filtering)

cat("=== SECTION 2: Loading Preprocessed Datasets ===\n")

expr_gse120584 <- tryCatch(
  readRDS("data/processed/GSE120584_expr_clean.rds"),
  error = function(e) {
    stop("Could not load GSE120584_expr_clean.rds. ",
         "Have you completed the Week 2 pipeline? Error: ", conditionMessage(e))
  }
)

metadata_120584 <- tryCatch(
  readRDS("data/processed/GSE120584_metadata_clean.rds"),
  error = function(e) {
    stop("Could not load GSE120584_metadata_clean.rds. ",
         "Error: ", conditionMessage(e))
  }
)

expr_gse46579 <- tryCatch(
  readRDS("data/processed/GSE46579_expr_rma.rds"),
  error = function(e) {
    stop("Could not load GSE46579_expr_rma.rds. ",
         "Have you completed the Week 2 microarray pipeline? ",
         "Error: ", conditionMessage(e))
  }
)

metadata_46579 <- tryCatch(
  readRDS("data/processed/GSE46579_metadata_clean.rds"),
  error = function(e) {
    stop("Could not load GSE46579_metadata_clean.rds. ",
         "Error: ", conditionMessage(e))
  }
)

# --------------------------------------------------------------------------
# Dataset summaries
# --------------------------------------------------------------------------
cat("\n--- GSE120584 (training, serum RNA-seq) ---\n")
cat("Expression matrix:", nrow(expr_gse120584), "features ×",
    ncol(expr_gse120584), "samples\n")
cat("Value range:", round(range(expr_gse120584), 2), "\n")
cat("Group distribution:\n")
print(table(metadata_120584$group))

cat("\n--- GSE46579 (validation, whole blood microarray) ---\n")
cat("Expression matrix:", nrow(expr_gse46579), "features ×",
    ncol(expr_gse46579), "samples\n")
cat("Value range:", round(range(expr_gse46579), 2), "\n")
cat("Group distribution:\n")
print(table(metadata_46579$group))

# --------------------------------------------------------------------------
# BIOLOGICAL CHECK:
# Before harmonization, verify that the two datasets look sensible:
#   - GSE120584 rows should be named like "hsa-miR-21-5p" or similar
#   - GSE46579 rows (from microarray probes) may have different naming formats,
#     possibly including precursor names or older strand notation ("*" suffix)
# --------------------------------------------------------------------------
cat("\nFirst 10 feature names in GSE120584:\n")
print(head(rownames(expr_gse120584), 10))

cat("\nFirst 10 feature names in GSE46579:\n")
print(head(rownames(expr_gse46579), 10))

cat("\nDo both sets contain 'hsa-miR' names?\n")
cat("  GSE120584: hsa-miR prefix count =",
    sum(grepl("^hsa-miR", rownames(expr_gse120584))), "\n")
cat("  GSE46579:  hsa-miR prefix count =",
    sum(grepl("^hsa-miR", rownames(expr_gse46579))), "\n")
# If GSE46579 has very few 'hsa-miR' names, the probe annotation may use a
# different format. Examine fData(gse46579) from the Week 2 GEO download to
# find the correct column mapping probe IDs to miRNA names.


# ==============================================================================
# SECTION 3: miRNA Name Harmonization Using miRBaseConverter
# ==============================================================================
# RATIONALE:
# miRBase has released 22 major versions since 2002. Naming conventions changed
# substantially between versions:
#   - Version 14: introduced -3p / -5p strand suffixes (replacing "*" notation)
#   - Version 18: removed some duplicated entries
#   - Version 21 → 22: ~400 name changes and ~80 removals
#
# Our two datasets use different miRBase versions in their annotations.
# Comparing miRNA names directly would miss many shared features simply because
# they are named differently. miRBaseConverter uses stable MIMAT accession numbers
# as version-independent identifiers.
#
# Reference: Xu T et al. (2018) miRBaseConverter: An R/Bioconductor Package for
# Converting and Retrieving miRNA Name, Accession, Sequence in Different Versions
# of miRBase. BMC Bioinformatics 19(Suppl 19):514.
# DOI: 10.1186/s12859-018-2531-5

cat("\n=== SECTION 3: miRNA Name Harmonization ===\n")
cat("Target version: miRBase v22 (current stable release)\n\n")

# --------------------------------------------------------------------------
# 3A. Detect current miRBase version of each dataset's names
# --------------------------------------------------------------------------
cat("--- Checking miRBase version for GSE120584 ---\n")
tryCatch({
  version_120584 <- checkMiRNAVersion(
    rownames(expr_gse120584),
    verbose = TRUE
  )
  cat("Detected version for GSE120584:", version_120584, "\n")
}, error = function(e) {
  cat("checkMiRNAVersion returned an error:", conditionMessage(e), "\n")
  cat("Proceeding with conversion attempt regardless.\n")
})

cat("\n--- Checking miRBase version for GSE46579 ---\n")
tryCatch({
  version_46579 <- checkMiRNAVersion(
    rownames(expr_gse46579),
    verbose = TRUE
  )
  cat("Detected version for GSE46579:", version_46579, "\n")
}, error = function(e) {
  cat("checkMiRNAVersion returned an error:", conditionMessage(e), "\n")
})

# --------------------------------------------------------------------------
# 3B. Convert GSE120584 names to miRBase v22
# --------------------------------------------------------------------------
# miRNA_NameToAccession() returns a data frame with:
#   OriginalName  — the input miRNA name
#   Accession     — stable MIMAT accession (version-independent)
#   VersionName   — the name in the requested miRBase version

cat("\n--- Converting GSE120584 names to miRBase v22 ---\n")

names_120584 <- rownames(expr_gse120584)
n_features_120584_before <- length(names_120584)

conversion_120584 <- miRNA_NameToAccession(
  names_120584,
  version = "v22"
)

# How many names were successfully converted?
converted_120584 <- conversion_120584[!is.na(conversion_120584$Accession), ]
failed_120584    <- conversion_120584[is.na(conversion_120584$Accession), ]

cat("GSE120584 name conversion results:\n")
cat("  Total input features:          ", n_features_120584_before, "\n")
cat("  Successfully converted to v22: ", nrow(converted_120584), "\n")
cat("  Could not map to v22:          ", nrow(failed_120584), "\n")
cat("  Conversion success rate:       ",
    round(nrow(converted_120584) / n_features_120584_before * 100, 1), "%\n")

if (nrow(failed_120584) > 0) {
  cat("\nFirst 10 names that could not be converted (check for non-miRNA entries):\n")
  print(head(failed_120584$OriginalName, 10))
}

# Names that changed between detected version and v22
changed_mask_120584 <- converted_120584$OriginalName != converted_120584$VersionName
n_changed_120584    <- sum(changed_mask_120584, na.rm = TRUE)
cat("\nNames that changed during conversion to v22:", n_changed_120584, "\n")
if (n_changed_120584 > 0) {
  cat("Examples of changed names (old → new):\n")
  changed_examples <- converted_120584[changed_mask_120584, ][1:min(5, n_changed_120584), ]
  print(data.frame(
    Old = changed_examples$OriginalName,
    New = changed_examples$VersionName
  ))
}

# --------------------------------------------------------------------------
# 3C. Convert GSE46579 names to miRBase v22
# --------------------------------------------------------------------------
cat("\n--- Converting GSE46579 names to miRBase v22 ---\n")

names_46579           <- rownames(expr_gse46579)
n_features_46579_before <- length(names_46579)

conversion_46579 <- miRNA_NameToAccession(
  names_46579,
  version = "v22"
)

converted_46579 <- conversion_46579[!is.na(conversion_46579$Accession), ]
failed_46579    <- conversion_46579[is.na(conversion_46579$Accession), ]

cat("GSE46579 name conversion results:\n")
cat("  Total input features:          ", n_features_46579_before, "\n")
cat("  Successfully converted to v22: ", nrow(converted_46579), "\n")
cat("  Could not map to v22:          ", nrow(failed_46579), "\n")
cat("  Conversion success rate:       ",
    round(nrow(converted_46579) / n_features_46579_before * 100, 1), "%\n")

changed_mask_46579 <- converted_46579$OriginalName != converted_46579$VersionName
n_changed_46579    <- sum(changed_mask_46579, na.rm = TRUE)
cat("Names that changed during conversion to v22:", n_changed_46579, "\n")

# Microarray annotations often contain many non-human or control probes.
# Inspect the failed set to determine whether failures are true miRNAs or
# probe-level artifacts (e.g., spike-in controls, blank probes).
if (nrow(failed_46579) > 0) {
  cat("\nFirst 10 names not mappable to v22:\n")
  print(head(failed_46579$OriginalName, 10))
  cat("(These may be non-human miRNAs, spike-in probes, or retired entries.)\n")
}

# --------------------------------------------------------------------------
# 3D. Handle duplicate MIMAT accessions (multiple probes mapping to same miRNA)
# --------------------------------------------------------------------------
# Some microarray platforms contain multiple probes for the same mature miRNA.
# When this occurs, summarize to the miRNA level by taking the mean across probes.

cat("\n--- Handling duplicate MIMAT accessions (GSE46579 microarray) ---\n")

# Attach accessions to expression matrix
expr_46579_annotated <- expr_gse46579[
  rownames(expr_gse46579) %in% converted_46579$OriginalName, ]

# Map accessions
acc_order <- converted_46579$Accession[
  match(rownames(expr_46579_annotated), converted_46579$OriginalName)]

# Check for duplicates
n_duplicates_46579 <- sum(duplicated(acc_order[!is.na(acc_order)]))
cat("Duplicate MIMAT accessions (multiple probes → same miRNA):",
    n_duplicates_46579, "\n")

if (n_duplicates_46579 > 0) {
  cat("Summarizing by taking mean across duplicate probes.\n")
  # Attach accession as a column for aggregation
  expr_46579_df        <- as.data.frame(t(expr_46579_annotated))
  expr_46579_df$MIMAT  <- acc_order
  expr_46579_df        <- expr_46579_df[!is.na(expr_46579_df$MIMAT), ]

  # Mean across probes with the same MIMAT accession (within each sample)
  expr_46579_agg <- expr_46579_df %>%
    group_by(MIMAT) %>%
    summarise(across(everything(), mean, na.rm = TRUE), .groups = "drop") %>%
    as.data.frame()

  rownames(expr_46579_agg) <- expr_46579_agg$MIMAT
  expr_46579_agg$MIMAT     <- NULL
  expr_46579_agg           <- t(as.matrix(expr_46579_agg))
  cat("GSE46579 after duplicate aggregation:", nrow(expr_46579_agg), "unique miRNAs\n")

} else {
  # No duplicates: just keep the mapped features
  expr_46579_agg  <- expr_46579_annotated
  rownames(expr_46579_agg) <- acc_order
  cat("No duplicate probes found; using features directly.\n")
}

# Similarly handle GSE120584
expr_120584_annotated <- expr_gse120584[
  rownames(expr_gse120584) %in% converted_120584$OriginalName, ]
acc_120584 <- converted_120584$Accession[
  match(rownames(expr_120584_annotated), converted_120584$OriginalName)]

expr_120584_df       <- as.data.frame(t(expr_120584_annotated))
expr_120584_df$MIMAT <- acc_120584
expr_120584_df       <- expr_120584_df[!is.na(expr_120584_df$MIMAT), ]

n_dup_120584 <- sum(duplicated(expr_120584_df$MIMAT))
cat("\nDuplicate MIMAT accessions in GSE120584:", n_dup_120584, "\n")

if (n_dup_120584 > 0) {
  expr_120584_agg <- expr_120584_df %>%
    group_by(MIMAT) %>%
    summarise(across(everything(), mean, na.rm = TRUE), .groups = "drop") %>%
    as.data.frame()
  rownames(expr_120584_agg) <- expr_120584_agg$MIMAT
  expr_120584_agg$MIMAT     <- NULL
  expr_120584_agg           <- t(as.matrix(expr_120584_agg))
} else {
  expr_120584_agg           <- expr_120584_annotated
  rownames(expr_120584_agg) <- acc_120584
}

cat("GSE120584 after deduplication:", nrow(expr_120584_agg), "unique miRNAs\n")


# ==============================================================================
# SECTION 4: Find Feature Intersection Between Platforms
# ==============================================================================
# RATIONALE:
# After converting both sets of names to v22-equivalent MIMAT accessions,
# we find which miRNAs are present in BOTH datasets. Only these "intersection"
# features can be used when training on one dataset and evaluating on the other.
#
# We also report what fraction of features was retained and specifically check
# for key AD-associated miRNAs documented in the published literature.

cat("\n=== SECTION 4: Feature Intersection ===\n")

# Feature sets (MIMAT accession IDs)
features_120584 <- rownames(expr_120584_agg)
features_46579  <- rownames(expr_46579_agg)

cat("Features in GSE120584 (after dedup): ", length(features_120584), "\n")
cat("Features in GSE46579  (after dedup): ", length(features_46579),  "\n")

# Intersection
common_features <- intersect(features_120584, features_46579)
n_common        <- length(common_features)

cat("\nFeatures in common (intersection):", n_common, "\n")
cat("Percent of GSE120584 features retained: ",
    round(n_common / length(features_120584) * 100, 1), "%\n")
cat("Percent of GSE46579 features retained:  ",
    round(n_common / length(features_46579) * 100, 1), "%\n")

if (n_common < 50) {
  warning("Fewer than 50 features in intersection. Check that name conversion ",
          "succeeded for both datasets. Examine the failed conversion sets above.")
}

# --------------------------------------------------------------------------
# 4A. Biological check: are key AD miRNAs present in the intersection?
# --------------------------------------------------------------------------
# These are well-documented AD-associated miRNAs from the published literature.
# If they are absent from the intersection, note this as a limitation.
#
# Literature references:
#   miR-21-5p:  elevated in AD serum (Dangla-Valls et al., multiple studies)
#   miR-29b-3p: downregulated in AD; targets BACE1 (Hebert et al. 2008 PNAS)
#   miR-155-5p: neuroinflammatory miRNA; elevated in AD microglia
#   miR-9-5p:   regulates APP processing; reduced in AD
#   miR-107:    targets BACE1; one of the earliest reported AD miRNA biomarkers
#   miR-181c-5p: reduced in plasma of AD patients in multiple cohorts

# Convert canonical AD miRNA names to MIMAT accessions for lookup
ad_mirnas_names <- c(
  "hsa-miR-21-5p", "hsa-miR-29b-3p", "hsa-miR-155-5p",
  "hsa-miR-9-5p", "hsa-miR-107", "hsa-miR-181c-5p",
  "hsa-miR-146a-5p", "hsa-miR-34a-5p", "hsa-miR-132-3p"
)

ad_mirna_acc <- tryCatch(
  miRNA_NameToAccession(ad_mirnas_names, version = "v22"),
  error = function(e) {
    cat("Could not look up AD miRNA accessions:", conditionMessage(e), "\n")
    NULL
  }
)

cat("\n--- Biological check: Key AD miRNAs in the intersection ---\n")
if (!is.null(ad_mirna_acc)) {
  for (i in seq_len(nrow(ad_mirna_acc))) {
    name <- ad_mirna_acc$OriginalName[i]
    acc  <- ad_mirna_acc$Accession[i]
    if (is.na(acc)) {
      cat("  ", name, "— accession NOT found in miRBase v22\n")
    } else if (acc %in% common_features) {
      cat("  ", name, "(", acc, ") — PRESENT in intersection\n")
    } else if (acc %in% features_120584) {
      cat("  ", name, "(", acc, ") — present in GSE120584 only (ABSENT from GSE46579)\n")
    } else if (acc %in% features_46579) {
      cat("  ", name, "(", acc, ") — present in GSE46579 only (ABSENT from GSE120584)\n")
    } else {
      cat("  ", name, "(", acc, ") — ABSENT from both datasets\n")
    }
  }
}
# INTERPRETATION:
# If miR-29b-3p is absent from GSE46579, this is a platform detection issue
# (the Affymetrix miRNA array may not have a probe for this miRNA, or it was
# not expressed above threshold in whole blood). Note such absences in the
# limitations section of any paper using these data for external validation.

# --------------------------------------------------------------------------
# 4B. Subset both matrices to intersection
# --------------------------------------------------------------------------
expr_120584_intersect <- expr_120584_agg[common_features, ]
expr_46579_intersect  <- expr_46579_agg[common_features,  ]

# Verify alignment
stopifnot(all(rownames(expr_120584_intersect) == rownames(expr_46579_intersect)))
cat("\nBoth expression matrices subsetted to", n_common, "common features.\n")
cat("Row name alignment confirmed.\n")

# --------------------------------------------------------------------------
# 4C. Convert MIMAT accessions back to v22 canonical names for readability
# --------------------------------------------------------------------------
# Working with MIMAT accessions is correct internally but hard to read.
# Convert final row names to human-readable v22 names for all outputs.

v22_name_lookup <- tryCatch(
  miRNA_AccessionToName(common_features, targetVersion = "v22"),
  error = function(e) {
    cat("AccessionToName conversion failed:", conditionMessage(e), "\n")
    cat("Keeping MIMAT accessions as row names.\n")
    NULL
  }
)

if (!is.null(v22_name_lookup) &&
    "TargetName" %in% colnames(v22_name_lookup)) {
  v22_names <- v22_name_lookup$TargetName
  # Replace NA with original MIMAT accession
  v22_names[is.na(v22_names)] <- common_features[is.na(v22_names)]
  rownames(expr_120584_intersect) <- v22_names
  rownames(expr_46579_intersect)  <- v22_names
  cat("Row names converted to miRBase v22 canonical names.\n")
  cat("First 5 row names after conversion:\n")
  print(head(rownames(expr_120584_intersect), 5))
} else {
  cat("Using MIMAT accessions as row names.\n")
}


# ==============================================================================
# SECTION 5: Per-Dataset Z-Score Standardization
# ==============================================================================
# RATIONALE:
# GSE120584 values are VST-transformed RNA-seq counts (typical range: 0–15)
# GSE46579 values are RMA-normalized microarray intensities (typical range: 2–14)
# These scales are not directly comparable due to platform differences.
#
# Per-dataset z-score standardization:
#   For each miRNA (row) in each dataset, subtract the within-dataset mean and
#   divide by the within-dataset standard deviation.
#   Result: each miRNA has mean 0 and SD 1 within its own dataset.
#
# KEY PRINCIPLE: Each dataset is standardized independently.
#   We do NOT pool both datasets before standardizing.
#   Pooling would allow the training dataset's distribution to influence the
#   standardization of the validation dataset — a subtle form of data leakage.
#   Standardizing separately preserves relative (within-dataset) differences
#   between AD and Control while removing absolute scale differences.

cat("\n=== SECTION 5: Per-Dataset Z-Score Standardization ===\n")

# --------------------------------------------------------------------------
# 5A. Row-wise (per-miRNA) z-score standardization
# --------------------------------------------------------------------------
# We standardize across samples within each dataset (row = miRNA, col = sample).
# This means each miRNA has mean=0, SD=1 ACROSS the samples in its own dataset.

z_score_rows <- function(mat) {
  # mat: rows = features (miRNAs), cols = samples
  # Returns: same dimensions with each row (miRNA) having mean=0, SD=1
  row_means <- rowMeans(mat, na.rm = TRUE)
  row_sds   <- apply(mat, 1, sd, na.rm = TRUE)

  # Warn if any miRNA has zero variance (constant across all samples)
  zero_var <- sum(row_sds == 0, na.rm = TRUE)
  if (zero_var > 0) {
    cat("WARNING:", zero_var, "features have zero variance and will be removed.\n")
    keep <- row_sds > 0
    mat       <- mat[keep, ]
    row_means <- row_means[keep]
    row_sds   <- row_sds[keep]
  }

  z_mat <- sweep(mat, 1, row_means, "-")
  z_mat <- sweep(z_mat, 1, row_sds, "/")
  return(z_mat)
}

expr_120584_z <- z_score_rows(expr_120584_intersect)
expr_46579_z  <- z_score_rows(expr_46579_intersect)

# --------------------------------------------------------------------------
# 5B. Verify standardization
# --------------------------------------------------------------------------
cat("\nVerification — GSE120584 z-scored:\n")
cat("  Row means (should all be ~0): min =",
    round(min(rowMeans(expr_120584_z)), 4),
    ", max =", round(max(rowMeans(expr_120584_z)), 4), "\n")
cat("  Row SDs   (should all be ~1): min =",
    round(min(apply(expr_120584_z, 1, sd)), 4),
    ", max =", round(max(apply(expr_120584_z, 1, sd)), 4), "\n")

cat("\nVerification — GSE46579 z-scored:\n")
cat("  Row means (should all be ~0): min =",
    round(min(rowMeans(expr_46579_z)), 4),
    ", max =", round(max(rowMeans(expr_46579_z)), 4), "\n")
cat("  Row SDs   (should all be ~1): min =",
    round(min(apply(expr_46579_z, 1, sd)), 4),
    ", max =", round(max(apply(expr_46579_z, 1, sd)), 4), "\n")

# --------------------------------------------------------------------------
# 5C. Visualize distributions pre- and post-standardization
# --------------------------------------------------------------------------
# Compare a representative miRNA (the one with the largest expression mean
# difference between platforms before standardization)

# Find miRNA with largest between-dataset mean difference before z-scoring
mean_diff_before <- abs(rowMeans(expr_120584_intersect) - rowMeans(expr_46579_intersect))
top_diff_feature <- names(which.max(mean_diff_before))

cat("\nMiRNA with largest pre-z-score platform mean difference:",
    top_diff_feature, "\n")
cat("  GSE120584 mean:", round(rowMeans(expr_120584_intersect)[top_diff_feature], 3), "\n")
cat("  GSE46579 mean: ", round(rowMeans(expr_46579_intersect)[top_diff_feature], 3), "\n")

# After z-scoring
cat("After z-scoring:\n")
cat("  GSE120584 mean:", round(rowMeans(expr_120584_z)[top_diff_feature], 3),
    "(should be ~0)\n")
cat("  GSE46579 mean: ", round(rowMeans(expr_46579_z)[top_diff_feature], 3),
    "(should be ~0)\n")

# PCA before and after z-scoring (just GSE120584 to track structure preservation)
pca_before <- prcomp(t(expr_120584_intersect), scale. = FALSE)
pca_after  <- prcomp(t(expr_120584_z),         scale. = FALSE)

var_before <- (pca_before$sdev^2) / sum(pca_before$sdev^2) * 100
var_after  <- (pca_after$sdev^2)  / sum(pca_after$sdev^2)  * 100

pca_df_before <- data.frame(
  PC1   = pca_before$x[, 1],
  PC2   = pca_before$x[, 2],
  Group = metadata_120584$group[match(colnames(expr_120584_intersect),
                                      metadata_120584$geo_accession)],
  Stage = "Before z-scoring"
)

pca_df_after <- data.frame(
  PC1   = pca_after$x[, 1],
  PC2   = pca_after$x[, 2],
  Group = metadata_120584$group[match(colnames(expr_120584_z),
                                      metadata_120584$geo_accession)],
  Stage = "After z-scoring"
)

pca_plot_data <- bind_rows(pca_df_before, pca_df_after)
pca_plot_data$Stage <- factor(pca_plot_data$Stage,
                               levels = c("Before z-scoring", "After z-scoring"))

p_pca_comparison <- ggplot(
  pca_plot_data,
  aes(x = PC1, y = PC2, colour = Group, shape = Group)
) +
  geom_point(size = 2.5, alpha = 0.8) +
  facet_wrap(~Stage, scales = "free") +
  scale_colour_manual(values = GROUP_COLOURS, na.value = "grey50") +
  labs(
    title = "PCA Before and After Per-Dataset Z-Score Standardization\n(GSE120584)",
    x     = "PC1", y = "PC2"
  ) +
  theme_bw(base_size = 11) +
  theme(strip.background = element_rect(fill = "grey90"),
        strip.text       = element_text(face = "bold"))

print(p_pca_comparison)
ggsave("qc_reports/pca_before_after_zscore.png",
       p_pca_comparison, width = 10, height = 4.5, dpi = 150)
cat("PCA comparison plot saved to qc_reports/pca_before_after_zscore.png\n")
# EXPECTED: The biological grouping (AD vs Control clustering) should be
#            preserved after z-scoring. The axis scales will change but the
#            relative positions of samples should be similar.


# ==============================================================================
# SECTION 6: ML Classifiers in R — Binary AD vs Control Classification
# ==============================================================================
# WHAT WE DO HERE:
#   1. Subset to AD vs Control samples (binary classification)
#   2. Train two models:
#        a. Random Forest (randomForest package, via caret)
#        b. LASSO Logistic Regression (glmnet, alpha = 1)
#   3. Use nested cross-validation: outer 5-fold for unbiased AUC estimation,
#      inner 3-fold for hyperparameter tuning.
#   4. Collect cross-validated predictions (probability of AD) for each fold.
#
# LEAKAGE PREVENTION:
#   - Feature matrices were z-scored PER DATASET (Section 5) — no leakage.
#   - All normalisation and model fitting happens INSIDE the outer CV loop;
#     here, because z-scoring was already done per-dataset, we only transpose
#     the training matrix once and feed it to caret.
#   - set.seed(42) ensures reproducibility.
#
# DATA CONVENTION: expr_120584_z has rows = miRNAs, columns = samples.
#   Transpose to samples × miRNAs for ML (caret and glmnet convention).

cat("\n=== SECTION 6: ML Classifiers (R) — AD vs Control ===\n")

set.seed(42)

# --------------------------------------------------------------------------
# 6A. Prepare binary training data (AD vs Control)
# --------------------------------------------------------------------------
# Subset metadata to AD and Control only; drop MCI for binary classification.
# (Three-class extension — AD/MCI/Control — is covered in Section 6F.)

meta_binary <- metadata_120584 %>%
  filter(group %in% c("Alzheimer's Disease", "Control")) %>%
  mutate(
    group_f = factor(
      ifelse(group == "Alzheimer's Disease", "AD", "Control"),
      levels = c("Control", "AD")
    )
  )

# Subset expression matrix to binary samples
# expr_120584_z: rows = miRNAs, cols = samples; transpose for ML
common_samples <- intersect(meta_binary$geo_accession, colnames(expr_120584_z))

X_train <- t(expr_120584_z[, common_samples])   # samples × miRNAs
y_train <- meta_binary$group_f[
  match(common_samples, meta_binary$geo_accession)]

cat("Binary classification dataset (AD vs Control):\n")
cat("  Samples:", nrow(X_train), "\n")
cat("  Features:", ncol(X_train), "\n")
cat("  Class distribution:\n")
print(table(y_train))

# --------------------------------------------------------------------------
# 6B. Define caret trainControl (repeated stratified cross-validation)
# --------------------------------------------------------------------------
# Outer 5-fold, 3 repeats for robust AUC estimation.
# classProbs = TRUE:   required to get predicted probabilities.
# twoClassSummary:     computes ROC/AUC, sensitivity, specificity.
# savePredictions:     keeps the per-fold held-out predictions.
#
# NOTE: We use "repeatedcv" for performance estimation.
# For hyperparameter selection inside each outer fold, caret automatically
# runs an inner CV defined by the same trainControl.
# This is the caret implementation of nested CV.

ctrl <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  verboseIter     = FALSE
)

cat("\ntrainControl: 5-fold × 3 repeats, classProbs, twoClassSummary\n")

# --------------------------------------------------------------------------
# 6C. Train Random Forest with nested hyperparameter tuning
# --------------------------------------------------------------------------
# caret's method="rf" calls randomForest() internally.
# Tuning parameter: mtry (number of features randomly sampled at each split).
# We supply a grid of mtry values; caret evaluates each via the inner CV.

set.seed(42)
cat("\nTraining Random Forest (nested CV) — this may take 2–5 minutes...\n")

rf_grid <- expand.grid(
  mtry = c(
    floor(sqrt(ncol(X_train))),       # default: sqrt(p) — standard for classification
    floor(ncol(X_train) / 3),         # p/3
    floor(ncol(X_train) / 5)          # p/5
  )
)

# Remove duplicates if n_features is small
rf_grid <- rf_grid[!duplicated(rf_grid$mtry), , drop = FALSE]

model_rf <- train(
  x          = X_train,
  y          = y_train,
  method     = "rf",
  metric     = "ROC",
  trControl  = ctrl,
  tuneGrid   = rf_grid,
  ntree      = 300,        # number of trees per forest; 300 is sufficient for CV
  importance = TRUE        # enable variable importance (used in Section 6E)
)

cat("\nRandom Forest CV results:\n")
print(model_rf$results[, c("mtry", "ROC", "Sens", "Spec")])
cat("Best mtry:", model_rf$bestTune$mtry, "\n")
cat("Best CV AUC:", round(max(model_rf$results$ROC, na.rm = TRUE), 4), "\n")

# --------------------------------------------------------------------------
# 6D. Train LASSO Logistic Regression
# --------------------------------------------------------------------------
# caret's method="glmnet" tunes two hyperparameters:
#   alpha  = mixing parameter (0 = ridge, 1 = LASSO, values in between = elastic net)
#   lambda = regularisation strength (larger → more shrinkage → fewer features)
# We fix alpha = 1 (pure LASSO) to enforce sparsity and select a small miRNA panel.
# lambda is tuned over the range that glmnet auto-generates.

set.seed(42)
cat("\nTraining LASSO Logistic Regression (nested CV)...\n")

glmnet_grid <- expand.grid(
  alpha  = 1,                                    # LASSO (alpha=1 fixed)
  lambda = exp(seq(log(0.001), log(1), length.out = 30))  # 30 lambda values
)

model_glm <- train(
  x         = X_train,
  y         = y_train,
  method    = "glmnet",
  metric    = "ROC",
  trControl = ctrl,
  tuneGrid  = glmnet_grid,
  family    = "binomial"
)

cat("\nLASSO Logistic Regression CV results (best lambda):\n")
best_glm <- model_glm$results[which.max(model_glm$results$ROC), ]
print(best_glm[, c("alpha", "lambda", "ROC", "Sens", "Spec")])
cat("Best lambda:", round(model_glm$bestTune$lambda, 6), "\n")
cat("Best CV AUC:", round(max(model_glm$results$ROC, na.rm = TRUE), 4), "\n")

# --------------------------------------------------------------------------
# 6E. Extract and save cross-validated predictions
# --------------------------------------------------------------------------
# model$pred contains the held-out predictions for every CV fold × repeat.
# Each row is one sample in one held-out fold.

extract_cv_preds <- function(model_obj, y_obs, sample_ids) {
  preds <- model_obj$pred
  if (is.null(preds))
    stop("No saved predictions. Set savePredictions='final' in trainControl.")

  # Keep best-hyperparameter rows only
  best_tune <- model_obj$bestTune
  for (param in names(best_tune)) {
    preds <- preds[preds[[param]] == best_tune[[param]], ]
  }

  # Average over repeats for each sample (there are 3 repeats × 5 folds)
  preds_agg <- preds %>%
    group_by(rowIndex) %>%
    summarise(
      prob_AD    = mean(AD),            # mean predicted P(AD) across repeats
      true_label = first(obs),          # observed class label
      .groups    = "drop"
    ) %>%
    arrange(rowIndex) %>%
    mutate(
      true_binary = as.integer(true_label == "AD"),
      sample_id   = sample_ids[rowIndex]
    )

  return(preds_agg)
}

cv_preds_rf  <- extract_cv_preds(model_rf,  y_train, common_samples)
cv_preds_glm <- extract_cv_preds(model_glm, y_train, common_samples)

cat("\nCV predictions extracted:\n")
cat("  Random Forest:        ", nrow(cv_preds_rf), "samples\n")
cat("  LASSO Logistic Reg.:  ", nrow(cv_preds_glm), "samples\n")

# Save cross-validated predictions
if (!dir.exists("results/Week5")) dir.create("results/Week5", recursive = TRUE)

write.csv(cv_preds_rf[, c("sample_id", "true_binary", "prob_AD")],
          "results/Week5/cv_predictions_rf.csv",
          row.names = FALSE)
write.csv(cv_preds_glm[, c("sample_id", "true_binary", "prob_AD")],
          "results/Week5/cv_predictions_glm.csv",
          row.names = FALSE)

cat("CV prediction files saved to results/Week5/\n")

# --------------------------------------------------------------------------
# 6F. Three-class classification: AD vs MCI vs Control
# --------------------------------------------------------------------------
# Use caret's multiClassSummary to get per-class metrics.
# Random Forest handles multiclass natively; glmnet uses one-vs-rest internally.

cat("\n--- Three-Class Classification (AD / MCI / Control) ---\n")

set.seed(42)

# Include all three groups
meta_3class <- metadata_120584 %>%
  filter(group %in% c("Alzheimer's Disease", "Mild Cognitive Impairment", "Control")) %>%
  mutate(
    group_f3 = factor(
      case_when(
        group == "Alzheimer's Disease"       ~ "AD",
        group == "Mild Cognitive Impairment" ~ "MCI",
        group == "Control"                   ~ "Control"
      ),
      levels = c("Control", "MCI", "AD")
    )
  )

samples_3class <- intersect(meta_3class$geo_accession, colnames(expr_120584_z))
X_3class <- t(expr_120584_z[, samples_3class])
y_3class <- meta_3class$group_f3[
  match(samples_3class, meta_3class$geo_accession)]

cat("Three-class dataset:\n")
print(table(y_3class))

ctrl_mc <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = "final",
  verboseIter     = FALSE
)

cat("Training Random Forest (3-class)...\n")
model_rf_3class <- train(
  x         = X_3class,
  y         = y_3class,
  method    = "rf",
  metric    = "AUC",
  trControl = ctrl_mc,
  tuneGrid  = rf_grid,
  ntree     = 300,
  importance = TRUE
)

cat("Three-class Random Forest CV:\n")
mc_results <- model_rf_3class$results[which.max(model_rf_3class$results$AUC), ]
cat("  Best AUC (macro):  ", round(mc_results$AUC, 4), "\n")

# Confusion matrix from 3-class predictions
cm_preds <- model_rf_3class$pred %>%
  filter(mtry == model_rf_3class$bestTune$mtry) %>%
  group_by(rowIndex) %>%
  slice(1) %>%    # one row per sample (use first repeat)
  ungroup()

cm_3class <- confusionMatrix(cm_preds$pred, cm_preds$obs, mode = "everything")
cat("\nThree-class confusion matrix (one repeat, representative):\n")
print(cm_3class$table)
cat("\nPer-class statistics:\n")
print(round(cm_3class$byClass[, c("Sensitivity", "Specificity", "F1")], 3))


# ==============================================================================
# SECTION 7: SHAP Feature Importance (fastshap)
# ==============================================================================
# WHY SHAP?
# caret's varImp() gives RF variable importance (mean Gini decrease), which is
# a useful sanity check but does not tell you:
#   - Whether high expression predicts AD or Control (direction)
#   - Patient-level explanations
#
# SHAP (SHapley Additive exPlanations) answers all of these questions.
# fastshap::explain() uses a sampling-based approximation that works with
# any black-box model, using a "pfun" that returns class probabilities.
#
# Reference: Greenwell B (2023). fastshap: Fast Approximate Shapley Values.
#   CRAN. https://cran.r-project.org/package=fastshap
#
# NOTE: fastshap uses a permutation-based approximation. Results are stochastic;
# set.seed(42) before calling explain() for reproducibility.
# Increase nsim for more precise SHAP estimates (default 1 is too few; 100 is good).

cat("\n=== SECTION 7: SHAP Feature Importance (fastshap) ===\n")

set.seed(42)

# Refit Random Forest on ALL binary training data (no CV) for SHAP computation.
# We use the best mtry found during cross-validation.
rf_final <- randomForest(
  x          = X_train,
  y          = y_train,
  ntree      = 500,
  mtry       = model_rf$bestTune$mtry,
  importance = TRUE
)

cat("Random Forest refitted on full training data (ntree=500).\n")

# --------------------------------------------------------------------------
# 7A. Define prediction function for fastshap
# --------------------------------------------------------------------------
# fastshap requires a function that takes a model and a matrix and returns
# a numeric vector (or matrix) of predictions. For binary classification,
# we return P(AD) — the probability of the positive class.

pfun_rf <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "AD"]
}

# --------------------------------------------------------------------------
# 7B. Compute SHAP values
# --------------------------------------------------------------------------
# explain() approximates SHAP values using Monte Carlo sampling.
# nsim = 100 gives a good balance of accuracy vs speed for ~100 features.
# For large feature sets (>500), try nsim = 50 first.
# X_train must be a matrix (not data frame) for fastshap.

cat("Computing SHAP values (nsim=100) — this takes ~1–3 minutes...\n")

set.seed(42)
shap_vals <- fastshap::explain(
  object    = rf_final,
  feature_names = colnames(X_train),
  X         = as.matrix(X_train),
  pred_fun  = pfun_rf,
  nsim      = 100,
  .progress = FALSE
)

# shap_vals is a matrix: samples × features
# Each entry = SHAP contribution of that miRNA for that patient

cat("SHAP values computed. Dimensions:", dim(shap_vals), "\n")
cat("  (rows = samples, columns = miRNA features)\n")

# --------------------------------------------------------------------------
# 7C. Global feature importance: mean |SHAP|
# --------------------------------------------------------------------------
# Mean absolute SHAP value across all samples = overall feature importance.
# A miRNA with large mean |SHAP| matters for many patients' predictions.

mean_abs_shap <- colMeans(abs(shap_vals))
shap_importance <- data.frame(
  miRNA        = names(mean_abs_shap),
  mean_abs_shap = as.numeric(mean_abs_shap)
) %>%
  arrange(desc(mean_abs_shap))

cat("\nTop 20 miRNAs by mean |SHAP| value:\n")
print(head(shap_importance, 20))

# Save full SHAP importance table (read by Week6_Interpretation.R)
write.csv(shap_importance,
          "results/Week5/shap_feature_importance.csv",
          row.names = FALSE)
cat("SHAP importance table saved to results/Week5/shap_feature_importance.csv\n")
cat("  (This file is read by Week6_Interpretation.R for composite ranking.)\n")

# --------------------------------------------------------------------------
# 7D. Beeswarm-style SHAP summary plot using ggplot2
# --------------------------------------------------------------------------
# For each of the top 20 miRNAs, plot individual sample SHAP values:
#   x-axis: SHAP value (positive = pushes toward AD, negative = toward Control)
#   y-axis: miRNA (ordered by mean |SHAP|, most important at top)
#   color:  z-scored expression value (blue = low expression, red = high)

top20_mirnas <- head(shap_importance$miRNA, 20)

shap_long <- as.data.frame(shap_vals[, top20_mirnas]) %>%
  mutate(sample_idx = seq_len(nrow(X_train))) %>%
  pivot_longer(cols = -sample_idx,
               names_to  = "miRNA",
               values_to = "shap_value")

# Attach expression values for color encoding
expr_long <- as.data.frame(X_train[, top20_mirnas]) %>%
  mutate(sample_idx = seq_len(nrow(X_train))) %>%
  pivot_longer(cols = -sample_idx,
               names_to  = "miRNA",
               values_to = "expression")

shap_plot_data <- left_join(shap_long, expr_long,
                             by = c("sample_idx", "miRNA"))

# Order miRNA factor by mean |SHAP| (top at top of y-axis)
shap_plot_data$miRNA <- factor(
  shap_plot_data$miRNA,
  levels = rev(top20_mirnas)   # rev() so highest importance is at top
)

# Jitter vertically within each miRNA row (beeswarm approximation)
set.seed(42)
shap_plot_data <- shap_plot_data %>%
  group_by(miRNA) %>%
  mutate(y_jitter = as.numeric(miRNA) + runif(n(), -0.35, 0.35)) %>%
  ungroup()

p_shap <- ggplot(shap_plot_data,
                 aes(x = shap_value, y = y_jitter, colour = expression)) +
  geom_point(size = 0.9, alpha = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
  scale_colour_gradient2(
    low      = "#4575B4",   # blue = low expression
    mid      = "white",
    high     = "#D73027",   # red = high expression
    midpoint = 0,
    name     = "Z-scored\nexpression"
  ) +
  scale_y_continuous(
    breaks = seq_along(levels(shap_plot_data$miRNA)),
    labels = levels(shap_plot_data$miRNA)
  ) +
  labs(
    x     = "SHAP value (positive = pushes toward AD prediction)",
    y     = NULL,
    title = "SHAP Beeswarm Plot — Random Forest AD vs Control\n(Top 20 miRNAs by mean |SHAP value|)"
  ) +
  theme_bw(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position  = "right"
  )

print(p_shap)
ggsave("results/Week5/shap_beeswarm.png",
       p_shap, width = 9, height = 7, dpi = 150)
cat("SHAP beeswarm plot saved to results/Week5/shap_beeswarm.png\n")

# --------------------------------------------------------------------------
# 7E. RF built-in variable importance (Gini) as a sanity check
# --------------------------------------------------------------------------
# Compare mean |SHAP| ranking to the Gini-based MeanDecreaseGini.
# Concordance between the two rankings confirms SHAP is working correctly.

rf_imp <- as.data.frame(importance(rf_final))
rf_imp$miRNA <- rownames(rf_imp)
rf_imp <- rf_imp %>% arrange(desc(MeanDecreaseGini))

cat("\nTop 10 miRNAs by RF Gini importance:\n")
print(head(rf_imp[, c("miRNA", "MeanDecreaseGini")], 10))
cat("\nTop 10 by mean |SHAP| (for comparison):\n")
print(head(shap_importance[, c("miRNA", "mean_abs_shap")], 10))


# ==============================================================================
# SECTION 8: External Validation — Apply Model to GSE46579
# ==============================================================================
# WHAT WE DO HERE:
#   1. Standardize GSE46579 metadata group labels to match GSE120584 convention
#   2. Subset to AD vs Control (binary validation)
#   3. Apply the RF model (trained on all GSE120584 binary data) to GSE46579
#   4. Compute external validation AUC
#   5. Build ROC objects for both training CV and external validation
#   6. Compare AUC values using DeLong's test (pROC::roc.test)
#
# NOTE on the right comparison:
#   We compare:
#     Training CV AUC  — from Section 6C (cross-validated, unbiased)
#     External Val AUC — from this section (completely independent cohort)
#   For independent cohorts, method = "bootstrap" is appropriate.
#   DeLong's method (exact) applies when the SAME patients are evaluated by
#   two different models. Our cohorts are different patients; bootstrap is used.

cat("\n=== SECTION 8: External Validation on GSE46579 ===\n")

# --------------------------------------------------------------------------
# 8A. Standardize GSE46579 group labels
# --------------------------------------------------------------------------
if (!"group" %in% colnames(metadata_46579)) {
  cat("'group' column not found in metadata_46579; attempting to parse.\n")
  if ("characteristics_ch1" %in% colnames(metadata_46579)) {
    metadata_46579$group <- trimws(gsub(".*: ", "", metadata_46579$characteristics_ch1))
  }
}

ad_label_pattern   <- "Alzheimer|alzheimer|AD$"
ctrl_label_pattern <- "Control|control|normal|Normal"

metadata_46579 <- metadata_46579 %>%
  mutate(
    group_std = case_when(
      grepl(ad_label_pattern,   group, ignore.case = TRUE) ~ "Alzheimer's Disease",
      grepl(ctrl_label_pattern, group, ignore.case = TRUE) ~ "Control",
      TRUE ~ as.character(group)
    )
  )

cat("GSE46579 group distribution (standardized):\n")
print(table(metadata_46579$group_std))

# --------------------------------------------------------------------------
# 8B. Prepare external validation matrix
# --------------------------------------------------------------------------
# Subset to AD vs Control only; match samples to expression matrix columns.

meta_val_binary <- metadata_46579 %>%
  filter(group_std %in% c("Alzheimer's Disease", "Control")) %>%
  mutate(
    group_f = factor(
      ifelse(group_std == "Alzheimer's Disease", "AD", "Control"),
      levels = c("Control", "AD")
    )
  )

val_samples <- intersect(meta_val_binary$geo_accession, colnames(expr_46579_z))
X_val <- t(expr_46579_z[, val_samples])      # samples × miRNAs
y_val <- meta_val_binary$group_f[
  match(val_samples, meta_val_binary$geo_accession)]

# Ensure feature alignment with training matrix
missing_features <- setdiff(colnames(X_train), colnames(X_val))
extra_features   <- setdiff(colnames(X_val), colnames(X_train))

if (length(missing_features) > 0) {
  cat("WARNING:", length(missing_features), "features in training but not validation.\n")
  cat("  Adding zero-filled columns for missing features.\n")
  miss_mat <- matrix(0, nrow = nrow(X_val), ncol = length(missing_features),
                     dimnames = list(rownames(X_val), missing_features))
  X_val <- cbind(X_val, miss_mat)
}
# Keep only training features (in training order)
X_val <- X_val[, colnames(X_train)]

cat("\nExternal validation set (binary AD vs Control):\n")
cat("  Samples:", nrow(X_val), "\n")
cat("  Features:", ncol(X_val), "\n")
cat("  Class distribution:\n")
print(table(y_val))

# --------------------------------------------------------------------------
# 8C. Predict on external validation cohort
# --------------------------------------------------------------------------
set.seed(42)

prob_val_rf  <- predict(rf_final, newdata = X_val, type = "prob")[, "AD"]

# LASSO model: use the best-lambda final model from caret
prob_val_glm <- predict(model_glm, newdata = X_val, type = "prob")[, "AD"]

y_val_binary <- as.integer(y_val == "AD")

cat("External validation predictions generated.\n")
cat("  RF mean P(AD) in AD samples:     ",
    round(mean(prob_val_rf[y_val == "AD"]), 3), "\n")
cat("  RF mean P(AD) in Control samples:",
    round(mean(prob_val_rf[y_val == "Control"]), 3), "\n")

# Save external validation predictions
val_preds_out <- data.frame(
  sample_id  = val_samples,
  true_label = y_val_binary,
  prob_AD_rf = prob_val_rf,
  prob_AD_glm= prob_val_glm
)
write.csv(val_preds_out,
          "results/Week5/external_validation_predictions.csv",
          row.names = FALSE)
cat("External validation predictions saved.\n")

# --------------------------------------------------------------------------
# 8D. Build ROC objects (pROC)
# --------------------------------------------------------------------------
roc_cv_rf <- roc(
  response  = cv_preds_rf$true_binary,
  predictor = cv_preds_rf$prob_AD,
  direction = "<",
  quiet     = TRUE
)

roc_val_rf <- roc(
  response  = y_val_binary,
  predictor = prob_val_rf,
  direction = "<",
  quiet     = TRUE
)

roc_cv_glm <- roc(
  response  = cv_preds_glm$true_binary,
  predictor = cv_preds_glm$prob_AD,
  direction = "<",
  quiet     = TRUE
)

roc_val_glm <- roc(
  response  = y_val_binary,
  predictor = prob_val_glm,
  direction = "<",
  quiet     = TRUE
)

cat("\n--- AUC Summary ---\n")
cat(sprintf("  Random Forest   — Training CV AUC: %.4f | Validation AUC: %.4f | Gap: %.4f\n",
            as.numeric(auc(roc_cv_rf)),
            as.numeric(auc(roc_val_rf)),
            as.numeric(auc(roc_cv_rf)) - as.numeric(auc(roc_val_rf))))
cat(sprintf("  LASSO Log. Reg. — Training CV AUC: %.4f | Validation AUC: %.4f | Gap: %.4f\n",
            as.numeric(auc(roc_cv_glm)),
            as.numeric(auc(roc_val_glm)),
            as.numeric(auc(roc_cv_glm)) - as.numeric(auc(roc_val_glm))))

# --------------------------------------------------------------------------
# 8E. Bootstrap confidence intervals (pROC::ci.auc)
# --------------------------------------------------------------------------
cat("\nComputing bootstrap confidence intervals (2000 resamples)...\n")

set.seed(42)
ci_cv_rf  <- ci.auc(roc_cv_rf,  method = "bootstrap", boot.n = 2000, conf.level = 0.95)
ci_val_rf <- ci.auc(roc_val_rf, method = "bootstrap", boot.n = 2000, conf.level = 0.95)
ci_cv_glm  <- ci.auc(roc_cv_glm,  method = "bootstrap", boot.n = 2000, conf.level = 0.95)
ci_val_glm <- ci.auc(roc_val_glm, method = "bootstrap", boot.n = 2000, conf.level = 0.95)

cat(sprintf("  Random Forest:\n"))
cat(sprintf("    Training CV AUC:         %.4f (95%% CI: %.4f – %.4f)\n",
            as.numeric(auc(roc_cv_rf)), ci_cv_rf[1], ci_cv_rf[3]))
cat(sprintf("    External Validation AUC: %.4f (95%% CI: %.4f – %.4f)\n",
            as.numeric(auc(roc_val_rf)), ci_val_rf[1], ci_val_rf[3]))

cat(sprintf("  LASSO:\n"))
cat(sprintf("    Training CV AUC:         %.4f (95%% CI: %.4f – %.4f)\n",
            as.numeric(auc(roc_cv_glm)), ci_cv_glm[1], ci_cv_glm[3]))
cat(sprintf("    External Validation AUC: %.4f (95%% CI: %.4f – %.4f)\n",
            as.numeric(auc(roc_val_glm)), ci_val_glm[1], ci_val_glm[3]))

# --------------------------------------------------------------------------
# 8F. Bootstrap AUC comparison test (DeLong's reasoning, bootstrap implementation)
# --------------------------------------------------------------------------
# DeLong's exact test is for PAIRED comparisons (same patients, two models).
# Because our two cohorts are INDEPENDENT (different patients), we use
# method = "bootstrap" which performs a permutation/bootstrap comparison.
# Ref: DeLong ER et al. (1988) Biometrics 44(3):837-845;
#      Robin X et al. (2011) pROC, BMC Bioinformatics 12:77.

cat("\n--- Bootstrap AUC Comparison: Training CV vs External Validation ---\n")

set.seed(42)
auc_test_rf <- roc.test(
  roc1        = roc_cv_rf,
  roc2        = roc_val_rf,
  method      = "bootstrap",
  boot.n      = 2000,
  alternative = "greater",    # H1: training AUC > validation AUC
  paired      = FALSE         # independent cohorts
)

cat("\nRandom Forest AUC comparison test:\n")
cat("  H0: Training CV AUC = External Validation AUC\n")
cat("  H1: Training CV AUC > External Validation AUC (one-sided)\n")
print(auc_test_rf)

cat("\nInterpretation:\n")
if (auc_test_rf$p.value < 0.05) {
  cat("  p < 0.05: Training AUC is significantly higher than validation AUC.\n")
  cat("  Some performance degradation on external cohort. This is expected:\n")
  cat("  platform differences (serum RNA-seq vs whole blood microarray) and\n")
  cat("  cohort heterogeneity contribute to the gap.\n")
} else {
  cat("  p >= 0.05: Cannot conclude significant difference in AUC between cohorts.\n")
  cat("  The model generalises comparably to the external cohort.\n")
}

# --------------------------------------------------------------------------
# 8G. ROC curve comparison plot (both models, both cohorts)
# --------------------------------------------------------------------------
# Build ggplot-compatible ROC data frame
roc_to_df <- function(roc_obj, model_label, cohort_label) {
  data.frame(
    FPR    = 1 - roc_obj$specificities,
    TPR    = roc_obj$sensitivities,
    Model  = model_label,
    Cohort = cohort_label,
    AUC    = round(as.numeric(auc(roc_obj)), 3)
  )
}

roc_all <- bind_rows(
  roc_to_df(roc_cv_rf,  "Random Forest", "Training CV"),
  roc_to_df(roc_val_rf, "Random Forest", "Validation"),
  roc_to_df(roc_cv_glm, "LASSO",         "Training CV"),
  roc_to_df(roc_val_glm,"LASSO",         "Validation")
) %>%
  mutate(
    Label = paste0(Model, " - ", Cohort, " (AUC=", AUC, ")"),
    Colour = case_when(
      Model == "Random Forest" & Cohort == "Training CV" ~ "#4575B4",
      Model == "Random Forest" & Cohort == "Validation"  ~ "#74ADD1",
      Model == "LASSO"         & Cohort == "Training CV" ~ "#D73027",
      Model == "LASSO"         & Cohort == "Validation"  ~ "#F46D43"
    ),
    Linetype = ifelse(Cohort == "Training CV", "solid", "dashed")
  )

p_roc <- ggplot(roc_all, aes(x = FPR, y = TPR,
                              colour = Label, linetype = Label)) +
  geom_line(linewidth = 1.1) +
  geom_abline(intercept = 0, slope = 1,
              colour = "grey50", linetype = "dotted", linewidth = 0.6) +
  scale_colour_manual(
    values = setNames(roc_all$Colour, roc_all$Label),
    name   = NULL
  ) +
  scale_linetype_manual(
    values = setNames(roc_all$Linetype, roc_all$Label),
    name   = NULL
  ) +
  labs(
    x     = "False Positive Rate (1 – Specificity)",
    y     = "True Positive Rate (Sensitivity)",
    title = "ROC Curves — Training CV vs External Validation\n(GSE120584 training | GSE46579 validation)"
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom",
        legend.text     = element_text(size = 8))

print(p_roc)
ggsave("results/Week5/roc_curves.png",
       p_roc, width = 7, height = 6.5, dpi = 150)
cat("ROC curve comparison plot saved to results/Week5/roc_curves.png\n")


# ==============================================================================
# SECTION 9: Calibration Plot
# ==============================================================================
# RATIONALE:
# AUC measures discrimination (can the model rank AD patients above Controls?)
# Calibration measures accuracy of probability estimates (if the model says
# P(AD) = 0.80, are approximately 80% of those patients truly AD?).
#
# A model can have excellent AUC but terrible calibration (e.g., it correctly
# ranks everyone but assigns probabilities that are all near 0.55 or 0.95).
# Clinicians need calibrated probabilities to make individual patient decisions.
#
# We implement calibration using manual probability binning.
# The rms::val.prob() function provides additional calibration statistics (E50,
# E90, Emax) if the rms package is available.

cat("\n=== SECTION 9: Calibration Plot ===\n")

# --------------------------------------------------------------------------
# 9A. Manual calibration plot using probability binning
# --------------------------------------------------------------------------
# Divide predicted probabilities into quantile-based bins.
# For each bin: mean predicted probability vs observed event rate.

calibration_plot_data <- function(true_labels, pred_probs,
                                  n_bins = 10, method = "quantile") {
  # method = "quantile": equal-frequency bins (preferred for imbalanced data)
  # method = "uniform": equal-width bins (0, 0.1, 0.2, ...)

  if (method == "quantile") {
    breaks <- quantile(pred_probs, probs = seq(0, 1, length.out = n_bins + 1))
    breaks <- unique(breaks)  # remove duplicates if distribution is peaked
  } else {
    breaks <- seq(0, 1, length.out = n_bins + 1)
  }

  bin_idx  <- cut(pred_probs, breaks = breaks, include.lowest = TRUE,
                   labels = FALSE)
  bin_df   <- data.frame(pred = pred_probs, true = true_labels, bin = bin_idx)

  calib_df <- bin_df %>%
    group_by(bin) %>%
    summarise(
      n             = n(),
      mean_pred     = mean(pred),
      observed_rate = mean(true),
      se            = sqrt(observed_rate * (1 - observed_rate) / n),
      .groups       = "drop"
    ) %>%
    filter(!is.na(bin))

  return(calib_df)
}

# Calibration for external validation (RF model)
calib_rf  <- calibration_plot_data(y_val_binary, prob_val_rf,  n_bins = 10)
calib_glm <- calibration_plot_data(y_val_binary, prob_val_glm, n_bins = 10)

calib_rf$model  <- "Random Forest"
calib_glm$model <- "LASSO"

calib_combined <- bind_rows(calib_rf, calib_glm)

p_calib <- ggplot(calib_combined,
                  aes(x = mean_pred, y = observed_rate,
                      colour = model, group = model)) +
  geom_abline(intercept = 0, slope = 1,
              linetype = "dashed", colour = "grey40", linewidth = 0.8) +
  geom_errorbar(
    aes(ymin = observed_rate - 1.96 * se,
        ymax = observed_rate + 1.96 * se),
    width = 0.02, alpha = 0.7
  ) +
  geom_point(aes(size = n), alpha = 0.85) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.9, alpha = 0.5) +
  scale_colour_manual(values = c("Random Forest" = "#4575B4",
                                 "LASSO"         = "#D73027"),
                      name   = "Model") +
  scale_size_continuous(name = "n (samples in bin)", range = c(2, 7)) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(
    x       = "Mean Predicted Probability P(AD)",
    y       = "Observed Event Rate (Fraction True AD)",
    title   = "Calibration Plot — External Validation (GSE46579)",
    caption = "Dashed line = perfect calibration. Error bars = 95% CI."
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")

print(p_calib)
ggsave("results/Week5/calibration_plot_external.png",
       p_calib, width = 6.5, height = 6.5, dpi = 150)
cat("Calibration plot saved to results/Week5/calibration_plot_external.png\n")

# --------------------------------------------------------------------------
# 9B. Quantitative calibration metrics (Brier score)
# --------------------------------------------------------------------------
# Brier score: mean squared error of probability predictions
# Range: 0 (perfect) to 0.25 (uninformative — equivalent to always predicting 0.5)

brier_rf  <- mean((y_val_binary - prob_val_rf)^2)
brier_glm <- mean((y_val_binary - prob_val_glm)^2)
prevalence <- mean(y_val_binary)
brier_null <- prevalence * (1 - prevalence)   # null model (predict prevalence)

cat("\nCalibration Statistics (External Validation):\n")
cat(sprintf("  Random Forest Brier Score: %.4f (null: %.4f | scaled: %.4f)\n",
            brier_rf,  brier_null, 1 - brier_rf  / brier_null))
cat(sprintf("  LASSO       Brier Score: %.4f (null: %.4f | scaled: %.4f)\n",
            brier_glm, brier_null, 1 - brier_glm / brier_null))
cat("  (Scaled Brier: 1 = perfect; 0 = null model; negative = worse than null)\n")

# --------------------------------------------------------------------------
# 9C. Optional: rms::val.prob for detailed calibration statistics
# --------------------------------------------------------------------------
if (requireNamespace("rms", quietly = TRUE)) {
  library(rms)
  cat("\nrms package available — computing detailed calibration statistics (RF).\n")
  tryCatch({
    cal_rms <- val.prob(
      p      = prob_val_rf,
      y      = y_val_binary,
      pl     = TRUE,
      logistic.cal = TRUE,
      main   = "val.prob Calibration — RF External Validation"
    )
    cat("Calibration statistics from rms::val.prob:\n")
    print(cal_rms)
  }, error = function(e) {
    cat("rms::val.prob error:", conditionMessage(e), "\n")
  })
} else {
  cat("rms package not available. Install with install.packages('rms') for\n")
  cat("additional calibration statistics (E50, E90, Emax, Hosmer-Lemeshow test).\n")
}


# ==============================================================================
# SECTION 10: Model Results Summary Table
# ==============================================================================
# Compile a publication-ready summary of all ML results.
# Includes: model, training CV AUC ± CI, validation AUC ± CI,
# sensitivity, specificity, PPV, NPV at Youden-optimal threshold.

cat("\n=== SECTION 10: Model Results Summary Table ===\n")

# Function to compute performance metrics from ROC object + predictions
compute_metrics <- function(roc_obj, pred_probs, true_labels) {

  auc_val  <- as.numeric(auc(roc_obj))
  ci_boot  <- tryCatch(
    as.numeric(ci.auc(roc_obj, method = "bootstrap", boot.n = 1000)),
    error = function(e) c(NA, auc_val, NA)
  )

  # Optimal threshold by Youden index
  coords_df <- tryCatch(
    coords(roc_obj, "best", best.method = "youden",
           ret = c("threshold", "sensitivity", "specificity")),
    error = function(e) data.frame(threshold = 0.5, sensitivity = NA, specificity = NA)
  )
  threshold <- coords_df$threshold[1]
  sens      <- coords_df$sensitivity[1]
  spec      <- coords_df$specificity[1]

  # PPV and NPV at the threshold
  pred_class <- as.integer(pred_probs >= threshold)
  TP <- sum(pred_class == 1 & true_labels == 1)
  TN <- sum(pred_class == 0 & true_labels == 0)
  FP <- sum(pred_class == 1 & true_labels == 0)
  FN <- sum(pred_class == 0 & true_labels == 1)

  ppv <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
  npv <- ifelse((TN + FN) > 0, TN / (TN + FN), NA)

  return(list(
    auc       = round(auc_val, 4),
    ci_lower  = round(ci_boot[1], 4),
    ci_upper  = round(ci_boot[3], 4),
    auc_str   = sprintf("%.3f (%.3f–%.3f)", auc_val, ci_boot[1], ci_boot[3]),
    threshold = round(threshold, 4),
    sens      = round(sens, 4),
    spec      = round(spec, 4),
    ppv       = round(ppv, 4),
    npv       = round(npv, 4)
  ))
}

set.seed(42)
m_cv_rf  <- compute_metrics(roc_cv_rf,  cv_preds_rf$prob_AD,  cv_preds_rf$true_binary)
m_val_rf <- compute_metrics(roc_val_rf, prob_val_rf,          y_val_binary)
m_cv_glm  <- compute_metrics(roc_cv_glm,  cv_preds_glm$prob_AD, cv_preds_glm$true_binary)
m_val_glm <- compute_metrics(roc_val_glm, prob_val_glm,         y_val_binary)

results_table <- data.frame(
  Model              = c("RandomForest", "LASSO"),
  Training_CV_AUC    = c(m_cv_rf$auc_str,  m_cv_glm$auc_str),
  Validation_AUC     = c(m_val_rf$auc_str, m_val_glm$auc_str),
  AUC_Gap            = c(round(m_cv_rf$auc - m_val_rf$auc, 4),
                         round(m_cv_glm$auc - m_val_glm$auc, 4)),
  Optimal_Threshold  = c(m_cv_rf$threshold,  m_cv_glm$threshold),
  Sensitivity_Val    = c(m_val_rf$sens,  m_val_glm$sens),
  Specificity_Val    = c(m_val_rf$spec,  m_val_glm$spec),
  PPV_Val            = c(m_val_rf$ppv,   m_val_glm$ppv),
  NPV_Val            = c(m_val_rf$npv,   m_val_glm$npv),
  stringsAsFactors   = FALSE
)

cat("\n=== COMPLETE MODEL RESULTS SUMMARY ===\n")
print(results_table)

write.csv(results_table,
          "results/Week5/model_performance_summary.csv",
          row.names = FALSE)
cat("\nSummary table saved to results/Week5/model_performance_summary.csv\n")

# Print formatted table if knitr is available
if (requireNamespace("knitr", quietly = TRUE)) {
  cat("\nFormatted table (knitr):\n")
  print(knitr::kable(results_table, format = "simple", align = "c"))
}

# AUC gap interpretation
cat("\nAUC gap interpretation:\n")
for (i in seq_len(nrow(results_table))) {
  gap <- results_table$AUC_Gap[i]
  interp <- if (gap < 0.05) "Excellent generalizability"
             else if (gap < 0.10) "Acceptable — some platform effects"
             else if (gap < 0.20) "Moderate degradation — investigate top features"
             else "Severe degradation — model may have overfit"
  cat(sprintf("  %-15s AUC gap = %.4f : %s\n",
              results_table$Model[i], gap, interp))
}


# ==============================================================================
# SECTION 11: Save Harmonized Data Objects
# ==============================================================================
# Save all harmonized data objects as .rds files for downstream analyses
# and reproducibility. These are the "clean, harmonized" versions that include
# only the intersection features, z-scored, and with v22 miRNA names.
#
# Week 6 reads:
#   data/processed/harmonized_expr.rds         — training expression matrix
#   data/processed/metadata_harmonized.rds     — training metadata

cat("\n=== SECTION 11: Save Harmonized Data Objects ===\n")

# Main harmonized expression matrices (z-scored, intersection features only)
saveRDS(expr_120584_z,
        "data/processed/GSE120584_expr_harmonized_zscore.rds")
saveRDS(expr_46579_z,
        "data/processed/GSE46579_expr_harmonized_zscore.rds")

# Canonical file names expected by Week6_Interpretation.R
saveRDS(expr_120584_z,
        "data/processed/harmonized_expr.rds")
saveRDS(metadata_120584,
        "data/processed/metadata_harmonized.rds")

# Name mapping tables (useful for tracing back to original names)
saveRDS(conversion_120584,
        "data/processed/GSE120584_mirbase_conversion.rds")
saveRDS(conversion_46579,
        "data/processed/GSE46579_mirbase_conversion.rds")

# Intersection feature list (with both MIMAT accessions and v22 names)
intersection_info <- data.frame(
  MIMAT_accession      = common_features,
  v22_name             = rownames(expr_120584_z),  # after back-conversion in Section 4C
  present_in_GSE120584 = TRUE,
  present_in_GSE46579  = TRUE
)

write.csv(intersection_info,
          "data/processed/feature_intersection.csv",
          row.names = FALSE)
saveRDS(intersection_info,
        "data/processed/feature_intersection.rds")

cat("Harmonized data objects saved:\n")
cat("  data/processed/harmonized_expr.rds                (training expr, z-scored)\n")
cat("  data/processed/metadata_harmonized.rds            (training metadata)\n")
cat("  data/processed/GSE120584_expr_harmonized_zscore.rds\n")
cat("  data/processed/GSE46579_expr_harmonized_zscore.rds\n")
cat("  data/processed/GSE120584_mirbase_conversion.rds\n")
cat("  data/processed/GSE46579_mirbase_conversion.rds\n")
cat("  data/processed/feature_intersection.csv\n")


# ==============================================================================
# SECTION 12: Session Summary
# ==============================================================================
# Print a complete summary of what was accomplished, key numbers, and file outputs.
# Save session information for reproducibility.

cat("\n")
cat("================================================================================\n")
cat("  WEEK 5 VALIDATION SCRIPT — SESSION SUMMARY\n")
cat("================================================================================\n")

cat("\n--- Data Harmonization Summary ---\n")
cat("Training dataset (GSE120584):\n")
cat("  Platform:           Illumina HiSeq 2500 small RNA-seq\n")
cat("  Sample type:        Serum\n")
cat("  Original features:  ", n_features_120584_before, "\n")
cat("  After v22 mapping:  ", nrow(converted_120584), "\n")
cat("  Names changed:      ", n_changed_120584, "\n")

cat("\nValidation dataset (GSE46579):\n")
cat("  Platform:           Affymetrix GeneChip miRNA 3.0 Array\n")
cat("  Sample type:        Whole blood\n")
cat("  Original features:  ", n_features_46579_before, "\n")
cat("  After v22 mapping:  ", nrow(converted_46579), "\n")
cat("  Names changed:      ", n_changed_46579, "\n")

cat("\nFeature intersection:\n")
cat("  Common features (MIMAT accession):    ", n_common, "\n")
cat("  Pct of GSE120584 retained:            ",
    round(n_common / nrow(converted_120584) * 100, 1), "%\n")
cat("  Pct of GSE46579 retained:             ",
    round(n_common / nrow(converted_46579) * 100, 1), "%\n")

cat("\nZ-score standardization: applied independently per dataset.\n")

cat("\n--- ML Model Summary ---\n")
cat("Binary classifier: AD vs Control (GSE120584 training)\n")
cat(sprintf("  Random Forest (best mtry=%d):\n", model_rf$bestTune$mtry))
cat(sprintf("    Training CV AUC: %s\n", m_cv_rf$auc_str))
cat(sprintf("    Validation AUC:  %s\n", m_val_rf$auc_str))
cat(sprintf("  LASSO (best lambda=%.5f):\n", model_glm$bestTune$lambda))
cat(sprintf("    Training CV AUC: %s\n", m_cv_glm$auc_str))
cat(sprintf("    Validation AUC:  %s\n", m_val_glm$auc_str))

cat("\nTop 5 miRNAs by SHAP importance (Random Forest):\n")
print(head(shap_importance[, c("miRNA", "mean_abs_shap")], 5))

cat("\n--- Files Written This Session ---\n")
cat("data/processed/:\n")
cat("  harmonized_expr.rds                   (training expr — read by Week 6)\n")
cat("  metadata_harmonized.rds               (training metadata — read by Week 6)\n")
cat("  GSE120584_expr_harmonized_zscore.rds\n")
cat("  GSE46579_expr_harmonized_zscore.rds\n")
cat("  feature_intersection.csv\n")
cat("results/Week5/:\n")
cat("  shap_feature_importance.csv           (miRNA, mean_abs_shap — read by Week 6)\n")
cat("  cv_predictions_rf.csv\n")
cat("  cv_predictions_glm.csv\n")
cat("  external_validation_predictions.csv\n")
cat("  model_performance_summary.csv\n")
cat("  roc_curves.png\n")
cat("  calibration_plot_external.png\n")
cat("  shap_beeswarm.png\n")
cat("qc_reports/:\n")
cat("  pca_before_after_zscore.png\n")

cat("\n--- Key Interpretation Reminders ---\n")
cat("AUC gap interpretation:\n")
cat("  < 0.05:    Excellent generalizability\n")
cat("  0.05-0.10: Acceptable; some platform effects\n")
cat("  0.10-0.20: Moderate degradation; investigate top features\n")
cat("  > 0.20:    Severe degradation; model may have overfit\n\n")
cat("If AUC gap is large: examine which top SHAP features are absent from the\n")
cat("  intersection. If key features are missing from GSE46579, the model could\n")
cat("  not express their full predictive capacity in validation.\n")

cat("\n--- Next Steps ---\n")
cat("1. Proceed to Week6_Interpretation.R for:\n")
cat("   - Integration of SHAP + DE results into composite miRNA ranking\n")
cat("   - multiMiR target database queries\n")
cat("   - KEGG/GO pathway enrichment (clusterProfiler)\n")
cat("   - STRINGdb protein-protein interaction network\n")
cat("2. For further methodological details, see Week5_AdvancedML_Validation.md.\n")

# Save session info for reproducibility
session_file <- "qc_reports/session_info_week5.txt"
sink(session_file)
cat("Week 5 Validation Script — Session Information\n")
cat("Date:", as.character(Sys.time()), "\n\n")
sessionInfo()
sink()
cat("\nSession info saved to", session_file, "\n")

cat("\n================================================================================\n")
cat("  Week 5 R Script Complete!\n")
cat("================================================================================\n")
