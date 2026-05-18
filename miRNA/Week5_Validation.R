################################################################################
# AI/ML in Biomarker Discovery — Week 5 Lab
# Title:   Cross-Platform Data Harmonization & External Validation
# Disease: Alzheimer's Disease | Biomarker: miRNA
# Audience: Wet-lab biologists — Weeks 1–4 pipelines assumed complete
#
# Learning Goals for This Script:
#   1. Load and inspect both training (GSE120584) and validation (GSE46579) datasets
#   2. Harmonize miRNA names across miRBase versions using miRBaseConverter
#   3. Find and report the feature intersection between two platforms
#   4. Apply per-dataset z-score standardization before cross-dataset testing
#   5. Export harmonized matrices for use in Python ML (Lab 5B)
#   6. Compare AUC values between training CV and external validation using DeLong's test
#   7. Produce calibration plots to assess probability accuracy
#   8. Compile a complete model results summary table
#   9. Save all output objects and report session information
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
# Bioconductor packages: BiocManager::install("miRBaseConverter")
# CRAN packages: install.packages("pROC")

suppressPackageStartupMessages({
  # Bioconductor
  library(miRBaseConverter)   # miRNA name versioning across miRBase releases

  # CRAN — statistics and modeling
  library(pROC)               # ROC curves, AUC, DeLong's test
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)

  # CRAN — optional, for calibration and report formatting
  # library(rms)              # val.prob() calibration; install if needed
  # library(knitr)            # kable() for formatted tables
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
for (d in c("data/processed", "results", "qc_reports")) {
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
# checkMiRNAVersion() scans the input names and guesses the most likely
# miRBase version. Use this to understand the naming provenance.

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
# SECTION 6: Export Harmonized Matrices for Python ML
# ==============================================================================
# The Python Lab 5B script loads these CSV files.
# Format convention:
#   - Rows    = samples (patients)
#   - Columns = miRNA features (plus any non-feature columns)
#   - First column = sample ID
#   CSV files include column headers and row names as the first column.

cat("\n=== SECTION 6: Export Harmonized Matrices for Python ===\n")

# --------------------------------------------------------------------------
# 6A. Transpose to samples × features (Python ML convention)
# --------------------------------------------------------------------------
expr_120584_export <- as.data.frame(t(expr_120584_z))
expr_46579_export  <- as.data.frame(t(expr_46579_z))

# Column names = miRNA names (same in both)
# Row names    = sample IDs (GSM accessions or study-specific IDs)

cat("Exported matrix dimensions:\n")
cat("  GSE120584: samples=", nrow(expr_120584_export),
    ", features=", ncol(expr_120584_export), "\n")
cat("  GSE46579:  samples=", nrow(expr_46579_export),
    ", features=", ncol(expr_46579_export), "\n")

# --------------------------------------------------------------------------
# 6B. Write expression matrices
# --------------------------------------------------------------------------
write.csv(expr_120584_export,
          "data/processed/GSE120584_harmonized.csv",
          row.names = TRUE)

write.csv(expr_46579_export,
          "data/processed/GSE46579_harmonized.csv",
          row.names = TRUE)

cat("Expression matrices exported.\n")

# --------------------------------------------------------------------------
# 6C. Export metadata with group labels
# --------------------------------------------------------------------------
# Include geo_accession (row ID) and group label.
# Add binary AD labels for convenience (Python model training).
metadata_120584_export <- metadata_120584 %>%
  select(geo_accession, group) %>%
  mutate(
    label_AD     = as.integer(group == "Alzheimer's Disease"),
    label_binary = as.integer(group != "Control")   # AD+MCI vs Control
  )

write.csv(metadata_120584_export,
          "data/processed/GSE120584_metadata_harmonized.csv",
          row.names = FALSE)

# For GSE46579, parse and export group labels similarly.
# Ensure group column exists; adapt column names to match Week 2 output.
if (!"group" %in% colnames(metadata_46579)) {
  # Attempt to parse from characteristics if not already clean
  cat("'group' column not found in metadata_46579; attempting to parse.\n")
  if ("characteristics_ch1" %in% colnames(metadata_46579)) {
    metadata_46579$group <- trimws(gsub(".*: ", "", metadata_46579$characteristics_ch1))
  }
}

# Standardize group labels to match GSE120584 convention
ad_label_pattern    <- "Alzheimer|alzheimer|AD$"
ctrl_label_pattern  <- "Control|control|normal|Normal"

metadata_46579 <- metadata_46579 %>%
  mutate(
    group_std = case_when(
      grepl(ad_label_pattern, group, ignore.case = TRUE)  ~ "Alzheimer's Disease",
      grepl(ctrl_label_pattern, group, ignore.case = TRUE) ~ "Control",
      TRUE ~ as.character(group)
    )
  )

metadata_46579_export <- metadata_46579 %>%
  select(geo_accession, group_std) %>%
  rename(group = group_std) %>%
  mutate(label_AD = as.integer(group == "Alzheimer's Disease"))

write.csv(metadata_46579_export,
          "data/processed/GSE46579_metadata_harmonized.csv",
          row.names = FALSE)

cat("\nMetadata files exported.\n")
cat("Files written to data/processed/:\n")
cat("  GSE120584_harmonized.csv              — z-scored expression (samples × miRNAs)\n")
cat("  GSE120584_metadata_harmonized.csv     — sample IDs and group labels\n")
cat("  GSE46579_harmonized.csv               — z-scored expression (samples × miRNAs)\n")
cat("  GSE46579_metadata_harmonized.csv      — sample IDs and group labels\n")


# ==============================================================================
# SECTION 7: DeLong AUC Comparison
# ==============================================================================
# RATIONALE:
# After running the Python ML pipeline (Lab 5B), we compare the training CV AUC
# to the external validation AUC. A simple point comparison ("0.88 vs 0.73")
# is insufficient — we need a statistical test that accounts for uncertainty.
#
# DeLong's method (DeLong ER et al., 1988 Biometrics) is the standard approach.
# It computes the variance-covariance structure of two ROC curves and tests
# whether their AUCs are equal. It is implemented in the pROC package.
#
# NOTE: For two NON-OVERLAPPING cohorts (our case — different patients in
# training and validation), the two ROC curves are independent, and a standard
# z-test on the difference in AUC values (with bootstrap CIs) is also valid.
# pROC's roc.test(method="bootstrap") handles this case.
# Use method="delong" when the same patients are evaluated by two different models.

cat("\n=== SECTION 7: DeLong AUC Comparison ===\n")
cat("Reading predicted probabilities from Python ML output...\n")

# These files are written by the Python Lab 5B script.
# If you haven't run Lab 5B yet, the files won't exist.
# We use tryCatch to handle this gracefully.

training_cv_pred_file <- "results/training_cv_predictions.csv"
validation_pred_file  <- "results/validation_predictions.csv"

load_preds_success <- tryCatch({
  pred_train_cv <- read.csv(training_cv_pred_file)
  pred_val_ext  <- read.csv(validation_pred_file)
  cat("Prediction files loaded successfully.\n")
  TRUE
}, error = function(e) {
  cat("Prediction files not found:", conditionMessage(e), "\n")
  cat("Complete Lab 5B Python analysis first, then re-run this section.\n")
  cat("Expected file format: columns = [sample_id, true_label, prob_AD]\n")
  FALSE
})

if (load_preds_success) {

  # --------------------------------------------------------------------------
  # 7A. Build ROC objects
  # --------------------------------------------------------------------------
  # pred_train_cv should have: true_label (0=Control, 1=AD), prob_AD
  # pred_val_ext  should have: true_label (0=Control, 1=AD), prob_AD

  roc_train <- roc(
    response  = pred_train_cv$true_label,
    predictor = pred_train_cv$prob_AD,
    direction = "<",   # lower prob → Control
    quiet     = TRUE
  )

  roc_val <- roc(
    response  = pred_val_ext$true_label,
    predictor = pred_val_ext$prob_AD,
    direction = "<",
    quiet     = TRUE
  )

  cat("\nTraining CV ROC:\n")
  print(roc_train)
  cat("\nExternal Validation ROC:\n")
  print(roc_val)

  # --------------------------------------------------------------------------
  # 7B. Confidence intervals for each AUC (bootstrap, 2000 resamples)
  # --------------------------------------------------------------------------
  cat("\nComputing bootstrap confidence intervals (2000 resamples)...\n")

  ci_train <- ci.auc(roc_train, method = "bootstrap", boot.n = 2000, conf.level = 0.95)
  ci_val   <- ci.auc(roc_val,   method = "bootstrap", boot.n = 2000, conf.level = 0.95)

  cat("Training CV AUC:           ", round(auc(roc_train), 4),
      " (95% CI:", round(ci_train[1], 4), "–", round(ci_train[3], 4), ")\n")
  cat("External Validation AUC:   ", round(auc(roc_val),   4),
      " (95% CI:", round(ci_val[1], 4), "–", round(ci_val[3], 4), ")\n")
  cat("AUC gap:                   ",
      round(auc(roc_train) - auc(roc_val), 4), "\n")

  # --------------------------------------------------------------------------
  # 7C. DeLong / bootstrap test for AUC comparison
  # --------------------------------------------------------------------------
  # For INDEPENDENT cohorts (different patients), use method = "bootstrap"
  # For paired comparisons (same patients, two models), use method = "delong"
  # We use bootstrap here because our two cohorts are independent.

  delong_test <- roc.test(
    roc1     = roc_train,
    roc2     = roc_val,
    method   = "bootstrap",
    boot.n   = 2000,
    alternative = "greater",   # one-sided: is training AUC > validation AUC?
    paired   = FALSE           # independent cohorts
  )

  cat("\nBootstrap AUC comparison test:\n")
  cat("  H0: Training CV AUC = External Validation AUC\n")
  cat("  H1: Training CV AUC > External Validation AUC (one-sided)\n")
  print(delong_test)
  cat("\nInterpretation:\n")
  if (delong_test$p.value < 0.05) {
    cat("  p < 0.05: Training CV AUC is significantly higher than external validation AUC.\n")
    cat("  This indicates the model's performance degrades when applied to the independent cohort.\n")
    cat("  Likely causes: platform differences, cohort heterogeneity, or overfitting.\n")
  } else {
    cat("  p >= 0.05: Cannot conclude significant difference between training and validation AUC.\n")
    cat("  The model generalizes comparably to the external cohort.\n")
  }

  # --------------------------------------------------------------------------
  # 7D. ROC curve comparison plot
  # --------------------------------------------------------------------------
  png("results/roc_comparison_training_vs_validation.png",
      width = 700, height = 600, res = 120)

  plot(roc_train,
       col   = "#4575B4",
       lwd   = 2.5,
       main  = "ROC Curve Comparison\nTraining CV vs External Validation",
       print.auc = FALSE)

  plot(roc_val,
       col   = "#D73027",
       lwd   = 2.5,
       add   = TRUE)

  legend("bottomright",
         legend = c(
           paste0("Training CV (AUC = ", round(auc(roc_train), 3), ")"),
           paste0("External Validation (AUC = ", round(auc(roc_val), 3), ")")
         ),
         col    = c("#4575B4", "#D73027"),
         lwd    = 2.5,
         bty    = "n",
         cex    = 0.9)

  dev.off()
  cat("ROC comparison plot saved to results/roc_comparison_training_vs_validation.png\n")

} else {
  cat("Skipping DeLong test — prediction files not available.\n")
  cat("Run Lab 5B (Python) and save predictions, then re-run Section 7.\n")
}


# ==============================================================================
# SECTION 8: Calibration Plot
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

cat("\n=== SECTION 8: Calibration Plot ===\n")

if (load_preds_success) {

  # --------------------------------------------------------------------------
  # 8A. Manual calibration plot using probability binning
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

    bin_idx   <- cut(pred_probs, breaks = breaks, include.lowest = TRUE,
                     labels = FALSE)
    bin_df    <- data.frame(pred = pred_probs, true = true_labels, bin = bin_idx)

    calib_df <- bin_df %>%
      group_by(bin) %>%
      summarise(
        n           = n(),
        mean_pred   = mean(pred),
        observed_rate = mean(true),
        se          = sqrt(observed_rate * (1 - observed_rate) / n),
        .groups     = "drop"
      ) %>%
      filter(!is.na(bin))

    return(calib_df)
  }

  calib_data <- calibration_plot_data(
    true_labels = pred_val_ext$true_label,
    pred_probs  = pred_val_ext$prob_AD,
    n_bins      = 10,
    method      = "quantile"
  )

  # Plot
  p_calib <- ggplot(calib_data, aes(x = mean_pred, y = observed_rate)) +
    geom_abline(intercept = 0, slope = 1,
                linetype = "dashed", colour = "grey40", linewidth = 0.8) +
    geom_point(aes(size = n), colour = "#4575B4", alpha = 0.8) +
    geom_errorbar(
      aes(ymin = observed_rate - 1.96 * se,
          ymax = observed_rate + 1.96 * se),
      width = 0.02, colour = "#4575B4", alpha = 0.7
    ) +
    geom_smooth(method = "loess", se = TRUE, colour = "#D73027",
                fill = "#D73027", alpha = 0.15, linewidth = 1) +
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
  ggsave("results/calibration_plot_external.png", p_calib,
         width = 6, height = 6, dpi = 150)
  cat("Calibration plot saved to results/calibration_plot_external.png\n")

  # --------------------------------------------------------------------------
  # 8B. Quantitative calibration metrics
  # --------------------------------------------------------------------------
  # Brier score: mean squared error of probability predictions
  # Range: 0 (perfect) to 0.25 (uninformative — equivalent to always predicting 0.5)
  # A "no-information" model predicting prevalence for all samples has Brier = prev*(1-prev)

  brier_score <- mean((pred_val_ext$true_label - pred_val_ext$prob_AD)^2)
  prevalence  <- mean(pred_val_ext$true_label)
  brier_null  <- prevalence * (1 - prevalence)  # null model (always predict prevalence)
  brier_scaled <- 1 - brier_score / brier_null   # scaled Brier (0 = null, 1 = perfect)

  cat("\nCalibration Statistics (External Validation):\n")
  cat("  Brier Score:        ", round(brier_score, 4),
      " (lower is better; 0 = perfect)\n")
  cat("  Brier Score (null): ", round(brier_null, 4), "\n")
  cat("  Scaled Brier Score: ", round(brier_scaled, 4),
      " (1 = perfect; 0 = null model; can be negative if worse than null)\n")

  # --------------------------------------------------------------------------
  # 8C. Optional: rms::val.prob for detailed calibration statistics
  # --------------------------------------------------------------------------
  if (requireNamespace("rms", quietly = TRUE)) {
    library(rms)
    cat("\nrms package available — computing detailed calibration statistics.\n")
    cal_rms <- val.prob(
      p      = pred_val_ext$prob_AD,
      y      = pred_val_ext$true_label,
      pl     = TRUE,           # produce calibration plot within rms
      logistic.cal = TRUE,     # overlay logistic calibration line
      main   = "val.prob Calibration (External Validation)"
    )
    cat("Calibration statistics from rms::val.prob:\n")
    print(cal_rms)
  } else {
    cat("rms package not available. Install with install.packages('rms') for\n")
    cat("additional calibration statistics (E50, E90, Emax, Hosmer-Lemeshow test).\n")
  }

} else {
  cat("Skipping calibration plot — prediction files not available.\n")
}


# ==============================================================================
# SECTION 9: Model Results Summary Table
# ==============================================================================
# This section compiles a publication-ready summary of all ML model results.
# The table includes: model name, training CV AUC ± 95%CI, external validation
# AUC ± 95%CI, sensitivity, specificity, PPV, NPV at optimal threshold.
#
# We read all model prediction files from the results/ directory.
# If multiple models were run in Python (XGBoost, Random Forest, Logistic
# Regression), each should have its own predictions file.

cat("\n=== SECTION 9: Model Results Summary Table ===\n")

# Function to compute performance metrics from predictions
compute_metrics <- function(true_labels, pred_probs, threshold = NULL) {

  # AUC with bootstrap CI
  roc_obj  <- roc(true_labels, pred_probs, direction = "<", quiet = TRUE)
  auc_val  <- as.numeric(auc(roc_obj))
  ci_boot  <- tryCatch(
    as.numeric(ci.auc(roc_obj, method = "bootstrap", boot.n = 1000)),
    error = function(e) c(NA, auc_val, NA)
  )

  # Optimal threshold by Youden index if not specified
  if (is.null(threshold)) {
    coords_df <- coords(roc_obj, "best", best.method = "youden",
                        ret = c("threshold", "sensitivity", "specificity"))
    threshold <- coords_df$threshold[1]
    sens      <- coords_df$sensitivity[1]
    spec      <- coords_df$specificity[1]
  } else {
    pred_class <- as.integer(pred_probs >= threshold)
    TP <- sum(pred_class == 1 & true_labels == 1)
    TN <- sum(pred_class == 0 & true_labels == 0)
    FP <- sum(pred_class == 1 & true_labels == 0)
    FN <- sum(pred_class == 0 & true_labels == 1)
    sens <- TP / (TP + FN)
    spec <- TN / (TN + FP)
  }

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

# --------------------------------------------------------------------------
# 9A. Compile results for all available models
# --------------------------------------------------------------------------
# List of models to include in the summary table.
# File naming convention: results/{model_name}_{cohort}_predictions.csv
# Each file has columns: true_label, prob_AD

model_names       <- c("XGBoost", "RandomForest", "LogisticRegression")
results_rows      <- list()

for (model in model_names) {

  train_file <- file.path("results",
                           paste0(tolower(gsub(" ", "", model)),
                                  "_training_cv_predictions.csv"))
  val_file   <- file.path("results",
                           paste0(tolower(gsub(" ", "", model)),
                                  "_validation_predictions.csv"))

  # Check for generic filenames if model-specific not found
  if (!file.exists(train_file)) {
    train_file <- "results/training_cv_predictions.csv"
  }
  if (!file.exists(val_file)) {
    val_file <- "results/validation_predictions.csv"
  }

  if (file.exists(train_file) && file.exists(val_file)) {

    pred_tr <- read.csv(train_file)
    pred_va <- read.csv(val_file)

    metrics_tr <- compute_metrics(pred_tr$true_label, pred_tr$prob_AD)
    metrics_va <- compute_metrics(pred_va$true_label, pred_va$prob_AD)

    results_rows[[model]] <- data.frame(
      Model              = model,
      Training_CV_AUC    = metrics_tr$auc_str,
      Validation_AUC     = metrics_va$auc_str,
      AUC_Gap            = round(metrics_tr$auc - metrics_va$auc, 4),
      Optimal_Threshold  = metrics_tr$threshold,  # threshold from training CV
      Sensitivity        = metrics_va$sens,
      Specificity        = metrics_va$spec,
      PPV                = metrics_va$ppv,
      NPV                = metrics_va$npv,
      stringsAsFactors   = FALSE
    )

    cat(sprintf("%-20s: Training AUC = %s | Validation AUC = %s | Gap = %.4f\n",
                model, metrics_tr$auc_str, metrics_va$auc_str,
                metrics_tr$auc - metrics_va$auc))

  } else {
    cat(sprintf("%-20s: Prediction files not found — skipping.\n", model))
  }
}

if (length(results_rows) > 0) {
  results_table <- do.call(rbind, results_rows)
  rownames(results_table) <- NULL

  cat("\n=== COMPLETE MODEL RESULTS SUMMARY ===\n")
  print(results_table)

  # Save as CSV
  write.csv(results_table,
            "results/model_results_summary.csv",
            row.names = FALSE)
  cat("\nSummary table saved to results/model_results_summary.csv\n")

  # Print formatted table (use knitr::kable if available for nicer output)
  if (requireNamespace("knitr", quietly = TRUE)) {
    cat("\nFormatted table (knitr):\n")
    print(knitr::kable(results_table, format = "simple", align = "c"))
  }

} else {
  cat("No prediction files found. Complete Lab 5B (Python) and then re-run.\n")
  cat("Once Python saves predictions to results/, this section will populate.\n")

  # Create a template table for documentation purposes
  template_table <- data.frame(
    Model             = c("XGBoost", "RandomForest", "LogisticRegression"),
    Training_CV_AUC   = c("[run Lab 5B]", "[run Lab 5B]", "[run Lab 5B]"),
    Validation_AUC    = c("[run Lab 5B]", "[run Lab 5B]", "[run Lab 5B]"),
    AUC_Gap           = c(NA, NA, NA),
    Sensitivity       = c(NA, NA, NA),
    Specificity       = c(NA, NA, NA),
    stringsAsFactors  = FALSE
  )

  write.csv(template_table,
            "results/model_results_summary_TEMPLATE.csv",
            row.names = FALSE)
  cat("Template table written to results/model_results_summary_TEMPLATE.csv\n")
}


# ==============================================================================
# SECTION 10: Save Harmonized Data Objects
# ==============================================================================
# Save all harmonized data objects as .rds files for use in downstream analyses
# and reproducibility. These are the "clean, harmonized" versions that include
# only the intersection features, z-scored, and with v22 miRNA names.

cat("\n=== SECTION 10: Save Harmonized Data Objects ===\n")

# Main harmonized expression matrices (z-scored, intersection features only)
saveRDS(expr_120584_z,
        "data/processed/GSE120584_expr_harmonized_zscore.rds")
saveRDS(expr_46579_z,
        "data/processed/GSE46579_expr_harmonized_zscore.rds")

# Name mapping tables (useful for tracing back to original names)
saveRDS(conversion_120584,
        "data/processed/GSE120584_mirbase_conversion.rds")
saveRDS(conversion_46579,
        "data/processed/GSE46579_mirbase_conversion.rds")

# Intersection feature list (with both MIMAT accessions and v22 names)
intersection_info <- data.frame(
  MIMAT_accession = common_features,
  v22_name        = rownames(expr_120584_z),  # after back-conversion in Section 4C
  present_in_GSE120584 = TRUE,
  present_in_GSE46579  = TRUE
)

write.csv(intersection_info,
          "data/processed/feature_intersection.csv",
          row.names = FALSE)
saveRDS(intersection_info,
        "data/processed/feature_intersection.rds")

cat("Harmonized data objects saved:\n")
cat("  data/processed/GSE120584_expr_harmonized_zscore.rds\n")
cat("  data/processed/GSE46579_expr_harmonized_zscore.rds\n")
cat("  data/processed/GSE120584_mirbase_conversion.rds\n")
cat("  data/processed/GSE46579_mirbase_conversion.rds\n")
cat("  data/processed/feature_intersection.csv\n")


# ==============================================================================
# SECTION 11: Session Summary
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

cat("\n--- Files Written This Session ---\n")
cat("data/processed/:\n")
cat("  GSE120584_harmonized.csv              (samples × features, z-scored)\n")
cat("  GSE120584_metadata_harmonized.csv     (sample IDs + group labels)\n")
cat("  GSE46579_harmonized.csv               (samples × features, z-scored)\n")
cat("  GSE46579_metadata_harmonized.csv      (sample IDs + group labels)\n")
cat("  GSE120584_expr_harmonized_zscore.rds  (R matrix object)\n")
cat("  GSE46579_expr_harmonized_zscore.rds   (R matrix object)\n")
cat("  feature_intersection.csv              (shared miRNAs with MIMAT and v22 names)\n")
cat("qc_reports/:\n")
cat("  pca_before_after_zscore.png           (QC plot)\n")

if (load_preds_success) {
  cat("results/:\n")
  cat("  roc_comparison_training_vs_validation.png\n")
  cat("  calibration_plot_external.png\n")
  cat("  model_results_summary.csv\n")
}

cat("\n--- Next Steps ---\n")
cat("1. Open Week5_AdvancedML_Validation.md for the full Lab 5B Python instructions.\n")
cat("2. In Python, load data/processed/GSE120584_harmonized.csv (training).\n")
cat("3. Run nested CV (Section 5.2.4) and XGBoost (Section 5.3).\n")
cat("4. Compute SHAP values (Section 5.4) — beeswarm and force plots.\n")
cat("5. Apply trained model to GSE46579_harmonized.csv (external validation).\n")
cat("6. Save prediction files to results/ and re-run Sections 7–9 of this script.\n")
cat("7. Interpret the AUC gap and biological plausibility of top miRNAs.\n")

cat("\n--- Key Interpretation Reminders ---\n")
cat("AUC gap interpretation:\n")
cat("  < 0.05:  Excellent generalizability\n")
cat("  0.05-0.10: Acceptable; some platform effects\n")
cat("  0.10-0.20: Moderate degradation; investigate top features\n")
cat("  > 0.20:  Severe degradation; model may have overfit\n\n")
cat("If AUC gap is large: examine which top SHAP features are absent from the\n")
cat("  intersection. If key features are missing from GSE46579, the model could\n")
cat("  not express their full predictive capacity in validation.\n")

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
cat("  Now proceed to Lab 5B (Python) to complete the ML analysis.\n")
cat("================================================================================\n")
