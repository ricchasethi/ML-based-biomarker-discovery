################################################################################
# AI/ML in Biomarker Discovery — Week 2 Lab
# Title:   Data Acquisition & Quality Control
# Disease: Alzheimer's Disease | Biomarker: miRNA
# Audience: Wet-lab biologists — basic R from Week 1 assumed
#
# Learning Goals for This Script:
#   1. Build a reproducible project directory structure
#   2. Download GEO datasets programmatically using GEOquery
#   3. Extract and parse sample metadata from GEO records
#   4. Run a QC pipeline on processed miRNA microarray data (GSE120584)
#   5. Run a QC pipeline on small RNA-seq count data (GSE46579)
#   6. Normalise both data types (check submitter normalisation; DESeq2 VST / TMM)
#   7. Detect and correct batch effects (ComBat / limma)
#   8. Detect hemolysis in blood-based miRNA data
#   9. Save clean, analysis-ready expression matrices for Week 3
#
# Datasets:
#   GSE120584 — Serum miRNA microarray (Toray 3D-Gene), 1,601 samples,
#               AD / MCI / NC / VaD / DLB — we use AD, MCI, NC      [PRIMARY]
#   GSE46579  — Whole-blood small RNA-seq (Illumina HiSeq 2000),
#               AD / Control                                         [VALIDATION]
#
# Run each section with Ctrl+Enter (Windows/Linux) or Cmd+Enter (Mac).
################################################################################


# ==============================================================================
# SECTION 1: Project Directory Setup
# ==============================================================================
# Good data science starts with an organised folder structure.
# Create this once; it persists for the entire 6-week course.
#
# Resulting structure:
#   data/
#     raw/        — everything downloaded from GEO, never modified
#     processed/  — clean matrices output from this script
#   qc_reports/   — QC plots and sample exclusion logs
#   results/      — outputs from Weeks 3–6

dirs <- c("data/raw", "data/processed", "qc_reports", "results")
for (d in dirs) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}
cat("Project directories ready.\n")

# Set your working directory to the course folder if not already there.
# Replace the path below with your actual course folder path.
# setwd("/path/to/your/AI_ML_Biomarker_Discovery")
getwd()  # Confirm current location


# ==============================================================================
# SECTION 2: Load All Packages
# ==============================================================================
# If any library() call fails, return to Week 1 Section 3 and reinstall.

suppressPackageStartupMessages({
  # Bioconductor
  library(GEOquery)          # GEO data download
  library(limma)             # Microarray normalisation & DE, removeBatchEffect
  library(DESeq2)            # RNA-seq count normalisation (VST) & DE
  library(edgeR)             # TMM normalisation, filterByExpr
  library(sva)               # ComBat batch correction

  # CRAN
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(pheatmap)
  library(RColorBrewer)
  library(gridExtra)
  library(readr)
  library(readxl)            # Read Excel files (GSE46579 count table)
})

cat("All packages loaded.\n")

# Colour palette used consistently for group labels throughout this script
GROUP_COLOURS <- c(
  "Control"                  = "#4575B4",   # blue
  "Mild Cognitive Impairment" = "#FEE090",  # amber
  "Alzheimer's Disease"      = "#D73027"    # red
)

# NULL-coalescing helper: a %||% b returns b when a is NULL
# (used in the QC logs when a hemolysis index could not be computed)
`%||%` <- function(a, b) if (!is.null(a)) a else b

# Jupyter: IRkernel's default 7x7in figure squashes per-sample plots.
# Widen it once here for the whole notebook (no effect in RStudio).
options(repr.plot.width = 14, repr.plot.height = 6)


# ==============================================================================
# SECTION 3: Download GSE120584 (Primary Microarray Dataset)
# ==============================================================================
# GSE120584: Serum miRNA expression in dementia (Shigemizu et al., 2019, Commun Biol).
# Platform: Toray 3D-Gene Human miRNA Oligo Chip (GPL21263) — a microarray
# Samples: 1,601 — AD (1,021), MCI (32), NC = normal control (288), VaD (91), DLB (169)
#
# For this microarray dataset, getGEO() gives us everything we need:
#   exprs(gse120584)  → the submitters' normalised log2 expression matrix
#   pData(gse120584)  → sample metadata
# (Raw per-sample scanner files in GSE120584_RAW.tar are not needed.)

cat("Downloading GSE120584 from GEO...\n")
cat("This may take 1–3 minutes depending on your internet speed.\n\n")

gse120584_list <- getGEO(
  "GSE120584",
  destdir    = "data/raw/",
  GSEMatrix  = TRUE,
  AnnotGPL   = TRUE
)

# GEOquery returns a list; one element per platform (GPL)
cat("Number of platforms in GSE120584:", length(gse120584_list), "\n")
gse120584 <- gse120584_list[[1]]   # extract the ExpressionSet
class(gse120584)                   # should be "ExpressionSet"

# The ExpressionSet has three linked compartments:
#   exprs(gse120584)  — normalised log2 expression matrix (miRNAs × samples)
#   pData(gse120584)  — phenotype data: clinical and technical metadata
#   fData(gse120584)  — feature data: miRNA probe annotations

cat("\nDimensions of expression slot (probes × samples):\n")
print(dim(exprs(gse120584)))


# ==============================================================================
# SECTION 4: Extract and Parse Sample Metadata (GSE120584)
# ==============================================================================
# GEO metadata is stored as free-text key:value pairs in "characteristics_ch1"
# columns. We need to parse these into clean, typed R variables.

metadata_raw <- pData(gse120584)

# See all available metadata column names
cat("Available metadata columns:\n")
print(colnames(metadata_raw))

# Inspect the characteristics columns that hold clinical information
cat("\nUnique values in characteristics_ch1:\n")
print(unique(metadata_raw$characteristics_ch1))

# ---- 4A. Parse group label ----
# Format in GEO: "diagnosis: AD" — this cohort has five diagnoses.
# This course compares AD, MCI and cognitively normal controls (NC);
# VaD (vascular dementia) and DLB (dementia with Lewy bodies) are set aside.

metadata <- metadata_raw
metadata$diagnosis <- trimws(gsub("diagnosis: ", "", metadata$characteristics_ch1))
print(table(metadata$diagnosis))

metadata <- metadata[metadata$diagnosis %in% c("NC", "MCI", "AD"), ]

# Subset the ExpressionSet itself, so exprs(gse120584) and pData(gse120584)
# hold ONLY the AD, MCI and NC samples from here on (columns = samples).
gse120584 <- gse120584[, metadata$geo_accession]
cat("\nAfter keeping AD / MCI / NC only:\n")
cat("  Samples in ExpressionSet:", ncol(gse120584), "\n")
cat("  Samples in metadata:     ", nrow(metadata), "\n")
print(table(pData(gse120584)$characteristics_ch1))

# Rename to the labels used in every later week (and in GROUP_COLOURS)
group_labels <- c(NC  = "Control",
                  MCI = "Mild Cognitive Impairment",
                  AD  = "Alzheimer's Disease")
metadata$group <- factor(group_labels[metadata$diagnosis], levels = group_labels)
unique(metadata$group)

# ---- 4B. Parse age and sex ----
# Adjust the column name and prefix pattern to match what GEO actually provides.
# Run unique(metadata$characteristics_ch1.1) to inspect first.

if ("characteristics_ch1.1" %in% colnames(metadata)) {
  metadata$age <- as.numeric(gsub("age: ", "", metadata$characteristics_ch1.1))
}
if ("characteristics_ch1.2" %in% colnames(metadata)) {
  metadata$sex <- trimws(gsub("Sex: |sex: ", "", metadata$characteristics_ch1.2))
}

# ---- 4C. Cohort summary ----
cat("\n=== Cohort Summary: GSE120584 ===\n")
print(table(metadata$group))
if ("sex" %in% colnames(metadata)) {
  cat("\nSex distribution per group:\n")
  print(table(metadata$sex, metadata$group))
}
if ("age" %in% colnames(metadata)) {
  cat("\nAge summary:\n")
  print(tapply(metadata$age, metadata$group, function(x) {
    c(N = sum(!is.na(x)), Mean = round(mean(x, na.rm = TRUE), 1),
      SD = round(sd(x, na.rm = TRUE), 1))
  }))
}

# BIOLOGICAL CHECK:
# Expected for a blood-based AD cohort:
#   Age: mean ~70–80 years; very few samples < 60
#   Sex: ~55–65% female (reflects AD demographic)
# Anything outside these ranges = likely metadata parsing error.
# This cohort is NOT balanced: ~1,021 AD vs 288 Control vs only 32 MCI, and
# controls are younger than AD patients (~72 vs ~79 years) — a confounder
# to watch in Week 3. In Weeks 4–5 judge models with balanced metrics
# (AUC, sensitivity/specificity), not raw accuracy.


# ==============================================================================
# SECTION 5: Extract the Expression Matrix (GSE120584)
# ==============================================================================
# getGEO() already loaded the submitters' processed matrix: exprs(gse120584).
# Values are background-subtracted, log2-transformed and normalised to three
# internal-control miRNAs (miR-149-3p, miR-2861, miR-4463) — there are no
# read counts in a microarray.
#
# Rows are miRBase accessions (MIMAT…); we replace them with readable miRNA
# names from fData(gse120584). The ExpressionSet was already reduced to the
# AD / MCI / Control samples in Section 4A.

expr_raw <- exprs(gse120584)
cat("Expression matrix (miRNAs × samples):", dim(expr_raw), "\n")
cat("Value range (log2):", round(range(expr_raw), 2), "\n")
cat("Missing values:", sum(is.na(expr_raw)), "\n")

# Rows are MIMAT accessions — swap in readable miRNA names from the feature data
feature_data <- fData(gse120584)
print(head(feature_data[, c("ID", "miRNA_ID_LIST")]))
stopifnot(identical(rownames(expr_raw), feature_data$ID))
rownames(expr_raw) <- feature_data$miRNA_ID_LIST
# A few probes detect several near-identical miRNAs,
# e.g. "hsa-miR-199a-3p, hsa-miR-199b-3p" — the name lists all of them.

# Make sure the columns are in the same order as the metadata rows.
# If they don't align, all downstream analyses will be wrong.
expr_raw <- expr_raw[, metadata$geo_accession]

cat("\nSamples in expression matrix:", ncol(expr_raw), "\n")
cat("Samples in metadata:         ", nrow(metadata), "\n")
cat("Order matches:", all(colnames(expr_raw) == metadata$geo_accession), "\n")
cat("\nPreview (first 5 miRNAs, first 4 samples):\n")
print(round(expr_raw[1:5, 1:4], 2))


# ==============================================================================
# SECTION 6: Microarray Quality Control (GSE120584)
# ==============================================================================
# We assess data quality before any analysis:
#   A. Signal distribution — do arrays have similar overall intensity?
#   B. Detected miRNA count — how many miRNAs rise above background per sample?
#   C. Detection by group — does one group systematically detect more miRNAs?
#
# How "not detected" is coded: the submitters replaced below-background signals
# with a floor value. In each sample, the lowest value is therefore the
# "not detected" floor, and many miRNAs sit exactly on it.

# ---- 6A. Signal Distribution per Array ----
group_colours_vec <- GROUP_COLOURS[as.character(metadata$group)]
ord <- order(metadata$group)   # plot samples grouped by diagnosis

par(mar = c(3, 5, 4, 2))
boxplot(
  expr_raw[, ord],
  main    = "Normalised log2 Expression per Array (GSE120584)",
  ylab    = "log2 expression",
  col     = group_colours_vec[ord],
  border  = group_colours_vec[ord],
  xaxt    = "n",               # >1,000 sample names are unreadable — hide them
  outline = FALSE
)
legend("topright", legend = names(GROUP_COLOURS),
       fill = GROUP_COLOURS, bty = "n", cex = 0.85)

median_signal <- apply(expr_raw, 2, median)
cat("\nPer-array median log2 expression:\n")
print(summary(median_signal))
# Medians are low because most miRNAs are not detected in serum —
# the median is often the "not detected" floor. Section 8 re-checks this
# after filtering.

# ---- 6B. Detected miRNAs per Sample ----
# A miRNA is "detected" in a sample if it is above that sample's floor (its minimum)
sample_floor <- apply(expr_raw, 2, min)
detected     <- sweep(expr_raw, 2, sample_floor + 0.01, ">")   # TRUE / FALSE matrix
detected_per_sample <- colSums(detected)

cat("Detected miRNAs per sample:\n")
print(summary(detected_per_sample))

# Flag samples far below the typical detection level (median − 3 MAD)
low_detect_cut  <- median(detected_per_sample) - 3 * mad(detected_per_sample)
low_detect_flag <- detected_per_sample < low_detect_cut

hist(detected_per_sample, breaks = 50, col = "#4575B4", border = "white",
     main = "Detected miRNAs per Sample (GSE120584)",
     xlab = "Number of miRNAs above background")
abline(v = low_detect_cut, col = "red", lty = 2, lwd = 1.5)

cat("\nSamples below", round(low_detect_cut), "detected miRNAs:", sum(low_detect_flag), "\n")

# ---- 6C. Detection by Group ----
par(mar = c(5, 5, 4, 2))
boxplot(
  detected_per_sample ~ metadata$group,
  main = "Detected miRNAs per Sample, by Group",
  xlab = NULL,
  ylab = "Number of miRNAs above background",
  col  = GROUP_COLOURS
)

cat("Median detected miRNAs per group:\n")
print(tapply(detected_per_sample, metadata$group, median))

# What to look for:
#   Groups should detect similar numbers of miRNAs.
#   A large, systematic difference between groups is more likely technical
#   (sample handling, storage time, array batch) than biological — and it can
#   masquerade as "disease signal" later. Note it and revisit it in Section 13.


# ==============================================================================
# SECTION 7: Low-Expression Filtering (GSE120584)
# ==============================================================================
# miRNAs that sit below background in most samples contribute only noise.
# We keep a miRNA if it is detected in at least 80% of the samples of
# at least one group. Checking group by group (like edgeR's filterByExpr()
# for RNA-seq, Section 18) avoids throwing away miRNAs that are switched on
# in only one diagnosis.

min_detect_rate <- 0.80   # fraction of a group's samples that must detect the miRNA

detect_rate_by_group <- sapply(levels(metadata$group), function(g)
  rowMeans(detected[, metadata$group == g, drop = FALSE]))

keep <- apply(detect_rate_by_group >= min_detect_rate, 1, any)

cat("miRNAs before filtering: ", nrow(expr_raw), "\n")
expr_filtered <- expr_raw[keep, ]
cat("miRNAs after filtering:  ", nrow(expr_filtered), "\n")
cat("Removed:                 ", sum(!keep), "low-expressed miRNAs\n")

# RULE: Never filter based on differential expression (p-value or fold change).
# Filtering must be based on expression level and detection rate only.
# Filtering by DE would introduce selection bias that inflates false positives.


# ==============================================================================
# SECTION 8: Normalisation (GSE120584)
# ==============================================================================
# The GEO matrix is already normalised by the submitters:
#   1. Background subtraction and log2 transform
#   2. Scaling each array to the mean of three internal-control miRNAs
#      (miR-149-3p, miR-2861, miR-4463) — like normalising to GAPDH in RT-qPCR
# Our job is to check that the normalisation worked, and to decide whether an
# extra between-array normalisation is needed.

# ---- 8A. Check the Submitters' Normalisation ----
# After good normalisation, the distributions of the detected (filtered) miRNAs
# should look similar across arrays and must not differ systematically by group.

median_filtered <- apply(expr_filtered, 2, median)
cat("Per-array median log2 expression (filtered miRNAs):\n")
print(summary(median_filtered))
cat("\nMedian by group:\n")
print(round(tapply(median_filtered, metadata$group, median), 2))

par(mar = c(3, 5, 4, 2))
boxplot(
  expr_filtered[, ord],
  main    = "Filtered miRNAs: log2 Expression per Array",
  ylab    = "log2 expression",
  col     = group_colours_vec[ord],
  border  = group_colours_vec[ord],
  xaxt    = "n",
  outline = FALSE
)
legend("topright", legend = names(GROUP_COLOURS),
       fill = GROUP_COLOURS, bty = "n", cex = 0.85)
# Boxes should be roughly aligned. A block of shifted boxes = possible batch effect.

# ---- 8B. Optional — Quantile Normalisation (limma) ----
# Quantile normalisation forces every array to have the same distribution.
# Use it only if 8A shows arrays with clearly shifted distributions:
# it can also erase genuine global differences between groups.

apply_quantile <- FALSE

if (apply_quantile) {
  expr_norm <- normalizeBetweenArrays(expr_filtered, method = "quantile")
  cat("Quantile normalisation applied.\n")
} else {
  expr_norm <- expr_filtered
  cat("Using the submitters' normalisation as provided.\n")
}

cat("Normalised matrix:", nrow(expr_norm), "miRNAs ×", ncol(expr_norm), "samples\n")


# ==============================================================================
# SECTION 9: Sample-to-Sample Correlation Heatmap (GSE120584)
# ==============================================================================
# After normalisation, pairwise Pearson correlations between samples
# should be high (> 0.90) and ideally cluster by biological group.
# A sample with uniformly low correlation to all others is a QC failure:
# we flag samples whose median correlation with the others is below 0.90.

cor_matrix <- cor(expr_norm, method = "pearson")

# Typical agreement of each sample with all the other samples
median_cor   <- apply(cor_matrix, 2, function(x) median(x[x < 1]))
low_cor_cut  <- 0.90
low_cor_flag <- median_cor < low_cor_cut

cat("Median correlation of each sample with the others:\n")
print(summary(median_cor))
cat("Samples with unusually low correlation (<", round(low_cor_cut, 3), "):",
    sum(low_cor_flag), "\n")

# Annotation sidebar showing group and sex
annotation_col <- data.frame(
  Group = metadata$group,
  row.names = colnames(cor_matrix)
)
if ("sex" %in% colnames(metadata)) {
  annotation_col$Sex <- metadata$sex
}

ann_colours <- list(Group = GROUP_COLOURS)

# >1,000 samples: the heatmap is written to a file (takes ~1 minute)
pheatmap(
  cor_matrix,
  annotation_col    = annotation_col,
  annotation_colors = ann_colours,
  color             = colorRampPalette(rev(brewer.pal(9, "RdBu")))(100),
  breaks            = seq(0.80, 1.0, length.out = 101),
  main              = "Sample-to-Sample Pearson Correlation (GSE120584)",
  show_rownames     = FALSE,
  show_colnames     = FALSE,
  filename          = "qc_reports/correlation_heatmap_GSE120584.png",
  width             = 10,
  height            = 8
)
cat("Correlation heatmap saved to qc_reports/correlation_heatmap_GSE120584.png\n")


# ==============================================================================
# SECTION 10: PCA — Visualise Data Structure and Batch Effects (GSE120584)
# ==============================================================================
# PCA reduces the expression matrix to 2 dimensions so we can see:
#   - Whether samples cluster by disease group (expected biological signal)
#   - Whether samples cluster by batch or other technical variable (batch effect)

pca_result  <- prcomp(t(expr_norm), scale. = TRUE)
var_exp     <- (pca_result$sdev^2) / sum(pca_result$sdev^2) * 100

pca_df <- data.frame(
  PC1   = pca_result$x[, 1],
  PC2   = pca_result$x[, 2],
  Group = metadata$group,
  row.names = rownames(pca_result$x)
)
if ("sex" %in% colnames(metadata))   pca_df$Sex   <- metadata$sex
if ("age" %in% colnames(metadata))   pca_df$Age   <- metadata$age

# PCA coloured by disease group
p_pca_group <- ggplot(pca_df, aes(x = PC1, y = PC2, colour = Group, shape = Group)) +
  geom_point(size = 2, alpha = 0.6) +
  scale_colour_manual(values = GROUP_COLOURS) +
  scale_shape_manual(values = c(16, 17, 15)) +
  labs(
    title = "PCA: GSE120584 — Coloured by Disease Group",
    x     = paste0("PC1 (", round(var_exp[1], 1), "% variance)"),
    y     = paste0("PC2 (", round(var_exp[2], 1), "% variance)")
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

print(p_pca_group)
ggsave("qc_reports/pca_group_GSE120584.png", p_pca_group, width = 7, height = 5, dpi = 150)

# INTERPRETATION GUIDE:
#   PC1 separates AD from Control    → strong biological signal — good!
#   PC1 separates by batch/date      → batch effect dominates — correct it (Section 13)
#   Outlier sample far from cluster  → failed QC — consider exclusion
#   Random scatter with no grouping  → very noisy data or genuine group similarity


# ==============================================================================
# SECTION 11: Hemolysis Detection (Serum miRNA)
# ==============================================================================
# Red blood cell lysis releases miR-451a and miR-23a-3p into serum/plasma,
# contaminating the circulating miRNA profile.
# The log2 ratio miR-451a / miR-23a-3p serves as a hemolysis index.
# Samples above the threshold should be flagged and excluded if possible.
#
# Reference: Murray et al. (2018) Cancer Epidemiol Biomarkers Prev
# DOI: 10.1158/1055-9965.EPI-17-0657
#
# Note for GSE120584: the submitters did not include miR-451a in the processed
# matrix, so the code below reports it as missing and skips the check. The same
# code runs unchanged on serum datasets that do include it.

mir451a_row  <- grep("miR-451a$|hsa-miR-451a", rownames(expr_norm), value = TRUE)[1]
mir23a_row   <- grep("miR-23a-3p|hsa-miR-23a-3p", rownames(expr_norm), value = TRUE)[1]

if (!is.na(mir451a_row) && !is.na(mir23a_row)) {
  hemolysis_index <- expr_norm[mir451a_row, ] - expr_norm[mir23a_row, ]
  # In log2 space: subtraction of values = ratio of signals

  metadata$hemolysis_index <- as.numeric(hemolysis_index)
  metadata$hemolyzed       <- metadata$hemolysis_index > 7
  # Threshold of 7 (log2 ratio) is a guide; adjust based on platform

  cat("Hemolysis index summary:\n")
  print(summary(metadata$hemolysis_index))
  cat("\nPotentially hemolyzed samples per group:\n")
  print(table(metadata$hemolyzed, metadata$group))

  # Box plot of hemolysis index by group
  hem_df <- data.frame(
    group            = metadata$group,
    hemolysis_index  = metadata$hemolysis_index
  )

  p_hem <- ggplot(hem_df, aes(x = group, y = hemolysis_index, fill = group)) +
    geom_boxplot(outlier.shape = 16, alpha = 0.8) +
    geom_hline(yintercept = 7, colour = "red", linetype = "dashed") +
    scale_fill_manual(values = GROUP_COLOURS) +
    labs(
      title   = "Hemolysis Index (miR-451a – miR-23a-3p, log2)",
      x       = NULL,
      y       = "Hemolysis Index",
      caption = "Red dashed line: exclusion threshold"
    ) +
    theme_bw(base_size = 12) +
    theme(legend.position = "none", axis.text.x = element_text(angle = 20, hjust = 1))

  print(p_hem)
  ggsave("qc_reports/hemolysis_index_GSE120584.png", p_hem, width = 6, height = 4, dpi = 150)

} else {
  cat("miR-451a or miR-23a-3p not found in this dataset — skipping hemolysis check.\n")
  cat("Found miR-451a:", !is.na(mir451a_row), "| Found miR-23a-3p:", !is.na(mir23a_row), "\n")
  metadata$hemolyzed <- FALSE
}


# ==============================================================================
# SECTION 12: Sample QC Decisions Log (GSE120584)
# ==============================================================================
# Document every exclusion decision with the reason.
# You will need this for your methods section and peer review responses.

qc_log <- data.frame(
  sample          = metadata$geo_accession,
  group           = metadata$group,
  detected_miRNAs = detected_per_sample[metadata$geo_accession],
  median_signal   = median_signal[metadata$geo_accession],
  median_cor      = median_cor[metadata$geo_accession],
  hemolysis_index = metadata$hemolysis_index %||% NA_real_,
  low_detection   = low_detect_flag[metadata$geo_accession],
  low_correlation = low_cor_flag[metadata$geo_accession],
  hemolyzed       = metadata$hemolyzed,
  pass_qc         = TRUE,
  exclude_reason  = "",
  stringsAsFactors = FALSE
)

# Flag failures (first reason found is recorded)
qc_log$pass_qc[qc_log$low_detection] <- FALSE
qc_log$exclude_reason[qc_log$low_detection] <- "Detected miRNAs < median - 3 MAD"

qc_log$exclude_reason[qc_log$low_correlation & qc_log$pass_qc] <- "Median correlation < 0.90"
qc_log$pass_qc[qc_log$low_correlation] <- FALSE

qc_log$exclude_reason[qc_log$hemolyzed & qc_log$pass_qc] <- "Hemolysis index > 7"
qc_log$pass_qc[qc_log$hemolyzed] <- FALSE

cat("\n=== QC Decision Summary ===\n")
print(table(qc_log$pass_qc, qc_log$group))
cat("\nFailed samples:\n")
print(qc_log[!qc_log$pass_qc, c("sample", "group", "exclude_reason")], row.names = FALSE)

# Save QC log
write.csv(qc_log, "qc_reports/sample_qc_decisions_GSE120584.csv", row.names = FALSE)
cat("QC log saved to qc_reports/sample_qc_decisions_GSE120584.csv\n")

# Apply exclusions
passing     <- qc_log$sample[qc_log$pass_qc]
metadata_qc <- metadata[metadata$geo_accession %in% passing, ]
expr_qc     <- expr_norm[, passing]

cat("\nSamples remaining after QC:\n")
print(table(metadata_qc$group))


# ==============================================================================
# SECTION 13: Batch Effect Detection and Correction (GSE120584)
# ==============================================================================
# Batch effects are systematic technical differences between groups of samples
# processed at different times or locations. They can masquerade as biology.
#
# This section shows:
#   A. How to detect batch effects with PCA
#   B. How to correct with ComBat (log-scale microarray data, as here)
#   C. How to estimate hidden batch variables with SVA
#
# NOTE: GSE120584's GEO metadata has no batch or processing-date column.
# The code below then uses a single placeholder batch and skips correction;
# if PCA shows structure unrelated to diagnosis, use SVA (13C).

# Common column names for batch: "batch", "extraction_date", "run", "lab"
batch_col <- intersect(colnames(metadata_qc),
                       c("batch", "extraction_date", "sequencing_run", "plate"))

if (length(batch_col) > 0) {
  cat("Batch variable found:", batch_col[1], "\n")
  metadata_qc$batch <- factor(metadata_qc[[batch_col[1]]])
} else {
  cat("No batch column found in metadata.\n")
  cat("If PCA shows clustering unrelated to disease group, use SVA (Section 13C).\n")
  metadata_qc$batch <- factor(rep("batch1", nrow(metadata_qc)))  # placeholder
}

# ---- 13A. Visualise Potential Batch Effects ----
pca_df_qc <- data.frame(
  PC1   = pca_result$x[passing, 1],
  PC2   = pca_result$x[passing, 2],
  Group = metadata_qc$group,
  Batch = metadata_qc$batch
)

p_batch <- ggplot(pca_df_qc, aes(x = PC1, y = PC2, colour = Group, shape = Batch)) +
  geom_point(size = 2, alpha = 0.6) +
  scale_colour_manual(values = GROUP_COLOURS) +
  labs(
    title = "PCA: Check for Batch Effects",
    x     = paste0("PC1 (", round(var_exp[1], 1), "% variance)"),
    y     = paste0("PC2 (", round(var_exp[2], 1), "% variance)")
  ) +
  theme_bw(base_size = 12)

print(p_batch)

# ---- 13B. ComBat Batch Correction ----
# ComBat adjusts batch-specific mean and variance per miRNA using empirical Bayes.
# The mod matrix tells ComBat what biological signal to PRESERVE.

if (nlevels(metadata_qc$batch) > 1) {

  mod  <- model.matrix(~ group, data = metadata_qc)  # protect disease group
  mod0 <- model.matrix(~ 1,     data = metadata_qc)  # null model

  expr_combat <- ComBat(
    dat        = expr_qc,           # normalised log2 expression matrix
    batch      = metadata_qc$batch, # batch labels
    mod        = mod,
    par.prior  = TRUE,              # parametric empirical Bayes (faster)
    prior.plots = FALSE
  )

  cat("ComBat batch correction applied.\n")

  # Verify: re-run PCA on corrected data
  pca_combat  <- prcomp(t(expr_combat), scale. = TRUE)
  var_combat  <- (pca_combat$sdev^2) / sum(pca_combat$sdev^2) * 100

  pca_df_combat <- data.frame(
    PC1   = pca_combat$x[, 1],
    PC2   = pca_combat$x[, 2],
    Group = metadata_qc$group,
    Batch = metadata_qc$batch
  )

  p_after_combat <- ggplot(pca_df_combat, aes(x = PC1, y = PC2,
                                               colour = Group, shape = Batch)) +
    geom_point(size = 2, alpha = 0.6) +
    scale_colour_manual(values = GROUP_COLOURS) +
    labs(
      title = "PCA After ComBat — Batch Effect Removed",
      x     = paste0("PC1 (", round(var_combat[1], 1), "% variance)"),
      y     = paste0("PC2 (", round(var_combat[2], 1), "% variance)")
    ) +
    theme_bw(base_size = 12)

  grid.arrange(p_batch, p_after_combat, ncol = 2)

  expr_final <- expr_combat

} else {
  cat("Only one batch detected — no batch correction applied.\n")
  expr_final <- expr_qc
}

# ---- 13C. SVA — Estimate Hidden Batch Variables ----
# Use this when batch metadata is missing but PCA shows unexplained structure.

# UNCOMMENT the block below if needed:
# expr_for_sva <- expr_qc      # already log2 and normalised
#
# mod_full <- model.matrix(~ group, data = metadata_qc)
# mod_null <- model.matrix(~ 1,     data = metadata_qc)
# n_sv     <- num.sv(expr_for_sva, mod_full, method = "leek")
# cat("Estimated number of surrogate variables:", n_sv, "\n")
# sva_obj  <- sva(expr_for_sva, mod_full, mod_null, n.sv = n_sv)
# # Add surrogate variables to metadata for use as covariates in the linear model
# for (i in seq_len(n_sv)) {
#   metadata_qc[[paste0("SV", i)]] <- sva_obj$sv[, i]
# }


# ==============================================================================
# SECTION 14: RLE Plot — Post-Normalisation Quality Check (GSE120584)
# ==============================================================================
# RLE (Relative Log Expression) should show boxes centred at zero
# with narrow, similar IQR across all samples after good normalisation.

row_medians <- apply(expr_final, 1, median)
rle_matrix  <- sweep(expr_final, 1, row_medians, "-")

ord_qc <- order(metadata_qc$group)

par(mar = c(3, 5, 4, 2))
boxplot(
  rle_matrix[, ord_qc],
  main    = "RLE Plot - Post-Normalisation (GSE120584)",
  ylab    = "Relative Log Expression",
  col     = GROUP_COLOURS[as.character(metadata_qc$group)][ord_qc],
  border  = GROUP_COLOURS[as.character(metadata_qc$group)][ord_qc],
  xaxt    = "n",
  ylim    = c(-2, 2),
  outline = FALSE
)
abline(h = 0, col = "red", lty = 2, lwd = 1.5)

# What to look for:
#   Median of each box near 0   → normalisation worked
#   Box width (IQR) similar     → comparable technical quality across samples
#   One box with large IQR      → possible failed/degraded sample
#   Systematic offset by group  → possible over-normalisation; check biology


# ==============================================================================
# SECTION 15: Download GSE46579 (Validation Small RNA-seq Dataset)
# ==============================================================================
# GSE46579: Whole-blood small RNA-seq, AD vs Control (Leidinger et al., 2013, Genome Biol).
# Platform: Illumina HiSeq 2000 (GPL11154)
# Samples: 70 — AD (48), Control (22)
# We preprocess this independently for use as an external validation set (Week 5).
#
# For RNA-seq, the GEO series matrix holds only metadata — exprs() is empty.
# The read counts (miRDeep2, miRBase v18) are in a supplementary Excel file:
#   GSE46579_AD_ngs_data_summarized.xls.gz
#     sheet "raw data"   — raw read counts
#     sheet "normalized" — the submitters' quantile-normalised counts
# We use the RAW counts: DESeq2 and edgeR need raw counts and normalise them
# themselves. Raw FASTQ files (on SRA) require HPC to process — beyond this course.

cat("\n--- Downloading GSE46579 (validation dataset) ---\n")

gse46579_list <- getGEO(
  "GSE46579",
  destdir   = "data/raw/",
  GSEMatrix = TRUE
)
gse46579 <- gse46579_list[[1]]
cat("Expression slot dimensions:", dim(exprs(gse46579)), "(empty for RNA-seq)\n")

# Download the supplementary count table (only once)
xls_gz <- "data/raw/GSE46579/GSE46579_AD_ngs_data_summarized.xls.gz"
if (!file.exists(xls_gz)) {
  supp_46 <- getGEOSuppFiles("GSE46579", makeDirectory = TRUE, baseDir = "data/raw/")
}

# readxl cannot open .gz files — decompress a copy next to it
xls_file <- sub("\\.gz$", "", xls_gz)
if (!file.exists(xls_file)) {
  R.utils::gunzip(xls_gz, destname = xls_file, remove = FALSE)
}
cat("Count table:", xls_file, "\n")
print(excel_sheets(xls_file))


# ==============================================================================
# SECTION 16: Parse Metadata and Load the Count Matrix (GSE46579)
# ==============================================================================
# The count table's columns are named "AD", "AD.1", …, "control", "control.1", …
# These match the GEO sample titles "AD sample 0", "AD sample 1", …,
# "control sample 0", … — we use the titles to attach GSM accessions.

# ---- 16A. Sample Metadata ----
metadata_46 <- pData(gse46579)
cat("Metadata columns for GSE46579:\n")
print(colnames(metadata_46))
print(table(metadata_46$`group:ch1`))

group_labels_46 <- c("control"           = "Control",
                     "alzheimer patient" = "Alzheimer's Disease")
metadata_46$group <- factor(group_labels_46[metadata_46$`group:ch1`],
                            levels = c("Control", "Alzheimer's Disease"))
# GEO stores every characteristic as text ("77", "female"): convert the types
metadata_46$age <- as.numeric(metadata_46$`age:ch1`)                      # age in years
metadata_46$sex <- factor(metadata_46$`gender:ch1`, levels = c("female", "male"))
str(metadata_46[, c("group", "age", "sex")])   # expect Factor, num, Factor

cat("\n=== Cohort Summary: GSE46579 ===\n")
print(table(metadata_46$group))
print(table(metadata_46$sex, metadata_46$group))
print(round(tapply(metadata_46$age, metadata_46$group, mean, na.rm = TRUE), 1))

# ---- 16B. Read the Raw Count Table ----
raw_counts_46 <- as.data.frame(
  read_excel(xls_file, sheet = "raw data", .name_repair = "minimal")
)
cat("Raw table dimensions (rows × columns):", dim(raw_counts_46), "\n")
print(raw_counts_46[1:5, 1:5])

# Column 1 holds "precursor:mature" names, e.g. "hsa-mir-30a:hsa-miR-30a-3p"
row_ids     <- raw_counts_46[[1]]
counts_46   <- as.matrix(raw_counts_46[, -1])

# Map column names to GEO titles, then to GSM accessions
col_titles  <- sub("^(AD|control)$", "\\1.0", colnames(counts_46))          # "AD" -> "AD.0"
col_titles  <- sub("^(AD|control)\\.([0-9]+)$", "\\1 sample \\2", col_titles) # "AD.12" -> "AD sample 12"
colnames(counts_46) <- metadata_46$geo_accession[match(col_titles, metadata_46$title)]
stopifnot(!anyNA(colnames(counts_46)))

# ---- 16C. One Row per Mature miRNA ----
# Keep known human miRNAs only (drops novel "brain-mir-…" predictions).
# The same mature miRNA can come from several precursor genes
# (e.g. miR-16-5p from mir-16-1 and mir-16-2). miRDeep2 counts the same
# reads under each precursor, so we take the maximum — summing would
# count those reads twice.

mature   <- sub(".*:", "", row_ids)          # "hsa-mir-30a:hsa-miR-30a-3p" -> "hsa-miR-30a-3p"
is_human <- grepl("^hsa-", mature)
cat("Rows that are novel (non-miRBase) predictions:", sum(!is_human), "\n")
cat("Duplicate mature miRNAs (several precursors):",
    sum(duplicated(mature[is_human])), "\n")

count_matrix_46 <- apply(counts_46[is_human, ], 2,
                         function(x) tapply(x, mature[is_human], max))
storage.mode(count_matrix_46) <- "integer"

# Put samples in the same order as the metadata
count_matrix_46 <- count_matrix_46[, metadata_46$geo_accession]

cat("\nCount matrix (miRNAs × samples):", dim(count_matrix_46), "\n")
cat("Order matches:", all(colnames(count_matrix_46) == metadata_46$geo_accession), "\n")
print(count_matrix_46[1:5, 1:4])


# ==============================================================================
# SECTION 17: RNA-seq Quality Control (GSE46579)
# ==============================================================================
# We assess three aspects of data quality before any analysis:
#   A. Library size — are total miRNA read counts adequate and similar across samples?
#   B. Detected miRNA count — are samples detecting comparable numbers of miRNAs?
#   C. Count distribution — do raw distributions look plausible?

# ---- 17A. Library Size ----
group_colours_46 <- GROUP_COLOURS[as.character(metadata_46$group)]
library_sizes    <- colSums(count_matrix_46)

par(mar = c(8, 5, 4, 2))
barplot(
  library_sizes / 1e6,
  main      = "Library Size per Sample (GSE46579)",
  ylab      = "Total miRNA Reads (millions)",
  col       = group_colours_46,
  las       = 2,
  cex.names = 0.55,
  border    = NA
)
abline(h = 0.5 * mean(library_sizes / 1e6), col = "red", lty = 2, lwd = 1.5)
legend("topright", legend = levels(metadata_46$group),
       fill = GROUP_COLOURS[levels(metadata_46$group)], bty = "n", cex = 0.85)

cat("\nLibrary size summary (millions of reads):\n")
print(summary(library_sizes / 1e6))

# Which miRNAs make up most of the reads?
read_share <- sweep(count_matrix_46, 2, library_sizes, "/") * 100
cat("\nMost abundant miRNAs (mean % of all miRNA reads per sample):\n")
print(round(sort(rowMeans(read_share), decreasing = TRUE)[1:5], 1))
# In whole blood, miR-486-5p — a red blood cell miRNA — takes ~90% of all
# reads. "Library size" therefore mostly measures miR-486-5p, and the
# reads left for all other miRNAs vary a lot between samples.

# Flag samples with library size < 50% of mean
low_lib_flag <- library_sizes < 0.5 * mean(library_sizes)
if (any(low_lib_flag)) {
  cat("WARNING: Samples with low library size (< 50% of mean):\n")
  print(names(library_sizes)[low_lib_flag])
} else {
  cat("All samples pass library size threshold.\n")
}

# ---- 17B. Detected miRNA Count ----
detected_46 <- colSums(count_matrix_46 > 0)

par(mar = c(8, 5, 4, 2))
barplot(
  detected_46,
  main      = "Detected miRNAs per Sample (count > 0)",
  ylab      = "Number of miRNAs with ≥ 1 read",
  col       = group_colours_46,
  las       = 2,
  cex.names = 0.55,
  border    = NA
)

cat("\nDetected miRNA count per sample:\n")
print(summary(detected_46))

# ---- 17C. Count Distribution (log2-transformed for visualisation) ----
log_counts <- log2(count_matrix_46 + 0.5)   # 0.5 pseudocount handles zeros

par(mar = c(8, 5, 4, 2))
boxplot(
  log_counts,
  main     = "Raw Count Distribution - log2(count + 0.5)",
  ylab     = "log2(count + 0.5)",
  col      = group_colours_46,
  las      = 2,
  cex.axis = 0.55,
  outline  = FALSE
)

# What to look for:
#   Boxes should look roughly similar in height and spread.
#   A box far below all others = failed sample.
#   Wide spread = high technical variation → normalisation will address this.


# ==============================================================================
# SECTION 18: Filtering and Normalisation (GSE46579)
# ==============================================================================
# miRNAs with very low counts across most samples contribute only noise.
# edgeR's filterByExpr() keeps features with enough counts in the smallest
# experimental group, so group-specific miRNAs are not filtered out.
#
# Then two complementary normalisations:
#   VST (DESeq2) — for visualisation, PCA, and ML feature engineering
#   TMM (edgeR)  — for differential expression in Week 4

# ---- 18A. Low-Count Filtering ----
cat("miRNAs before filtering: ", nrow(count_matrix_46), "\n")

keep_46 <- filterByExpr(
  count_matrix_46,
  group           = metadata_46$group,
  min.count       = 10,     # minimum 10 reads in smallest group
  min.total.count = 15      # minimum 15 reads across all samples
)

count_filtered_46 <- count_matrix_46[keep_46, ]
cat("miRNAs after filtering:  ", nrow(count_filtered_46), "\n")
cat("Removed:                 ", sum(!keep_46), "low-count miRNAs\n")

# ---- 18B. DESeq2 — Size Factor Normalisation + VST ----
# DESeq2's median-of-ratios method estimates a size factor per sample.
# VST (Variance Stabilizing Transformation) then produces log2-scale values
# suitable for linear methods (PCA, clustering, logistic regression).

dds_46 <- DESeqDataSetFromMatrix(
  countData = count_filtered_46,
  colData   = metadata_46,
  design    = ~ sex + age + group    # DESeq2 suggests centring/scaling age; Week 4 does this
)

# Relevel so Control is the reference group in all comparisons
dds_46$group <- relevel(dds_46$group, ref = "Control")

# Estimate size factors (normalisation coefficients)
dds_46 <- estimateSizeFactors(dds_46)
cat("\nDESeq2 size factors:\n")
print(summary(sizeFactors(dds_46)))
# In most datasets size factors sit close to 1.0. Here they span a much wider
# range because sequencing depth differed a lot between samples (1–50 million
# reads). Median-of-ratios uses the typical miRNA, so the one dominant
# miRNA (miR-486-5p) does not distort it. Very extreme values (< 0.2 or > 5)
# deserve a look in the QC log.

# VST: blind = TRUE uses no design information → unbiased QC.
# vst() is a fast shortcut that needs > 1,000 features; miRNA datasets have
# a few hundred, so we call the full varianceStabilizingTransformation().
vst_46      <- varianceStabilizingTransformation(dds_46, blind = TRUE)
expr_vst_46 <- assay(vst_46)

cat("\nVST-transformed matrix dimensions:", dim(expr_vst_46), "\n")
cat("Value range after VST:", round(range(expr_vst_46), 2), "\n")

# Post-normalisation box plot
par(mar = c(8, 5, 4, 2))
boxplot(
  expr_vst_46,
  main     = "Post-VST Expression Distribution (GSE46579)",
  ylab     = "VST-transformed expression",
  col      = group_colours_46,
  las      = 2,
  cex.axis = 0.55,
  outline  = FALSE
)
# Boxes should now be aligned. Any persistent outlier box warrants investigation.

# ---- 18C. TMM Normalisation (edgeR) — for DE in Week 4 ----
dge_46 <- DGEList(counts = count_filtered_46, group = metadata_46$group)
dge_46 <- calcNormFactors(dge_46, method = "TMM")

cat("\nTMM normalisation factors:\n")
print(summary(dge_46$samples$norm.factors))
# Typically most values are 0.9–1.1. Here they spread much wider (≈0.2–3)
# because miR-486-5p's share of the reads differs between samples: when one
# miRNA takes more of the reads, every other miRNA gets fewer. Correcting this
# "composition effect" is exactly what TMM is for — it is not by itself a
# sign of a failed sample.

# log2-CPM values (for visualisation — not used in the DE model directly)
cpm_tmm_46 <- cpm(dge_46, normalized.lib.sizes = TRUE, log = TRUE, prior.count = 0.5)


# ==============================================================================
# SECTION 19: Sample QC — Correlation, PCA and QC Log (GSE46579)
# ==============================================================================
# Same checks as for GSE120584, on the VST-normalised data.
#
# Hemolysis note: GSE46579 is WHOLE BLOOD, which contains red blood cells.
# Red-cell miRNAs (miR-486-5p, miR-92a-3p, miR-451a) are part of the sample
# by design, so the serum hemolysis index (Section 11) does not apply here.

# ---- 19A. Sample-to-Sample Correlation ----
cor_46 <- cor(expr_vst_46, method = "pearson")

median_cor_46   <- apply(cor_46, 2, function(x) median(x[x < 1]))
low_cor_flag_46 <- median_cor_46 < 0.90
cat("Median correlation of each sample with the others:\n")
print(summary(median_cor_46))
cat("Samples with median correlation < 0.90:", sum(low_cor_flag_46), "\n")

ann_46 <- data.frame(Group = metadata_46$group,
                     row.names = colnames(cor_46))

pheatmap(
  cor_46,
  annotation_col    = ann_46,
  annotation_colors = list(Group = GROUP_COLOURS[levels(metadata_46$group)]),
  color             = colorRampPalette(rev(brewer.pal(9, "RdBu")))(100),
  breaks            = seq(0.80, 1.0, length.out = 101),
  main              = "Sample-to-Sample Pearson Correlation (GSE46579, post-VST)",
  fontsize_row      = 6,
  fontsize_col      = 6,
  show_rownames     = FALSE,
  filename          = "qc_reports/correlation_heatmap_GSE46579.png",
  width = 8, height = 6
)
cat("Correlation heatmap saved to qc_reports/correlation_heatmap_GSE46579.png\n")

# ---- 19B. PCA ----
pca_46     <- prcomp(t(expr_vst_46), scale. = TRUE)
var_exp_46 <- (pca_46$sdev^2) / sum(pca_46$sdev^2) * 100

pca_df_46 <- data.frame(
  PC1   = pca_46$x[, 1],
  PC2   = pca_46$x[, 2],
  Group = metadata_46$group
)

p_pca_46 <- ggplot(pca_df_46, aes(x = PC1, y = PC2, colour = Group, shape = Group)) +
  geom_point(size = 3, alpha = 0.85) +
  scale_colour_manual(values = GROUP_COLOURS) +
  scale_shape_manual(values = c(16, 15)) +
  labs(
    title = "PCA: GSE46579 — Coloured by Disease Group",
    x     = paste0("PC1 (", round(var_exp_46[1], 1), "% variance)"),
    y     = paste0("PC2 (", round(var_exp_46[2], 1), "% variance)")
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

print(p_pca_46)
ggsave("qc_reports/pca_group_GSE46579.png", p_pca_46, width = 7, height = 5, dpi = 150)

# ---- 19C. Whole-Blood Check: Red-Cell miRNAs ----
rbc_mirnas <- intersect(c("hsa-miR-486-5p", "hsa-miR-92a-3p", "hsa-miR-451a"),
                        rownames(count_filtered_46))
rbc_share  <- colSums(count_filtered_46[rbc_mirnas, , drop = FALSE]) /
              colSums(count_filtered_46) * 100
cat("Share of reads from red-cell miRNAs (", paste(rbc_mirnas, collapse = ", "), ") %:\n")
print(summary(rbc_share))
print(round(tapply(rbc_share, metadata_46$group, median), 1))
# A very large share is expected in whole blood, so we do not exclude samples
# on this basis. But if the share differed clearly between AD and Control, it
# would point to a difference in blood cell composition — a confounder.

# ---- 19D. QC Decisions Log ----
qc_log_46 <- data.frame(
  sample          = metadata_46$geo_accession,
  group           = metadata_46$group,
  library_size    = library_sizes[metadata_46$geo_accession],
  detected_miRNAs = detected_46[metadata_46$geo_accession],
  median_cor      = median_cor_46[metadata_46$geo_accession],
  low_library     = low_lib_flag[metadata_46$geo_accession],
  low_correlation = low_cor_flag_46[metadata_46$geo_accession],
  pass_qc         = TRUE,
  exclude_reason  = "",
  stringsAsFactors = FALSE
)

qc_log_46$pass_qc[qc_log_46$low_library] <- FALSE
qc_log_46$exclude_reason[qc_log_46$low_library] <- "Library size < 50% of mean"

qc_log_46$exclude_reason[qc_log_46$low_correlation & qc_log_46$pass_qc] <- "Median correlation < 0.90"
qc_log_46$pass_qc[qc_log_46$low_correlation] <- FALSE

cat("\n=== QC Decision Summary: GSE46579 ===\n")
print(table(qc_log_46$pass_qc, qc_log_46$group))
cat("\nFailed samples:\n")
print(qc_log_46[!qc_log_46$pass_qc, c("sample", "group", "exclude_reason")], row.names = FALSE)

write.csv(qc_log_46, "qc_reports/sample_qc_decisions_GSE46579.csv", row.names = FALSE)
cat("QC log saved to qc_reports/sample_qc_decisions_GSE46579.csv\n")

# Apply exclusions
passing_46           <- qc_log_46$sample[qc_log_46$pass_qc]
metadata_46_qc       <- metadata_46[metadata_46$geo_accession %in% passing_46, ]
count_filtered_46_qc <- count_filtered_46[, passing_46]
expr_vst_46_qc       <- expr_vst_46[, passing_46]
dds_46_qc            <- dds_46[, passing_46]

cat("\nSamples remaining after QC:\n")
print(table(metadata_46_qc$group))


# ==============================================================================
# SECTION 20: Final Clean Data Checkpoint and Save
# ==============================================================================
# After completing QC, normalisation, and batch correction, save the clean
# data objects that Week 3 will load directly. Never modify these files
# manually — all changes must go through this reproducible pipeline.

# Pre-save audit — verify everything is in order
cat("\n========================================\n")
cat("  WEEK 2 CLEAN DATA AUDIT\n")
cat("========================================\n")

cat("\n-- GSE120584 (serum miRNA microarray, primary training set) --\n")
cat("Final expression matrix:", nrow(expr_final), "miRNAs ×",
    ncol(expr_final), "samples\n")
cat("Value range:", round(range(expr_final), 2), "\n")
cat("Group distribution:\n")
print(table(metadata_qc$group))

cat("\n-- GSE46579 (whole-blood small RNA-seq, external validation set) --\n")
cat("Final VST matrix:", nrow(expr_vst_46_qc), "miRNAs ×",
    ncol(expr_vst_46_qc), "samples\n")
cat("Value range:", round(range(expr_vst_46_qc), 2), "\n")
cat("Group distribution:\n")
print(table(metadata_46_qc$group))

# Save objects
# GSE120584 — microarray: normalised log2 matrix (no counts exist)
saveRDS(expr_final,           "data/processed/GSE120584_expr_clean.rds")
saveRDS(metadata_qc,          "data/processed/GSE120584_metadata_clean.rds")

# GSE46579 — RNA-seq: VST matrix for ML, filtered counts + DESeq2 object for DE
saveRDS(expr_vst_46_qc,       "data/processed/GSE46579_expr_vst.rds")
saveRDS(count_filtered_46_qc, "data/processed/GSE46579_counts_filtered.rds")
saveRDS(dds_46_qc,            "data/processed/GSE46579_dds.rds")     # for DESeq2 in Week 4
saveRDS(metadata_46_qc,       "data/processed/GSE46579_metadata_clean.rds")

cat("\nClean data saved to data/processed/\n")

# Save final session info for reproducibility
sink("qc_reports/session_info_week2.txt")
sessionInfo()
sink()

cat("\n========================================\n")
cat("  Week 2 Complete!\n")
cat("  Files saved to data/processed/\n")
cat("  QC reports saved to qc_reports/\n")
cat("========================================\n")
cat("\nNEXT WEEK (Week 3):\n")
cat("  - Load expr_clean.rds and metadata_clean.rds\n")
cat("  - PCA, t-SNE, UMAP for dimensionality reduction\n")
cat("  - Unsupervised clustering and heatmaps\n")
cat("  - Identify batch effects not caught in Week 2\n")
cat("  - Build publication-quality visualisations\n")
