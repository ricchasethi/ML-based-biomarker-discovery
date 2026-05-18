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
#   4. Run a full QC pipeline on RNA-seq count data (GSE120584)
#   5. Run a full QC pipeline on microarray CEL data (GSE46579)
#   6. Normalize both data types (DESeq2/TMM and RMA)
#   7. Detect and correct batch effects (ComBat / limma)
#   8. Detect hemolysis in blood-based miRNA data
#   9. Save a clean, analysis-ready expression matrix for Week 3
#
# Datasets:
#   GSE120584 — Serum small RNA-seq, 3 groups: AD / MCI / Control  [PRIMARY]
#   GSE46579  — Whole blood Affymetrix microarray, AD / Control     [VALIDATION]
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
  library(DESeq2)            # RNA-seq count normalization & DE
  library(edgeR)             # TMM normalization, filterByExpr
  library(limma)             # Microarray DE, removeBatchEffect
  library(affy)              # Read Affymetrix CEL files (older arrays)
  library(oligo)             # Read Affymetrix CEL files (miRNA arrays)
  library(sva)               # ComBat batch correction

  # CRAN
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(pheatmap)
  library(RColorBrewer)
  library(gridExtra)
  library(readr)
})

cat("All packages loaded.\n")

# Colour palette used consistently for group labels throughout this script
GROUP_COLOURS <- c(
  "Control"                  = "#4575B4",   # blue
  "Mild Cognitive Impairment" = "#FEE090",  # amber
  "Alzheimer's Disease"      = "#D73027"    # red
)


# ==============================================================================
# SECTION 3: Download GSE120584 (Primary RNA-seq Dataset)
# ==============================================================================
# GSE120584: Serum small RNA-seq in AD, MCI, and healthy controls.
# Platform: Illumina HiSeq 2500 (GPL19117)
# Groups: AD (n≈48), MCI (n≈50), Control (n≈50)
#
# GEOquery downloads two things:
#   1. GSEMatrix = TRUE  → parsed expression matrix + metadata (ExpressionSet)
#   2. getGEOSuppFiles() → raw supplementary files (count matrix for RNA-seq)

cat("Downloading GSE120584 metadata from GEO...\n")
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
#   exprs(gse120584)  — expression/count matrix (miRNAs × samples)
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

# --- Parse group label ---
# Expected format: "disease state: Alzheimer's Disease"
metadata <- metadata_raw
metadata$group <- gsub("disease state: ", "", metadata$characteristics_ch1)
metadata$group <- trimws(metadata$group)
metadata$group <- factor(
  metadata$group,
  levels = c("Control", "Mild Cognitive Impairment", "Alzheimer's Disease")
)

# --- Parse age ---
# Adjust the column name and prefix pattern to match what GEO actually provides.
# Run unique(metadata_raw$characteristics_ch1.1) to inspect first.
if ("characteristics_ch1.1" %in% colnames(metadata_raw)) {
  metadata$age <- as.numeric(
    gsub("age: ", "", metadata_raw$characteristics_ch1.1)
  )
}

# --- Parse sex ---
if ("characteristics_ch1.2" %in% colnames(metadata_raw)) {
  metadata$sex <- gsub("Sex: |sex: ", "", metadata_raw$characteristics_ch1.2)
  metadata$sex <- trimws(metadata$sex)
}

# --- Cohort summary ---
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
#   Groups: roughly balanced AD/MCI/Control
# Anything outside these ranges = likely metadata parsing error.


# ==============================================================================
# SECTION 5: Download RNA-seq Count Matrix (GSE120584)
# ==============================================================================
# For RNA-seq datasets, GEO also stores the processed count matrix as a
# supplementary text file. We download and load that file here.
# Raw FASTQ files (on SRA) require HPC to process — beyond this course scope.

cat("Downloading supplementary count matrix for GSE120584...\n")
getGEOSuppFiles("GSE120584",
                makeDirectory = TRUE,
                baseDir       = "data/raw/")

# List downloaded files
supp_files <- list.files("data/raw/GSE120584/", full.names = TRUE)
cat("Downloaded files:\n")
print(supp_files)

# Identify the count matrix file (usually .txt.gz)
count_file <- grep("count|Count|matrix|Matrix", supp_files, value = TRUE)[1]
cat("Loading count matrix from:", count_file, "\n")

# Load the count matrix
# Adjust sep and header arguments if your file uses different delimiters
count_matrix <- read.table(
  count_file,
  header      = TRUE,
  row.names   = 1,
  sep         = "\t",
  check.names = FALSE
)

cat("Count matrix dimensions (miRNAs × samples):", dim(count_matrix), "\n")
cat("Preview (first 5 miRNAs, first 4 samples):\n")
print(count_matrix[1:5, 1:4])

# Ensure sample order matches metadata
# If they don't align, all downstream analyses will be wrong.
common_samples <- intersect(colnames(count_matrix), metadata$geo_accession)
count_matrix   <- count_matrix[, common_samples]
metadata       <- metadata[metadata$geo_accession %in% common_samples, ]

cat("\nSamples in count matrix:", ncol(count_matrix), "\n")
cat("Samples in metadata:     ", nrow(metadata), "\n")
cat("Order matches:", all(colnames(count_matrix) == metadata$geo_accession), "\n")


# ==============================================================================
# SECTION 6: RNA-seq Quality Control
# ==============================================================================
# We assess three aspects of data quality before any analysis:
#   A. Library size — are total read counts adequate and similar across samples?
#   B. Detected miRNA count — are samples detecting comparable numbers of miRNAs?
#   C. Count distribution — do raw distributions look plausible?

# ---- 6A. Library Size ----
library_sizes <- colSums(count_matrix)

# Colour bars by group
group_colours_vec <- GROUP_COLOURS[as.character(metadata$group)]

par(mar = c(8, 5, 4, 2))
barplot(
  library_sizes / 1e6,
  main      = "Library Size per Sample (GSE120584)",
  ylab      = "Total Reads (millions)",
  col       = group_colours_vec,
  las       = 2,
  cex.names = 0.55,
  border    = NA
)
abline(h = 0.5 * mean(library_sizes / 1e6), col = "red", lty = 2, lwd = 1.5)
legend("topright", legend = names(GROUP_COLOURS),
       fill = GROUP_COLOURS, bty = "n", cex = 0.85)

cat("\nLibrary size summary (millions of reads):\n")
print(summary(library_sizes / 1e6))

# Flag samples with library size < 50% of mean
low_lib_flag <- library_sizes < 0.5 * mean(library_sizes)
if (any(low_lib_flag)) {
  cat("WARNING: Samples with low library size (< 50% of mean):\n")
  print(names(library_sizes)[low_lib_flag])
} else {
  cat("All samples pass library size threshold.\n")
}

# ---- 6B. Detected miRNA Count ----
detected_per_sample <- colSums(count_matrix > 0)

par(mar = c(8, 5, 4, 2))
barplot(
  detected_per_sample,
  main      = "Detected miRNAs per Sample (count > 0)",
  ylab      = "Number of miRNAs with ≥ 1 read",
  col       = group_colours_vec,
  las       = 2,
  cex.names = 0.55,
  border    = NA
)

cat("\nDetected miRNA count per sample:\n")
print(summary(detected_per_sample))

# ---- 6C. Count Distribution (log2-transformed for visualisation) ----
log_counts <- log2(count_matrix + 0.5)   # 0.5 pseudocount handles zeros

par(mar = c(8, 5, 4, 2))
boxplot(
  log_counts,
  main     = "Raw Count Distribution — log2(count + 0.5)",
  ylab     = "log2(count + 0.5)",
  col      = group_colours_vec,
  las      = 2,
  cex.axis = 0.55,
  outline  = FALSE
)

# Boxes should look roughly similar in height and spread.
# A box far below all others = failed sample.
# Wide spread = high technical variation → normalization will address this.


# ==============================================================================
# SECTION 7: Low-Count Filtering (RNA-seq)
# ==============================================================================
# miRNAs with very low counts across most samples contribute only noise.
# edgeR's filterByExpr() applies a biologically motivated filter:
# it keeps features with enough counts in the smallest experimental group.
# This avoids filtering out miRNAs that are group-specific.

cat("miRNAs before filtering: ", nrow(count_matrix), "\n")

keep <- filterByExpr(
  count_matrix,
  group          = metadata$group,
  min.count      = 10,     # minimum 10 reads in smallest group
  min.total.count = 15     # minimum 15 reads across all samples
)

count_filtered <- count_matrix[keep, ]
cat("miRNAs after filtering:  ", nrow(count_filtered), "\n")
cat("Removed:                 ", sum(!keep), "low-count miRNAs\n")

# RULE: Never filter based on differential expression (p-value or fold change).
# Filtering must be based on expression level and detection rate only.
# Filtering by DE would introduce selection bias that inflates false positives.


# ==============================================================================
# SECTION 8: RNA-seq Normalization
# ==============================================================================
# We use two complementary methods:
#   TMM (edgeR)   — for differential expression in Week 4
#   VST (DESeq2)  — for visualization, PCA, and ML feature engineering

# ---- 8A. DESeq2 — Size Factor Normalization + VST ----
# DESeq2's median-of-ratios method estimates a size factor per sample.
# VST (Variance Stabilizing Transformation) then produces log-scale values
# suitable for linear methods (PCA, clustering, logistic regression).

dds <- DESeqDataSetFromMatrix(
  countData = count_filtered,
  colData   = metadata,
  design    = ~ group          # adjust to ~ sex + age + group when covariates available
)

# Relevel so Control is the reference group in all comparisons
dds$group <- relevel(dds$group, ref = "Control")

# Estimate size factors (normalisation coefficients)
dds <- estimateSizeFactors(dds)
cat("\nDESeq2 size factors (should be close to 1.0 for most samples):\n")
print(round(sizeFactors(dds), 3))

# VST: for visualization and ML input (blind=TRUE = no design info used → unbiased QC)
vst_data <- vst(dds, blind = TRUE)
expr_vst  <- assay(vst_data)

# For very small N (< 30 samples per group), use rlog instead of VST:
# rlog_data <- rlog(dds, blind = TRUE)
# expr_rlog  <- assay(rlog_data)

cat("\nVST-transformed expression matrix dimensions:", dim(expr_vst), "\n")
cat("Value range after VST:", round(range(expr_vst), 2), "\n")
# Expected: roughly 0–15 for serum miRNA-seq

# Post-normalization box plot
par(mar = c(8, 5, 4, 2))
boxplot(
  expr_vst,
  main     = "Post-VST Expression Distribution",
  ylab     = "VST-transformed expression",
  col      = group_colours_vec,
  las      = 2,
  cex.axis = 0.55,
  outline  = FALSE
)
# Boxes should now be aligned. Any persistent outlier box warrants investigation.

# ---- 8B. TMM Normalization (edgeR) — for DE in Week 4 ----
dge <- DGEList(counts = count_filtered, group = metadata$group)
dge <- calcNormFactors(dge, method = "TMM")

cat("\nTMM normalization factors:\n")
print(round(dge$samples$norm.factors, 3))
# Most values should be 0.9–1.1. Extreme values (< 0.7 or > 1.4) flag problematic samples.

# Extract log2-CPM values (for visualisation — not used in DE model directly)
cpm_tmm <- cpm(dge, normalized.lib.sizes = TRUE, log = TRUE, prior.count = 0.5)


# ==============================================================================
# SECTION 9: Sample-to-Sample Correlation Heatmap
# ==============================================================================
# After normalization, pairwise Pearson correlations between all samples
# should be high (> 0.90) and cluster by biological group.
# Any sample with uniformly low correlation to all others is a QC failure.

cor_matrix <- cor(expr_vst, method = "pearson")

# Annotation sidebar showing group and sex
annotation_col <- data.frame(
  Group = metadata$group,
  row.names = colnames(cor_matrix)
)
if ("sex" %in% colnames(metadata)) {
  annotation_col$Sex <- metadata$sex
}

ann_colours <- list(Group = GROUP_COLOURS)

pheatmap(
  cor_matrix,
  annotation_col  = annotation_col,
  annotation_colors = ann_colours,
  color           = colorRampPalette(rev(brewer.pal(9, "RdBu")))(100),
  breaks          = seq(0.80, 1.0, length.out = 101),
  main            = "Sample-to-Sample Pearson Correlation (post-VST)",
  fontsize_row    = 5,
  fontsize_col    = 5,
  show_rownames   = FALSE,
  filename        = "qc_reports/correlation_heatmap_GSE120584.png",
  width           = 10,
  height          = 8
)
cat("Correlation heatmap saved to qc_reports/correlation_heatmap_GSE120584.png\n")


# ==============================================================================
# SECTION 10: PCA — Visualise Data Structure and Batch Effects
# ==============================================================================
# PCA reduces the expression matrix to 2 dimensions so we can see:
#   - Whether samples cluster by disease group (expected biological signal)
#   - Whether samples cluster by batch or other technical variable (batch effect)

pca_result  <- prcomp(t(expr_vst), scale. = TRUE)
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
  geom_point(size = 3, alpha = 0.85) +
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
#   PC1 separates by batch/date      → batch effect dominates — correct it (Section 12)
#   Outlier sample far from cluster  → failed QC — consider exclusion
#   Random scatter with no grouping  → very noisy data or genuine group similarity


# ==============================================================================
# SECTION 11: Hemolysis Detection (Blood-Based miRNA Specific)
# ==============================================================================
# Red blood cell lysis releases miR-451a and miR-23a-3p into serum/plasma,
# contaminating the circulating miRNA profile.
# The log2 ratio miR-451a / miR-23a-3p serves as a hemolysis index.
# Samples above the threshold should be flagged and excluded if possible.
#
# Reference: Murray et al. (2018) Cancer Epidemiol Biomarkers Prev
# DOI: 10.1158/1055-9965.EPI-17-0657

mir451a_row  <- grep("miR-451a$|hsa-miR-451a", rownames(expr_vst), value = TRUE)[1]
mir23a_row   <- grep("miR-23a-3p|hsa-miR-23a-3p", rownames(expr_vst), value = TRUE)[1]

if (!is.na(mir451a_row) && !is.na(mir23a_row)) {
  hemolysis_index <- expr_vst[mir451a_row, ] - expr_vst[mir23a_row, ]
  # In log2 space: subtraction of values = division of raw counts

  metadata$hemolysis_index <- as.numeric(hemolysis_index)
  metadata$hemolyzed       <- metadata$hemolysis_index > 7
  # Threshold of 7 (in log2-VST space) is a guide; adjust based on platform

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
      title   = "Hemolysis Index (miR-451a – miR-23a-3p, log2 VST)",
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
  cat("Detected miRNA names (first 20):\n")
  print(head(rownames(expr_vst), 20))
  metadata$hemolyzed <- FALSE
}


# ==============================================================================
# SECTION 12: Sample QC Decisions Log
# ==============================================================================
# Document every exclusion decision with the reason.
# You will need this for your methods section and peer review responses.

qc_log <- data.frame(
  sample         = metadata$geo_accession,
  group          = metadata$group,
  library_size   = colSums(count_matrix[, metadata$geo_accession]),
  detected_miRNAs = colSums(count_matrix[, metadata$geo_accession] > 0),
  hemolysis_index = metadata$hemolysis_index %||% NA_real_,
  low_library    = library_sizes[metadata$geo_accession] < 0.5 * mean(library_sizes),
  hemolyzed      = metadata$hemolyzed,
  pass_qc        = TRUE,
  exclude_reason = "",
  stringsAsFactors = FALSE
)

# Flag failures
qc_log$pass_qc[qc_log$low_library] <- FALSE
qc_log$exclude_reason[qc_log$low_library] <- "Library size < 50% of mean"

qc_log$pass_qc[qc_log$hemolyzed & qc_log$pass_qc] <- FALSE
qc_log$exclude_reason[qc_log$hemolyzed & qc_log$exclude_reason == ""] <- "Hemolysis index > 7"

cat("\n=== QC Decision Summary ===\n")
print(table(qc_log$pass_qc, qc_log$group))
cat("\nFailed samples:\n")
print(qc_log[!qc_log$pass_qc, c("sample", "group", "exclude_reason")])

# Save QC log
write.csv(qc_log, "qc_reports/sample_qc_decisions_GSE120584.csv", row.names = FALSE)
cat("QC log saved to qc_reports/sample_qc_decisions_GSE120584.csv\n")

# Helper function for the %||% operator used above (NULL coalescing)
`%||%` <- function(a, b) if (!is.null(a)) a else b

# Apply exclusions
passing <- qc_log$sample[qc_log$pass_qc]
count_filtered_qc <- count_filtered[, passing]
metadata_qc       <- metadata[metadata$geo_accession %in% passing, ]
expr_vst_qc       <- expr_vst[, passing]

cat("\nSamples remaining after QC:\n")
print(table(metadata_qc$group))


# ==============================================================================
# SECTION 13: Batch Effect Detection and Correction
# ==============================================================================
# Batch effects are systematic technical differences between groups of samples
# processed at different times or locations. They can masquerade as biology.
#
# This section shows:
#   A. How to detect batch effects with PCA and RLE
#   B. How to correct with ComBat (recommended for RNA-seq)
#   C. How to correct with limma::removeBatchEffect (for microarray)
#
# NOTE: If your dataset has no batch variable in its metadata, skip to 13C
# which uses SVA to estimate hidden batch variables.

# --- Check if batch information exists in your metadata ---
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
# Add batch to PCA plot
pca_df_qc <- data.frame(
  PC1   = pca_result$x[passing, 1],
  PC2   = pca_result$x[passing, 2],
  Group = metadata_qc$group,
  Batch = metadata_qc$batch
)

p_batch <- ggplot(pca_df_qc, aes(x = PC1, y = PC2, colour = Group, shape = Batch)) +
  geom_point(size = 3, alpha = 0.85) +
  scale_colour_manual(values = GROUP_COLOURS) +
  labs(
    title = "PCA: Check for Batch Effects",
    x     = paste0("PC1 (", round(var_exp[1], 1), "% variance)"),
    y     = paste0("PC2 (", round(var_exp[2], 1), "% variance)")
  ) +
  theme_bw(base_size = 12)

print(p_batch)

# ---- 13B. ComBat Batch Correction (recommended for RNA-seq) ----
# ComBat adjusts batch-specific mean and variance per miRNA using empirical Bayes.
# The mod matrix tells ComBat what biological signal to PRESERVE.

if (nlevels(metadata_qc$batch) > 1) {

  mod  <- model.matrix(~ group, data = metadata_qc)  # protect disease group
  mod0 <- model.matrix(~ 1,     data = metadata_qc)  # null model

  expr_combat <- ComBat(
    dat        = expr_vst_qc,       # VST-normalised expression matrix
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
    geom_point(size = 3, alpha = 0.85) +
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
  expr_final <- expr_vst_qc
}

# ---- 13C. SVA — Estimate Hidden Batch Variables ----
# Use this when batch metadata is missing but PCA shows unexplained structure.

# UNCOMMENT the block below if needed:
# dds_qc <- DESeqDataSetFromMatrix(
#   countData = count_filtered_qc,
#   colData   = metadata_qc,
#   design    = ~ group
# )
# dds_qc <- estimateSizeFactors(dds_qc)
# vst_qc  <- vst(dds_qc, blind = FALSE)
# expr_for_sva <- assay(vst_qc)
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
# SECTION 14: RLE Plot — Post-Normalization Quality Check
# ==============================================================================
# RLE (Relative Log Expression) should show boxes centred at zero
# with narrow, similar IQR across all samples after good normalization.

row_medians <- apply(expr_final, 1, median)
rle_matrix  <- sweep(expr_final, 1, row_medians, "-")

par(mar = c(8, 5, 4, 2))
boxplot(
  rle_matrix,
  main     = "RLE Plot — Post-Normalization (GSE120584)",
  ylab     = "Relative Log Expression",
  col      = GROUP_COLOURS[as.character(metadata_qc$group)],
  las      = 2,
  cex.axis = 0.5,
  ylim     = c(-2, 2),
  outline  = FALSE
)
abline(h = 0, col = "red", lty = 2, lwd = 1.5)

# What to look for:
#   Median of each box near 0   → normalization worked
#   Box width (IQR) similar     → comparable technical quality across samples
#   One box with large IQR      → possible failed/degraded sample
#   Systematic offset by group  → possible over-normalization; check biology


# ==============================================================================
# SECTION 15: Microarray Pipeline — GSE46579 (Validation Dataset)
# ==============================================================================
# GSE46579: Whole blood Affymetrix microarray, AD vs Control.
# We preprocess this independently for use as an external validation set (Week 5).

cat("\n--- Downloading GSE46579 (validation dataset) ---\n")

gse46579_list <- getGEO(
  "GSE46579",
  destdir   = "data/raw/",
  GSEMatrix = TRUE,
  AnnotGPL  = TRUE
)
gse46579 <- gse46579_list[[1]]

metadata_46 <- pData(gse46579)
cat("Metadata columns for GSE46579:\n")
print(colnames(metadata_46))

# Parse group labels (inspect unique values first and adjust regex accordingly)
unique_chars <- unique(metadata_46$characteristics_ch1)
cat("Unique characteristics_ch1 values:\n")
print(unique_chars)

metadata_46$group <- gsub(".*: ", "", metadata_46$characteristics_ch1)
metadata_46$group <- trimws(metadata_46$group)
cat("\nGroup distribution in GSE46579:\n")
print(table(metadata_46$group))

# ---- Download CEL files ----
cat("Downloading raw CEL files for GSE46579...\n")
getGEOSuppFiles("GSE46579",
                makeDirectory = TRUE,
                baseDir       = "data/raw/")

# Unpack the .tar archive
tar_file <- list.files("data/raw/GSE46579/",
                       pattern = "\\.tar$",
                       full.names = TRUE)[1]

if (!is.na(tar_file)) {
  untar(tar_file, exdir = "data/raw/GSE46579/CEL_files/")
  cat("CEL files extracted.\n")
}

# List CEL files
cel_files <- list.files(
  "data/raw/GSE46579/CEL_files/",
  pattern    = "\\.CEL(\\.gz)?$",
  full.names = TRUE
)
cat("Number of CEL files:", length(cel_files), "\n")


# ==============================================================================
# SECTION 16: Load and QC Affymetrix Microarray Data
# ==============================================================================
# CEL files store raw probe-level fluorescence intensities.
# We use oligo (for modern miRNA arrays) or affy (for older 3' IVT arrays).

cat("Loading CEL files into R...\n")
raw_affy <- tryCatch(
  read.celfiles(cel_files),   # oligo package
  error = function(e) {
    cat("oligo::read.celfiles failed:", conditionMessage(e), "\n")
    cat("Trying affy::ReadAffy...\n")
    ReadAffy(filenames = cel_files)   # affy package fallback
  }
)

cat("CEL data loaded. Dimensions (probes × samples):", dim(exprs(raw_affy)), "\n")

# ---- 16A. Raw Intensity Box Plot ----
par(mar = c(8, 5, 4, 2))
boxplot(
  raw_affy,
  main     = "Raw Probe Intensities (GSE46579)",
  col      = c("#4575B4", "#D73027")[as.numeric(factor(metadata_46$group))],
  las      = 2,
  cex.axis = 0.6,
  ylab     = "log2 Intensity"
)
# All boxes should have similar heights. Major differences = normalization required.

# ---- 16B. Manual RLE on Raw Data ----
log_raw    <- log2(exprs(raw_affy) + 1)
row_med_raw <- apply(log_raw, 1, median)
rle_raw    <- sweep(log_raw, 1, row_med_raw, "-")

par(mar = c(8, 5, 4, 2))
boxplot(
  rle_raw,
  main     = "RLE Plot — Raw Microarray Data (GSE46579)",
  col      = c("#4575B4", "#D73027")[as.numeric(factor(metadata_46$group))],
  las      = 2,
  cex.axis = 0.6,
  ylab     = "RLE",
  ylim     = c(-2, 2),
  outline  = FALSE
)
abline(h = 0, col = "red", lty = 2, lwd = 1.5)

# ---- 16C. Sample-to-Sample Correlation Heatmap (Microarray) ----
cor_raw <- cor(log_raw, method = "pearson")

ann_46  <- data.frame(Group = metadata_46$group,
                      row.names = colnames(cor_raw))

pheatmap(
  cor_raw,
  annotation_col = ann_46,
  color          = colorRampPalette(rev(brewer.pal(9, "RdBu")))(100),
  breaks         = seq(0.85, 1.0, length.out = 101),
  main           = "Sample Correlation — Raw Microarray (GSE46579)",
  fontsize_row   = 6,
  fontsize_col   = 6,
  show_rownames  = FALSE,
  filename       = "qc_reports/correlation_heatmap_GSE46579_raw.png",
  width = 8, height = 6
)
cat("Microarray raw correlation heatmap saved.\n")


# ==============================================================================
# SECTION 17: RMA Normalization (Microarray)
# ==============================================================================
# RMA (Robust Multi-array Average) performs three steps:
#   1. Background correction  — removes non-specific hybridization signal
#   2. Quantile normalization — forces identical distribution across all arrays
#   3. Summarization          — combines probe intensities into one value per miRNA
#
# Output is already log2-transformed. Values typically range 2–14.

cat("Applying RMA normalization to GSE46579...\n")
rma_norm <- rma(raw_affy)    # from oligo or affy package

expr_rma <- exprs(rma_norm)
cat("RMA-normalized matrix dimensions:", dim(expr_rma), "\n")
cat("Value range:", round(range(expr_rma), 2), "\n")

# Post-normalization QC
par(mar = c(8, 5, 4, 2))
boxplot(
  expr_rma,
  main     = "Post-RMA Normalized Intensities (GSE46579)",
  col      = c("#4575B4", "#D73027")[as.numeric(factor(metadata_46$group))],
  las      = 2,
  cex.axis = 0.6,
  ylab     = "RMA-normalized log2 Intensity"
)
# After RMA, all boxes should be almost identical — quantile normalization guarantees this.

# RLE post-normalization
row_med_rma <- apply(expr_rma, 1, median)
rle_rma     <- sweep(expr_rma, 1, row_med_rma, "-")

par(mar = c(8, 5, 4, 2))
boxplot(
  rle_rma,
  main     = "RLE Post-RMA Normalization (GSE46579)",
  col      = c("#4575B4", "#D73027")[as.numeric(factor(metadata_46$group))],
  las      = 2,
  cex.axis = 0.6,
  ylab     = "RLE",
  ylim     = c(-2, 2),
  outline  = FALSE
)
abline(h = 0, col = "red", lty = 2, lwd = 1.5)


# ==============================================================================
# SECTION 18: Reference-Gene Normalization (Optional — Microarray)
# ==============================================================================
# Reference-gene normalization scales each sample relative to a stably
# expressed internal control miRNA (analogous to GAPDH in RT-qPCR).
# Wang et al. (2015) showed this outperforms quantile normalization
# when global expression shifts are possible.
#
# Stable blood reference miRNAs: miR-93-5p, miR-191-5p
# Spike-in controls (best): cel-miR-39, cel-miR-54 (if present in data)

ref_candidates <- c("hsa-miR-93-5p", "hsa-miR-191-5p", "hsa-miR-16-5p")
ref_found      <- intersect(ref_candidates, rownames(expr_rma))

if (length(ref_found) > 0) {
  ref_mirna <- ref_found[1]
  cat("Using reference miRNA:", ref_mirna, "\n")

  ref_vals     <- expr_rma[ref_mirna, ]
  mean_ref     <- mean(ref_vals)
  scale_fac    <- ref_vals - mean_ref   # log2 scale: subtraction = division

  expr_rgb <- sweep(expr_rma, 2, scale_fac, "-")

  cat("Reference miRNA expression after RGB normalization (should be constant):\n")
  print(round(summary(expr_rgb[ref_mirna, ]), 3))
} else {
  cat("None of the candidate reference miRNAs found.\n")
  cat("Probe names in dataset (first 20):\n")
  print(head(rownames(expr_rma), 20))
  expr_rgb <- expr_rma   # fall back to RMA-only normalization
}


# ==============================================================================
# SECTION 19: Hemolysis Detection (Microarray)
# ==============================================================================

mir451a_ma <- grep("miR-451a|hsa.miR.451a", rownames(expr_rma), value = TRUE)[1]
mir23a_ma  <- grep("miR-23a-3p|hsa.miR.23a.3p", rownames(expr_rma), value = TRUE)[1]

if (!is.na(mir451a_ma) && !is.na(mir23a_ma)) {
  hemolysis_46          <- expr_rma[mir451a_ma, ] - expr_rma[mir23a_ma, ]
  metadata_46$hemolysis <- as.numeric(hemolysis_46)
  metadata_46$hemolyzed <- metadata_46$hemolysis > 7

  cat("Hemolyzed samples in GSE46579:\n")
  print(table(metadata_46$hemolyzed, metadata_46$group))
} else {
  cat("Hemolysis miRNAs not found in GSE46579 probes. Skipping.\n")
  metadata_46$hemolyzed <- FALSE
}


# ==============================================================================
# SECTION 20: Final Clean Data Checkpoint and Save
# ==============================================================================
# After completing QC, normalization, and batch correction, save the clean
# data objects that Week 3 will load directly. Never modify these files
# manually — all changes must go through this reproducible pipeline.

# Pre-save audit — verify everything is in order
cat("\n========================================\n")
cat("  WEEK 2 CLEAN DATA AUDIT\n")
cat("========================================\n")

cat("\n-- GSE120584 (RNA-seq, primary training set) --\n")
cat("Final expression matrix:", nrow(expr_final), "miRNAs ×",
    ncol(expr_final), "samples\n")
cat("Value range:", round(range(expr_final), 2), "\n")
cat("Group distribution:\n")
print(table(metadata_qc$group))

cat("\n-- GSE46579 (microarray, external validation set) --\n")
cat("Final expression matrix:", nrow(expr_rma), "miRNAs ×",
    ncol(expr_rma), "samples\n")
cat("Value range:", round(range(expr_rma), 2), "\n")
cat("Group distribution:\n")
print(table(metadata_46$group))

# Save objects
saveRDS(expr_final,    "data/processed/GSE120584_expr_clean.rds")
saveRDS(metadata_qc,   "data/processed/GSE120584_metadata_clean.rds")
saveRDS(count_filtered_qc, "data/processed/GSE120584_counts_filtered.rds")
saveRDS(dds,           "data/processed/GSE120584_dds.rds")     # for DESeq2 in Week 4

saveRDS(expr_rma,      "data/processed/GSE46579_expr_rma.rds")
saveRDS(metadata_46,   "data/processed/GSE46579_metadata_clean.rds")

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
