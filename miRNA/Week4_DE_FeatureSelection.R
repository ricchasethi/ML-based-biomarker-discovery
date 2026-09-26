# =============================================================================
# Week 4: Differential Expression & Feature Selection
# AI/ML in Biomarker Discovery — miRNA in Alzheimer's Disease
# =============================================================================
#
# COURSE CONTEXT:
#   This script performs the R-side work for Week 4: formal differential
#   expression (DE) analysis to rank miRNAs by statistical evidence for
#   AD-associated changes, followed by filter-based feature selection and
#   export of the feature matrix for ML classifiers (Lab 4B).
#
# TWO DATASETS, TWO DATA TYPES, TWO DE METHODS:
#   GSE120584 — serum miRNA microarray (primary dataset; ~1,300 samples)
#               Normalised log2 values → limma (linear model + empirical Bayes)
#   GSE46579  — whole-blood small RNA-seq (validation dataset; ~65 samples)
#               Raw read counts → DESeq2 (negative binomial model for counts)
#
# THREE COMPARISONS on GSE120584:
#   1. Alzheimer's Disease (AD) vs Control
#   2. Mild Cognitive Impairment (MCI) vs Control
#   3. AD vs MCI
# ONE COMPARISON on GSE46579 (it has no MCI group):
#   AD vs Control
#
# FEATURE SELECTION (R-side, univariate filter):
#   Mann-Whitney U test (Wilcoxon rank-sum) per miRNA
#   Benjamini-Hochberg FDR correction
#   Export ranked feature matrix for ML (Lab 4B)
#
# PREREQUISITES (run Weeks 2 & 3 scripts first):
#   data/processed/GSE120584_expr_clean.rds
#   data/processed/GSE120584_metadata_clean.rds
#   data/processed/GSE120584_expr_varianceFiltered.rds
#   data/processed/GSE46579_counts_filtered.rds
#   data/processed/GSE46579_metadata_clean.rds
#
# OUTPUTS (written to results/):
#   de_results_limma_AD_vs_Control.csv  (+ _MCI_vs_Control, _AD_vs_MCI)   GSE120584
#   volcano_limma_AD_vs_Control.png     (+ _MCI_vs_Control, _AD_vs_MCI)
#   ma_plot_limma_AD_vs_Control.png
#   de_results_deseq2_AD_vs_Control.csv                                    GSE46579
#   volcano_deseq2_AD_vs_Control.png
#   ma_plot_deseq2_AD_vs_Control.png
#   overlap_limma_deseq2_AD.csv
#   mwu_filter_features.csv
#   consensus_features_Week4.csv
#   Week4/DE_results_GSE120584.csv      (miRNA, log2FC, padj — read by Week 6)
#
# Exported for ML (Lab 4B):
#   data/processed/GSE120584_expr_forML.csv
#   data/processed/GSE120584_labels_binary.csv  (AD=1, Control=0)
#
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Package loading
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(limma)        # linear models for DE on log-scale (microarray) data
  library(DESeq2)       # negative binomial DE for RNA-seq count data
  library(edgeR)        # DGEList and filterByExpr helper functions
  library(ggplot2)      # publication-quality plots
  library(ggrepel)      # non-overlapping labels on volcano plots
  library(dplyr)        # data manipulation (filter, arrange, mutate)
})

# Create results directories if they do not exist
dir.create("results/Week4", recursive = TRUE, showWarnings = FALSE)

# Shared colour palette
GROUP_COLOURS <- c(
  "Control"                   = "#4575B4",
  "Mild Cognitive Impairment" = "#FEE090",
  "Alzheimer's Disease"       = "#D73027"
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────

cat("\n====================================================\n")
cat("  Week 4 — Differential Expression & Feature Selection\n")
cat("====================================================\n\n")

# GSE120584: microarray — normalised log2 matrix for limma
expr_clean <- readRDS("data/processed/GSE120584_expr_clean.rds")
metadata   <- readRDS("data/processed/GSE120584_metadata_clean.rds")
expr_vf    <- readRDS("data/processed/GSE120584_expr_varianceFiltered.rds")

cat("GSE120584 (microarray):\n")
cat("  Normalised log2 matrix:", nrow(expr_clean), "miRNAs ×",
    ncol(expr_clean), "samples\n")
cat("  Variance-filtered matrix (Week 3):", nrow(expr_vf), "miRNAs ×",
    ncol(expr_vf), "samples\n")
cat("  Groups:\n")
print(table(metadata$group))
stopifnot(all(colnames(expr_clean) == metadata$geo_accession))

# GSE46579: RNA-seq — raw filtered counts for DESeq2
rnaseq_available <- file.exists("data/processed/GSE46579_counts_filtered.rds") &&
                    file.exists("data/processed/GSE46579_metadata_clean.rds")

if (rnaseq_available) {
  counts_46 <- readRDS("data/processed/GSE46579_counts_filtered.rds")
  meta_46   <- readRDS("data/processed/GSE46579_metadata_clean.rds")
  cat("\nGSE46579 (small RNA-seq):\n")
  cat("  Filtered count matrix:", nrow(counts_46), "miRNAs ×",
      ncol(counts_46), "samples\n")
  cat("  Groups:\n")
  print(table(meta_46$group))
  stopifnot(all(colnames(counts_46) == meta_46$geo_accession))
} else {
  cat("\nGSE46579 not found — skipping DESeq2 validation analysis.\n")
  cat("  (Run Week 2 script first to generate GSE46579 processed files)\n")
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — limma setup for GSE120584: design matrix
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY limma FOR MICROARRAY:
#   Microarray values are continuous log2 intensities (already normalised in
#   Week 2), so a linear model is appropriate. limma fits one linear model per
#   miRNA and then "borrows strength" across all miRNAs (empirical Bayes) to
#   stabilise each miRNA's variance estimate.
#
# DESIGN MATRIX:
#   ~ 0 + group + sex + age
#   "0 +" gives one column per group (Control, MCI, AD) so comparisons can be
#   written as simple differences, e.g. AD − Control.
#   sex and age are covariates. Week 3 (Section 12) showed that age explains
#   less of PC1 than disease does (partial R² 0.6% vs 3.4%) but more of PC2
#   (1.8% vs 0.2%), and correlates with PC1–PC3 (|r| ≈ 0.17). Because controls
#   are also younger than AD patients (~72 vs ~79 years), age is a confounder
#   and must be adjusted for. Sex is cheap to include (one column).
#
# Group labels contain spaces and an apostrophe, which are awkward in
#   formulas, so we use the short names Control / MCI / AD inside the design.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 2: limma setup (GSE120584) ─────────────────────────────────\n\n")

short_names <- c("Control"                   = "Control",
                 "Mild Cognitive Impairment" = "MCI",
                 "Alzheimer's Disease"       = "AD")
metadata$group_short <- factor(short_names[as.character(metadata$group)],
                               levels = c("Control", "MCI", "AD"))

if (all(c("age", "sex") %in% colnames(metadata)) &&
    !anyNA(metadata$age) && !anyNA(metadata$sex)) {
  design <- model.matrix(~ 0 + group_short + sex + age, data = metadata)
  cat("Using design: ~ 0 + group + sex + age (covariates included)\n")
} else {
  design <- model.matrix(~ 0 + group_short, data = metadata)
  cat("Using design: ~ 0 + group (no covariates available)\n")
}
colnames(design) <- sub("^group_short", "", colnames(design))

cat("Design matrix:", nrow(design), "samples ×", ncol(design), "columns\n")
print(colnames(design))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Run limma
# ─────────────────────────────────────────────────────────────────────────────
#
# Three steps:
#   (1) lmFit          — fit the linear model to every miRNA
#   (2) contrasts.fit  — compute the three group comparisons
#   (3) eBayes         — moderate the variance estimates across miRNAs
#                        (trend = TRUE: allow variance to depend on expression
#                        level, as it does on arrays)
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 3: Running limma ───────────────────────────────────────────\n\n")

fit <- lmFit(expr_clean, design)

contrast_matrix <- makeContrasts(
  AD_vs_Control  = AD  - Control,
  MCI_vs_Control = MCI - Control,
  AD_vs_MCI      = AD  - MCI,
  levels = design
)
fit2 <- contrasts.fit(fit, contrast_matrix)
fit2 <- eBayes(fit2, trend = TRUE)

cat("limma fit complete. Contrasts:\n")
print(colnames(contrast_matrix))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Extract DE results (GSE120584)
# ─────────────────────────────────────────────────────────────────────────────
#
# topTable() returns, per miRNA:
#   logFC      — log2 fold change (difference of group means on the log2 scale)
#   AveExpr    — average log2 expression across all samples
#   t, P.Value — moderated t-statistic and raw p-value
#   adj.P.Val  — Benjamini-Hochberg FDR
#
# Significance: FDR < 0.05 AND |log2FC| > 0.3 (≈ 1.23-fold change).
#
# WHY 0.3 HERE, BUT 0.5 FOR RNA-seq (Section 6)?
#   Microarray fold changes are compressed: background signal and a limited
#   dynamic range shrink differences between groups. In GSE120584 hundreds of
#   miRNAs reach FDR < 0.05, yet none changes by more than ~1.35-fold, so the
#   usual RNA-seq cut-off of 0.5 would keep nothing. At the same time, with
#   >1,000 samples even tiny differences become "significant", so a
#   fold-change cut-off is still needed to keep the list meaningful.
#   Always choose the cut-off with the platform in mind — and report it.
#
# CAUTION when reading the top hits: many are low-abundance miRNAs sitting
#   just above background, all higher in AD. Week 2 (Section 6C) showed that
#   AD samples detect more miRNAs than Controls — a possible technical
#   difference. Treat such hits with care until validated in GSE46579.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 4: Extracting DE results ───────────────────────────────────\n\n")

get_limma_table <- function(fit_obj, coef_name) {
  tab <- topTable(fit_obj, coef = coef_name, number = Inf,
                  adjust.method = "BH", sort.by = "P")
  tab$miRNA <- rownames(tab)
  tab[, c("miRNA", setdiff(colnames(tab), "miRNA"))]
}

lfc_cut <- 0.3   # |log2 fold change| cut-off for this microarray dataset

is_sig <- function(tab) {
  tab[tab$adj.P.Val < 0.05 & abs(tab$logFC) > lfc_cut, ]
}

# ── 4a: AD vs Control ────────────────────────────────────────────────────
res_AD_df <- get_limma_table(fit2, "AD_vs_Control")
sig_AD    <- is_sig(res_AD_df)

cat("=== AD vs Control (limma, GSE120584) ===\n")
cat("Total miRNAs tested:", nrow(res_AD_df), "\n")
cat("Significant (FDR < 0.05, |log2FC| > 0.3):", nrow(sig_AD), "\n")
cat("  Upregulated in AD:   ", sum(sig_AD$logFC > 0), "\n")
cat("  Downregulated in AD: ", sum(sig_AD$logFC < 0), "\n")
cat("\nTop 15 DE miRNAs (AD vs Control):\n")
print(head(sig_AD[, c("miRNA", "logFC", "AveExpr", "P.Value", "adj.P.Val")], 15),
      row.names = FALSE)

write.csv(res_AD_df, "results/de_results_limma_AD_vs_Control.csv", row.names = FALSE)

# Simplified table read by Week6_Interpretation.R (columns: miRNA, log2FC, padj)
write.csv(
  data.frame(miRNA  = res_AD_df$miRNA,
             log2FC = res_AD_df$logFC,
             padj   = res_AD_df$adj.P.Val,
             pvalue = res_AD_df$P.Value,
             AveExpr = res_AD_df$AveExpr),
  "results/Week4/DE_results_GSE120584.csv", row.names = FALSE
)

# ── 4b: MCI vs Control ───────────────────────────────────────────────────
res_MCI_df <- get_limma_table(fit2, "MCI_vs_Control")
sig_MCI    <- is_sig(res_MCI_df)

cat("\n=== MCI vs Control ===\n")
cat("Significant (FDR < 0.05, |log2FC| > 0.3):", nrow(sig_MCI), "\n")
cat("  (Only ~32 MCI samples: expect less power than for AD vs Control)\n")
write.csv(res_MCI_df, "results/de_results_limma_MCI_vs_Control.csv", row.names = FALSE)

# ── 4c: AD vs MCI ────────────────────────────────────────────────────────
res_AD_MCI_df <- get_limma_table(fit2, "AD_vs_MCI")
sig_AD_MCI    <- is_sig(res_AD_MCI_df)

cat("\n=== AD vs MCI ===\n")
cat("Significant (FDR < 0.05, |log2FC| > 0.3):", nrow(sig_AD_MCI), "\n")
write.csv(res_AD_MCI_df, "results/de_results_limma_AD_vs_MCI.csv", row.names = FALSE)

# Three-way summary
cat("\n=== Three-Way DE Summary ===\n")
cat("  MCI vs Control (early-detection candidates): ", nrow(sig_MCI), "\n")
cat("  AD vs Control  (disease-stage markers):      ", nrow(sig_AD), "\n")
cat("  AD vs MCI      (progression markers):        ", nrow(sig_AD_MCI), "\n")

# Progressive markers: same direction in both MCI vs Ctrl AND AD vs Ctrl
if (nrow(sig_MCI) > 0 && nrow(sig_AD) > 0) {
  common_both <- intersect(sig_MCI$miRNA, sig_AD$miRNA)
  mci_dir     <- sign(sig_MCI$logFC[match(common_both, sig_MCI$miRNA)])
  ad_dir      <- sign(sig_AD$logFC[match(common_both, sig_AD$miRNA)])
  progressive <- common_both[mci_dir == ad_dir]
  cat("\n  Progressive (same-direction DE in both MCI and AD):", length(progressive), "\n")
  if (length(progressive) > 0) {
    cat("  Progressive markers:", paste(progressive, collapse = ", "), "\n")
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Volcano and MA plots (limma, GSE120584)
# ─────────────────────────────────────────────────────────────────────────────
#
# VOLCANO PLOT: log2FC (x-axis) vs -log10(p-value) (y-axis)
#   Points in upper-right: strongly upregulated AND highly significant
#   Points in upper-left: strongly downregulated AND highly significant
#   Key convention: colour by significance + direction; label top 15 by FDR
#
# MA PLOT: average expression (A) vs log2FC (M)
#   A well-normalised dataset shows the cloud of points centred at M = 0
#   across all expression levels.  A systematic trend at low expression
#   indicates a normalisation artefact.
#
# make_volcano() works for both limma and DESeq2 tables: you tell it which
#   columns hold the fold change, p-value and FDR.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 5: Volcano and MA plots (limma) ─────────────────────────────\n\n")

make_volcano <- function(de_df, lfc_col, p_col, fdr_col, title, outfile,
                         case = "AD", reference = "Control", lfc_cut = 0.5) {
  up_lab   <- paste("Up in", case)
  down_lab <- paste("Down in", case)

  plot_df <- de_df[!is.na(de_df[[p_col]]), ]
  plot_df$significance <- "Not Significant"
  plot_df$significance[!is.na(plot_df[[fdr_col]]) & plot_df[[fdr_col]] < 0.05 &
                       plot_df[[lfc_col]] >  lfc_cut] <- up_lab
  plot_df$significance[!is.na(plot_df[[fdr_col]]) & plot_df[[fdr_col]] < 0.05 &
                       plot_df[[lfc_col]] < -lfc_cut] <- down_lab
  plot_df$significance <- factor(plot_df$significance,
                                 levels = c("Not Significant", up_lab, down_lab))

  # Top 15 for labelling (by smallest FDR)
  label_df <- head(plot_df[order(plot_df[[fdr_col]]), ], 15)

  p <- ggplot(plot_df,
              aes(x = .data[[lfc_col]],
                  y = -log10(.data[[p_col]] + 1e-300),
                  colour = significance)) +
    geom_point(alpha = 0.55, size = 1.5) +
    geom_point(data = label_df, size = 2.5, alpha = 0.9) +
    geom_vline(xintercept = c(-lfc_cut, lfc_cut), linetype = "dashed",
               colour = "grey40", linewidth = 0.4) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed",
               colour = "grey40", linewidth = 0.4) +
    geom_text_repel(data = label_df,
                    aes(label = miRNA), size = 2.8,
                    max.overlaps = 20, box.padding = 0.4,
                    colour = "black") +
    scale_colour_manual(
      values = setNames(c("grey70", "#D73027", "#4575B4"),
                        c("Not Significant", up_lab, down_lab)),
      drop = FALSE) +
    labs(title   = title,
         x       = paste0("log2 Fold Change (", case, " / ", reference, ")"),
         y       = "-log10(p-value)",
         colour  = NULL,
         caption = paste0("Dashed lines: |log2FC| > ", lfc_cut,
                          " and p < 0.05 (uncorrected)")) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold"), legend.position = "top")

  ggsave(outfile, p, width = 8, height = 6, dpi = 150)
  cat("Saved:", outfile, "\n")
  invisible(p)
}

make_volcano(res_AD_df, "logFC", "P.Value", "adj.P.Val",
             title   = "Volcano Plot: AD vs Control (limma, GSE120584)",
             outfile = "results/volcano_limma_AD_vs_Control.png",
             lfc_cut = lfc_cut)

make_volcano(res_MCI_df, "logFC", "P.Value", "adj.P.Val",
             title   = "Volcano Plot: MCI vs Control (limma, GSE120584)",
             outfile = "results/volcano_limma_MCI_vs_Control.png",
             case    = "MCI", lfc_cut = lfc_cut)

make_volcano(res_AD_MCI_df, "logFC", "P.Value", "adj.P.Val",
             title     = "Volcano Plot: AD vs MCI (limma, GSE120584)",
             outfile   = "results/volcano_limma_AD_vs_MCI.png",
             reference = "MCI", lfc_cut = lfc_cut)

# MA plot for AD vs Control
make_ma_plot <- function(de_df, a_values, lfc_col, fdr_col, title, xlab, outfile,
                         lfc_cut = 0.5) {
  sig_class <- ifelse(!is.na(de_df[[fdr_col]]) & de_df[[fdr_col]] < 0.05 &
                        abs(de_df[[lfc_col]]) > lfc_cut,
                      ifelse(de_df[[lfc_col]] > 0, "Up in AD", "Down in AD"),
                      "Not Significant")
  plot_df <- data.frame(A = a_values, M = de_df[[lfc_col]],
                        significance = factor(sig_class,
                          levels = c("Not Significant", "Up in AD", "Down in AD")))

  p <- ggplot(plot_df, aes(x = A, y = M, colour = significance)) +
    geom_point(alpha = 0.5, size = 1.2) +
    geom_hline(yintercept = 0, colour = "black", linewidth = 0.5) +
    geom_hline(yintercept = c(-lfc_cut, lfc_cut), linetype = "dashed",
               colour = "grey40", linewidth = 0.4) +
    scale_colour_manual(values = c("Not Significant" = "grey70",
                                   "Up in AD"        = "#D73027",
                                   "Down in AD"      = "#4575B4"),
                        drop = FALSE) +
    labs(title   = title,
         x       = xlab,
         y       = "log2 Fold Change (AD/Control)  [M]",
         colour  = NULL,
         caption = "Centred cloud at M=0 across all A values = good normalisation") +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold"), legend.position = "top")

  ggsave(outfile, p, width = 8, height = 5, dpi = 150)
  cat("Saved:", outfile, "\n")
  invisible(p)
}

make_ma_plot(res_AD_df, res_AD_df$AveExpr, "logFC", "adj.P.Val",
             title   = "MA Plot: AD vs Control (limma, GSE120584)",
             xlab    = "Average log2 Expression  [A]",
             outfile = "results/ma_plot_limma_AD_vs_Control.png",
             lfc_cut = lfc_cut)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — DESeq2 pipeline for GSE46579 (small RNA-seq validation)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY DESeq2 FOR RNA-seq:
#   Raw counts follow a negative binomial distribution (variance > mean due
#   to overdispersion from biological variation between samples).
#   DESeq2 models this explicitly, unlike a simple t-test which assumes
#   normality. It needs RAW counts — never give it VST or log values.
#
# DESIGN FORMULA:
#   ~ sex + age + group   (age is centred and scaled — DESeq2 fits more
#                          reliably when numeric covariates are on a small scale)
# RELEVEL: Control is the reference so the fold change is AD / Control.
#
# WHY SHRINKAGE (lfcShrink):
#   Low-count miRNAs have very noisy fold change estimates.  A miRNA with
#   1 count in one group and 3 counts in another looks like a 3-fold change,
#   but this is almost certainly noise.  lfcShrink pulls extreme fold changes
#   from noisy features toward zero while preserving reliable estimates from
#   well-detected features.
#   type = "apeglm" is best for a single coefficient (AD vs Control).
# ─────────────────────────────────────────────────────────────────────────────

if (rnaseq_available) {
  cat("\n─── Section 6: DESeq2 pipeline (GSE46579 RNA-seq) ──────────────────────\n\n")

  meta_46$group <- factor(meta_46$group, levels = c("Control", "Alzheimer's Disease"))

  if (all(c("age", "sex") %in% colnames(meta_46)) &&
      !anyNA(meta_46$age) && !anyNA(meta_46$sex)) {
    meta_46$age_scaled <- as.numeric(scale(meta_46$age))
    meta_46$sex        <- factor(meta_46$sex)
    design_46 <- ~ sex + age_scaled + group
    cat("Using design: ~ sex + age + group (covariates included)\n")
  } else {
    design_46 <- ~ group
    cat("Using design: ~ group (no covariates available)\n")
  }

  dds_46 <- DESeqDataSetFromMatrix(
    countData = counts_46,
    colData   = meta_46,
    design    = design_46
  )
  dds_46$group <- relevel(dds_46$group, ref = "Control")

  # DESeq() runs: size factors → dispersions → negative binomial Wald tests
  dds_46 <- DESeq(dds_46)
  cat("Estimated model coefficients:\n")
  print(resultsNames(dds_46))

  coef_ad <- grep("^group_Alzheimer", resultsNames(dds_46), value = TRUE)

  # apeglm is a separate Bioconductor package; fall back to "normal" if missing
  shrink_type <- if (requireNamespace("apeglm", quietly = TRUE)) "apeglm" else "normal"
  if (shrink_type == "normal") {
    cat("NOTE: apeglm not installed — using type = 'normal' shrinkage.\n",
        "     Install with BiocManager::install('apeglm') for the recommended method.\n")
  }
  res_46 <- lfcShrink(dds_46, coef = coef_ad, type = shrink_type)

  res_46_df <- as.data.frame(res_46)
  res_46_df$miRNA <- rownames(res_46_df)
  res_46_df <- res_46_df[order(res_46_df$padj, na.last = TRUE),
                         c("miRNA", setdiff(colnames(res_46_df), "miRNA"))]

  sig_46 <- res_46_df[!is.na(res_46_df$padj) & res_46_df$padj < 0.05 &
                      abs(res_46_df$log2FoldChange) > 0.5, ]

  cat("\n=== GSE46579 DESeq2 Results: AD vs Control ===\n")
  cat("Total miRNAs tested:", nrow(res_46_df), "\n")
  cat("Significant (FDR < 0.05, |log2FC| > 0.5):", nrow(sig_46), "\n")
  cat("  Upregulated in AD:  ", sum(sig_46$log2FoldChange > 0), "\n")
  cat("  Downregulated in AD:", sum(sig_46$log2FoldChange < 0), "\n")
  cat("\nTop 15 DE miRNAs (DESeq2, GSE46579):\n")
  print(head(sig_46[, c("miRNA", "baseMean", "log2FoldChange", "lfcSE",
                        "pvalue", "padj")], 15), row.names = FALSE)

  write.csv(res_46_df, "results/de_results_deseq2_AD_vs_Control.csv", row.names = FALSE)
  cat("\nFull DESeq2 results saved to results/de_results_deseq2_AD_vs_Control.csv\n")

  make_volcano(res_46_df, "log2FoldChange", "pvalue", "padj",
               title   = "Volcano Plot: AD vs Control (DESeq2, GSE46579 RNA-seq)",
               outfile = "results/volcano_deseq2_AD_vs_Control.png")

  make_ma_plot(res_46_df, log2(res_46_df$baseMean + 1), "log2FoldChange", "padj",
               title   = "MA Plot: AD vs Control (DESeq2, GSE46579)",
               xlab    = "log2(Mean Normalised Count + 1)  [A]",
               outfile = "results/ma_plot_deseq2_AD_vs_Control.png")

} else {
  cat("\n─── Section 6: Skipped (GSE46579 files not found) ───────────────────────\n")
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Cross-dataset overlap analysis
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: miRNAs that are independently significant in BOTH the serum microarray
#   (GSE120584) and the whole-blood RNA-seq (GSE46579) dataset are far more
#   trustworthy than those significant in only one dataset. Dataset-specific
#   findings can reflect platform artefacts, cohort-specific confounding, or
#   batch effects. The overlap is the most conservative feature list.
#
# CAVEAT: here we match miRNAs by name. The two datasets use different
#   miRBase versions (GSE46579: v18), so some shared miRNAs have different
#   names and are missed. Week 5 harmonises names properly with
#   miRBaseConverter. Serum and whole blood also differ biologically, so
#   expect a modest overlap.
# ─────────────────────────────────────────────────────────────────────────────

if (rnaseq_available && exists("sig_46")) {
  cat("\n─── Section 7: Cross-dataset overlap analysis ───────────────────────────\n\n")

  shared_tested <- intersect(res_AD_df$miRNA, res_46_df$miRNA)
  cat("miRNAs tested in both datasets (exact name match):", length(shared_tested), "\n\n")

  overlap_both <- intersect(sig_AD$miRNA, sig_46$miRNA)
  only_limma   <- setdiff(sig_AD$miRNA, sig_46$miRNA)
  only_deseq2  <- setdiff(sig_46$miRNA, sig_AD$miRNA)

  cat("=== Cross-Dataset DE Overlap: AD vs Control ===\n")
  cat("limma significant  (GSE120584):", length(sig_AD$miRNA), "\n")
  cat("DESeq2 significant (GSE46579): ", length(sig_46$miRNA), "\n")
  cat("Overlap (both datasets):       ", length(overlap_both), "\n")
  cat("Only in limma:                 ", length(only_limma), "\n")
  cat("Only in DESeq2:                ", length(only_deseq2), "\n")

  if (length(overlap_both) > 0) {
    cat("\nOverlapping miRNAs:\n")
    print(overlap_both)

    # Save overlap with FC from both datasets
    overlap_df <- merge(
      sig_AD[sig_AD$miRNA %in% overlap_both, c("miRNA", "logFC", "adj.P.Val")],
      sig_46[sig_46$miRNA %in% overlap_both, c("miRNA", "log2FoldChange", "padj")],
      by = "miRNA"
    )
    colnames(overlap_df)[2:5] <- c("logFC_limma",   "padj_limma",
                                   "log2FC_DESeq2", "padj_DESeq2")
    overlap_df$direction_consistent <-
      sign(overlap_df$logFC_limma) == sign(overlap_df$log2FC_DESeq2)

    write.csv(overlap_df, "results/overlap_limma_deseq2_AD.csv", row.names = FALSE)
    cat("\nOverlap table saved to results/overlap_limma_deseq2_AD.csv\n")
    cat("Same direction in both datasets:", sum(overlap_df$direction_consistent),
        "of", nrow(overlap_df), "\n")
  }

  # Simple Venn diagram using base R text output
  cat("\n=== Venn Diagram (text) ===\n")
  cat("┌──────────────────────────────────────────────┐\n")
  cat("│ limma only  │  Both  │   DESeq2 only          │\n")
  cat(sprintf("│ %-12d│  %-5d │   %-20d│\n",
              length(only_limma), length(overlap_both), length(only_deseq2)))
  cat("└──────────────────────────────────────────────┘\n")

} else {
  cat("\n─── Section 7: Skipped (GSE46579 not available) ─────────────────────────\n")
  overlap_both <- character(0)
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Univariate filter feature selection (Mann-Whitney U in R)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: Before exporting the feature matrix for ML, we apply a fast
#   univariate filter to remove the most obviously uninformative miRNAs.
#   The Mann-Whitney U test (= Wilcoxon rank-sum test) is non-parametric:
#   it does not assume normally distributed expression values.
#
# The result: a ranked miRNA list based on p-value, used by the ML lab
#   (Lab 4B) for further processing.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 8: Mann-Whitney U filter feature selection ──────────────────\n\n")

# Use the Week 3 variance-filtered log2 matrix; restrict to AD vs Control
ad_ctrl_mask <- metadata$group %in% c("Control", "Alzheimer's Disease")
expr_bin     <- expr_vf[, metadata$geo_accession[ad_ctrl_mask]]
meta_bin     <- metadata[ad_ctrl_mask, ]

cat("Binary comparison subset: AD vs Control\n")
cat("Samples:", ncol(expr_bin), "—",
    sum(meta_bin$group == "Alzheimer's Disease"), "AD,",
    sum(meta_bin$group == "Control"), "Control\n")
cat("Running Mann-Whitney U test for each of", nrow(expr_bin), "miRNAs...\n")

# Compute Mann-Whitney U p-value for every miRNA
mw_results <- data.frame(
  miRNA  = rownames(expr_bin),
  W      = NA_real_,
  pvalue = NA_real_,
  stringsAsFactors = FALSE
)

for (i in seq_len(nrow(expr_bin))) {
  group0 <- expr_bin[i, meta_bin$group == "Control"]
  group1 <- expr_bin[i, meta_bin$group == "Alzheimer's Disease"]
  wt     <- wilcox.test(group0, group1, exact = FALSE)
  mw_results$W[i]      <- wt$statistic
  mw_results$pvalue[i] <- wt$p.value
}

# Benjamini-Hochberg FDR correction
mw_results$padj <- p.adjust(mw_results$pvalue, method = "BH")
mw_results <- mw_results[order(mw_results$pvalue), ]

# Per-miRNA mean expression in each group (for direction annotation)
mw_results$mean_AD   <- rowMeans(expr_bin[mw_results$miRNA,
                                           meta_bin$group == "Alzheimer's Disease"])
mw_results$mean_Ctrl <- rowMeans(expr_bin[mw_results$miRNA,
                                           meta_bin$group == "Control"])
mw_results$direction  <- ifelse(mw_results$mean_AD > mw_results$mean_Ctrl,
                                "Up in AD", "Down in AD")

cat("\n=== Top 20 miRNAs by Mann-Whitney U p-value ===\n")
print(head(mw_results[, c("miRNA", "pvalue", "padj", "direction")], 20), row.names = FALSE)
cat("\nMiRNAs with FDR < 0.05:", sum(mw_results$padj < 0.05, na.rm = TRUE), "\n")
cat("miRNAs with FDR < 0.20:", sum(mw_results$padj < 0.20, na.rm = TRUE), "\n")

write.csv(mw_results, "results/mwu_filter_features.csv", row.names = FALSE)
cat("Mann-Whitney U ranked feature list saved to results/mwu_filter_features.csv\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Consensus feature table (limma + MWU)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: A miRNA that appears in the top features from MULTIPLE independent
#   ranking methods is far more likely to represent a true biological signal.
#   limma adjusts for age and sex; MWU is non-parametric and assumption-light.
#   Agreement between them — and with the independent RNA-seq cohort — is the
#   starting point for Week 5's classifiers.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 9: Consensus feature table ─────────────────────────────────\n\n")

# Top 50 from each method
top50_limma <- head(res_AD_df$miRNA, 50)
top50_mwu   <- head(mw_results$miRNA, 50)

# Assign appearance counts
all_features <- unique(c(top50_limma, top50_mwu))
consensus_df <- data.frame(
  miRNA    = all_features,
  in_limma = all_features %in% top50_limma,
  in_MWU   = all_features %in% top50_mwu,
  stringsAsFactors = FALSE
)
consensus_df$n_methods <- as.integer(consensus_df$in_limma) +
                          as.integer(consensus_df$in_MWU)

# Add cross-dataset overlap flag if available
if (length(overlap_both) > 0) {
  consensus_df$in_DESeq2_overlap <- all_features %in% overlap_both
  consensus_df$n_methods <- consensus_df$n_methods +
                            as.integer(consensus_df$in_DESeq2_overlap)
}

# Add limma fold change and FDR
consensus_df <- merge(
  consensus_df,
  res_AD_df[, c("miRNA", "logFC", "adj.P.Val")],
  by = "miRNA", all.x = TRUE
)
colnames(consensus_df)[colnames(consensus_df) == "logFC"]     <- "log2FC_limma"
colnames(consensus_df)[colnames(consensus_df) == "adj.P.Val"] <- "padj_limma"

consensus_df <- consensus_df[order(-consensus_df$n_methods,
                                    consensus_df$padj_limma,
                                    na.last = TRUE), ]

cat("=== Consensus Feature Summary ===\n")
cat("In top 50 of BOTH limma and MWU:",
    sum(consensus_df$in_limma & consensus_df$in_MWU), "miRNAs\n")
cat("In top 50 of only one method:   ",
    sum(xor(consensus_df$in_limma, consensus_df$in_MWU)), "miRNAs\n")
cat("\nTop 20 consensus features (appearing in most methods):\n")
print(head(consensus_df[, c("miRNA", "n_methods", "log2FC_limma", "padj_limma",
                            "in_limma", "in_MWU")], 20), row.names = FALSE)

write.csv(consensus_df, "results/consensus_features_Week4.csv", row.names = FALSE)
cat("\nFull consensus feature table saved to results/consensus_features_Week4.csv\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Export feature matrix for ML (Lab 4B)
# ─────────────────────────────────────────────────────────────────────────────
#
# The ML lab needs:
#   (1) Feature matrix: samples × miRNAs  (CSV, miRNAs as columns)
#   (2) Sample labels: binary AD=1 / Control=0
#
# STRATEGY: Export the variance-filtered log2 matrix for the AD vs Control
#   binary subset.  The lab applies its own Mann-Whitney filter
#   (reproducing Section 8 logic) before ML training.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 10: Export feature matrix for ML ────────────────────────────\n\n")

# Transpose: ML tools expect samples as rows, features as columns
expr_forML <- as.data.frame(t(expr_bin))  # nrow = samples, ncol = miRNAs
labels_binary <- data.frame(
  sample = meta_bin$geo_accession,
  group  = meta_bin$group,
  label  = as.integer(meta_bin$group == "Alzheimer's Disease")
)

write.csv(expr_forML,    "data/processed/GSE120584_expr_forML.csv",
          row.names = TRUE)
write.csv(labels_binary, "data/processed/GSE120584_labels_binary.csv",
          row.names = FALSE)

cat("Feature matrix exported:\n")
cat("  data/processed/GSE120584_expr_forML.csv\n")
cat(sprintf("  Dimensions: %d samples × %d miRNAs\n",
            nrow(expr_forML), ncol(expr_forML)))
cat("\nBinary labels exported:\n")
cat("  data/processed/GSE120584_labels_binary.csv\n")
cat(sprintf("  AD: %d, Control: %d\n",
            sum(labels_binary$label == 1), sum(labels_binary$label == 0)))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Session summary
# ─────────────────────────────────────────────────────────────────────────────

cat("\n====================================================\n")
cat("  Week 4 — DE & Feature Selection Summary\n")
cat("====================================================\n\n")

cat("limma results (GSE120584, serum microarray):\n")
cat(sprintf("  AD vs Control  — significant: %d (up: %d, down: %d)\n",
            nrow(sig_AD), sum(sig_AD$logFC > 0), sum(sig_AD$logFC < 0)))
cat(sprintf("  MCI vs Control — significant: %d\n", nrow(sig_MCI)))
cat(sprintf("  AD vs MCI      — significant: %d\n", nrow(sig_AD_MCI)))

if (rnaseq_available && exists("sig_46")) {
  cat(sprintf("\nDESeq2 results (GSE46579, whole-blood RNA-seq):\n"))
  cat(sprintf("  AD vs Control  — significant: %d\n", nrow(sig_46)))
  cat(sprintf("  Cross-dataset overlap:        %d\n", length(overlap_both)))
}

cat(sprintf("\nMann-Whitney U filter (AD vs Control):\n"))
cat(sprintf("  FDR < 0.05: %d  |  FDR < 0.20: %d\n",
            sum(mw_results$padj < 0.05, na.rm = TRUE),
            sum(mw_results$padj < 0.20, na.rm = TRUE)))

cat(sprintf("\nConsensus features (in ≥ 2 methods): %d\n",
            sum(consensus_df$n_methods >= 2, na.rm = TRUE)))

cat("\nFiles written to results/:\n")
cat("  de_results_limma_AD_vs_Control.csv\n")
cat("  de_results_limma_MCI_vs_Control.csv\n")
cat("  de_results_limma_AD_vs_MCI.csv\n")
cat("  volcano_limma_*.png  |  ma_plot_limma_AD_vs_Control.png\n")
cat("  Week4/DE_results_GSE120584.csv   (read by Week 6)\n")
if (rnaseq_available && exists("sig_46")) {
  cat("  de_results_deseq2_AD_vs_Control.csv\n")
  cat("  volcano_deseq2_AD_vs_Control.png  |  ma_plot_deseq2_AD_vs_Control.png\n")
  if (length(overlap_both) > 0) cat("  overlap_limma_deseq2_AD.csv\n")
}
cat("  mwu_filter_features.csv\n")
cat("  consensus_features_Week4.csv\n")

cat("\nFiles for ML (Lab 4B):\n")
cat("  data/processed/GSE120584_expr_forML.csv\n")
cat("  data/processed/GSE120584_labels_binary.csv\n")

cat("\n─────────────────────────────────────────────────────\n")
cat("PROCEED TO:\n")
cat("  Week 5  — Open Week5_Validation.R in RStudio\n")
cat("─────────────────────────────────────────────────────\n\n")

sessionInfo()
